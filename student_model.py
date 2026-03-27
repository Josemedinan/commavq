from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class StudentPredictorConfig:
  vocab_size: int = 1024
  tokens_per_frame: int = 128
  context_frames: int = 8
  d_model: int = 160
  temporal_heads: int = 4
  temporal_layers: int = 2
  spatial_heads: int = 4
  spatial_layers: int = 4
  ff_mult: int = 4
  dropout: float = 0.1
  norm_first: bool = False
  adapter_rank: int = 0

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "StudentPredictorConfig":
    return cls(**data)


class StudentFramePredictor(nn.Module):
  def __init__(self, config: StudentPredictorConfig = StudentPredictorConfig()) -> None:
    super().__init__()
    self.config = config

    self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
    self.pos_embed = nn.Embedding(config.tokens_per_frame, config.d_model)
    self.frame_embed = nn.Embedding(config.context_frames, config.d_model)
    self.input_dropout = nn.Dropout(config.dropout)

    temporal_layer = nn.TransformerEncoderLayer(
      d_model=config.d_model,
      nhead=config.temporal_heads,
      dim_feedforward=config.d_model * config.ff_mult,
      dropout=config.dropout,
      activation="gelu",
      batch_first=True,
      norm_first=config.norm_first,
    )
    self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=config.temporal_layers)

    spatial_layer = nn.TransformerEncoderLayer(
      d_model=config.d_model,
      nhead=config.spatial_heads,
      dim_feedforward=config.d_model * config.ff_mult,
      dropout=config.dropout,
      activation="gelu",
      batch_first=True,
      norm_first=config.norm_first,
    )
    self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=config.spatial_layers)
    self.final_norm = nn.LayerNorm(config.d_model)
    self.out_proj = nn.Linear(config.d_model, config.vocab_size)
    self.adapter_rank = int(config.adapter_rank)
    if self.adapter_rank > 0:
      self.adapter_down = nn.Linear(config.d_model, self.adapter_rank)
      self.adapter_up = nn.Linear(self.adapter_rank, config.vocab_size)
      self.adapter_act = nn.GELU()
      nn.init.zeros_(self.adapter_up.weight)
      nn.init.zeros_(self.adapter_up.bias)
    else:
      self.adapter_down = None
      self.adapter_up = None
      self.adapter_act = None

    self.register_buffer("_position_ids", torch.arange(config.tokens_per_frame, dtype=torch.long), persistent=False)
    self.register_buffer("_frame_ids", torch.arange(config.context_frames, dtype=torch.long), persistent=False)

  def forward(self, context_frames: torch.Tensor) -> torch.Tensor:
    if context_frames.ndim != 3:
      raise ValueError(f"Expected context tensor with 3 dims [batch, context, tokens], got {tuple(context_frames.shape)}")

    batch, context_frames_count, tokens_per_frame = context_frames.shape
    if tokens_per_frame != self.config.tokens_per_frame:
      raise ValueError(
        f"Expected {self.config.tokens_per_frame} tokens per frame, got {tokens_per_frame}",
      )
    if context_frames_count != self.config.context_frames:
      raise ValueError(
        f"Expected exactly {self.config.context_frames} context frames, got {context_frames_count}",
      )

    token_ids = context_frames.long()
    pos_ids = self._position_ids[:tokens_per_frame]
    frame_ids = self._frame_ids[:context_frames_count]

    x = self.tok_embed(token_ids)
    x = x + self.pos_embed(pos_ids).view(1, 1, tokens_per_frame, self.config.d_model)
    x = x + self.frame_embed(frame_ids).view(1, context_frames_count, 1, self.config.d_model)
    x = self.input_dropout(x)

    # Temporal mixing happens independently for each spatial position.
    x = x.permute(0, 2, 1, 3).reshape(batch * tokens_per_frame, context_frames_count, self.config.d_model)
    x = self.temporal_encoder(x)
    x = x[:, -1, :].reshape(batch, tokens_per_frame, self.config.d_model)

    # Spatial refinement lets positions exchange information before the final logits.
    x = self.spatial_encoder(x)
    x = self.final_norm(x)
    logits = self.out_proj(x)
    if self.adapter_rank > 0:
      logits = logits + self.adapter_up(self.adapter_act(self.adapter_down(x)))
    return logits


def count_parameters(model: nn.Module) -> int:
  return sum(int(parameter.numel()) for parameter in model.parameters())


def save_student_checkpoint(
  path: str | Path,
  model: StudentFramePredictor,
  *,
  extra: dict[str, Any] | None = None,
) -> None:
  payload = {
    "config": model.config.to_dict(),
    "state_dict": model.state_dict(),
  }
  if extra:
    payload["extra"] = extra
  torch.save(payload, Path(path))


def load_student_checkpoint(
  path: str | Path,
  *,
  map_location: str | torch.device = "cpu",
) -> tuple[StudentFramePredictor, dict[str, Any]]:
  payload = torch.load(Path(path), map_location=map_location, weights_only=False)
  config_dict = payload.get("config")
  if not isinstance(config_dict, dict):
    raise ValueError("Student checkpoint is missing a config dictionary")
  model = StudentFramePredictor(StudentPredictorConfig.from_dict(config_dict))
  state_dict = payload.get("state_dict")
  if not isinstance(state_dict, dict):
    raise ValueError("Student checkpoint is missing a state_dict")
  model.load_state_dict(state_dict, strict=True)
  return model, payload


def summarize_checkpoint(path: str | Path) -> str:
  model, payload = load_student_checkpoint(path)
  summary = {
    "path": str(Path(path)),
    "parameters": count_parameters(model),
    "config": model.config.to_dict(),
    "extra": payload.get("extra", {}),
  }
  return json.dumps(summary, indent=2, sort_keys=True)
