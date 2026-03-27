from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .transforms import VOCAB_SIZE
from .student_quantization import load_student_model_artifact


def select_torch_device(torch_module, preference: str) -> str:
  preference = preference.lower()
  if preference == "auto":
    if torch_module.cuda.is_available():
      return "cuda"
    if getattr(torch_module.backends, "mps", None) is not None and torch_module.backends.mps.is_available():
      return "mps"
    return "cpu"
  if preference == "cuda" and not torch_module.cuda.is_available():
    raise RuntimeError("CUDA requested but not available")
  if preference == "mps":
    if getattr(torch_module.backends, "mps", None) is None or not torch_module.backends.mps.is_available():
      raise RuntimeError("MPS requested but not available")
  return preference


def select_model_dtype(torch_module, precision: str):
  precision = precision.lower()
  if precision == "float16":
    return torch_module.float16
  if precision == "bfloat16":
    return torch_module.bfloat16
  return torch_module.float32


@dataclass
class StudentRuntimeConfig:
  model_path: str = "artifacts/student_model_q8.bin"
  device: str = "auto"
  precision: str = "float32"
  batch_size: int = 16
  temperature: float | None = None


class StudentRuntime:
  def __init__(self, config: StudentRuntimeConfig = StudentRuntimeConfig()) -> None:
    import torch

    self._torch = torch
    self.config = config
    self.device = select_torch_device(torch, config.device)
    self.model_dtype = select_model_dtype(torch, config.precision)
    self.model_path = Path(config.model_path)
    if not self.model_path.exists():
      raise FileNotFoundError(f"Student model artifact not found at {self.model_path}")
    self.model, self.metadata = load_student_model_artifact(self.model_path, map_location="cpu")
    self.model.to(device=self.device, dtype=self.model_dtype)
    self.model.eval()
    extra = self.metadata.get("extra", {})
    recommended_temperature = extra.get("recommended_temperature", 1.0)
    self.temperature = float(config.temperature if config.temperature is not None else recommended_temperature)
    if self.temperature <= 0.0:
      raise ValueError(f"Student runtime temperature must be positive, got {self.temperature}")
    position_temperatures = extra.get("position_temperatures")
    if position_temperatures is None:
      self.position_temperatures = None
    else:
      position_array = np.asarray(position_temperatures, dtype=np.float32)
      if position_array.shape != (self.model.config.tokens_per_frame,):
        raise ValueError(
          f"position_temperatures must have shape ({self.model.config.tokens_per_frame},), got {position_array.shape}",
        )
      if not np.all(np.isfinite(position_array)) or np.any(position_array <= 0.0):
        raise ValueError("position_temperatures must contain only finite positive values")
      self.position_temperatures = torch.from_numpy(position_array).to(device=self.device, dtype=torch.float32)
    position_logit_bias = extra.get("position_logit_bias")
    if position_logit_bias is None:
      self.position_logit_bias = None
    else:
      bias_array = np.asarray(position_logit_bias, dtype=np.float32)
      expected_shape = (self.model.config.tokens_per_frame, VOCAB_SIZE)
      if bias_array.shape != expected_shape:
        raise ValueError(f"position_logit_bias must have shape {expected_shape}, got {bias_array.shape}")
      if not np.all(np.isfinite(bias_array)):
        raise ValueError("position_logit_bias must contain only finite values")
      self.position_logit_bias = torch.from_numpy(bias_array).to(device=self.device, dtype=torch.float32)
    if self.device != "cpu":
      torch.set_float32_matmul_precision("high")

  def _temperature_scaled_logits(self, logits):
    scaled = logits.float()
    if self.position_logit_bias is not None:
      scaled = scaled + self.position_logit_bias.view(1, self.model.config.tokens_per_frame, VOCAB_SIZE)
    if self.position_temperatures is not None:
      scaled = scaled / self.position_temperatures.view(1, -1, 1)
      if self.temperature != 1.0:
        scaled = scaled / self.temperature
      return scaled
    return scaled / self.temperature

  @property
  def context_frames(self) -> int:
    return int(self.model.config.context_frames)

  def predict_probs(self, context_batch: np.ndarray) -> tuple[np.ndarray, float]:
    torch = self._torch
    context_array = np.asarray(context_batch, dtype=np.int64)
    start_time = time.perf_counter()
    context_tensor = torch.from_numpy(context_array).to(device=self.device, dtype=torch.long)
    with torch.inference_mode():
      logits = self._temperature_scaled_logits(self.model(context_tensor))
      probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs.astype(np.float64, copy=False), time.perf_counter() - start_time

  def predict_logits(self, context_batch: np.ndarray) -> tuple[np.ndarray, float]:
    torch = self._torch
    context_array = np.asarray(context_batch, dtype=np.int64)
    start_time = time.perf_counter()
    context_tensor = torch.from_numpy(context_array).to(device=self.device, dtype=torch.long)
    with torch.inference_mode():
      logits = self._temperature_scaled_logits(self.model(context_tensor)).cpu().numpy()
    return logits.astype(np.float32, copy=False), time.perf_counter() - start_time

  def summary(self) -> dict[str, object]:
    return {
      "device": self.device,
      "precision": self.config.precision,
      "batch_size": self.config.batch_size,
      "model_path": str(self.model_path),
      "model_bits": self.metadata.get("bits"),
      "context_frames": self.context_frames,
      "vocab_size": VOCAB_SIZE,
      "temperature": self.temperature,
      "has_position_temperatures": self.position_temperatures is not None,
      "has_position_logit_bias": self.position_logit_bias is not None,
    }
