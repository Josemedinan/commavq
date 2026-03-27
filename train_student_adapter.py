#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from student_model import (
  StudentFramePredictor,
  StudentPredictorConfig,
  count_parameters,
  load_student_checkpoint,
  save_student_checkpoint,
)
from strong_compression.dataset import load_segments
from strong_compression.student_data import StudentBatch, StudentFrameDataset, collate_student_batch, split_segments
from strong_compression.student_runtime import select_model_dtype, select_torch_device


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Train a small residual adapter on top of the current student")
  parser.add_argument("--shards", nargs="+", required=True, help="Input shard .tar.gz paths")
  parser.add_argument("--init-checkpoint", required=True, help="Base student checkpoint")
  parser.add_argument("--adapter-rank", type=int, default=16, help="Residual adapter bottleneck rank")
  parser.add_argument("--train-segments", type=int, default=32, help="Number of segments used for training")
  parser.add_argument("--val-segments", type=int, default=8, help="Number of segments used for validation")
  parser.add_argument("--max-frames", type=int, default=1200, help="Frames per segment to load")
  parser.add_argument("--limit-per-shard", type=int, default=2, help="Optional cap on segments to load from each shard")
  parser.add_argument("--seed-frames", type=int, default=0, help="Initial raw frames during training/eval")
  parser.add_argument("--epochs", type=int, default=1, help="Adapter fine-tuning epochs")
  parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
  parser.add_argument("--eval-batch-size", type=int, default=16, help="Validation batch size")
  parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
  parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
  parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
  parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Torch device preference")
  parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32", help="Model dtype")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--output", required=True, help="Output checkpoint path")
  parser.add_argument("--metrics-json", default=None, help="Optional metrics JSON output path")
  return parser


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def _bits_per_token(loss_value: float) -> float:
  return float(loss_value / math.log(2.0))


def _move_batch(batch: StudentBatch, device: str) -> StudentBatch:
  return StudentBatch(
    context=batch.context.to(device=device, dtype=torch.long),
    target=batch.target.to(device=device, dtype=torch.long),
    segment_indices=batch.segment_indices.to(device=device),
    frame_indices=batch.frame_indices.to(device=device),
    example_indices=batch.example_indices.to(device=device),
  )


def evaluate_model(model: StudentFramePredictor, loader: DataLoader, *, device: str) -> dict[str, float]:
  model.eval()
  total_loss = 0.0
  total_tokens = 0
  total_correct = 0
  start_time = time.perf_counter()
  with torch.inference_mode():
    for raw_batch in loader:
      batch = _move_batch(raw_batch, device)
      logits = model(batch.context).float()
      loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch.target.reshape(-1))
      total_loss += float(loss.item()) * int(batch.target.numel())
      total_correct += int((logits.argmax(dim=-1) == batch.target).sum().item())
      total_tokens += int(batch.target.numel())
  seconds = time.perf_counter() - start_time
  avg_loss = total_loss / max(total_tokens, 1)
  return {
    "loss": avg_loss,
    "bits_per_token": _bits_per_token(avg_loss),
    "accuracy": total_correct / max(total_tokens, 1),
    "tokens": total_tokens,
    "seconds": seconds,
    "tokens_per_second": total_tokens / max(seconds, 1e-9),
  }


def build_adapter_model(init_checkpoint: str, adapter_rank: int, *, device: str, dtype) -> tuple[StudentFramePredictor, dict[str, Any]]:
  base_model, payload = load_student_checkpoint(init_checkpoint, map_location="cpu")
  config_dict = base_model.config.to_dict()
  config_dict["adapter_rank"] = int(adapter_rank)
  model = StudentFramePredictor(StudentPredictorConfig.from_dict(config_dict))
  missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)
  if unexpected:
    raise ValueError(f"Unexpected keys while loading base checkpoint into adapter model: {unexpected}")
  if base_model.config.adapter_rank == int(adapter_rank):
    expected_missing: set[str] = set()
  elif int(base_model.config.adapter_rank) == 0 and int(adapter_rank) > 0:
    expected_missing = {"adapter_down.weight", "adapter_down.bias", "adapter_up.weight", "adapter_up.bias"}
  else:
    expected_missing = set()
  if set(missing) != expected_missing:
    raise ValueError(f"Unexpected missing keys for adapter model: {missing}")
  model = model.to(device=device, dtype=dtype)
  for name, parameter in model.named_parameters():
    parameter.requires_grad = name.startswith("adapter_")
  return model, payload


def main() -> None:
  args = build_parser().parse_args()
  set_seed(args.seed)

  device = select_torch_device(torch, args.device)
  model_dtype = select_model_dtype(torch, args.precision)
  if device != "cpu":
    torch.set_float32_matmul_precision("high")

  segments = load_segments(
    args.shards,
    root=".",
    limit_segments=args.train_segments + args.val_segments,
    limit_per_shard=args.limit_per_shard,
    max_frames=args.max_frames,
  )
  train_segments, val_segments = split_segments(
    segments,
    train_segments=args.train_segments,
    val_segments=args.val_segments,
  )

  model, payload = build_adapter_model(args.init_checkpoint, args.adapter_rank, device=device, dtype=model_dtype)
  train_dataset = StudentFrameDataset(
    train_segments,
    context_frames=model.config.context_frames,
    seed_frames=args.seed_frames,
  )
  val_dataset = StudentFrameDataset(
    val_segments,
    context_frames=model.config.context_frames,
    seed_frames=args.seed_frames,
  )
  train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_student_batch,
  )
  val_loader = DataLoader(
    val_dataset,
    batch_size=args.eval_batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_student_batch,
  )

  trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
  optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

  initial_val = evaluate_model(model, val_loader, device=device)
  print(json.dumps({"epoch": 0, "phase": "init", "val_bits_per_token": initial_val["bits_per_token"], "val_loss": initial_val["loss"]}, sort_keys=True))

  best_val_bits = float(initial_val["bits_per_token"])
  history: list[dict[str, Any]] = []
  output_path = Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  extra = dict(payload.get("extra", {}))
  extra.update({
    "adapter_rank": args.adapter_rank,
    "adapter_init_checkpoint": args.init_checkpoint,
    "seed_frames": args.seed_frames,
    "train_segments": args.train_segments,
    "val_segments": args.val_segments,
    "max_frames": args.max_frames,
    "limit_per_shard": args.limit_per_shard,
    "trainable_parameters": sum(int(parameter.numel()) for parameter in trainable),
    "base_parameters": count_parameters(model) - sum(int(parameter.numel()) for parameter in trainable),
  })
  save_student_checkpoint(output_path, model, extra=extra)

  for epoch in range(1, args.epochs + 1):
    model.train()
    epoch_start = time.perf_counter()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for raw_batch in train_loader:
      batch = _move_batch(raw_batch, device)
      optimizer.zero_grad(set_to_none=True)
      logits = model(batch.context)
      loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch.target.reshape(-1))
      loss.backward()
      if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
      optimizer.step()

      total_loss += float(loss.detach().cpu().item()) * int(batch.target.numel())
      total_correct += int((logits.argmax(dim=-1) == batch.target).sum().detach().cpu().item())
      total_tokens += int(batch.target.numel())

    train_loss = total_loss / max(total_tokens, 1)
    val_metrics = evaluate_model(model, val_loader, device=device)
    epoch_report = {
      "epoch": epoch,
      "train_bits_per_token": _bits_per_token(train_loss),
      "train_accuracy": total_correct / max(total_tokens, 1),
      "val_bits_per_token": val_metrics["bits_per_token"],
      "val_accuracy": val_metrics["accuracy"],
      "epoch_seconds": time.perf_counter() - epoch_start,
    }
    history.append(epoch_report)
    print(json.dumps(epoch_report, sort_keys=True))

    if val_metrics["bits_per_token"] < best_val_bits:
      best_val_bits = float(val_metrics["bits_per_token"])
      best_extra = dict(extra)
      best_extra.update({
        "best_val_bits_per_token": best_val_bits,
        "history": history,
        "recommended_temperature": extra.get("recommended_temperature", 1.0),
      })
      save_student_checkpoint(output_path, model, extra=best_extra)

  metrics = {
    "init_checkpoint": args.init_checkpoint,
    "output": str(output_path),
    "adapter_rank": args.adapter_rank,
    "train_segments": args.train_segments,
    "val_segments": args.val_segments,
    "seed_frames": args.seed_frames,
    "best_val_bits_per_token": best_val_bits,
    "trainable_parameters": sum(int(parameter.numel()) for parameter in trainable),
    "history": history,
    "initial_val": initial_val,
  }
  if args.metrics_json:
    Path(args.metrics_json).write_text(json.dumps(metrics, indent=2, sort_keys=True))
  print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
