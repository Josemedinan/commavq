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
from torch.utils.data import DataLoader, WeightedRandomSampler

from student_model import StudentFramePredictor, count_parameters, load_student_checkpoint, save_student_checkpoint
from strong_compression.dataset import load_segments
from strong_compression.student_data import StudentBatch, StudentFrameDataset, collate_student_batch, split_segments
from strong_compression.student_runtime import select_model_dtype, select_torch_device


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Final high-ROI fine-tuning pass for the current student predictor")
  parser.add_argument("--shards", nargs="+", required=True, help="Input shard .tar.gz paths")
  parser.add_argument("--init-checkpoint", default="artifacts/student_model.pt", help="Starting student checkpoint")
  parser.add_argument("--train-segments", type=int, default=96, help="Number of segments used for training")
  parser.add_argument("--val-segments", type=int, default=16, help="Number of segments used for validation")
  parser.add_argument("--max-frames", type=int, default=32, help="Frames per segment to load")
  parser.add_argument("--limit-per-shard", type=int, default=None, help="Optional cap on how many segments to load from each shard")
  parser.add_argument("--seed-frames", type=int, default=None, help="How many initial frames stay raw during training/eval; defaults to model context")
  parser.add_argument("--epochs", type=int, default=3, help="Fine-tuning epochs")
  parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
  parser.add_argument("--eval-batch-size", type=int, default=32, help="Validation batch size")
  parser.add_argument("--learning-rate", type=float, default=5e-5, help="AdamW learning rate")
  parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay")
  parser.add_argument("--grad-clip", type=float, default=1.0, help="Clip gradients to this norm")
  parser.add_argument("--scheduler", choices=["constant", "cosine"], default="constant", help="Learning-rate schedule")
  parser.add_argument("--curriculum", choices=["none", "student_error"], default="none", help="Optional cheap curriculum")
  parser.add_argument("--curriculum-power", type=float, default=1.5, help="Exponent applied to normalized difficulty")
  parser.add_argument("--curriculum-min-weight", type=float, default=0.5, help="Floor for curriculum sampling weights")
  parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Torch device preference")
  parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32", help="Model dtype")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--output", required=True, help="Where to write the best checkpoint")
  parser.add_argument("--metrics-json", default=None, help="Optional path for the training metrics JSON")
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


def evaluate_model(
  model: StudentFramePredictor,
  loader: DataLoader,
  *,
  device: str,
  temperature: float = 1.0,
) -> dict[str, float]:
  model.eval()
  total_loss = 0.0
  total_tokens = 0
  total_correct = 0
  tokens_per_frame = int(model.config.tokens_per_frame)
  start_time = time.perf_counter()
  with torch.inference_mode():
    for raw_batch in loader:
      batch = _move_batch(raw_batch, device)
      logits = model(batch.context).float() / float(temperature)
      loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch.target.reshape(-1))
      total_loss += float(loss.item()) * int(batch.target.numel())
      total_correct += int((logits.argmax(dim=-1) == batch.target).sum().item())
      total_tokens += int(batch.target.numel())
  seconds = time.perf_counter() - start_time
  avg_loss = total_loss / max(total_tokens, 1)
  bits_per_token = _bits_per_token(avg_loss)
  return {
    "loss": avg_loss,
    "bits_per_token": bits_per_token,
    "bits_per_frame": bits_per_token * tokens_per_frame if total_tokens else 0.0,
    "accuracy": total_correct / max(total_tokens, 1),
    "tokens": total_tokens,
    "seconds": seconds,
    "tokens_per_second": total_tokens / max(seconds, 1e-9),
  }


def build_student_error_weights(
  model: StudentFramePredictor,
  dataset: StudentFrameDataset,
  *,
  device: str,
  eval_batch_size: int,
  power: float,
  min_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
  loader = DataLoader(
    dataset,
    batch_size=eval_batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_student_batch,
  )
  difficulties = np.ones(len(dataset), dtype=np.float32)
  model.eval()
  start_time = time.perf_counter()
  with torch.inference_mode():
    for raw_batch in loader:
      batch = _move_batch(raw_batch, device)
      logits = model(batch.context).float()
      log_probs = F.log_softmax(logits, dim=-1)
      losses = F.nll_loss(
        log_probs.reshape(-1, log_probs.shape[-1]),
        batch.target.reshape(-1),
        reduction="none",
      ).reshape(batch.target.shape[0], -1).mean(dim=1)
      difficulties[batch.example_indices.cpu().numpy()] = losses.cpu().numpy().astype(np.float32, copy=False)
  mean_difficulty = float(np.mean(difficulties))
  normalized = difficulties / max(mean_difficulty, 1e-9)
  weights = np.clip(np.power(normalized, power, dtype=np.float32), min_weight, None)
  report = {
    "seconds": time.perf_counter() - start_time,
    "mean_loss_nats": mean_difficulty,
    "mean_bits_per_token": _bits_per_token(mean_difficulty),
    "min_weight": float(np.min(weights)),
    "max_weight": float(np.max(weights)),
    "mean_weight": float(np.mean(weights)),
    "power": power,
  }
  return weights.astype(np.float64, copy=False), report


def _build_scheduler(optimizer: torch.optim.Optimizer, *, epochs: int, mode: str):
  if mode == "cosine":
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
  return None


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

  model, payload = load_student_checkpoint(args.init_checkpoint, map_location="cpu")
  model = model.to(device=device, dtype=model_dtype)

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
  val_loader = DataLoader(
    val_dataset,
    batch_size=args.eval_batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_student_batch,
  )

  sampler = None
  curriculum_report = None
  if args.curriculum == "student_error":
    weights, curriculum_report = build_student_error_weights(
      model,
      train_dataset,
      device=device,
      eval_batch_size=args.eval_batch_size,
      power=args.curriculum_power,
      min_weight=args.curriculum_min_weight,
    )
    sampler = WeightedRandomSampler(weights=weights.tolist(), num_samples=len(train_dataset), replacement=True)

  train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=sampler is None,
    sampler=sampler,
    num_workers=0,
    collate_fn=collate_student_batch,
  )

  initial_val = evaluate_model(model, val_loader, device=device)
  print(json.dumps({"epoch": 0, "phase": "init", "val_bits_per_token": initial_val["bits_per_token"], "val_loss": initial_val["loss"]}, sort_keys=True))

  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
  scheduler = _build_scheduler(optimizer, epochs=args.epochs, mode=args.scheduler)

  best_val_bits = float(initial_val["bits_per_token"])
  history: list[dict[str, Any]] = []
  output_path = Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  save_student_checkpoint(
    output_path,
    model,
    extra={
      "best_val_bits_per_token": best_val_bits,
      "history": history,
      "train_segments": args.train_segments,
      "val_segments": args.val_segments,
      "max_frames": args.max_frames,
      "limit_per_shard": args.limit_per_shard,
      "seed_frames": args.seed_frames if args.seed_frames is not None else model.config.context_frames,
      "seed": args.seed,
      "curriculum": curriculum_report,
      "init_checkpoint": args.init_checkpoint,
      "recommended_temperature": payload.get("extra", {}).get("recommended_temperature", 1.0),
    },
  )

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()

      total_loss += float(loss.item()) * int(batch.target.numel())
      total_correct += int((logits.argmax(dim=-1) == batch.target).sum().item())
      total_tokens += int(batch.target.numel())

    if scheduler is not None:
      scheduler.step()
    train_loss = total_loss / max(total_tokens, 1)
    val_metrics = evaluate_model(model, val_loader, device=device)
    epoch_seconds = time.perf_counter() - epoch_start
    metrics = {
      "epoch": epoch,
      "train_loss": train_loss,
      "train_bits_per_token": _bits_per_token(train_loss),
      "train_accuracy": total_correct / max(total_tokens, 1),
      "val_loss": val_metrics["loss"],
      "val_bits_per_token": val_metrics["bits_per_token"],
      "val_accuracy": val_metrics["accuracy"],
      "epoch_seconds": epoch_seconds,
      "learning_rate": optimizer.param_groups[0]["lr"],
    }
    history.append(metrics)
    print(json.dumps(metrics, sort_keys=True))

    if float(val_metrics["bits_per_token"]) < best_val_bits:
      best_val_bits = float(val_metrics["bits_per_token"])
      save_student_checkpoint(
        output_path,
        model,
        extra={
          "best_val_bits_per_token": best_val_bits,
          "history": history,
          "train_segments": args.train_segments,
          "val_segments": args.val_segments,
          "max_frames": args.max_frames,
          "limit_per_shard": args.limit_per_shard,
          "seed_frames": args.seed_frames if args.seed_frames is not None else model.config.context_frames,
          "seed": args.seed,
          "curriculum": curriculum_report,
          "init_checkpoint": args.init_checkpoint,
          "recommended_temperature": payload.get("extra", {}).get("recommended_temperature", 1.0),
        },
      )

  summary = {
    "output": str(output_path),
    "checkpoint_bytes": output_path.stat().st_size if output_path.exists() else None,
    "parameters": count_parameters(model),
    "model_size_bytes_estimate": count_parameters(model) * 4,
    "best_val_bits_per_token": best_val_bits,
    "initial_val_bits_per_token": initial_val["bits_per_token"],
    "train_examples": len(train_dataset),
    "val_examples": len(val_dataset),
    "limit_per_shard": args.limit_per_shard,
    "seed_frames": args.seed_frames if args.seed_frames is not None else model.config.context_frames,
    "device": device,
    "precision": args.precision,
    "curriculum": curriculum_report,
    "config": model.config.to_dict(),
    "history": history,
  }
  if args.metrics_json:
    Path(args.metrics_json).write_text(json.dumps(summary, indent=2, sort_keys=True))
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
