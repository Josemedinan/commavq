#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from strong_compression.student_codec import decompress_student_archive, load_archive
from strong_compression.student_runtime import StudentRuntime, StudentRuntimeConfig


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Decompress a student-coded commaVQ archive")
  parser.add_argument("--input", required=True, help="Input archive path")
  parser.add_argument("--model", required=True, help="Student checkpoint or quantized artifact")
  parser.add_argument("--output-dir", required=True, help="Directory where .npy segments will be written")
  parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Torch device preference")
  parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32", help="Model precision")
  parser.add_argument("--batch-size", type=int, default=16, help="Batch size across segments")
  parser.add_argument("--temperature", type=float, default=None, help="Optional logit temperature override; defaults to checkpoint metadata or 1.0")
  return parser


def main() -> None:
  args = build_parser().parse_args()
  archive_bytes = load_archive(args.input)
  runtime = StudentRuntime(
    StudentRuntimeConfig(
      model_path=args.model,
      device=args.device,
      precision=args.precision,
      batch_size=args.batch_size,
      temperature=args.temperature,
    ),
  )
  segments, report = decompress_student_archive(
    archive_bytes,
    model_path=args.model,
    runtime=runtime,
    device_override=None,
  )
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  for segment in segments:
    with open(output_dir / segment["name"], "wb") as handle:
      np.save(handle, segment["tokens"])
  summary = {
    **report,
    "model": args.model,
    "device": args.device,
    "precision": args.precision,
    "batch_size": args.batch_size,
    "temperature": runtime.temperature,
  }
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
