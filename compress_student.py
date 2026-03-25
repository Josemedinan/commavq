#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from strong_compression.dataset import load_segments
from strong_compression.student_codec import StudentCompressionConfig, compress_student_segments, save_archive
from strong_compression.student_runtime import StudentRuntimeConfig


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Compress commaVQ segments with the frame-level student predictor")
  parser.add_argument("--shards", nargs="+", required=True, help="Input shard .tar.gz paths")
  parser.add_argument("--model", required=True, help="Student checkpoint or quantized artifact")
  parser.add_argument("--output", required=True, help="Output student archive path")
  parser.add_argument("--limit-segments", type=int, default=None, help="Limit number of segments to compress")
  parser.add_argument("--max-frames", type=int, default=None, help="Truncate each segment to the first N frames")
  parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Torch device preference")
  parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32", help="Model precision")
  parser.add_argument("--batch-size", type=int, default=16, help="Batch size across segments")
  parser.add_argument("--temperature", type=float, default=None, help="Optional logit temperature override; defaults to checkpoint metadata or 1.0")
  parser.add_argument("--count-total", type=int, default=32768, help="Total arithmetic count after probability quantization")
  return parser


def main() -> None:
  args = build_parser().parse_args()
  segments = load_segments(args.shards, root=".", limit_segments=args.limit_segments, max_frames=args.max_frames)
  config = StudentCompressionConfig(
    runtime=StudentRuntimeConfig(
      model_path=args.model,
      device=args.device,
      precision=args.precision,
      batch_size=args.batch_size,
      temperature=args.temperature,
    ),
    count_total=args.count_total,
  )
  archive_bytes, report = compress_student_segments(segments, config=config)
  save_archive(args.output, archive_bytes)
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
