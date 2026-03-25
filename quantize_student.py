#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from strong_compression.student_quantization import quantize_student_checkpoint


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Quantize a trained student predictor")
  parser.add_argument("--input", required=True, help="Path to the float student checkpoint")
  parser.add_argument("--output", default="artifacts/student_model_q8.bin", help="Quantized artifact output path")
  parser.add_argument("--bits", type=int, default=8, help="Quantization bit width")
  return parser


def main() -> None:
  args = build_parser().parse_args()
  Path(args.output).parent.mkdir(parents=True, exist_ok=True)
  report = quantize_student_checkpoint(args.input, args.output, bits=args.bits)
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
