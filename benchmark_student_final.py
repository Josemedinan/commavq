#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from student_model import count_parameters
from strong_compression.dataset import load_segments
from strong_compression.student_codec import StudentCompressionConfig, compress_student_segments, decompress_student_archive, measure_student_bits_per_token
from strong_compression.student_quantization import load_student_model_artifact, quantize_student_checkpoint
from strong_compression.student_runtime import StudentRuntime, StudentRuntimeConfig
from strong_compression.student_submission import build_student_submission_zip


TARGET_RATIO = 2.2
TARGET_ARCHIVE_BPT = 10.0 / TARGET_RATIO


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Benchmark the final student submission path")
  parser.add_argument("--shards", nargs="+", required=True, help="Input shard .tar.gz paths")
  parser.add_argument("--model", required=True, help="Student checkpoint or q8 artifact")
  parser.add_argument("--limit-segments", type=int, default=4, help="Maximum number of segments to load")
  parser.add_argument("--max-frames", type=int, default=1200, help="Maximum frames per segment")
  parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Torch device preference")
  parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32", help="Model precision")
  parser.add_argument("--batch-size", type=int, default=16, help="Batch size across segments")
  parser.add_argument("--temperature", type=float, default=None, help="Optional temperature override")
  parser.add_argument("--count-total", type=int, default=1 << 15, help="Arithmetic coder total count")
  parser.add_argument("--build-submission", action="store_true", help="Also build and measure a self-contained submission zip")
  parser.add_argument("--output-json", default=None, help="Optional JSON report output path")
  return parser


def _materialize_q8(model_path: str) -> tuple[str, dict[str, Any] | None]:
  if Path(model_path).suffix == ".bin":
    return model_path, None
  with tempfile.TemporaryDirectory(prefix="student_final_q8_") as tmpdir:
    q8_path = Path(tmpdir) / f"{Path(model_path).stem}_q8.bin"
    report = quantize_student_checkpoint(model_path, q8_path, bits=8)
    preserved = Path("artifacts") / q8_path.name
    preserved.parent.mkdir(parents=True, exist_ok=True)
    preserved.write_bytes(q8_path.read_bytes())
    return str(preserved), report


def main() -> None:
  args = build_parser().parse_args()
  segments = load_segments(
    args.shards,
    root=".",
    limit_segments=args.limit_segments,
    max_frames=args.max_frames,
  )
  runtime = StudentRuntime(
    StudentRuntimeConfig(
      model_path=args.model,
      device=args.device,
      precision=args.precision,
      batch_size=args.batch_size,
      temperature=args.temperature,
    ),
  )

  nll_report = measure_student_bits_per_token(segments, runtime=runtime)
  archive_bytes, compress_report = compress_student_segments(
    segments,
    config=StudentCompressionConfig(runtime=runtime.config, count_total=args.count_total),
    runtime=runtime,
  )
  decoded_segments, decode_report = decompress_student_archive(
    archive_bytes,
    model_path=args.model,
    runtime=runtime,
  )
  exact_match = all(np.array_equal(original.tokens, decoded["tokens"]) for original, decoded in zip(segments, decoded_segments))

  model, metadata = load_student_model_artifact(args.model, map_location="cpu")
  q8_path, quantize_report = _materialize_q8(args.model)
  q8_bytes = Path(q8_path).stat().st_size

  submission_report = None
  if args.build_submission:
    with tempfile.TemporaryDirectory(prefix="student_final_submission_") as tmpdir:
      tmp_root = Path(tmpdir)
      archive_path = tmp_root / "data.bin"
      archive_path.write_bytes(archive_bytes)
      submission_zip = Path("artifacts") / f"{Path(args.model).stem}_submission.zip"
      submission_report = build_student_submission_zip(archive_path, q8_path, submission_zip)

  report = {
    "target_ratio": TARGET_RATIO,
    "target_archive_bpt": TARGET_ARCHIVE_BPT,
    "segment_count": len(segments),
    "max_frames": args.max_frames,
    "model_path": args.model,
    "q8_model_path": q8_path,
    "parameters": count_parameters(model),
    "float_model_bytes": Path(args.model).stat().st_size,
    "q8_model_bytes": q8_bytes,
    "recommended_temperature": float(metadata.get("extra", {}).get("recommended_temperature", 1.0)),
    "runtime": runtime.summary(),
    "predicted_bits_per_token": nll_report["predicted_bits_per_token"],
    "effective_bits_per_token": nll_report["effective_bits_per_token"],
    "archive_bits_per_token": compress_report["bits_per_token"],
    "compression_ratio": compress_report["compression_ratio"],
    "gap_archive_minus_target_bpt": float(compress_report["bits_per_token"]) - TARGET_ARCHIVE_BPT,
    "archive_bytes": len(archive_bytes),
    "payload_bytes": compress_report["payload_bytes"],
    "seed_bytes": compress_report["seed_bytes"],
    "archive_overhead_bytes": compress_report["archive_overhead_bytes"],
    "compress_seconds": compress_report["compress_seconds"],
    "decompress_seconds": decode_report["decompress_seconds"],
    "compress_tokens_per_second": compress_report["encode_tokens_per_second"],
    "decompress_tokens_per_second": decode_report["decode_tokens_per_second"],
    "exact_match": exact_match,
    "quantize_report": quantize_report,
    "submission_report": submission_report,
  }
  if args.output_json:
    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True))
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
