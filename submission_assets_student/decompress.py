#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

from strong_compression.student_codec import decompress_student_archive, load_archive
from strong_compression.student_runtime import StudentRuntime, StudentRuntimeConfig


def _resolve_model_path() -> Path:
  for candidate in ("student_model_q8.bin", "student_model.pt"):
    path = HERE / candidate
    if path.exists():
      return path
  raise FileNotFoundError("Expected student model artifact next to decompress.py")


def main() -> None:
  archive_path = HERE / "data.bin"
  model_path = _resolve_model_path()
  output_dir = Path(os.environ.get("OUTPUT_DIR", HERE / "decompressed"))

  runtime = StudentRuntime(
    StudentRuntimeConfig(
      model_path=str(model_path),
      device=os.environ.get("COMMAVQ_DEVICE", "auto"),
      precision=os.environ.get("COMMAVQ_PRECISION", "float32"),
      batch_size=int(os.environ.get("COMMAVQ_BATCH_SIZE", "16")),
      temperature=float(os.environ["COMMAVQ_TEMPERATURE"]) if "COMMAVQ_TEMPERATURE" in os.environ else None,
    ),
  )
  archive_bytes = load_archive(archive_path)
  segments, report = decompress_student_archive(
    archive_bytes,
    model_path=str(model_path),
    runtime=runtime,
  )

  output_dir.mkdir(parents=True, exist_ok=True)
  for segment in segments:
    output_path = output_dir / segment["name"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
      np.save(handle, segment["tokens"])

  print(json.dumps({
    "segments": len(segments),
    "tokens": report["total_tokens"],
    "decompress_seconds": report["decompress_seconds"],
    "model_path": str(model_path.name),
    "device": runtime.device,
    "temperature": runtime.temperature,
  }, sort_keys=True))


if __name__ == "__main__":
  main()
