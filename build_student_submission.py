#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from strong_compression.student_quantization import load_student_model_artifact
from strong_compression.student_submission import build_student_submission_zip, materialize_student_submission_tree


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Build a self-contained student submission zip")
  parser.add_argument("--archive", required=True, help="Path to the student-compressed data.bin payload")
  parser.add_argument("--model", required=True, help="Path to the student checkpoint or q8 artifact")
  parser.add_argument("--output-zip", required=True, help="Where to write the submission zip")
  parser.add_argument("--output-tree", default=None, help="Optional directory where the standalone submission tree will be materialized")
  return parser


def main() -> None:
  args = build_parser().parse_args()
  report = build_student_submission_zip(args.archive, args.model, args.output_zip)
  _model, metadata = load_student_model_artifact(args.model, map_location="cpu")
  report["model_recommended_temperature"] = float(metadata.get("extra", {}).get("recommended_temperature", 1.0))
  if args.output_tree:
    tree_report = materialize_student_submission_tree(args.archive, args.model, args.output_tree)
    report["tree_dir"] = tree_report["tree_dir"]
    report["tree_bytes"] = tree_report["tree_bytes"]
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
