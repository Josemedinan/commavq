import unittest
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from student_model import StudentFramePredictor, StudentPredictorConfig, save_student_checkpoint
from strong_compression.dataset import load_segments
from strong_compression.student_codec import StudentCompressionConfig, compress_student_segments, decompress_student_archive
from strong_compression.student_runtime import StudentRuntime, StudentRuntimeConfig


class LosslessRoundtripTest(unittest.TestCase):
  def test_student_lossless_roundtrip(self) -> None:
    segments = load_segments(
      ["commavq_data/data-0000.tar.gz"],
      root=".",
      limit_segments=1,
      max_frames=4,
    )
    with tempfile.TemporaryDirectory(prefix="student_test_") as tmpdir:
      checkpoint_path = Path(tmpdir) / "student_model.pt"
      model = StudentFramePredictor(
        StudentPredictorConfig(
          context_frames=2,
          d_model=64,
          temporal_layers=1,
          spatial_layers=1,
          temporal_heads=4,
          spatial_heads=4,
          dropout=0.0,
        ),
      )
      save_student_checkpoint(checkpoint_path, model)
      runtime = StudentRuntime(
        StudentRuntimeConfig(
          model_path=str(checkpoint_path),
          device="cpu",
          precision="float32",
          batch_size=1,
        ),
      )
      archive_bytes, report = compress_student_segments(
        segments,
        config=StudentCompressionConfig(runtime=runtime.config),
        runtime=runtime,
      )
      decoded_segments, decode_report = decompress_student_archive(
        archive_bytes,
        model_path=str(checkpoint_path),
        runtime=runtime,
      )

    self.assertEqual(report["compressed_bytes"], len(archive_bytes))
    self.assertEqual(len(decoded_segments), len(segments))
    self.assertEqual(decode_report["segments"], len(segments))
    for original, decoded in zip(segments, decoded_segments):
      self.assertEqual(original.name, decoded["name"])
      self.assertTrue(np.array_equal(original.tokens, decoded["tokens"]))


if __name__ == "__main__":
  unittest.main()
