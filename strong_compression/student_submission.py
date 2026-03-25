from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECOMPRESS_TEMPLATE = ROOT / "submission_assets_student" / "decompress.py"
SUBMISSION_INIT = """from .student_codec import decompress_student_archive, load_archive\n\n__all__ = [\"decompress_student_archive\", \"load_archive\"]\n"""
PACKAGE_FILES = [
  ("student_model.py", ROOT / "student_model.py"),
  ("strong_compression/arithmetic.py", ROOT / "strong_compression" / "arithmetic.py"),
  ("strong_compression/bitpack.py", ROOT / "strong_compression" / "bitpack.py"),
  ("strong_compression/dataset.py", ROOT / "strong_compression" / "dataset.py"),
  ("strong_compression/student_archive.py", ROOT / "strong_compression" / "student_archive.py"),
  ("strong_compression/student_codec.py", ROOT / "strong_compression" / "student_codec.py"),
  ("strong_compression/student_quantization.py", ROOT / "strong_compression" / "student_quantization.py"),
  ("strong_compression/student_runtime.py", ROOT / "strong_compression" / "student_runtime.py"),
  ("strong_compression/transforms.py", ROOT / "strong_compression" / "transforms.py"),
]


def canonical_model_name(model_path: str | Path) -> str:
  path = Path(model_path)
  return "student_model_q8.bin" if path.suffix == ".bin" else "student_model.pt"


def materialize_student_submission_tree(
  archive_path: str | Path,
  model_path: str | Path,
  output_dir: str | Path,
) -> dict[str, object]:
  archive_path = Path(archive_path)
  model_path = Path(model_path)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(archive_path, output_dir / "data.bin")
  shutil.copyfile(model_path, output_dir / canonical_model_name(model_path))
  shutil.copyfile(DECOMPRESS_TEMPLATE, output_dir / "decompress.py")
  init_path = output_dir / "strong_compression" / "__init__.py"
  init_path.parent.mkdir(parents=True, exist_ok=True)
  init_path.write_text(SUBMISSION_INIT)
  for arcname, path in PACKAGE_FILES:
    destination = output_dir / arcname
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
  return {
    "tree_dir": str(output_dir),
    "tree_bytes": sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file()),
  }


def _write_submission_zip(
  archive_path: Path,
  model_path: Path,
  output_zip: Path,
  *,
  data_compress_type: int,
) -> dict[str, object]:
  output_zip.parent.mkdir(parents=True, exist_ok=True)
  with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
    handle.write(archive_path, arcname="data.bin", compress_type=data_compress_type)
    handle.write(model_path, arcname=canonical_model_name(model_path), compress_type=zipfile.ZIP_DEFLATED)
    handle.write(DECOMPRESS_TEMPLATE, arcname="decompress.py", compress_type=zipfile.ZIP_DEFLATED)
    handle.writestr("strong_compression/__init__.py", SUBMISSION_INIT, compress_type=zipfile.ZIP_DEFLATED)
    for arcname, path in PACKAGE_FILES:
      handle.write(path, arcname=arcname, compress_type=zipfile.ZIP_DEFLATED)
  return {
    "zip_path": str(output_zip),
    "zip_bytes": output_zip.stat().st_size,
    "data_compress_type": "stored" if data_compress_type == zipfile.ZIP_STORED else "deflated",
  }


def build_student_submission_zip(
  archive_path: str | Path,
  model_path: str | Path,
  output_zip: str | Path,
) -> dict[str, object]:
  archive_path = Path(archive_path)
  model_path = Path(model_path)
  output_zip = Path(output_zip)
  with tempfile.TemporaryDirectory(prefix="student_submission_zip_") as tmpdir:
    tmp_root = Path(tmpdir)
    stored_path = tmp_root / "student_submission_stored.zip"
    deflated_path = tmp_root / "student_submission_deflated.zip"
    stored = _write_submission_zip(archive_path, model_path, stored_path, data_compress_type=zipfile.ZIP_STORED)
    deflated = _write_submission_zip(archive_path, model_path, deflated_path, data_compress_type=zipfile.ZIP_DEFLATED)
    best = stored if stored["zip_bytes"] <= deflated["zip_bytes"] else deflated
    chosen_path = stored_path if best is stored else deflated_path
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(chosen_path, output_zip)
  return {
    **best,
    "zip_path": str(output_zip),
    "archive_bytes": archive_path.stat().st_size,
    "model_bytes": model_path.stat().st_size,
    "decompress_template_bytes": DECOMPRESS_TEMPLATE.stat().st_size,
    "package_py_bytes": len(SUBMISSION_INIT.encode("utf-8")) + sum(path.stat().st_size for _, path in PACKAGE_FILES),
  }
