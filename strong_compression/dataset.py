import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class Segment:
  name: str
  tokens: np.ndarray
  shard: str

  @property
  def frames(self) -> int:
    return int(self.tokens.shape[0])


def _normalize_shard_path(root: Path, shard: str | Path) -> Path:
  shard_path = Path(shard)
  if shard_path.is_absolute():
    return shard_path
  if shard_path.exists():
    return shard_path
  return root / shard_path


def load_segments(
  shard_paths: Iterable[str | Path],
  *,
  root: str | Path = ".",
  limit_segments: int | None = None,
  max_frames: int | None = None,
) -> list[Segment]:
  root_path = Path(root)
  segments: list[Segment] = []
  for shard in shard_paths:
    shard_path = _normalize_shard_path(root_path, shard)
    with tarfile.open(shard_path, "r:gz") as tar:
      names = sorted(name for name in tar.getnames() if name.endswith(".token.npy"))
      for name in names:
        with tar.extractfile(name) as handle:
          tokens = np.load(io.BytesIO(handle.read()))
        if max_frames is not None:
          tokens = tokens[:max_frames]
        segments.append(Segment(name=name, tokens=tokens.astype(np.int16, copy=False), shard=shard_path.name))
        if limit_segments is not None and len(segments) >= limit_segments:
          return segments
  return segments
