from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import Segment
from .transforms import TOKENS_PER_FRAME, flatten_tokens


@dataclass
class StudentBatch:
  context: torch.Tensor
  target: torch.Tensor
  segment_indices: torch.Tensor
  frame_indices: torch.Tensor
  example_indices: torch.Tensor


class StudentFrameDataset(Dataset[dict[str, Any]]):
  def __init__(self, segments: list[Segment], *, context_frames: int) -> None:
    self.segments = segments
    self.context_frames = int(context_frames)
    self.flat_segments = [flatten_tokens(segment.tokens).astype(np.int32, copy=False) for segment in segments]
    self.examples: list[tuple[int, int]] = []
    for segment_index, flat in enumerate(self.flat_segments):
      for frame_index in range(self.context_frames, int(flat.shape[0])):
        self.examples.append((segment_index, frame_index))

  def __len__(self) -> int:
    return len(self.examples)

  def __getitem__(self, index: int) -> dict[str, Any]:
    segment_index, frame_index = self.examples[index]
    flat = self.flat_segments[segment_index]
    context = flat[frame_index - self.context_frames:frame_index]
    target = flat[frame_index]
    return {
      "context": context.astype(np.int64, copy=False),
      "target": target.astype(np.int64, copy=False),
      "segment_index": int(segment_index),
      "frame_index": int(frame_index),
      "example_index": int(index),
    }


def collate_student_batch(items: list[dict[str, Any]]) -> StudentBatch:
  context = torch.from_numpy(np.stack([item["context"] for item in items]).astype(np.int64, copy=False))
  target = torch.from_numpy(np.stack([item["target"] for item in items]).astype(np.int64, copy=False))
  segment_indices = torch.tensor([item["segment_index"] for item in items], dtype=torch.long)
  frame_indices = torch.tensor([item["frame_index"] for item in items], dtype=torch.long)
  example_indices = torch.tensor([item["example_index"] for item in items], dtype=torch.long)
  return StudentBatch(
    context=context,
    target=target,
    segment_indices=segment_indices,
    frame_indices=frame_indices,
    example_indices=example_indices,
  )


def split_segments(
  segments: list[Segment],
  *,
  train_segments: int,
  val_segments: int,
) -> tuple[list[Segment], list[Segment]]:
  if train_segments <= 0:
    raise ValueError("train_segments must be positive")
  if val_segments <= 0:
    raise ValueError("val_segments must be positive")
  if train_segments + val_segments > len(segments):
    raise ValueError(
      f"Requested {train_segments + val_segments} segments but only loaded {len(segments)}",
    )
  return segments[:train_segments], segments[train_segments:train_segments + val_segments]


def flat_contexts_and_targets(segments: list[Segment], *, context_frames: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
  contexts: list[np.ndarray] = []
  targets: list[np.ndarray] = []
  for segment in segments:
    flat = flatten_tokens(segment.tokens).astype(np.int32, copy=False)
    for frame_index in range(context_frames, int(flat.shape[0])):
      contexts.append(flat[frame_index - context_frames:frame_index])
      targets.append(flat[frame_index])
  return contexts, targets


def segment_token_count(segments: list[Segment]) -> int:
  return sum(int(segment.tokens.size) for segment in segments)


def predicted_token_count(segments: list[Segment], *, context_frames: int) -> int:
  total = 0
  for segment in segments:
    frames = int(segment.tokens.shape[0])
    total += max(frames - context_frames, 0) * TOKENS_PER_FRAME
  return total
