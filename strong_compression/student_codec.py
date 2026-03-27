from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .arithmetic import ArithmeticDecoder, ArithmeticEncoder, CodelengthStats
from .bitpack import pack_uint10, unpack_uint10
from .dataset import Segment
from .student_archive import build_student_archive, parse_student_archive
from .student_runtime import StudentRuntime, StudentRuntimeConfig
from .transforms import TOKENS_PER_FRAME, flatten_tokens


STUDENT_ARCHIVE_MODE = "arith_student_frame"


@dataclass
class StudentCompressionConfig:
  runtime: StudentRuntimeConfig = field(default_factory=StudentRuntimeConfig)
  count_total: int = 1 << 15
  seed_frames: int | None = None


def _quantize_prob_rows_cpu(prob_rows: np.ndarray, total: int) -> np.ndarray:
  rows = np.asarray(prob_rows, dtype=np.float64)
  flat = rows.reshape(-1, 1024).copy()
  flat = np.clip(flat, 0.0, None)

  row_sums = flat.sum(axis=1, keepdims=True)
  invalid = (~np.isfinite(row_sums[:, 0])) | (row_sums[:, 0] <= 0.0)
  if np.any(invalid):
    flat[invalid] = 1.0
    row_sums = flat.sum(axis=1, keepdims=True)
  flat /= row_sums

  remaining = total - 1024
  scaled = flat * remaining
  extra = np.floor(scaled).astype(np.int32)
  leftovers = remaining - extra.sum(axis=1)
  fractions = scaled - extra
  counts = extra + 1

  token_order = np.arange(1024, dtype=np.int32)
  for row_index, leftover in enumerate(leftovers.tolist()):
    if leftover <= 0:
      continue
    order = np.lexsort((token_order, -fractions[row_index]))
    counts[row_index, order[:leftover]] += 1

  cumulative = np.zeros((counts.shape[0], 1025), dtype=np.int32)
  np.cumsum(counts, axis=1, out=cumulative[:, 1:])
  return cumulative.reshape(rows.shape[:-1] + (1025,))


def _group_segment_indices_by_frames(flat_segments: list[np.ndarray]) -> dict[int, list[int]]:
  groups: dict[int, list[int]] = defaultdict(list)
  for index, flat_tokens in enumerate(flat_segments):
    groups[int(flat_tokens.shape[0])].append(index)
  return dict(sorted(groups.items()))


def _effective_seed_frames(configured_seed_frames: int | None, *, context_frames: int, frames: int) -> int:
  default_seed_frames = context_frames if configured_seed_frames is None else int(configured_seed_frames)
  if default_seed_frames < 0:
    raise ValueError(f"seed_frames must be non-negative, got {default_seed_frames}")
  return min(default_seed_frames, frames)


def _build_padded_context(batch_flat: np.ndarray, *, frame_index: int, context_frames: int) -> np.ndarray:
  if frame_index <= 0:
    return np.zeros((batch_flat.shape[0], context_frames, TOKENS_PER_FRAME), dtype=np.int32)
  start = max(0, frame_index - context_frames)
  context = batch_flat[:, start:frame_index]
  available = int(context.shape[1])
  if available == context_frames:
    return context
  pad_count = context_frames - available
  if available > 0:
    pad = np.repeat(context[:, :1], pad_count, axis=1)
  else:
    pad = np.zeros((batch_flat.shape[0], pad_count, TOKENS_PER_FRAME), dtype=np.int32)
  return np.concatenate([pad, context], axis=1)


def _encode_row(encoder: ArithmeticEncoder, stats: CodelengthStats, cumulative: np.ndarray, tokens: np.ndarray, total: int) -> None:
  for position, token in enumerate(tokens.tolist()):
    low = int(cumulative[position, token])
    high = int(cumulative[position, token + 1])
    encoder.encode(low, high, total)
    stats.add(high - low, total)


def measure_student_bits_per_token(
  segments: list[Segment],
  *,
  runtime: StudentRuntime,
  seed_frames: int | None = None,
) -> dict[str, float | int]:
  total_bits = 0.0
  total_predicted_tokens = 0
  total_tokens = 0
  total_seconds = 0.0

  flat_segments = [flatten_tokens(segment.tokens) for segment in segments]
  for frames, indices in _group_segment_indices_by_frames(flat_segments).items():
    effective_seed_frames = _effective_seed_frames(seed_frames, context_frames=runtime.context_frames, frames=frames)
    batch_size = runtime.config.batch_size
    for batch_start in range(0, len(indices), batch_size):
      batch_indices = indices[batch_start:batch_start + batch_size]
      batch_flat = np.stack([flat_segments[index] for index in batch_indices]).astype(np.int32, copy=False)
      total_tokens += int(batch_flat.size)
      for frame_index in range(effective_seed_frames, frames):
        context = _build_padded_context(batch_flat, frame_index=frame_index, context_frames=runtime.context_frames)
        probs, seconds = runtime.predict_probs(context)
        total_seconds += seconds
        targets = batch_flat[:, frame_index].astype(np.int64, copy=False)
        batch_axis = np.arange(targets.shape[0], dtype=np.int64)[:, None]
        position_axis = np.arange(TOKENS_PER_FRAME, dtype=np.int64)[None, :]
        target_probs = np.clip(probs[batch_axis, position_axis, targets], 1e-12, 1.0)
        total_bits += float((-np.log2(target_probs)).sum())
        total_predicted_tokens += int(targets.size)

  raw_seed_tokens = total_tokens - total_predicted_tokens
  effective_bits = total_bits + raw_seed_tokens * 10.0
  return {
    "predicted_bits_per_token": total_bits / max(total_predicted_tokens, 1),
    "effective_bits_per_token": effective_bits / max(total_tokens, 1),
    "predict_seconds": total_seconds,
    "predicted_tokens": total_predicted_tokens,
    "total_tokens": total_tokens,
  }


def compress_student_segments(
  segments: list[Segment],
  *,
  config: StudentCompressionConfig | None = None,
  runtime: StudentRuntime | None = None,
) -> tuple[bytes, dict[str, Any]]:
  config = config or StudentCompressionConfig()
  if config.count_total <= 1024:
    raise ValueError("count_total must exceed vocab size 1024")
  runtime = runtime or StudentRuntime(config.runtime)

  flat_segments = [flatten_tokens(segment.tokens).astype(np.int32, copy=False) for segment in segments]
  encoders = [ArithmeticEncoder() for _ in segments]
  stats = [CodelengthStats() for _ in segments]
  segment_reports = []
  predict_seconds = 0.0
  arithmetic_seconds = 0.0

  start_time = time.perf_counter()
  for frames, indices in _group_segment_indices_by_frames(flat_segments).items():
    seed_frames = _effective_seed_frames(config.seed_frames, context_frames=runtime.context_frames, frames=frames)
    for batch_start in range(0, len(indices), runtime.config.batch_size):
      batch_indices = indices[batch_start:batch_start + runtime.config.batch_size]
      batch_flat = np.stack([flat_segments[index] for index in batch_indices]).astype(np.int32, copy=False)
      for frame_index in range(seed_frames, frames):
        context = _build_padded_context(batch_flat, frame_index=frame_index, context_frames=runtime.context_frames)
        probs, seconds = runtime.predict_probs(context)
        predict_seconds += seconds
        cumulative = _quantize_prob_rows_cpu(probs, config.count_total)
        frame_tokens = batch_flat[:, frame_index]
        arithmetic_start = time.perf_counter()
        for batch_offset, segment_index in enumerate(batch_indices):
          _encode_row(
            encoders[segment_index],
            stats[segment_index],
            cumulative[batch_offset],
            frame_tokens[batch_offset],
            config.count_total,
          )
        arithmetic_seconds += time.perf_counter() - arithmetic_start

  records = []
  payload_bytes = 0
  seed_bytes_total = 0
  total_tokens = 0
  predicted_tokens = 0
  for index, segment in enumerate(segments):
    flat = flat_segments[index]
    frames = int(flat.shape[0])
    seed_frames = _effective_seed_frames(config.seed_frames, context_frames=runtime.context_frames, frames=frames)
    seed_tokens = flat[:seed_frames]
    seed_bytes = pack_uint10(seed_tokens)
    payload = encoders[index].finish()
    payload_bytes += len(payload)
    seed_bytes_total += len(seed_bytes)
    total_tokens += int(flat.size)
    predicted_tokens += max(frames - seed_frames, 0) * TOKENS_PER_FRAME
    records.append({
      "name": segment.name,
      "frames": frames,
      "seed_frames": seed_frames,
      "seed_bytes": seed_bytes,
      "payload": payload,
    })
    segment_reports.append({
      "name": segment.name,
      "frames": frames,
      "seed_frames": seed_frames,
      "seed_bytes": len(seed_bytes),
      "payload_bytes": len(payload),
      "predicted_bits_per_token": stats[index].total_bits_estimate / max(stats[index].symbols, 1),
    })

  archive_bytes, archive_breakdown = build_student_archive(
    records,
    count_total=config.count_total,
    context_frames=runtime.context_frames,
  )
  total_time = time.perf_counter() - start_time
  logical_original_bytes = total_tokens * 10 / 8
  raw_seed_tokens = total_tokens - predicted_tokens

  report = {
    "total_segments": len(segments),
    "total_tokens": total_tokens,
    "predicted_tokens": predicted_tokens,
    "raw_seed_tokens": raw_seed_tokens,
    "seed_frames": config.seed_frames if config.seed_frames is not None else runtime.context_frames,
    "logical_original_bytes": logical_original_bytes,
    "compressed_bytes": len(archive_bytes),
    "payload_bytes": payload_bytes,
    "seed_bytes": seed_bytes_total,
    "archive_overhead_bytes": len(archive_bytes) - payload_bytes - seed_bytes_total,
    "compression_ratio": logical_original_bytes / max(len(archive_bytes), 1),
    "bits_per_token": (len(archive_bytes) * 8) / max(total_tokens, 1),
    "payload_bits_per_predicted_token": (payload_bytes * 8) / max(predicted_tokens, 1),
    "seed_bits_per_total_token": (seed_bytes_total * 8) / max(total_tokens, 1),
    "compress_seconds": total_time,
    "predict_seconds": predict_seconds,
    "arithmetic_seconds": arithmetic_seconds,
    "encode_tokens_per_second": total_tokens / max(total_time, 1e-9),
    "runtime": runtime.summary(),
    "count_total": config.count_total,
    "archive_breakdown": archive_breakdown,
    "segments": segment_reports,
  }
  return archive_bytes, report


def _decode_symbol_row(decoder: ArithmeticDecoder, cumulative: np.ndarray, total: int) -> int:
  target = decoder.get_target(total)
  symbol = int(np.searchsorted(cumulative, target, side="right") - 1)
  low = int(cumulative[symbol])
  high = int(cumulative[symbol + 1])
  decoder.update(low, high, total)
  return symbol


def decompress_student_archive(
  data: bytes,
  *,
  model_path: str | Path,
  runtime: StudentRuntime | None = None,
  device_override: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  header, records, archive_breakdown = parse_student_archive(data)
  runtime = runtime or StudentRuntime(
    StudentRuntimeConfig(
      model_path=str(model_path),
      device=device_override or "auto",
    ),
  )
  if runtime.context_frames != int(header["context_frames"]):
    raise ValueError(
      f"Student model expects {runtime.context_frames} context frames but archive uses {header['context_frames']}",
    )

  count_total = int(header["count_total"])
  start_time = time.perf_counter()
  predict_seconds = 0.0
  arithmetic_seconds = 0.0

  flat_outputs: list[np.ndarray] = []
  decoder_states: list[ArithmeticDecoder] = []
  grouped_indices: dict[int, list[int]] = defaultdict(list)
  for index, record in enumerate(records):
    frames = int(record["frames"])
    seed_frames = int(record["seed_frames"])
    flat = np.zeros((frames, TOKENS_PER_FRAME), dtype=np.int16)
    if seed_frames > 0:
      seed_values = unpack_uint10(record["seed_bytes"], seed_frames * TOKENS_PER_FRAME)
      flat[:seed_frames] = seed_values.reshape(seed_frames, TOKENS_PER_FRAME).astype(np.int16, copy=False)
    flat_outputs.append(flat)
    decoder_states.append(ArithmeticDecoder(record["payload"]))
    grouped_indices[frames].append(index)

  for frames in sorted(grouped_indices):
    indices = grouped_indices[frames]
    for batch_start in range(0, len(indices), runtime.config.batch_size):
      batch_indices = indices[batch_start:batch_start + runtime.config.batch_size]
      batch_flat = np.stack([flat_outputs[index] for index in batch_indices]).astype(np.int32, copy=True)
      batch_decoders = [decoder_states[index] for index in batch_indices]
      batch_seed_frames = min(int(records[index]["seed_frames"]) for index in batch_indices)
      for frame_index in range(batch_seed_frames, frames):
        incomplete_offsets = [offset for offset, record_index in enumerate(batch_indices) if frame_index < int(records[record_index]["seed_frames"])]
        if incomplete_offsets:
          continue
        context = _build_padded_context(batch_flat, frame_index=frame_index, context_frames=runtime.context_frames)
        probs, seconds = runtime.predict_probs(context)
        predict_seconds += seconds
        cumulative = _quantize_prob_rows_cpu(probs, count_total)
        arithmetic_start = time.perf_counter()
        for batch_offset, decoder in enumerate(batch_decoders):
          decoded_frame = batch_flat[batch_offset, frame_index]
          for position in range(TOKENS_PER_FRAME):
            decoded_frame[position] = _decode_symbol_row(decoder, cumulative[batch_offset, position], count_total)
        arithmetic_seconds += time.perf_counter() - arithmetic_start
      for batch_offset, record_index in enumerate(batch_indices):
        flat_outputs[record_index][:] = batch_flat[batch_offset].astype(np.int16, copy=False)

  segments = [
    {
      "name": record["name"],
      "tokens": flat_outputs[index].reshape(int(record["frames"]), 8, 16),
      "mode": STUDENT_ARCHIVE_MODE,
    }
    for index, record in enumerate(records)
  ]
  total_tokens = sum(int(segment["tokens"].size) for segment in segments)
  total_time = time.perf_counter() - start_time
  report = {
    "segments": len(segments),
    "total_tokens": total_tokens,
    "decompress_seconds": total_time,
    "predict_seconds": predict_seconds,
    "arithmetic_seconds": arithmetic_seconds,
    "decode_tokens_per_second": total_tokens / max(total_time, 1e-9),
    "runtime": runtime.summary(),
    "archive_breakdown": archive_breakdown,
  }
  return segments, report


def save_archive(path: str | Path, data: bytes) -> None:
  Path(path).write_bytes(data)


def load_archive(path: str | Path) -> bytes:
  return Path(path).read_bytes()
