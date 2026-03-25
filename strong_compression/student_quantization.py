from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from student_model import StudentFramePredictor, StudentPredictorConfig, load_student_checkpoint


MAGIC = b"STQ1"
HEADER_STRUCT = struct.Struct("<4sBBI")


@dataclass
class QuantizedStudentStats:
  original_bytes: int
  quantized_bytes: int
  compressed_bytes: int
  parameters: int
  bits: int

  def to_dict(self) -> dict[str, int]:
    return {
      "original_bytes": self.original_bytes,
      "quantized_bytes": self.quantized_bytes,
      "compressed_bytes": self.compressed_bytes,
      "parameters": self.parameters,
      "bits": self.bits,
    }


def _encode_varint(value: int) -> bytes:
  out = bytearray()
  while value >= 0x80:
    out.append((value & 0x7F) | 0x80)
    value >>= 7
  out.append(value)
  return bytes(out)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
  value = 0
  shift = 0
  while True:
    if offset >= len(data):
      raise ValueError("Quantized student artifact truncated")
    byte = data[offset]
    offset += 1
    value |= (byte & 0x7F) << shift
    if (byte & 0x80) == 0:
      return value, offset
    shift += 7


def _tensor_original_bytes(tensor: torch.Tensor) -> int:
  return int(tensor.numel() * tensor.element_size())


def quantize_state_dict(
  state_dict: dict[str, torch.Tensor],
  *,
  bits: int,
) -> tuple[list[dict[str, Any]], QuantizedStudentStats]:
  if bits != 8:
    raise ValueError("Only 8-bit quantization is implemented in this first student round")

  quantized_entries: list[dict[str, Any]] = []
  original_bytes = 0
  quantized_bytes = 0
  parameters = 0
  for name, tensor in state_dict.items():
    flat = tensor.detach().cpu().float().contiguous().view(-1).numpy()
    parameters += int(flat.size)
    original_bytes += _tensor_original_bytes(tensor)
    min_val = float(np.min(flat))
    max_val = float(np.max(flat))
    if max_val == min_val:
      scale = 1.0
      quantized = np.zeros_like(flat, dtype=np.uint8)
    else:
      scale = (max_val - min_val) / 255.0
      quantized = np.clip(np.round((flat - min_val) / scale), 0, 255).astype(np.uint8)
    quantized_bytes += int(quantized.nbytes)
    quantized_entries.append({
      "name": name,
      "shape": tuple(int(dim) for dim in tensor.shape),
      "min_val": min_val,
      "scale": float(scale),
      "data": quantized,
    })

  stats = QuantizedStudentStats(
    original_bytes=original_bytes,
    quantized_bytes=quantized_bytes,
    compressed_bytes=0,
    parameters=parameters,
    bits=bits,
  )
  return quantized_entries, stats


def pack_quantized_student(
  config: StudentPredictorConfig,
  quantized_entries: list[dict[str, Any]],
  *,
  bits: int,
  extra: dict[str, Any] | None = None,
) -> bytes:
  config_blob = json.dumps(
    {
      "config": config.to_dict(),
      "extra": extra or {},
    },
    sort_keys=True,
  ).encode("utf-8")

  body = bytearray()
  body.extend(_encode_varint(len(config_blob)))
  body.extend(config_blob)
  body.extend(_encode_varint(len(quantized_entries)))
  for entry in quantized_entries:
    name_bytes = entry["name"].encode("utf-8")
    body.extend(_encode_varint(len(name_bytes)))
    body.extend(name_bytes)
    shape = entry["shape"]
    body.extend(_encode_varint(len(shape)))
    for dim in shape:
      body.extend(_encode_varint(int(dim)))
    body.extend(struct.pack("<ff", float(entry["min_val"]), float(entry["scale"])))
    data = entry["data"].tobytes()
    body.extend(_encode_varint(len(data)))
    body.extend(data)

  compressed_body = zlib.compress(bytes(body), level=9)
  return HEADER_STRUCT.pack(MAGIC, 1, bits, len(body)) + compressed_body


def unpack_quantized_student(data: bytes) -> tuple[StudentFramePredictor, dict[str, Any]]:
  if len(data) < HEADER_STRUCT.size:
    raise ValueError("Quantized student artifact is truncated")

  magic, version, bits, _raw_len = HEADER_STRUCT.unpack_from(data, 0)
  if magic != MAGIC:
    raise ValueError("Invalid quantized student artifact magic")
  if version != 1:
    raise ValueError(f"Unsupported quantized student artifact version: {version}")
  if bits != 8:
    raise ValueError(f"Unsupported quantized student bit-width: {bits}")

  raw = zlib.decompress(data[HEADER_STRUCT.size:])
  offset = 0
  config_len, offset = _decode_varint(raw, offset)
  config_blob = raw[offset:offset + config_len]
  offset += config_len
  config_payload = json.loads(config_blob.decode("utf-8"))
  config = StudentPredictorConfig.from_dict(config_payload["config"])
  model = StudentFramePredictor(config)

  num_entries, offset = _decode_varint(raw, offset)
  state_dict: dict[str, torch.Tensor] = {}
  for _ in range(num_entries):
    name_len, offset = _decode_varint(raw, offset)
    name = raw[offset:offset + name_len].decode("utf-8")
    offset += name_len
    ndim, offset = _decode_varint(raw, offset)
    shape = []
    for _ in range(ndim):
      dim, offset = _decode_varint(raw, offset)
      shape.append(dim)
    min_val, scale = struct.unpack_from("<ff", raw, offset)
    offset += 8
    data_len, offset = _decode_varint(raw, offset)
    q_bytes = raw[offset:offset + data_len]
    offset += data_len
    q_array = np.frombuffer(q_bytes, dtype=np.uint8).astype(np.float32)
    if float(scale) == 0.0:
      restored = np.full(q_array.shape, float(min_val), dtype=np.float32)
    else:
      restored = q_array * float(scale) + float(min_val)
    state_dict[name] = torch.from_numpy(restored.reshape(shape))

  model.load_state_dict(state_dict, strict=True)
  metadata = {
    "config": config_payload["config"],
    "extra": config_payload.get("extra", {}),
    "bits": int(bits),
  }
  return model, metadata


def quantize_student_checkpoint(
  checkpoint_path: str | Path,
  output_path: str | Path,
  *,
  bits: int = 8,
) -> dict[str, Any]:
  model, payload = load_student_checkpoint(checkpoint_path, map_location="cpu")
  quantized_entries, stats = quantize_state_dict(model.state_dict(), bits=bits)
  artifact = pack_quantized_student(model.config, quantized_entries, bits=bits, extra=payload.get("extra", {}))
  output_path = Path(output_path)
  output_path.write_bytes(artifact)
  stats.compressed_bytes = output_path.stat().st_size
  return {
    "checkpoint_path": str(checkpoint_path),
    "output_path": str(output_path),
    "config": model.config.to_dict(),
    "stats": stats.to_dict(),
  }


def load_student_model_artifact(
  path: str | Path,
  *,
  map_location: str | torch.device = "cpu",
) -> tuple[StudentFramePredictor, dict[str, Any]]:
  path = Path(path)
  data = path.read_bytes()
  if data[:4] == MAGIC:
    model, metadata = unpack_quantized_student(data)
    return model.to(map_location), metadata
  model, payload = load_student_checkpoint(path, map_location=map_location)
  metadata = {
    "config": model.config.to_dict(),
    "extra": payload.get("extra", {}),
    "bits": 32,
  }
  return model, metadata
