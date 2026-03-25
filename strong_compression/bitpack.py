from __future__ import annotations

import numpy as np


def pack_uint10(values: np.ndarray) -> bytes:
  flat = np.asarray(values, dtype=np.uint16).reshape(-1)
  out = bytearray()
  bit_buffer = 0
  bit_count = 0
  for value in flat.tolist():
    if not 0 <= int(value) < 1024:
      raise ValueError(f"Value {value} is out of range for 10-bit packing")
    bit_buffer |= int(value) << bit_count
    bit_count += 10
    while bit_count >= 8:
      out.append(bit_buffer & 0xFF)
      bit_buffer >>= 8
      bit_count -= 8
  if bit_count:
    out.append(bit_buffer & 0xFF)
  return bytes(out)


def unpack_uint10(data: bytes, count: int) -> np.ndarray:
  values = np.empty(count, dtype=np.uint16)
  bit_buffer = 0
  bit_count = 0
  byte_index = 0
  for index in range(count):
    while bit_count < 10:
      if byte_index >= len(data):
        raise ValueError("Packed 10-bit stream is truncated")
      bit_buffer |= data[byte_index] << bit_count
      bit_count += 8
      byte_index += 1
    values[index] = bit_buffer & 0x3FF
    bit_buffer >>= 10
    bit_count -= 10
  return values
