from __future__ import annotations

import re
import struct
import zlib
from typing import Any

MAGIC = b"SVQ1"
HEADER_STRUCT = struct.Struct("<4sBBHIIII")
FLAG_NAMES_ZLIB = 1 << 0
FLAG_TABLE_ZLIB = 1 << 1
PACKED_NAME_PATTERN = re.compile(r"^([0-9a-f]{32})_([0-9]+)\.token\.npy$")


def _encode_varint(value: int) -> bytes:
  if value < 0:
    raise ValueError("varint only supports non-negative integers")
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
      raise ValueError("Archive varint truncated")
    byte = data[offset]
    offset += 1
    value |= (byte & 0x7F) << shift
    if (byte & 0x80) == 0:
      return value, offset
    shift += 7
    if shift > 63:
      raise ValueError("Archive varint is too large")


def _encode_name_stream(name: str) -> bytes:
  match = PACKED_NAME_PATTERN.match(name)
  if match:
    digest_hex, suffix = match.groups()
    out = bytearray()
    out.append(1)
    out.extend(bytes.fromhex(digest_hex))
    out.extend(_encode_varint(int(suffix)))
    return bytes(out)

  name_bytes = name.encode("utf-8")
  out = bytearray()
  out.append(0)
  out.extend(_encode_varint(len(name_bytes)))
  out.extend(name_bytes)
  return bytes(out)


def _decode_name_stream(data: bytes, offset: int) -> tuple[str, int]:
  if offset >= len(data):
    raise ValueError("Archive name stream truncated")

  kind = data[offset]
  offset += 1
  if kind == 1:
    digest = data[offset:offset + 16]
    if len(digest) != 16:
      raise ValueError("Archive packed name truncated")
    offset += 16
    suffix, offset = _decode_varint(data, offset)
    return f"{digest.hex()}_{suffix}.token.npy", offset
  if kind == 0:
    length, offset = _decode_varint(data, offset)
    name_bytes = data[offset:offset + length]
    if len(name_bytes) != length:
      raise ValueError("Archive utf-8 name truncated")
    offset += length
    return name_bytes.decode("utf-8"), offset
  raise ValueError(f"Unsupported archive name kind: {kind}")


def build_student_archive(
  records: list[dict[str, Any]],
  *,
  count_total: int,
  context_frames: int,
) -> tuple[bytes, dict[str, int]]:
  names_blob = bytearray()
  table_blob = bytearray()
  seed_bytes_total = 0
  payload_bytes_total = 0

  for record in records:
    names_blob.extend(_encode_name_stream(record["name"]))
    table_blob.extend(_encode_varint(int(record["frames"])))
    table_blob.extend(_encode_varint(int(record["seed_frames"])))
    table_blob.extend(_encode_varint(len(record["seed_bytes"])))
    table_blob.extend(_encode_varint(len(record["payload"])))
    seed_bytes_total += len(record["seed_bytes"])
    payload_bytes_total += len(record["payload"])

  flags = 0
  raw_names = bytes(names_blob)
  raw_table = bytes(table_blob)
  compressed_names = zlib.compress(raw_names, level=9)
  compressed_table = zlib.compress(raw_table, level=9)

  if len(compressed_names) < len(raw_names):
    flags |= FLAG_NAMES_ZLIB
    stored_names = compressed_names
  else:
    stored_names = raw_names

  if len(compressed_table) < len(raw_table):
    flags |= FLAG_TABLE_ZLIB
    stored_table = compressed_table
  else:
    stored_table = raw_table

  header = HEADER_STRUCT.pack(
    MAGIC,
    1,
    flags,
    int(context_frames),
    int(count_total),
    len(records),
    len(stored_names),
    len(stored_table),
  )

  out = bytearray()
  out.extend(header)
  out.extend(stored_names)
  out.extend(stored_table)
  for record in records:
    out.extend(record["seed_bytes"])
    out.extend(record["payload"])

  breakdown = {
    "fixed_header_bytes": HEADER_STRUCT.size,
    "names_bytes": len(stored_names),
    "table_bytes": len(stored_table),
    "seed_bytes": seed_bytes_total,
    "payload_bytes": payload_bytes_total,
    "total_bytes": len(out),
  }
  return bytes(out), breakdown


def parse_student_archive(data: bytes) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
  if len(data) < HEADER_STRUCT.size:
    raise ValueError("Student archive truncated before header")

  magic, version, flags, context_frames, count_total, num_records, names_len, table_len = HEADER_STRUCT.unpack_from(data, 0)
  if magic != MAGIC:
    raise ValueError("Invalid student archive magic")
  if version != 1:
    raise ValueError(f"Unsupported student archive version: {version}")

  offset = HEADER_STRUCT.size
  names_blob = data[offset:offset + names_len]
  if len(names_blob) != names_len:
    raise ValueError("Student archive truncated in names blob")
  offset += names_len
  table_blob = data[offset:offset + table_len]
  if len(table_blob) != table_len:
    raise ValueError("Student archive truncated in table blob")
  offset += table_len

  raw_names = zlib.decompress(names_blob) if (flags & FLAG_NAMES_ZLIB) else names_blob
  raw_table = zlib.decompress(table_blob) if (flags & FLAG_TABLE_ZLIB) else table_blob
  name_offset = 0
  table_offset = 0
  records: list[dict[str, Any]] = []
  seed_bytes_total = 0
  payload_bytes_total = 0

  for _ in range(num_records):
    name, name_offset = _decode_name_stream(raw_names, name_offset)
    frames, table_offset = _decode_varint(raw_table, table_offset)
    seed_frames, table_offset = _decode_varint(raw_table, table_offset)
    seed_len, table_offset = _decode_varint(raw_table, table_offset)
    payload_len, table_offset = _decode_varint(raw_table, table_offset)
    seed_bytes = data[offset:offset + seed_len]
    if len(seed_bytes) != seed_len:
      raise ValueError("Student archive truncated in raw seed bytes")
    offset += seed_len
    payload = data[offset:offset + payload_len]
    if len(payload) != payload_len:
      raise ValueError("Student archive truncated in arithmetic payload")
    offset += payload_len
    seed_bytes_total += seed_len
    payload_bytes_total += payload_len
    records.append({
      "name": name,
      "frames": int(frames),
      "seed_frames": int(seed_frames),
      "seed_bytes": seed_bytes,
      "payload": payload,
    })

  header = {
    "context_frames": int(context_frames),
    "count_total": int(count_total),
  }
  breakdown = {
    "fixed_header_bytes": HEADER_STRUCT.size,
    "names_bytes": len(names_blob),
    "table_bytes": len(table_blob),
    "seed_bytes": seed_bytes_total,
    "payload_bytes": payload_bytes_total,
    "total_bytes": offset,
  }
  return header, records, breakdown
