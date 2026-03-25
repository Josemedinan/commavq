from dataclasses import dataclass


STATE_BITS = 32
FULL_RANGE = 1 << STATE_BITS
HALF_RANGE = FULL_RANGE >> 1
QUARTER_RANGE = HALF_RANGE >> 1
THREE_QUARTER_RANGE = QUARTER_RANGE * 3


class BitOutputStream:
  def __init__(self) -> None:
    self._bytes = bytearray()
    self._current_byte = 0
    self._num_bits_filled = 0

  def write(self, bit: int) -> None:
    self._current_byte = (self._current_byte << 1) | bit
    self._num_bits_filled += 1
    if self._num_bits_filled == 8:
      self._bytes.append(self._current_byte)
      self._current_byte = 0
      self._num_bits_filled = 0

  def finish(self) -> bytes:
    if self._num_bits_filled > 0:
      self._current_byte <<= (8 - self._num_bits_filled)
      self._bytes.append(self._current_byte)
      self._current_byte = 0
      self._num_bits_filled = 0
    return bytes(self._bytes)


class BitInputStream:
  def __init__(self, data: bytes) -> None:
    self._data = data
    self._index = 0
    self._current_byte = 0
    self._num_bits_remaining = 0

  def read(self) -> int:
    if self._num_bits_remaining == 0:
      if self._index < len(self._data):
        self._current_byte = self._data[self._index]
        self._index += 1
      else:
        self._current_byte = 0
      self._num_bits_remaining = 8
    self._num_bits_remaining -= 1
    return (self._current_byte >> self._num_bits_remaining) & 1


class ArithmeticEncoder:
  def __init__(self) -> None:
    self.low = 0
    self.high = FULL_RANGE - 1
    self.pending_bits = 0
    self.stream = BitOutputStream()

  def _shift(self, bit: int) -> None:
    self.stream.write(bit)
    for _ in range(self.pending_bits):
      self.stream.write(bit ^ 1)
    self.pending_bits = 0

  def encode(self, low_count: int, high_count: int, total: int) -> None:
    if not (0 <= low_count < high_count <= total):
      raise ValueError("Invalid arithmetic coding interval")
    current_range = self.high - self.low + 1
    self.high = self.low + (current_range * high_count // total) - 1
    self.low = self.low + (current_range * low_count // total)

    while True:
      if self.high < HALF_RANGE:
        self._shift(0)
      elif self.low >= HALF_RANGE:
        self._shift(1)
        self.low -= HALF_RANGE
        self.high -= HALF_RANGE
      elif self.low >= QUARTER_RANGE and self.high < THREE_QUARTER_RANGE:
        self.pending_bits += 1
        self.low -= QUARTER_RANGE
        self.high -= QUARTER_RANGE
      else:
        break
      self.low = (self.low << 1) & (FULL_RANGE - 1)
      self.high = ((self.high << 1) & (FULL_RANGE - 1)) | 1

  def finish(self) -> bytes:
    self.pending_bits += 1
    if self.low < QUARTER_RANGE:
      self._shift(0)
    else:
      self._shift(1)
    return self.stream.finish()


class ArithmeticDecoder:
  def __init__(self, data: bytes) -> None:
    self.low = 0
    self.high = FULL_RANGE - 1
    self.stream = BitInputStream(data)
    self.code = 0
    for _ in range(STATE_BITS):
      self.code = (self.code << 1) | self.stream.read()

  def get_target(self, total: int) -> int:
    current_range = self.high - self.low + 1
    return ((self.code - self.low + 1) * total - 1) // current_range

  def update(self, low_count: int, high_count: int, total: int) -> None:
    if not (0 <= low_count < high_count <= total):
      raise ValueError("Invalid arithmetic coding interval")
    current_range = self.high - self.low + 1
    self.high = self.low + (current_range * high_count // total) - 1
    self.low = self.low + (current_range * low_count // total)

    while True:
      if self.high < HALF_RANGE:
        pass
      elif self.low >= HALF_RANGE:
        self.low -= HALF_RANGE
        self.high -= HALF_RANGE
        self.code -= HALF_RANGE
      elif self.low >= QUARTER_RANGE and self.high < THREE_QUARTER_RANGE:
        self.low -= QUARTER_RANGE
        self.high -= QUARTER_RANGE
        self.code -= QUARTER_RANGE
      else:
        break
      self.low = (self.low << 1) & (FULL_RANGE - 1)
      self.high = ((self.high << 1) & (FULL_RANGE - 1)) | 1
      self.code = ((self.code << 1) & (FULL_RANGE - 1)) | self.stream.read()


@dataclass
class CodelengthStats:
  total_bits_estimate: float = 0.0
  symbols: int = 0

  def add(self, probability_mass: int, total: int) -> None:
    import math
    self.total_bits_estimate += -math.log2(probability_mass / total)
    self.symbols += 1
