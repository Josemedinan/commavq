import lzma

import numpy as np


VOCAB_SIZE = 1024
TOKENS_PER_FRAME = 128


def flatten_tokens(tokens: np.ndarray) -> np.ndarray:
  return tokens.reshape(tokens.shape[0], TOKENS_PER_FRAME).astype(np.int32, copy=False)


def baseline_serialize(tokens: np.ndarray) -> bytes:
  return tokens.astype(np.int16, copy=False).reshape(-1, TOKENS_PER_FRAME).T.ravel().tobytes()


def baseline_deserialize(data: bytes, frames: int) -> np.ndarray:
  values = np.frombuffer(lzma.decompress(data), dtype=np.int16)
  return values.reshape(TOKENS_PER_FRAME, frames).T.reshape(frames, 8, 16)


def baseline_compress(tokens: np.ndarray) -> bytes:
  return lzma.compress(baseline_serialize(tokens), preset=lzma.PRESET_EXTREME | 9)
