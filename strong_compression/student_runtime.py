from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .transforms import VOCAB_SIZE
from .student_quantization import load_student_model_artifact


def select_torch_device(torch_module, preference: str) -> str:
  preference = preference.lower()
  if preference == "auto":
    if torch_module.cuda.is_available():
      return "cuda"
    if getattr(torch_module.backends, "mps", None) is not None and torch_module.backends.mps.is_available():
      return "mps"
    return "cpu"
  if preference == "cuda" and not torch_module.cuda.is_available():
    raise RuntimeError("CUDA requested but not available")
  if preference == "mps":
    if getattr(torch_module.backends, "mps", None) is None or not torch_module.backends.mps.is_available():
      raise RuntimeError("MPS requested but not available")
  return preference


def select_model_dtype(torch_module, precision: str):
  precision = precision.lower()
  if precision == "float16":
    return torch_module.float16
  if precision == "bfloat16":
    return torch_module.bfloat16
  return torch_module.float32


@dataclass
class StudentRuntimeConfig:
  model_path: str = "artifacts/student_model_q8.bin"
  device: str = "auto"
  precision: str = "float32"
  batch_size: int = 16
  temperature: float | None = None


class StudentRuntime:
  def __init__(self, config: StudentRuntimeConfig = StudentRuntimeConfig()) -> None:
    import torch

    self._torch = torch
    self.config = config
    self.device = select_torch_device(torch, config.device)
    self.model_dtype = select_model_dtype(torch, config.precision)
    self.model_path = Path(config.model_path)
    if not self.model_path.exists():
      raise FileNotFoundError(f"Student model artifact not found at {self.model_path}")
    self.model, self.metadata = load_student_model_artifact(self.model_path, map_location="cpu")
    self.model.to(device=self.device, dtype=self.model_dtype)
    self.model.eval()
    recommended_temperature = self.metadata.get("extra", {}).get("recommended_temperature", 1.0)
    self.temperature = float(config.temperature if config.temperature is not None else recommended_temperature)
    if self.temperature <= 0.0:
      raise ValueError(f"Student runtime temperature must be positive, got {self.temperature}")
    if self.device != "cpu":
      torch.set_float32_matmul_precision("high")

  @property
  def context_frames(self) -> int:
    return int(self.model.config.context_frames)

  def predict_probs(self, context_batch: np.ndarray) -> tuple[np.ndarray, float]:
    torch = self._torch
    context_array = np.asarray(context_batch, dtype=np.int64)
    start_time = time.perf_counter()
    context_tensor = torch.from_numpy(context_array).to(device=self.device, dtype=torch.long)
    with torch.inference_mode():
      logits = self.model(context_tensor)
      probs = torch.softmax(logits.float() / self.temperature, dim=-1).cpu().numpy()
    return probs.astype(np.float64, copy=False), time.perf_counter() - start_time

  def predict_logits(self, context_batch: np.ndarray) -> tuple[np.ndarray, float]:
    torch = self._torch
    context_array = np.asarray(context_batch, dtype=np.int64)
    start_time = time.perf_counter()
    context_tensor = torch.from_numpy(context_array).to(device=self.device, dtype=torch.long)
    with torch.inference_mode():
      logits = (self.model(context_tensor).float() / self.temperature).cpu().numpy()
    return logits.astype(np.float32, copy=False), time.perf_counter() - start_time

  def summary(self) -> dict[str, object]:
    return {
      "device": self.device,
      "precision": self.config.precision,
      "batch_size": self.config.batch_size,
      "model_path": str(self.model_path),
      "model_bits": self.metadata.get("bits"),
      "context_frames": self.context_frames,
      "vocab_size": VOCAB_SIZE,
      "temperature": self.temperature,
    }
