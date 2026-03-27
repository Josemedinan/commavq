# Student Final Results

## Final Candidate

Best self-contained model:

- `artifacts/student_adapter16_full_ft2_q8.bin`

Exact held-out benchmark:

- `artifacts/benchmark_student_adapter16_full_ft2_q8_c1073741824_heldout.json`

Core metrics:

| model | predicted bpt | archive bpt | ratio | archive bytes | payload bytes | exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `student_adapter16_full_ft2_q8` | `3.9952` | `3.9968` | `2.502x` | `306954` | `306834` | `true` |

This is the current best local result for the self-contained student submission path.

## Method

The winning path keeps decode fully local and GPT-free:

1. train the frame-level student on full `1200`-frame segments
2. compress with deterministic padded context using `seed_frames = 0`
3. keep position-aware calibration in checkpoint metadata
4. freeze the long-context backbone
5. train a tiny residual logit adapter with rank `16`
6. quantize the final checkpoint to q8 and package it for submission

The key change over the earlier `~2.48x` line was the adapter. It gave a small but real cross-entropy reduction while adding only a compact correction layer on top of the existing model.

## Files Kept In Repo

Only the final reproducible artifacts are retained:

- `artifacts/student_adapter16_full_ft2.pt`
- `artifacts/student_adapter16_full_ft2_metrics.json`
- `artifacts/student_adapter16_full_ft2_q8.bin`
- `artifacts/benchmark_student_adapter16_full_ft2_q8_c1073741824_heldout.json`

Main code for this path:

- `student_model.py`
- `train_student_final.py`
- `train_student_adapter.py`
- `compress_student.py`
- `decompress_student.py`
- `benchmark_student_final.py`
- `build_student_submission.py`
- `strong_compression/student_codec.py`
- `strong_compression/student_runtime.py`

## Recommended Use

Compression:

```bash
python compress_student.py \
  --shards /path/to/data-0000.tar.gz \
  --model artifacts/student_adapter16_full_ft2_q8.bin \
  --output data.bin \
  --device auto \
  --precision float32 \
  --batch-size 16
```

Decompression:

```bash
python decompress_student.py \
  --input data.bin \
  --model artifacts/student_adapter16_full_ft2_q8.bin \
  --output-dir out_tokens \
  --device auto \
  --precision float32 \
  --batch-size 16
```

The q8 artifact already carries the recommended temperature metadata, and the default arithmetic resolution in the compressor is set to the winning high-precision setting.

## Honest Summary

This repository now contains a clean final path that:

- is strictly lossless
- is self-contained at decode time
- reaches `2.502x` locally on the exact held-out benchmark
- keeps only the final model and benchmark artifacts instead of the full exploration trail
