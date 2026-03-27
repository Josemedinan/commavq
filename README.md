<div align="center">
<h1>commaVQ challenge</h1>

<h3>
  <a href="https://comma.ai/leaderboard">Leaderboard</a>
  <span> · </span>
  <a href="https://comma.ai/jobs">comma.ai/jobs</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Discord</a>
  <span> · </span>
  <a href="https://x.com/comma_ai">X</a>
</h3>

</div>


| Source Video    | Compressed Video | Future Prediction |
| --------------- | ---------------- |------------------ |
| <video src="https://github.com/commaai/commavq/assets/29985433/91894bf7-592b-4204-b3f2-3e805984045c">  |  <video src="https://github.com/commaai/commavq/assets/29985433/3a799ac8-781e-461c-bf14-c15cea42b985">    |  <video src="https://github.com/commaai/commavq/assets/29985433/f6f7699b-b6cb-4f9c-80c9-8e00d75fbfae"> |

A world model is a model that can predict the next state of the world given the observed previous states and actions.

World models are essential to training all kinds of intelligent agents, especially self-driving models.

commaVQ contains:
- encoder/decoder models used to heavily compress driving scenes
- a world model trained on 3,000,000 minutes of driving videos
- a dataset of 100,000 minutes of compressed driving videos

# Task

## Lossless compression challenge: make me smaller! $500 challenge
Losslessly compress 5,000 minutes of driving video "tokens". Go to [./compression/](./compression/) to start

**Prize: highest compression rate on 5,000 minutes of driving video (~915MB) - Challenge ended July, 1st 2024 11:59pm AOE**

Submit a single zip file containing the compressed data and a python script to decompress it into its original form using [this form](https://forms.gle/US88Hg7UR6bBuW3BA). Top solutions are listed on [comma's official leaderboard](https://comma.ai/leaderboard).

<!-- TABLE-START -->
<table class="ranked">
 <thead>
  <tr>
   <th>
   </th>
   <th>
    score
   </th>
   <th>
    name
   </th>
   <th>
    method
   </th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>
   </td>
   <td>
    3.4
   </td>
   <td>
    <a href="https://github.com/szabolcs-cs">
     szabolcs-cs
    </a>
   </td>
   <td>
    self-compressing neural network
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.9
   </td>
   <td>
    <a href="https://github.com/BradyWynn">
     BradyWynn
    </a>
   </td>
   <td>
    arithmetic coding with GPT
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.7
   </td>
   <td>
    <a href="https://github.com/ylevental">
     ylevental
    </a>
   </td>
   <td>
    arithmetic coding with GPT
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.6
   </td>
   <td>
    <a href="https://github.com/pkourouklidis">
     pkourouklidis
    </a>
    👑
   </td>
   <td>
    arithmetic coding with GPT
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.3
   </td>
   <td>
    anonymous
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.3
   </td>
   <td>
    <a href="https://github.com/rostislav">
     rostislav
    </a>
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.2
   </td>
   <td>
    anonymous
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.2
   </td>
   <td>
    anonymous
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.2
   </td>
   <td>
    <a href="https://github.com/0x41head">
     0x41head
    </a>
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.2
   </td>
   <td>
    <a href="https://github.com/tillinf">
     tillinf
    </a>
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.2
   </td>
   <td>
    <a href="https://github.com/ylevental">
     ylevental
    </a>
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.2
   </td>
   <td>
    <a href="https://github.com/nuniesmith">
     nuniesmith
    </a>
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    1.6
   </td>
   <td>
    baseline
   </td>
   <td>
    lzma
   </td>
  </tr>
 </tbody>
</table>
<!-- TABLE-END -->

## Overview
A VQ-VAE [1,2] was used to heavily compress each video frame into 128 "tokens" of 10 bits each. Each entry of the dataset is a "segment" of compressed driving video, i.e. 1min of frames at 20 FPS. Each file is of shape 1200x8x16 and saved as int16.

A world model [3] was trained to predict the next token given a context of past tokens. This world model is a Generative Pre-trained Transformer (GPT) [4] trained on 3,000,000 minutes of driving videos following a similar recipe to [5].

## Examples
[./notebooks/encode.ipynb](./notebooks/encode.ipynb) and [./notebooks/decode.ipynb](./notebooks/decode.ipynb) for an example of how to visualize the dataset using a segment of driving video from [comma's drive to Taco Bell](https://blog.comma.ai/taco-bell/)

[./notebooks/gpt.ipynb](./notebooks/gpt.ipynb) for an example of how to use the world model to imagine future frames.

[./compression/compress.py](./compression/compress.py) for an example of how to compress the tokens using lzma

## Local Student Submission Improvements

This repository now also includes a self-contained student-based lossless compressor that was tuned specifically to push the local compression rate beyond `2.5x` without requiring the original GPT checkpoint at decode time.

### Method Used

The final path keeps the original challenge format and lossless requirement, but replaces the baseline compressor with:

- a frame-level student predictor in [./student_model.py](./student_model.py)
- a small residual logit adapter trained on top of that predictor
- deterministic arithmetic coding in [./strong_compression/student_codec.py](./strong_compression/student_codec.py)
- a compact archive format in [./strong_compression/student_archive.py](./strong_compression/student_archive.py)
- an autocontained submission builder in [./build_student_submission.py](./build_student_submission.py)

The key improvement sequence was:

- train the student on full `1200`-frame segments instead of short windows
- switch compression to deterministic padded context with `seed_frames = 0`
- calibrate logits with position-aware temperature and bias terms
- add a tiny residual adapter head that only learns a correction on top of the best long-context student

That combination reduced cross-entropy enough for the final q8 artifact to cross `2.5x` locally while keeping decode self-contained.

### Why This Solution Is Good

- It is strictly lossless: decoded output matches the original tokens bit-for-bit.
- It is autocontained: decode does not download or depend on the GPT checkpoint.
- It is practical on Apple Silicon: the final path runs on a MacBook Air M3 with `mps` or `cpu`.
- It is compact: the final quantized student artifact is small enough to package cleanly inside the submission zip.

### Final Local Result

The best local submission candidate is documented in [./RESULTS_STUDENT_LONG_FINAL.md](./RESULTS_STUDENT_LONG_FINAL.md).

Headline numbers:

- `4 x 1200` frames held-out: `3.9968` archive bits/token, `2.5020x`
- exact roundtrip: `true`
- predictor path remains fully local and GPT-free at decode time

Main files for that path:

- [./compress_student.py](./compress_student.py)
- [./decompress_student.py](./decompress_student.py)
- [./quantize_student.py](./quantize_student.py)
- [./train_student_final.py](./train_student_final.py)
- [./train_student_adapter.py](./train_student_adapter.py)
- [./benchmark_student_final.py](./benchmark_student_final.py)
- final q8 artifact: `artifacts/student_adapter16_full_ft2_q8.bin`

### Future Improvements

The most promising next steps are still aligned with this same path:

- continue adapter fine-tuning on a broader full-length shard mix
- test slightly larger adapter ranks only if they beat the current `16`-rank line on exact held-out bpt
- improve calibration and benchmark coverage on larger shard sets
- reduce submission size further without changing the lossless decode path

## Download the dataset
- Using huggingface datasets
```python
import numpy as np
from datasets import load_dataset
# load the first shard
data_files = {'train': ['data-0000.tar.gz']}
ds = load_dataset('commaai/commavq', data_files=data_files)
tokens = np.array(ds['train'][0]['token.npy'])
poses = np.array(ds['train'][0]['pose.npy'])
```
- Manually download from huggingface datasets repository: https://huggingface.co/datasets/commaai/commavq

## References
[1] Van Den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation learning." Advances in neural information processing systems 30 (2017).

[2] Esser, Patrick, Robin Rombach, and Bjorn Ommer. "Taming transformers for high-resolution image synthesis." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

[3] https://worldmodels.github.io/

[4] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

[5] Micheli, Vincent, Eloi Alonso, and François Fleuret. "Transformers are Sample-Efficient World Models." The Eleventh International Conference on Learning Representations. 2022.
