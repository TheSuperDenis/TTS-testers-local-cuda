# GPU TTS Docker Launcher

Standalone Windows PowerShell launcher for local, Dockerized CUDA TTS voice-clone testing and inference-speed comparison.

PowerShell acts as the frontend menu. Docker runs one selected model container at a time, writes generated WAV/JSON files to `output`, and removes the selected one-off container when you quit so the model unloads from GPU memory.

## What This Is For

This repo is for local CUDA TTS testing: load one model at a time, synthesize text against a local reference voice, and compare how fast inference runs on your own GPU.

Included model containers:

- `xtts`: XTTS-v2 voice clone.
- `qwen`: Qwen3-TTS 12Hz 0.6B Base voice clone.
- `miotts`: MioTTS 0.1B voice clone.
- `luxtts`: LuxTTS 100M voice clone.

The repo does not include voice samples, transcripts, model weights, generated outputs, API keys, or Docker image exports.

This is a local testing bench, not a production API service or hosted TTS product.

## License

This project is source-available for local testing and evaluation only. It is not open-source licensed.

You may clone, modify, build, and run it locally for testing. You may not use it commercially, provide it as a hosted service, or redistribute built Docker images, containers, model weights, downloaded caches, generated service bundles, or other built artifacts.

See [LICENSE](LICENSE) for the full terms. Third-party models and dependencies keep their own licenses and usage restrictions.

## Hardware Target

The default build targets NVIDIA RTX 50-series / Blackwell GPUs using PyTorch CUDA 13 wheels. The primary tested target is an RTX 5070.

Default CUDA stack:

```text
torch==2.11.0
torchaudio==2.11.0
https://download.pytorch.org/whl/cu130
```

The launcher detects an RTX 5070 with `nvidia-smi` and exports `TTS_GPU` so Docker exposes only that GPU to the selected container. Inside each container, `CUDA_VISIBLE_DEVICES=0` makes the selected GPU appear as device 0.

For another compatible NVIDIA GPU, pass a Docker GPU device id or UUID:

```powershell
.\run.ps1 -Gpu 0
```

To force CUDA 12.8 wheels instead of CUDA 13:

```powershell
.\run.ps1 -Cuda128
```

## RTX 5070 Test Notes

These are informal local measurements from short voice-clone prompts on an RTX 5070. They are included only as rough comparison points for local CUDA inference speed; your results will vary with prompt length, model cache warmth, Docker/image state, driver stack, and reference audio.

No full machine details are published here.

| Model | Test text length | Model load / prep observed | Generated audio | Synthesis elapsed | RTF | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LuxTTS 100M | 87 chars | ~18.2s load + voice prompt prep | 7.55s | 3.94s | 0.52x | Fast after load; tuned slower to reduce skipped words. |
| MioTTS 0.1B | 84 chars | ~10.9s load + codec + voice embedding | 5.28s | 12.54s | 2.37x | Token cap reduced drawn-out vowel loops. |
| Qwen3-TTS 0.6B Base | 64 chars | ~35.6s load + voice prompt prep | 4.88s | 19.61s | 4.02x | Runs on SDPA; `flash_attention_3` failed on the RTX 5070 target. |

## First Run

1. Add your local voice reference files. These are intentionally ignored by git.

Recommended filenames:

```text
reference.wav
reference.txt
```

`reference.wav` should be a clean sample of the target voice. `reference.txt` should contain the exact words spoken in that WAV.

Private voice-clone inputs are runtime inputs only. Do not put proprietary voice samples, transcripts, API keys, or per-user clone settings in Python files or commits. If another app stores the selected clone settings in a database, pass those resolved local file paths into this launcher at runtime with `-Speaker` and `-ReferenceText`, or through the model-specific environment variables.

2. Open PowerShell in this folder.

3. Run:

```powershell
.\run.ps1
```

The menu asks which model to run. After the selected model loads, type text and press Enter.

Do not type synthesis text while a model is still downloading or loading. Wait for the model-specific `READY` line and prompt:

```text
xttsv2>
qwen0.6btts>
miotts0.1b>
luxtts100m>
```

Outputs are written locally and ignored by git:

```text
output/
output/qwen/
output/miotts/
output/luxtts/
```

Model downloads are cached under `.cache` and are also ignored by git.

## Interactive Commands

Paste multiple lines:

```text
:paste
```

Finish pasted text with a line containing only:

```text
:end
```

Synthesize a mounted prompt file:

```text
:file /workspace/prompt.txt
```

Quit and unload the selected container:

```text
:q
```

## Model Tuning

LuxTTS defaults are tuned for slower, less-skippy output:

```text
LUXTTS_SPEED=0.68
LUXTTS_NUM_STEPS=6
LUXTTS_T_SHIFT=0.35
LUXTTS_PROMPT_DURATION=1000.0
```

`LUXTTS_PROMPT_DURATION=1000.0` tells LuxTTS to use the full reference WAV with the full reference transcript.

MioTTS defaults are tuned to reduce drawn-out vowel/syllable loops:

```text
MIOTTS_MAX_NEW_TOKENS=520
MIOTTS_MAX_TOKEN_OVERHEAD=80
MIOTTS_MAX_TOKENS_PER_CHAR=3.0
MIOTTS_NO_REPEAT_NGRAM_SIZE=6
MIOTTS_REPETITION_PENALTY=1.15
MIOTTS_TEMPERATURE=0.55
MIOTTS_TOP_P=0.85
```

Qwen includes the official CUDA 13 `flash-attn-3` wheel plus a compatibility shim for Qwen's older `flash_attn.flash_attn_interface` import. Keep Qwen's main attention mode at `sdpa`; `flash_attention_3` was tested on the RTX 5070 target and failed during generation with a CUDA kernel-image error.

## Useful Options

```powershell
.\run.ps1 -Model xtts
.\run.ps1 -Model qwen
.\run.ps1 -Model miotts
.\run.ps1 -Model luxtts
.\run.ps1 -Speaker "reference.wav" -ReferenceText "reference.txt" -Language en
.\run.ps1 -NoBuild
.\run.ps1 -Model qwen -Rebuild
```

Speaker files can live in this folder or in `voices`. This folder is mounted read-only at `/workspace`, and `voices` is mounted at `/app/voices`.

Do not use `docker compose up --build` for normal use. The model services are behind Compose profiles to prevent all models from starting together. Use `.\run.ps1` as the frontend.
