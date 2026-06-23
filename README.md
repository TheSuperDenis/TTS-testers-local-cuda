# GPU TTS Docker Launcher

Standalone Windows PowerShell launcher for local, Dockerized CUDA TTS voice-clone and preset-voice testing with inference-speed comparison.

PowerShell acts as the frontend menu. Docker runs one selected model container at a time, writes generated WAV/JSON files to `output`, and removes the selected one-off container when you quit so the model unloads from GPU memory.

## What This Is For

This repo is for local CUDA TTS testing: load one model at a time, synthesize text against a local reference voice or configured preset voice, and compare how fast inference runs on your own GPU.

Included model containers:

- `xtts`: XTTS-v2 voice clone. No official female preset catalog is bundled.
- `qwen`: Qwen3-TTS 12Hz 0.6B Base voice clone, plus official Qwen CustomVoice female presets.
- `miotts`: MioTTS 0.1B voice clone, plus the official MioTTS English female preset embedding.
- `luxtts`: LuxTTS 100M voice clone. No official female preset catalog is bundled.
- `piper`: Piper TTS preset voice picker with American female and British female options. Piper does not do zero-shot voice cloning.
- `kokoro`: Kokoro-82M preset voice picker with all American female and British female presets from the model. Kokoro does not do zero-shot voice cloning.
- `kitten`: KittenTTS 80M preset voice picker with all female presets from the model. KittenTTS does not do zero-shot voice cloning.
- `chatterbox`: Resemble AI Chatterbox 500M voice clone, plus the official built-in default voice fallback.
- `sopro`: SoproTTS 135M English voice clone. No official female preset catalog is bundled.
- `pockettts`: Kyutai PocketTTS 100M voice clone, plus selected official English built-in voices. PocketTTS is CPU-first upstream.
- `moss`: OpenMOSS MOSS-TTS-Nano 100M ONNX voice clone, plus the official English Female built-in voices Ava and Bella.

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
| Piper en_GB Cori high | 60 chars | ~2.8s cached preset voice load | 4.0s | 1.08s | 0.28x | Preset British English voice; no zero-shot clone prep. |
| PocketTTS Alba | 59 chars | ~10.4s load + 1.2s voice load | 5.8s | 3.1s | 0.54x | CPU-first preset path; no RTX 5070 GPU reservation. |
| SoproTTS 135M clone | 45 chars | ~45.3s load + 3.3s voice prep | 3.4s | 2.4s | 0.71x | Real voice-clone path; faster synthesis after load than most clone models here. |
| Kokoro-82M bf_emma | 63 chars | ~7.2s cached preset voice load | 4.4s | 5.1s | 1.15x | Preset British English voice; no zero-shot clone prep. |
| KittenTTS 80M Bella | 59 chars | ~8.0s model + preset voice load | 5.4s | 8.4s | 1.56x | Preset female voice; ONNX Runtime CUDA. |
| MioTTS 0.1B | 84 chars | ~10.9s load + codec + voice embedding | 5.28s | 12.54s | 2.37x | Token cap reduced drawn-out vowel loops. |
| Qwen3-TTS 0.6B Base | 64 chars | ~35.6s load + voice prompt prep | 4.88s | 19.61s | 4.02x | Runs on SDPA; `flash_attention_3` failed on the RTX 5070 target. |
| Chatterbox 500M clone | 50 chars | ~24.4s warm load + 17.4s voice prep | 4.5s | 20.7s | 4.58x | Real voice-clone path works, but this first test was not near real time. |

## First Run

1. Add your local voice reference files. These are intentionally ignored by git.

Recommended filenames:

```text
reference.wav
reference.txt
```

`reference.wav` should be a clean sample of the target voice. `reference.txt` should contain the exact words spoken in that WAV. Chatterbox, SoproTTS, PocketTTS, and MOSS-TTS-Nano use the WAV but do not need the transcript. Piper, Kokoro, and KittenTTS do not use these files because they use configured preset voices instead of zero-shot cloning.

Private voice-clone inputs are runtime inputs only. Do not put proprietary voice samples, transcripts, API keys, or per-user clone settings in Python files or commits. If another app stores the selected clone settings in a database, pass those resolved local file paths into this launcher at runtime with `-Speaker` and `-ReferenceText`, or through the model-specific environment variables.

2. Open PowerShell in this folder.

3. Run:

```powershell
.\run.ps1
```

The menu asks which model to run. For clone-capable models, the first voice option is always your local reference voice clone. If the upstream model publishes usable female presets, those appear after the clone option. If you choose Piper, Kokoro, or KittenTTS, the launcher asks which preset voice to load. After the selected model loads, type text and press Enter.

Do not type synthesis text while a model is still downloading or loading. Wait for the model-specific `READY` line and prompt:

```text
xttsv2>
qwen0.6btts>
miotts0.1b>
luxtts100m>
piper>
kokoro82m>
kittentts>
chatterbox>
soprotts>
pockettts>
mossttsnano>
```

Outputs are written locally and ignored by git:

```text
output/
output/qwen/
output/miotts/
output/luxtts/
output/piper/
output/kokoro/
output/kitten/
output/chatterbox/
output/sopro/
output/pockettts/
output/moss/
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

For Qwen, the picker now offers:

| Voice option | Mode | Notes |
| --- | --- | --- |
| `reference-clone` | Base model voice clone | Uses your local reference WAV and exact transcript. |
| `serena` | CustomVoice preset | Official Qwen preset: warm, gentle young female voice. Native language: Chinese. |
| `vivian` | CustomVoice preset | Official Qwen preset: bright, slightly edgy young female voice. Native language: Chinese. |
| `sohee` | CustomVoice preset | Official Qwen preset: warm Korean female voice with rich emotion. Native language: Korean. |
| `ono-anna` | CustomVoice preset | Official Qwen preset: playful Japanese female voice with a light, nimble timbre. Native language: Japanese. |

Qwen's published CustomVoice list does not include an American female or British female preset. The launcher includes the official female presets only and keeps `reference-clone` as option 1 for your own American/British reference voice.

```powershell
.\run.ps1 -Model qwen -QwenVoice reference-clone
.\run.ps1 -Model qwen -QwenVoice serena
```

For MioTTS, the picker now offers:

| Voice option | Mode | Notes |
| --- | --- | --- |
| `reference-clone` | Voice clone | Uses your local reference WAV. |
| `en-female` | Official preset | Downloads and caches the public `en_female` preset embedding from `Aratako/MioTTS-Inference`. |

```powershell
.\run.ps1 -Model miotts -MioVoice reference-clone
.\run.ps1 -Model miotts -MioVoice en-female
```

Chatterbox uses Resemble AI's Chatterbox 500M TTS model. The picker offers voice cloning first. Chatterbox does not publish named American female or British female presets in the official package, so the only non-clone option exposed here is the built-in default conditioning from the model.

| Voice option | Mode | Notes |
| --- | --- | --- |
| `reference-clone` | Voice clone | Uses your local reference WAV. Chatterbox does not need `reference.txt`. |
| `builtin-default` | Built-in fallback | Uses the official built-in conditioning. It is not labeled upstream as American, British, female, breathy, or lower-pitched. |

```powershell
.\run.ps1 -Model chatterbox -ChatterboxVoice reference-clone
.\run.ps1 -Model chatterbox -ChatterboxVoice builtin-default
```

Chatterbox defaults:

```text
CHATTERBOX_CHUNK_CHARS=220
CHATTERBOX_PAUSE_MS=220
CHATTERBOX_EXAGGERATION=0.45
CHATTERBOX_CFG_WEIGHT=0.35
CHATTERBOX_TEMPERATURE=0.8
CHATTERBOX_TOP_P=1.0
CHATTERBOX_MIN_P=0.05
CHATTERBOX_REPETITION_PENALTY=1.2
```

The Chatterbox splitter preserves pasted dialogue lines as separate turns and splits long sentences around clauses before falling back to word splits.

SoproTTS uses Samuel Vitorino's Sopro 135M English TTS model. The picker offers voice cloning only. SoproTTS does not publish named American female or British female presets in the official package, and its API requires a reference audio path or prepared reference tokens.

| Voice option | Mode | Notes |
| --- | --- | --- |
| `reference-clone` | Voice clone | Uses your local reference WAV. SoproTTS does not need `reference.txt`. |

```powershell
.\run.ps1 -Model sopro -SoproVoice reference-clone
```

SoproTTS defaults:

```text
SOPRO_MODEL_ID=samuel-vitorino/sopro
SOPRO_CHUNK_CHARS=220
SOPRO_PAUSE_MS=220
SOPRO_MAX_FRAMES=400
SOPRO_REF_SECONDS=12.0
SOPRO_STYLE_STRENGTH=1.2
SOPRO_TEMPERATURE=1.05
SOPRO_TOP_P=0.9
```

The SoproTTS splitter preserves pasted dialogue lines as separate turns and splits long sentences around clauses before falling back to word splits.

PocketTTS uses Kyutai's 100M-parameter PocketTTS model. The picker offers voice cloning first, followed by selected official English built-in voices whose names read as likely female voices. Upstream does not label these voices as American, British, female, breathy, sultry, or lower-pitched, so the catalog does not claim those attributes.

PocketTTS is intentionally CPU-first. Kyutai's README says the model runs on CPU, does not require a GPU PyTorch build, and did not show a GPU speedup in their tests. This Docker profile therefore does not reserve the RTX 5070 even though it uses the same PowerShell frontend and output workflow as the CUDA profiles.

Voice cloning requires access to Kyutai's gated PocketTTS voice-cloning weights. Accept the terms on the Hugging Face model page first. After that, either log in locally or set `HF_TOKEN` in the PowerShell session before launching. The launcher will read the token from the current environment or the normal local Hugging Face login file and pass it into Docker for that run. Do not write the token into repo files.

```powershell
uvx hf auth login

# or:
$env:HF_TOKEN = "your-local-hugging-face-token"
.\run.ps1 -Model pockettts -PocketVoice reference-clone
```

| Voice option | Mode | Notes |
| --- | --- | --- |
| `reference-clone` | Voice clone | Uses your local reference WAV. PocketTTS does not need `reference.txt`. |
| `alba` | Built-in voice | Official English PocketTTS voice. |
| `anna` | Built-in voice | Official English PocketTTS voice. |
| `azelma` | Built-in voice | Official English PocketTTS voice. |
| `cosette` | Built-in voice | Official English PocketTTS voice. |
| `eponine` | Built-in voice | Official English PocketTTS voice. |
| `eve` | Built-in voice | Official English PocketTTS voice. |
| `fantine` | Built-in voice | Official English PocketTTS voice. |
| `jane` | Built-in voice | Official English PocketTTS voice. |
| `mary` | Built-in voice | Official English PocketTTS voice. |
| `vera` | Built-in voice | Official English PocketTTS voice. |

```powershell
.\run.ps1 -Model pockettts -PocketVoice reference-clone
.\run.ps1 -Model pockettts -PocketVoice alba
```

PocketTTS defaults:

```text
POCKETTTS_LANGUAGE=english
POCKETTTS_CHUNK_CHARS=220
POCKETTTS_PAUSE_MS=220
POCKETTTS_MAX_TOKENS=320
POCKETTTS_FRAMES_AFTER_EOS=24
POCKETTTS_TRUNCATE_REF=1
```

The PocketTTS splitter preserves pasted dialogue lines as separate turns and splits long sentences around clauses before falling back to word splits.

MOSS-TTS-Nano uses OpenMOSS's 100M-parameter ONNX runtime, not the larger MOSS-TTS 8B stack. The picker offers voice cloning first, followed by the official built-in voices labeled English Female in the MOSS-TTS-Nano manifest. Upstream labels these as English Female, but does not label them American, British, breathy, sultry, or lower-pitched.

The MOSS-Nano path uses ONNX Runtime. This Docker profile defaults to `MOSS_EXECUTION_PROVIDER=cuda` for the RTX 5070 CUDA test bench and installs the matching CUDA PyTorch libraries so ONNX Runtime can load CUDA 13 dependencies. Set `MOSS_EXECUTION_PROVIDER=cpu` if the CUDA provider is not stable on your driver stack.

| Voice option | Mode | Notes |
| --- | --- | --- |
| `reference-clone` | Voice clone | Uses your local reference WAV. MOSS-TTS-Nano does not need `reference.txt`. |
| `ava` | Built-in voice | Official MOSS-TTS-Nano voice labeled English Female. |
| `bella` | Built-in voice | Official MOSS-TTS-Nano voice labeled English Female. |

```powershell
.\run.ps1 -Model moss -MossVoice reference-clone
.\run.ps1 -Model moss -MossVoice ava

$env:MOSS_EXECUTION_PROVIDER = "cpu"
.\run.ps1 -Model moss -MossVoice bella
```

MOSS-TTS-Nano defaults:

```text
MOSS_EXECUTION_PROVIDER=cuda
MOSS_CPU_THREADS=4
MOSS_MAX_NEW_FRAMES=375
MOSS_VOICE_CLONE_MAX_TEXT_TOKENS=75
MOSS_SAMPLE_MODE=fixed
MOSS_REALTIME_STREAMING_DECODE=1
```

MOSS-TTS-Nano performs its own token-budget chunking during synthesis. The wrapper preloads the selected reference or built-in voice before showing `READY`, then writes WAV and JSON files under `output/moss`.

Piper uses the public `rhasspy/piper-voices` ONNX voice set. The PowerShell launcher includes a small curated picker and chooses the highest available quality for each listed voice.

| Group | Voice | Quality used |
| --- | --- | --- |
| American female | Amy | medium |
| American female | HFC Female | medium |
| American female | Kathleen | low |
| American female | Kristin | medium |
| American female | Lessac | high |
| American female | LJSpeech | high |
| British female | Alba | medium |
| British female | Aru | medium |
| British female | Cori | high |
| British female | Jenny Dioco | medium |
| British female | Southern English Female | low |

You can also skip the picker and launch a known catalog id:

```powershell
.\run.ps1 -Model piper -PiperVoice gb-cori
.\run.ps1 -Model piper -PiperVoice us-ljspeech
```

The catalog lives in `piper_service/voice_catalog.json`. Advanced users can still override the exact files with `PIPER_MODEL_FILE`, `PIPER_CONFIG_FILE`, and `PIPER_VOICE_NAME`.

Piper defaults are tuned for dialogue-style pacing instead of one long fast line:

```text
PIPER_CHUNK_CHARS=160
PIPER_PAUSE_MS=260
PIPER_SENTENCE_SILENCE=0.22
PIPER_LENGTH_SCALE=1.08
```

`PIPER_CHUNK_CHARS` controls the maximum text size for each synthesized WAV part. The Piper splitter preserves pasted dialogue lines as separate turns and splits long sentences around clauses before falling back to word splits.

Piper is wired for ONNX Runtime CUDA. The CUDA 13 path uses the current ONNX Runtime CUDA 13 nightly feed plus the PyTorch CUDA stack so it can be tested on the RTX 5070 target. If you need the CUDA 12.8 path, run the launcher with `-Cuda128`.

Kokoro-82M uses the official `hexgrad/Kokoro-82M` voice presets. The launcher includes all American female and British female presets listed for the model:

| Group | Voice id | Voice | Grade |
| --- | --- | --- | --- |
| American female | `af_heart` | Heart | A |
| American female | `af_alloy` | Alloy | C |
| American female | `af_aoede` | Aoede | C+ |
| American female | `af_bella` | Bella | A- |
| American female | `af_jessica` | Jessica | D |
| American female | `af_kore` | Kore | C+ |
| American female | `af_nicole` | Nicole | B- |
| American female | `af_nova` | Nova | C |
| American female | `af_river` | River | D |
| American female | `af_sarah` | Sarah | C+ |
| American female | `af_sky` | Sky | C- |
| British female | `bf_alice` | Alice | D |
| British female | `bf_emma` | Emma | B- |
| British female | `bf_isabella` | Isabella | C |
| British female | `bf_lily` | Lily | D |

You can skip the Kokoro picker and launch a known voice id:

```powershell
.\run.ps1 -Model kokoro -KokoroVoice af_heart
.\run.ps1 -Model kokoro -KokoroVoice bf_emma
```

Kokoro defaults are tuned to avoid rushing on longer input:

```text
KOKORO_CHUNK_CHARS=220
KOKORO_PAUSE_MS=220
KOKORO_SPEED=0.95
```

The Kokoro splitter preserves pasted dialogue lines as separate turns and splits long sentences around clauses before falling back to word splits.

KittenTTS uses the official `KittenML/kitten-tts-mini-0.8` 80M ONNX model by default. The launcher includes all female presets listed for the model:

| Voice id | Voice | Internal id |
| --- | --- | --- |
| `bella` | Bella | `expr-voice-2-f` |
| `luna` | Luna | `expr-voice-3-f` |
| `rosie` | Rosie | `expr-voice-4-f` |
| `kiki` | Kiki | `expr-voice-5-f` |

You can skip the KittenTTS picker and launch a known voice id:

```powershell
.\run.ps1 -Model kitten -KittenVoice bella
.\run.ps1 -Model kitten -KittenVoice luna
```

KittenTTS defaults:

```text
KITTEN_MODEL_ID=KittenML/kitten-tts-mini-0.8
KITTEN_CHUNK_CHARS=220
KITTEN_PAUSE_MS=220
KITTEN_SPEED=1.0
```

KittenTTS is wired for ONNX Runtime CUDA with the same CUDA 13 path used by Piper. If you need the CUDA 12.8 path, run the launcher with `-Cuda128`.

## Useful Options

```powershell
.\run.ps1 -Model xtts
.\run.ps1 -Model qwen
.\run.ps1 -Model qwen -QwenVoice serena
.\run.ps1 -Model miotts
.\run.ps1 -Model miotts -MioVoice en-female
.\run.ps1 -Model luxtts
.\run.ps1 -Model piper
.\run.ps1 -Model piper -PiperVoice gb-cori
.\run.ps1 -Model kokoro
.\run.ps1 -Model kokoro -KokoroVoice af_heart
.\run.ps1 -Model kitten
.\run.ps1 -Model kitten -KittenVoice bella
.\run.ps1 -Model chatterbox
.\run.ps1 -Model chatterbox -ChatterboxVoice builtin-default
.\run.ps1 -Model sopro
.\run.ps1 -Model sopro -SoproVoice reference-clone
.\run.ps1 -Model pockettts
.\run.ps1 -Model pockettts -PocketVoice alba
.\run.ps1 -Model moss
.\run.ps1 -Model moss -MossVoice ava
.\run.ps1 -Speaker "reference.wav" -ReferenceText "reference.txt" -Language en
.\run.ps1 -NoBuild
.\run.ps1 -Model pockettts -Rebuild
```

Speaker files can live in this folder or in `voices`. This folder is mounted read-only at `/workspace`, and `voices` is mounted at `/app/voices`.

Do not use `docker compose up --build` for normal use. The model services are behind Compose profiles to prevent all models from starting together. Use `.\run.ps1` as the frontend.
