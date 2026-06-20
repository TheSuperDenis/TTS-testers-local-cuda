from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from xtts_service.text_splitter import split_long_text


DEFAULT_MODEL = "YatharthS/LuxTTS"
SAMPLE_RATE = 48000
PROMPT = "luxtts100m> "


@dataclass
class RuntimeConfig:
    model_id: str
    output_dir: Path
    ref_audio: Path
    ref_text_file: Path
    chunk_chars: int
    pause_ms: int
    keep_parts: bool
    prompt_duration: float
    rms: float
    num_steps: int
    guidance_scale: float
    t_shift: float
    speed: float
    return_smooth: bool


@dataclass
class LuxRuntime:
    torch: object
    model: object
    feature_extractor: object
    vocos: object
    tokenizer: object
    encoded_prompt: dict[str, object]
    reference_text: str
    device: object


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = RuntimeConfig(
            model_id=args.model_id,
            output_dir=Path(args.output_dir),
            ref_audio=required_path(args.ref_audio, "reference audio"),
            ref_text_file=required_path(args.ref_text_file, "reference transcript"),
            chunk_chars=args.chunk_chars,
            pause_ms=args.pause_ms,
            keep_parts=args.keep_parts,
            prompt_duration=args.prompt_duration,
            rms=args.rms,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            t_shift=args.t_shift,
            speed=args.speed,
            return_smooth=args.return_smooth,
        )
    except ValueError as exc:
        print(f"LuxTTS startup failed: {exc}", file=sys.stderr, flush=True)
        return 1

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.interactive:
        print(
            "LuxTTS 100M is loading. Wait for the READY line and luxtts100m> prompt before typing.",
            flush=True,
        )

    try:
        runtime = load_runtime(config)
    except Exception as exc:
        print(f"LuxTTS startup failed: {exc}", file=sys.stderr, flush=True)
        return 1

    if args.text:
        return synthesize(runtime, config, args.text)

    if args.prompt_file:
        text = Path(args.prompt_file).read_text(encoding="utf-8")
        return synthesize(runtime, config, text)

    if args.interactive:
        return interactive_loop(runtime, config)

    print("No text was provided. Use --interactive, --text, or --prompt-file.", file=sys.stderr)
    return 2


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive LuxTTS voice-clone generator.")
    parser.add_argument("--interactive", action="store_true", help="Read text from stdin repeatedly.")
    parser.add_argument("--text", help="Synthesize one text string and exit.")
    parser.add_argument("--prompt-file", help="Read one UTF-8 text file and exit.")
    parser.add_argument("--model-id", default=os.getenv("LUXTTS_MODEL_ID", DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=os.getenv("LUXTTS_OUTPUT_DIR", "/app/output/luxtts"))
    parser.add_argument("--ref-audio", default=os.getenv("LUXTTS_REF_AUDIO", "/workspace/reference.wav"))
    parser.add_argument("--ref-text-file", default=os.getenv("LUXTTS_REF_TEXT_FILE", "/workspace/reference.txt"))
    parser.add_argument("--chunk-chars", type=int, default=int(os.getenv("LUXTTS_CHUNK_CHARS", "160")))
    parser.add_argument("--pause-ms", type=int, default=int(os.getenv("LUXTTS_PAUSE_MS", "180")))
    parser.add_argument("--prompt-duration", type=float, default=float(os.getenv("LUXTTS_PROMPT_DURATION", "1000.0")))
    parser.add_argument("--rms", type=float, default=float(os.getenv("LUXTTS_RMS", "0.01")))
    parser.add_argument("--num-steps", type=int, default=int(os.getenv("LUXTTS_NUM_STEPS", "6")))
    parser.add_argument("--guidance-scale", type=float, default=float(os.getenv("LUXTTS_GUIDANCE_SCALE", "3.0")))
    parser.add_argument("--t-shift", type=float, default=float(os.getenv("LUXTTS_T_SHIFT", "0.35")))
    parser.add_argument("--speed", type=float, default=float(os.getenv("LUXTTS_SPEED", "0.68")))
    parser.add_argument("--return-smooth", action="store_true", default=env_bool("LUXTTS_RETURN_SMOOTH", False))
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args(argv)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def required_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    raise ValueError(f"{label} not found at {path}")


def load_runtime(config: RuntimeConfig) -> LuxRuntime:
    import torch
    from huggingface_hub import snapshot_download
    from linacodec.vocoder.vocos import Vocos
    from torch.nn.utils import parametrize
    from zipvoice.models.zipvoice_distill import ZipVoiceDistill
    from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
    from zipvoice.utils.checkpoint import load_checkpoint
    from zipvoice.utils.feature import VocosFbank

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the container. Check Docker GPU support and the GPU mask.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    visible = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    print(f"CUDA ready: torch {torch.__version__}, CUDA {torch.version.cuda}, visible GPUs: {visible}", flush=True)

    device = torch.device("cuda", 0)
    print(f"Loading {config.model_id}...", flush=True)
    loaded_at = time.perf_counter()
    model_path = Path(snapshot_download(config.model_id))
    tokenizer = EmiliaTokenizer(token_file=str(model_path / "tokens.txt"))
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    model = ZipVoiceDistill(
        **model_config["model"],
        **tokenizer_config,
    )
    load_checkpoint(filename=str(model_path / "model.pt"), model=model, strict=True)
    model = model.to(device).eval()

    feature_extractor = VocosFbank()
    vocos = Vocos.from_hparams(str(model_path / "vocoder" / "config.yaml")).to(device)
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")
    vocos.load_state_dict(torch.load(model_path / "vocoder" / "vocos.bin", map_location=device))
    vocos.freq_range = 12000
    print(f"Model loaded in {time.perf_counter() - loaded_at:.1f}s.", flush=True)

    reference_text = config.ref_text_file.read_text(encoding="utf-8").strip()
    print(f"Voice clone sample: {config.ref_audio}", flush=True)
    print(f"Voice reference transcript: {config.ref_text_file}", flush=True)
    prompt_started = time.perf_counter()
    encoded_prompt = encode_prompt_from_transcript(
        torch=torch,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device,
        audio_path=config.ref_audio,
        reference_text=reference_text,
        duration=config.prompt_duration,
        rms=config.rms,
    )
    print(f"Voice clone prompt prepared in {time.perf_counter() - prompt_started:.1f}s.", flush=True)

    return LuxRuntime(
        torch=torch,
        model=model,
        feature_extractor=feature_extractor,
        vocos=vocos,
        tokenizer=tokenizer,
        encoded_prompt=encoded_prompt,
        reference_text=reference_text,
        device=device,
    )


def encode_prompt_from_transcript(
    *,
    torch,
    feature_extractor,
    tokenizer,
    device,
    audio_path: Path,
    reference_text: str,
    duration: float,
    rms: float,
) -> dict[str, object]:
    import librosa
    from zipvoice.utils.infer import rms_norm

    prompt_wav, _ = librosa.load(str(audio_path), sr=24000, duration=duration)
    prompt_wav = torch.from_numpy(prompt_wav).unsqueeze(0)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, rms)
    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=24000).to(device)
    prompt_features = prompt_features.unsqueeze(0) * 0.1
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
    prompt_tokens = tokenizer.texts_to_token_ids([reference_text])
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_features_lens": prompt_features_lens,
        "prompt_features": prompt_features,
        "prompt_rms": prompt_rms,
    }


def interactive_loop(runtime: LuxRuntime, config: RuntimeConfig) -> int:
    print("READY: LuxTTS 100M is listening. Type text, :paste, :file /workspace/prompt.txt, or :q.", flush=True)

    while True:
        try:
            line = read_prompt_line(PROMPT).strip()
        except EOFError:
            print()
            return 0

        if not line:
            continue
        if line in {":q", ":quit", "quit", "exit"}:
            return 0
        if line == ":paste":
            text = read_multiline()
        elif line.startswith(":file "):
            path = Path(line[6:].strip().strip('"'))
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                print(f"Could not read {path}: {exc}", flush=True)
                continue
        else:
            text = line

        synthesize(runtime, config, text)


def read_prompt_line(prompt: str) -> str:
    print(prompt, end="", flush=True)
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line


def read_multiline() -> str:
    print("Paste text. Finish with a line containing only :end.", flush=True)
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == ":end":
            return "\n".join(lines)
        lines.append(line)


def synthesize(runtime: LuxRuntime, config: RuntimeConfig, text: str) -> int:
    chunks = split_long_text(text, max_chars=config.chunk_chars, language="en")
    if not chunks:
        print("Nothing to synthesize.", flush=True)
        return 0

    started = time.perf_counter()
    base_name = output_name(text)
    parts_dir = config.output_dir / ".parts" / base_name
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    chunk_timings: list[dict[str, float | int]] = []

    print(f"Synthesizing {len(text)} chars in {len(chunks)} chunk(s)...", flush=True)
    for index, chunk in enumerate(chunks, start=1):
        part_path = parts_dir / f"{index:04d}.wav"
        chunk_started = time.perf_counter()
        audio = generate_chunk(runtime, config, chunk)
        write_wav(part_path, audio, SAMPLE_RATE)
        part_paths.append(part_path)
        elapsed = time.perf_counter() - chunk_started
        chunk_timings.append({"chunk_index": index, "elapsed_seconds": round(elapsed, 3)})
        print(f"  chunk {index}/{len(chunks)}: {elapsed:.2f}s", flush=True)

    output_path = config.output_dir / f"{base_name}.wav"
    audio_seconds = concatenate_wavs(part_paths, output_path, pause_ms=config.pause_ms)
    elapsed = time.perf_counter() - started
    rtf = elapsed / audio_seconds if audio_seconds > 0 else 0.0
    write_metadata(
        output_path,
        config=config,
        runtime=runtime,
        source_text=text,
        chunks=chunks,
        chunk_timings=chunk_timings,
        audio_seconds=audio_seconds,
        elapsed_seconds=elapsed,
        rtf=rtf,
    )
    print(
        f"Done: {output_path} | audio {audio_seconds:.1f}s | elapsed {elapsed:.1f}s | RTF {rtf:.2f}x",
        flush=True,
    )

    if not config.keep_parts:
        shutil.rmtree(parts_dir, ignore_errors=True)
        try:
            parts_dir.parent.rmdir()
        except OSError:
            pass
    return 0


def generate_chunk(runtime: LuxRuntime, config: RuntimeConfig, text: str):
    from zipvoice.modeling_utils import generate

    prompt_tokens = runtime.encoded_prompt["prompt_tokens"]
    prompt_features_lens = runtime.encoded_prompt["prompt_features_lens"]
    prompt_features = runtime.encoded_prompt["prompt_features"]
    prompt_rms = runtime.encoded_prompt["prompt_rms"]

    runtime.vocos.return_48k = not config.return_smooth
    with runtime.torch.inference_mode():
        audio = generate(
            prompt_tokens,
            prompt_features_lens,
            prompt_features,
            prompt_rms,
            text,
            runtime.model,
            runtime.vocos,
            runtime.tokenizer,
            num_step=config.num_steps,
            guidance_scale=config.guidance_scale,
            t_shift=config.t_shift,
            speed=config.speed,
        )
    return audio.cpu()


def write_wav(path: Path, waveform, sample_rate: int) -> None:
    import soundfile as sf

    if hasattr(waveform, "detach"):
        waveform = waveform.detach().float().cpu()
    if hasattr(waveform, "dim") and waveform.dim() == 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    if hasattr(waveform, "numpy"):
        waveform = waveform.numpy()
    sf.write(path, waveform, sample_rate, subtype="PCM_16")


def concatenate_wavs(part_paths: list[Path], output_path: Path, pause_ms: int) -> float:
    import numpy as np
    import soundfile as sf

    if not part_paths:
        raise ValueError("No WAV parts were generated.")

    with sf.SoundFile(part_paths[0]) as first:
        sample_rate = first.samplerate
        channels = first.channels

    total_frames = 0
    pause_frames = int(sample_rate * max(pause_ms, 0) / 1000)

    with sf.SoundFile(output_path, mode="w", samplerate=sample_rate, channels=channels, subtype="PCM_16") as target:
        for index, part_path in enumerate(part_paths):
            with sf.SoundFile(part_path) as source:
                if source.samplerate != sample_rate or source.channels != channels:
                    raise RuntimeError(f"{part_path} does not match the first chunk audio format.")

                while True:
                    data = source.read(frames=65536, dtype="float32", always_2d=True)
                    if len(data) == 0:
                        break
                    target.write(data)
                    total_frames += len(data)

            if pause_frames and index + 1 < len(part_paths):
                target.write(np.zeros((pause_frames, channels), dtype="float32"))
                total_frames += pause_frames

    return total_frames / sample_rate


def write_metadata(
    output_path: Path,
    *,
    config: RuntimeConfig,
    runtime: LuxRuntime,
    source_text: str,
    chunks: list[str],
    chunk_timings: list[dict[str, float | int]],
    audio_seconds: float,
    elapsed_seconds: float,
    rtf: float,
) -> None:
    metadata = {
        "audio_seconds": round(audio_seconds, 3),
        "chunk_count": len(chunks),
        "chunk_timings": chunk_timings,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "guidance_scale": config.guidance_scale,
        "model_id": config.model_id,
        "num_steps": config.num_steps,
        "pause_ms": config.pause_ms,
        "prompt_duration": config.prompt_duration,
        "realtime_factor": round(rtf, 4),
        "return_smooth": config.return_smooth,
        "rms": config.rms,
        "sample_rate": SAMPLE_RATE,
        "source_text": source_text,
        "speaker_reference_text": runtime.reference_text,
        "speaker_reference_text_path": str(config.ref_text_file),
        "speaker_wav": str(config.ref_audio),
        "speed": config.speed,
        "t_shift": config.t_shift,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def output_name(text: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    words = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    slug = words[:48].strip("-") or "luxtts"
    return f"{timestamp}-{slug}"


if __name__ == "__main__":
    raise SystemExit(main())
