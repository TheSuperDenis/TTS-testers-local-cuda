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

from .text_splitter import split_long_text


DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
PROMPT = "xttsv2> "


@dataclass
class RuntimeConfig:
    model_name: str
    output_dir: Path
    speaker_wav: Path | None
    speaker_text: Path | None
    language: str
    chunk_chars: int
    pause_ms: int
    keep_parts: bool
    allow_cpu: bool


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = RuntimeConfig(
        model_name=args.model_name,
        output_dir=Path(args.output_dir),
        speaker_wav=_speaker_path(args.speaker_wav),
        speaker_text=_speaker_text_path(args.speaker_text),
        language=args.language,
        chunk_chars=args.chunk_chars,
        pause_ms=args.pause_ms,
        keep_parts=args.keep_parts,
        allow_cpu=args.allow_cpu,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.interactive:
        print("XTTS-v2 is loading. Wait for the READY line and xttsv2> prompt before typing.", flush=True)

    try:
        runtime = load_runtime(config)
    except Exception as exc:
        print(f"XTTS startup failed: {exc}", file=sys.stderr, flush=True)
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
    parser = argparse.ArgumentParser(description="Interactive XTTS-v2 long-audio generator.")
    parser.add_argument("--interactive", action="store_true", help="Read text from stdin repeatedly.")
    parser.add_argument("--text", help="Synthesize one text string and exit.")
    parser.add_argument("--prompt-file", help="Read one UTF-8 text file and exit.")
    parser.add_argument("--model-name", default=os.getenv("XTTS_MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=os.getenv("XTTS_OUTPUT_DIR", "/app/output"))
    parser.add_argument("--speaker-wav", default=os.getenv("XTTS_SPEAKER_WAV", "/app/voices/default.wav"))
    parser.add_argument("--speaker-text", default=os.getenv("XTTS_SPEAKER_TEXT", ""))
    parser.add_argument("--language", default=os.getenv("XTTS_LANGUAGE", "en"))
    parser.add_argument("--chunk-chars", type=int, default=int(os.getenv("XTTS_CHUNK_CHARS", "250")))
    parser.add_argument("--pause-ms", type=int, default=int(os.getenv("XTTS_PAUSE_MS", "180")))
    parser.add_argument("--keep-parts", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args(argv)


def _speaker_path(value: str | None) -> Path | None:
    if not value:
        return None

    path = Path(value)
    if path.exists():
        return path

    print(
        f"Reference voice not found at {path}. XTTS will try the model default voice; "
        "put a WAV file in voices/default.wav for voice cloning.",
        flush=True,
    )
    return None


def _speaker_text_path(value: str | None) -> Path | None:
    if not value:
        return None

    path = Path(value)
    if path.exists():
        return path

    print(f"Reference transcript not found at {path}. Continuing with the voice sample only.", flush=True)
    return None


def load_runtime(config: RuntimeConfig):
    import torch
    from TTS.api import TTS

    if torch.cuda.is_available():
        device = "cuda"
        visible = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        print(f"CUDA ready: torch {torch.__version__}, CUDA {torch.version.cuda}, visible GPUs: {visible}", flush=True)
    elif config.allow_cpu:
        device = "cpu"
        print("CUDA is not available; running on CPU because --allow-cpu was set.", flush=True)
    else:
        raise RuntimeError("CUDA is not available inside the container. Check Docker GPU support and the GPU mask.")

    print(f"Loading {config.model_name}...", flush=True)
    loaded_at = time.perf_counter()
    tts = TTS(config.model_name).to(device)
    print(f"Model loaded in {time.perf_counter() - loaded_at:.1f}s on {device}.", flush=True)
    if config.speaker_wav:
        print(f"Voice clone sample: {config.speaker_wav}", flush=True)
    if config.speaker_text:
        print(f"Voice reference transcript: {config.speaker_text}", flush=True)
    return tts


def interactive_loop(runtime, config: RuntimeConfig) -> int:
    print("READY: XTTS-v2 is listening. Type text, :paste, :file /app/prompt.txt, or :q.", flush=True)

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


def synthesize(runtime, config: RuntimeConfig, text: str) -> int:
    chunks = split_long_text(text, max_chars=config.chunk_chars, language=config.language)
    if not chunks:
        print("Nothing to synthesize.", flush=True)
        return 0

    started = time.perf_counter()
    base_name = output_name(text)
    parts_dir = config.output_dir / ".parts" / base_name
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []

    print(f"Synthesizing {len(text)} chars in {len(chunks)} chunk(s)...", flush=True)
    for index, chunk in enumerate(chunks, start=1):
        part_path = parts_dir / f"{index:04d}.wav"
        chunk_started = time.perf_counter()
        kwargs = {
            "text": chunk,
            "file_path": str(part_path),
            "language": config.language,
        }
        if config.speaker_wav:
            kwargs["speaker_wav"] = str(config.speaker_wav)

        runtime.tts_to_file(**kwargs)
        part_paths.append(part_path)
        print(f"  chunk {index}/{len(chunks)}: {time.perf_counter() - chunk_started:.1f}s", flush=True)

    output_path = config.output_dir / f"{base_name}.wav"
    audio_seconds = concatenate_wavs(part_paths, output_path, pause_ms=config.pause_ms)
    elapsed = time.perf_counter() - started
    rtf = elapsed / audio_seconds if audio_seconds > 0 else 0.0
    write_metadata(
        output_path,
        config=config,
        source_text=text,
        chunks=chunks,
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
    source_text: str,
    chunks: list[str],
    audio_seconds: float,
    elapsed_seconds: float,
    rtf: float,
) -> None:
    reference_text = None
    if config.speaker_text:
        try:
            reference_text = config.speaker_text.read_text(encoding="utf-8").strip()
        except OSError:
            reference_text = None

    metadata = {
        "audio_seconds": round(audio_seconds, 3),
        "chunk_count": len(chunks),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "language": config.language,
        "model_name": config.model_name,
        "pause_ms": config.pause_ms,
        "realtime_factor": round(rtf, 4),
        "source_text": source_text,
        "speaker_reference_text": reference_text,
        "speaker_reference_text_path": str(config.speaker_text) if config.speaker_text else None,
        "speaker_wav": str(config.speaker_wav) if config.speaker_wav else None,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def output_name(text: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    words = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    slug = words[:48].strip("-") or "xtts"
    return f"{timestamp}-{slug}"


if __name__ == "__main__":
    raise SystemExit(main())
