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


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_CUSTOM_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
PROMPT = "qwen0.6btts> "


@dataclass
class RuntimeConfig:
    voice_mode: str
    model_id: str
    output_dir: Path
    ref_audio: Path | None
    ref_text_file: Path | None
    language: str
    chunk_chars: int
    pause_ms: int
    keep_parts: bool
    attn_implementation: str
    custom_speaker: str
    custom_instruct: str
    custom_voice_name: str


@dataclass
class QwenRuntime:
    model: object
    voice_mode: str
    voice_clone_prompt: object | None
    reference_text: str
    custom_speaker: str
    custom_instruct: str
    custom_voice_name: str


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        voice_mode = normalize_voice_mode(args.voice_mode)
        model_id = DEFAULT_CUSTOM_MODEL if voice_mode == "custom_voice" and args.model_id == DEFAULT_MODEL else args.model_id
        config = RuntimeConfig(
            voice_mode=voice_mode,
            model_id=model_id,
            output_dir=Path(args.output_dir),
            ref_audio=required_path(args.ref_audio, "reference audio") if voice_mode == "clone" else optional_path(args.ref_audio),
            ref_text_file=required_path(args.ref_text_file, "reference transcript") if voice_mode == "clone" else optional_path(args.ref_text_file),
            language=args.language,
            chunk_chars=args.chunk_chars,
            pause_ms=args.pause_ms,
            keep_parts=args.keep_parts,
            attn_implementation=args.attn_implementation,
            custom_speaker=args.custom_speaker,
            custom_instruct=args.custom_instruct,
            custom_voice_name=args.custom_voice_name,
        )
    except ValueError as exc:
        print(f"Qwen startup failed: {exc}", file=sys.stderr, flush=True)
        return 1

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.interactive:
        print(
            "Qwen3-TTS 0.6B is loading. Wait for the READY line and qwen0.6btts> prompt before typing.",
            flush=True,
        )

    try:
        runtime = load_runtime(config)
    except Exception as exc:
        print(f"Qwen startup failed: {exc}", file=sys.stderr, flush=True)
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
    parser = argparse.ArgumentParser(description="Interactive Qwen3-TTS 0.6B voice-clone generator.")
    parser.add_argument("--interactive", action="store_true", help="Read text from stdin repeatedly.")
    parser.add_argument("--text", help="Synthesize one text string and exit.")
    parser.add_argument("--prompt-file", help="Read one UTF-8 text file and exit.")
    parser.add_argument("--voice-mode", default=os.getenv("QWEN_VOICE_MODE", "clone"))
    parser.add_argument("--model-id", default=os.getenv("QWEN_MODEL_ID", DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=os.getenv("QWEN_OUTPUT_DIR", "/app/output/qwen"))
    parser.add_argument("--ref-audio", default=os.getenv("QWEN_REF_AUDIO", "/workspace/reference.wav"))
    parser.add_argument("--ref-text-file", default=os.getenv("QWEN_REF_TEXT_FILE", "/workspace/reference.txt"))
    parser.add_argument("--language", default=os.getenv("QWEN_LANGUAGE", "English"))
    parser.add_argument("--chunk-chars", type=int, default=int(os.getenv("QWEN_CHUNK_CHARS", "360")))
    parser.add_argument("--pause-ms", type=int, default=int(os.getenv("QWEN_PAUSE_MS", "180")))
    parser.add_argument("--attn-implementation", default=os.getenv("QWEN_ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--custom-speaker", default=os.getenv("QWEN_CUSTOM_SPEAKER", "Serena"))
    parser.add_argument("--custom-instruct", default=os.getenv("QWEN_CUSTOM_INSTRUCT", ""))
    parser.add_argument("--custom-voice-name", default=os.getenv("QWEN_CUSTOM_VOICE_NAME", "Serena"))
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args(argv)


def normalize_voice_mode(value: str) -> str:
    normalized = (value or "clone").strip().lower().replace("-", "_")
    if normalized in {"clone", "reference", "reference_clone", "voice_clone"}:
        return "clone"
    if normalized in {"custom", "custom_voice", "preset"}:
        return "custom_voice"
    raise ValueError(f"unsupported Qwen voice mode '{value}'. Use clone or custom_voice.")


def required_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    raise ValueError(f"{label} not found at {path}")


def optional_path(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def load_runtime(config: RuntimeConfig) -> QwenRuntime:
    import torch
    from qwen_tts import Qwen3TTSModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the container. Check Docker GPU support and the GPU mask.")

    visible = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    print(f"CUDA ready: torch {torch.__version__}, CUDA {torch.version.cuda}, visible GPUs: {visible}", flush=True)
    print(f"Loading {config.model_id}...", flush=True)

    loaded_at = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        config.model_id,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=config.attn_implementation,
    )
    print(f"Model loaded in {time.perf_counter() - loaded_at:.1f}s.", flush=True)

    if config.voice_mode == "custom_voice":
        print(
            f"Using Qwen CustomVoice preset: {config.custom_voice_name} ({config.custom_speaker})",
            flush=True,
        )
        if config.custom_instruct:
            print(f"Qwen style instruction: {config.custom_instruct}", flush=True)
        return QwenRuntime(
            model=model,
            voice_mode=config.voice_mode,
            voice_clone_prompt=None,
            reference_text="",
            custom_speaker=config.custom_speaker,
            custom_instruct=config.custom_instruct,
            custom_voice_name=config.custom_voice_name,
        )

    if config.ref_audio is None or config.ref_text_file is None:
        raise RuntimeError("Qwen voice cloning requires both a reference WAV and a reference transcript.")

    reference_text = config.ref_text_file.read_text(encoding="utf-8").strip()
    print(f"Voice clone sample: {config.ref_audio}", flush=True)
    print(f"Voice reference transcript: {config.ref_text_file}", flush=True)

    prompt_started = time.perf_counter()
    print("Preparing Qwen voice-clone prompt. This can take a minute on first load...", flush=True)
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(config.ref_audio),
        ref_text=reference_text,
        x_vector_only_mode=False,
    )
    print(f"Voice clone prompt prepared in {time.perf_counter() - prompt_started:.1f}s.", flush=True)
    return QwenRuntime(
        model=model,
        voice_mode=config.voice_mode,
        voice_clone_prompt=prompt,
        reference_text=reference_text,
        custom_speaker="",
        custom_instruct="",
        custom_voice_name="Reference voice clone",
    )


def interactive_loop(runtime: QwenRuntime, config: RuntimeConfig) -> int:
    print("READY: Qwen3-TTS 0.6B is listening. Type text, :paste, :file /workspace/prompt.txt, or :q.", flush=True)

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


def synthesize(runtime: QwenRuntime, config: RuntimeConfig, text: str) -> int:
    chunks = split_long_text(text, max_chars=config.chunk_chars, language="en")
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
        if runtime.voice_mode == "custom_voice":
            wavs, sample_rate = runtime.model.generate_custom_voice(
                text=chunk,
                language=config.language,
                speaker=runtime.custom_speaker,
                instruct=runtime.custom_instruct or None,
                non_streaming_mode=True,
            )
        else:
            wavs, sample_rate = runtime.model.generate_voice_clone(
                text=chunk,
                language=config.language,
                voice_clone_prompt=runtime.voice_clone_prompt,
            )
        write_wav(part_path, wavs[0], sample_rate)
        part_paths.append(part_path)
        print(f"  chunk {index}/{len(chunks)}: {time.perf_counter() - chunk_started:.1f}s", flush=True)

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


def write_wav(path: Path, waveform, sample_rate: int) -> None:
    import soundfile as sf

    if hasattr(waveform, "detach"):
        waveform = waveform.detach().float().cpu().numpy()
    sf.write(path, waveform, sample_rate)


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
    runtime: QwenRuntime,
    source_text: str,
    chunks: list[str],
    audio_seconds: float,
    elapsed_seconds: float,
    rtf: float,
) -> None:
    metadata = {
        "audio_seconds": round(audio_seconds, 3),
        "attn_implementation": config.attn_implementation,
        "chunk_count": len(chunks),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "language": config.language,
        "model_id": config.model_id,
        "pause_ms": config.pause_ms,
        "realtime_factor": round(rtf, 4),
        "source_text": source_text,
        "voice_mode": runtime.voice_mode,
        "custom_speaker": runtime.custom_speaker,
        "custom_instruct": runtime.custom_instruct,
        "custom_voice_name": runtime.custom_voice_name,
        "speaker_reference_text": runtime.reference_text,
        "speaker_reference_text_path": str(config.ref_text_file) if config.ref_text_file else "",
        "speaker_wav": str(config.ref_audio) if config.ref_audio else "",
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def output_name(text: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    words = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    slug = words[:48].strip("-") or "qwen"
    return f"{timestamp}-{slug}"


if __name__ == "__main__":
    raise SystemExit(main())
