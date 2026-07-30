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
from importlib.metadata import version
from pathlib import Path

from xtts_service.text_splitter import split_long_text


DEFAULT_MODEL_NAME = "supertonic-3"
DEFAULT_VOICE_ID = "F1"
DEFAULT_VOICE_NAME = "Female F1"
DEFAULT_LANGUAGE = "en"
DEFAULT_CHUNK_CHARS = 220
DEFAULT_PAUSE_MS = 220
DEFAULT_TOTAL_STEPS = 8
DEFAULT_SPEED = 1.0
PROMPT = "supertonic3> "
SESSION_NAMES = ("dp_ort", "text_enc_ort", "vector_est_ort", "vocoder_ort")


@dataclass
class RuntimeConfig:
    model_name: str
    output_dir: Path
    model_dir: Path
    voice_id: str
    voice_name: str
    language: str
    chunk_chars: int
    pause_ms: int
    total_steps: int
    speed: float
    keep_parts: bool
    use_cuda: bool


@dataclass
class SupertonicRuntime:
    tts: object
    voice_style: object
    backend: str
    sample_rate: int
    package_version: str
    onnxruntime_version: str
    onnxruntime_providers: list[str]
    session_providers: dict[str, list[str]]
    torch_version: str
    torch_cuda: str
    visible_gpus: list[str]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = RuntimeConfig(
        model_name=args.model_name,
        output_dir=Path(args.output_dir),
        model_dir=Path(args.model_dir),
        voice_id=args.voice_id,
        voice_name=args.voice_name,
        language=args.language,
        chunk_chars=args.chunk_chars,
        pause_ms=args.pause_ms,
        total_steps=args.total_steps,
        speed=args.speed,
        keep_parts=args.keep_parts,
        use_cuda=args.cuda,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir.mkdir(parents=True, exist_ok=True)
    if args.interactive:
        print(
            "Supertonic 3 is loading. Wait for the READY line and supertonic3> prompt before typing.",
            flush=True,
        )

    try:
        runtime = load_runtime(config)
    except Exception as exc:
        print(f"Supertonic 3 startup failed: {exc}", file=sys.stderr, flush=True)
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
    parser = argparse.ArgumentParser(description="Interactive Supertonic 3 preset-voice generator.")
    parser.add_argument("--interactive", action="store_true", help="Read text from stdin repeatedly.")
    parser.add_argument("--text", help="Synthesize one text string and exit.")
    parser.add_argument("--prompt-file", help="Read one UTF-8 text file and exit.")
    parser.add_argument("--model-name", default=os.getenv("SUPERTONIC_MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--output-dir", default=os.getenv("SUPERTONIC_OUTPUT_DIR", "/app/output/supertonic"))
    parser.add_argument(
        "--model-dir",
        default=os.getenv("SUPERTONIC_MODEL_DIR", "/models/supertonic-cache/supertonic3"),
    )
    parser.add_argument("--voice-id", default=os.getenv("SUPERTONIC_VOICE_ID", DEFAULT_VOICE_ID))
    parser.add_argument("--voice-name", default=os.getenv("SUPERTONIC_VOICE_NAME", DEFAULT_VOICE_NAME))
    parser.add_argument("--language", default=os.getenv("SUPERTONIC_LANGUAGE", DEFAULT_LANGUAGE))
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=int(os.getenv("SUPERTONIC_CHUNK_CHARS", str(DEFAULT_CHUNK_CHARS))),
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=int(os.getenv("SUPERTONIC_PAUSE_MS", str(DEFAULT_PAUSE_MS))),
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=int(os.getenv("SUPERTONIC_TOTAL_STEPS", str(DEFAULT_TOTAL_STEPS))),
    )
    parser.add_argument("--speed", type=float, default=float(os.getenv("SUPERTONIC_SPEED", str(DEFAULT_SPEED))))
    parser.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=env_bool("SUPERTONIC_USE_CUDA", True),
    )
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args(argv)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def select_onnx_providers(available_providers: list[str], use_cuda: bool) -> list[str]:
    if not use_cuda:
        if "CPUExecutionProvider" not in available_providers:
            raise RuntimeError(f"ONNX Runtime does not expose CPUExecutionProvider: {available_providers}")
        return ["CPUExecutionProvider"]

    if "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError(
            "Supertonic CUDA was requested, but onnxruntime does not expose CUDAExecutionProvider. "
            "Rebuild the supertonic image with the CUDA ONNX Runtime package."
        )
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def load_runtime(config: RuntimeConfig) -> SupertonicRuntime:
    import onnxruntime as ort
    import torch

    available_providers = list(ort.get_available_providers())
    provider_list = select_onnx_providers(available_providers, config.use_cuda)
    backend = "cuda" if config.use_cuda else "cpu"
    visible_gpus: list[str] = []

    if config.use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside the container. Check Docker GPU support and the GPU mask.")
        visible_gpus = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        torch.empty((1,), device="cuda")
        print(
            f"CUDA ready: torch {torch.__version__}, CUDA {torch.version.cuda}, visible GPUs: {visible_gpus}",
            flush=True,
        )
    else:
        print(f"Using CPU: torch {torch.__version__}", flush=True)

    print(f"ONNX Runtime {ort.__version__} providers: {available_providers}", flush=True)
    print(f"Using Supertonic 3 preset voice: {config.voice_name} ({config.voice_id})", flush=True)
    print("Supertonic 3 uses fixed preset styles and does not use the local reference WAV or transcript.", flush=True)
    print(f"Loading {config.model_name} on {backend}...", flush=True)

    import supertonic.config as supertonic_config
    import supertonic.loader as supertonic_loader

    supertonic_config.DEFAULT_ONNX_PROVIDERS = provider_list
    supertonic_loader.DEFAULT_ONNX_PROVIDERS = provider_list

    from supertonic import TTS

    loaded_at = time.perf_counter()
    tts = TTS(
        model=config.model_name,
        model_dir=config.model_dir,
        auto_download=True,
    )
    voice_style = tts.get_voice_style(config.voice_id)
    session_providers = get_session_provider_map(tts.model)
    if config.use_cuda:
        missing_cuda = [
            name
            for name, providers in session_providers.items()
            if "CUDAExecutionProvider" not in providers
        ]
        if missing_cuda:
            raise RuntimeError(
                "Supertonic sessions did not use CUDAExecutionProvider: "
                + ", ".join(missing_cuda)
            )

    print(f"Session providers: {session_providers}", flush=True)
    print(f"Model and voice loaded in {time.perf_counter() - loaded_at:.1f}s.", flush=True)
    return SupertonicRuntime(
        tts=tts,
        voice_style=voice_style,
        backend=backend,
        sample_rate=int(tts.sample_rate),
        package_version=version("supertonic"),
        onnxruntime_version=str(ort.__version__),
        onnxruntime_providers=available_providers,
        session_providers=session_providers,
        torch_version=str(torch.__version__),
        torch_cuda=str(torch.version.cuda),
        visible_gpus=visible_gpus,
    )


def get_session_provider_map(engine: object) -> dict[str, list[str]]:
    providers: dict[str, list[str]] = {}
    for name in SESSION_NAMES:
        session = getattr(engine, name, None)
        if session is None or not hasattr(session, "get_providers"):
            raise RuntimeError(f"Supertonic engine is missing the expected ONNX session '{name}'.")
        providers[name] = list(session.get_providers())
    return providers


def interactive_loop(runtime: SupertonicRuntime, config: RuntimeConfig) -> int:
    print("READY: Supertonic 3 is listening. Type text, :paste, :file /workspace/prompt.txt, or :q.", flush=True)

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


def synthesize(runtime: SupertonicRuntime, config: RuntimeConfig, text: str) -> int:
    chunks = split_supertonic_text(text, max_chars=config.chunk_chars, language=config.language)
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
        write_supertonic_wav(runtime, config, chunk, part_path)
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


def write_supertonic_wav(
    runtime: SupertonicRuntime,
    config: RuntimeConfig,
    text: str,
    output_path: Path,
) -> None:
    import numpy as np
    import soundfile as sf

    waveform, _duration = runtime.tts.synthesize(
        text=text,
        voice_style=runtime.voice_style,
        total_steps=config.total_steps,
        speed=config.speed,
        max_chunk_length=config.chunk_chars,
        silence_duration=0.0,
        lang=config.language,
        verbose=False,
    )
    audio = np.asarray(waveform, dtype=np.float32).squeeze()
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    if audio.size == 0:
        raise RuntimeError("Supertonic 3 did not generate any audio for this chunk.")
    sf.write(output_path, audio, runtime.sample_rate, subtype="PCM_16")


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
    runtime: SupertonicRuntime,
    source_text: str,
    chunks: list[str],
    chunk_timings: list[dict[str, float | int]],
    audio_seconds: float,
    elapsed_seconds: float,
    rtf: float,
) -> None:
    metadata = {
        "audio_seconds": round(audio_seconds, 3),
        "backend": runtime.backend,
        "chunk_chars": config.chunk_chars,
        "chunk_count": len(chunks),
        "chunk_timings": chunk_timings,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "language": config.language,
        "model_name": config.model_name,
        "onnxruntime_providers": runtime.onnxruntime_providers,
        "onnxruntime_version": runtime.onnxruntime_version,
        "pause_ms": config.pause_ms,
        "realtime_factor": round(rtf, 4),
        "sample_rate": runtime.sample_rate,
        "session_providers": runtime.session_providers,
        "source_text": source_text,
        "speed": config.speed,
        "supertonic_version": runtime.package_version,
        "torch_cuda": runtime.torch_cuda,
        "torch_version": runtime.torch_version,
        "total_steps": config.total_steps,
        "use_cuda": config.use_cuda,
        "visible_gpus": runtime.visible_gpus,
        "voice_id": config.voice_id,
        "voice_name": config.voice_name,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def split_supertonic_text(
    text: str,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    language: str = DEFAULT_LANGUAGE,
) -> list[str]:
    if max_chars < 80:
        raise ValueError("max_chars must be at least 80")

    chunks: list[str] = []
    for unit in dialogue_units(text):
        chunks.extend(split_long_text(unit, max_chars=max_chars, language=language))
    return chunks


def dialogue_units(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    non_empty = [line for line in lines if line]
    if len(non_empty) > 1:
        return non_empty

    compact_text = re.sub(r"\s+", " ", normalized).strip()
    return [compact_text] if compact_text else []


def output_name(text: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    words = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    slug = words[:48].strip("-") or "supertonic"
    return f"{timestamp}-{slug}"


if __name__ == "__main__":
    raise SystemExit(main())
