from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from xtts_service.text_splitter import split_long_text


DEFAULT_MODEL_REPO = "rhasspy/piper-voices"
DEFAULT_MODEL_REVISION = "v1.0.0"
DEFAULT_MODEL_FILE = "en/en_GB/cori/high/en_GB-cori-high.onnx"
DEFAULT_CONFIG_FILE = "en/en_GB/cori/high/en_GB-cori-high.onnx.json"
DEFAULT_VOICE_NAME = "en_GB Cori high"
DEFAULT_CHUNK_CHARS = 160
DEFAULT_PAUSE_MS = 260
DEFAULT_SENTENCE_SILENCE = 0.22
DEFAULT_LENGTH_SCALE = 1.08
PROMPT = "piper> "


@dataclass
class RuntimeConfig:
    model_repo: str
    model_revision: str
    model_file: str
    config_file: str
    voice_name: str
    output_dir: Path
    model_dir: Path
    chunk_chars: int
    pause_ms: int
    keep_parts: bool
    use_cuda: bool
    sentence_silence: float
    length_scale: float | None
    noise_scale: float | None
    noise_w: float | None


@dataclass
class PiperRuntime:
    voice: object
    model_path: Path
    config_path: Path
    providers: list[str]
    torch_version: str
    torch_cuda: str
    visible_gpus: list[str]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = RuntimeConfig(
        model_repo=args.model_repo,
        model_revision=args.model_revision,
        model_file=args.model_file,
        config_file=args.config_file,
        voice_name=args.voice_name,
        output_dir=Path(args.output_dir),
        model_dir=Path(args.model_dir),
        chunk_chars=args.chunk_chars,
        pause_ms=args.pause_ms,
        keep_parts=args.keep_parts,
        use_cuda=args.cuda,
        sentence_silence=args.sentence_silence,
        length_scale=args.length_scale,
        noise_scale=args.noise_scale,
        noise_w=args.noise_w,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir.mkdir(parents=True, exist_ok=True)
    if args.interactive:
        print(
            "Piper TTS is loading. Wait for the READY line and piper> prompt before typing.",
            flush=True,
        )

    try:
        runtime = load_runtime(config)
    except Exception as exc:
        print(f"Piper startup failed: {exc}", file=sys.stderr, flush=True)
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
    parser = argparse.ArgumentParser(description="Interactive Piper preset-voice generator.")
    parser.add_argument("--interactive", action="store_true", help="Read text from stdin repeatedly.")
    parser.add_argument("--text", help="Synthesize one text string and exit.")
    parser.add_argument("--prompt-file", help="Read one UTF-8 text file and exit.")
    parser.add_argument("--model-repo", default=os.getenv("PIPER_MODEL_REPO", DEFAULT_MODEL_REPO))
    parser.add_argument("--model-revision", default=os.getenv("PIPER_MODEL_REVISION", DEFAULT_MODEL_REVISION))
    parser.add_argument("--model-file", default=os.getenv("PIPER_MODEL_FILE", DEFAULT_MODEL_FILE))
    parser.add_argument("--config-file", default=os.getenv("PIPER_CONFIG_FILE", DEFAULT_CONFIG_FILE))
    parser.add_argument("--voice-name", default=os.getenv("PIPER_VOICE_NAME", DEFAULT_VOICE_NAME))
    parser.add_argument("--output-dir", default=os.getenv("PIPER_OUTPUT_DIR", "/app/output/piper"))
    parser.add_argument("--model-dir", default=os.getenv("PIPER_MODEL_DIR", "/models/piper"))
    parser.add_argument("--chunk-chars", type=int, default=int(os.getenv("PIPER_CHUNK_CHARS", str(DEFAULT_CHUNK_CHARS))))
    parser.add_argument("--pause-ms", type=int, default=int(os.getenv("PIPER_PAUSE_MS", str(DEFAULT_PAUSE_MS))))
    parser.add_argument(
        "--sentence-silence",
        type=float,
        default=float(os.getenv("PIPER_SENTENCE_SILENCE", str(DEFAULT_SENTENCE_SILENCE))),
    )
    parser.add_argument(
        "--length-scale",
        type=optional_float,
        default=optional_env_float("PIPER_LENGTH_SCALE", DEFAULT_LENGTH_SCALE),
    )
    parser.add_argument("--noise-scale", type=optional_float, default=optional_env_float("PIPER_NOISE_SCALE"))
    parser.add_argument("--noise-w", type=optional_float, default=optional_env_float("PIPER_NOISE_W"))
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=env_bool("PIPER_USE_CUDA", True))
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args(argv)


def optional_float(value: str) -> float | None:
    if value.strip() == "":
        return None
    return float(value)


def optional_env_float(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    if value is None:
        return default
    if value.strip() == "":
        return None
    return float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_runtime(config: RuntimeConfig) -> PiperRuntime:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    torch_version = ""
    torch_cuda = ""
    visible_gpus: list[str] = []
    if config.use_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside the container. Check Docker GPU support and the GPU mask.")
        visible_gpus = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        torch_version = str(torch.__version__)
        torch_cuda = str(torch.version.cuda)
        print(f"CUDA ready: torch {torch_version}, CUDA {torch_cuda}, visible GPUs: {visible_gpus}", flush=True)
        torch.empty((1,), device="cuda")

    providers = list(ort.get_available_providers())
    print(f"ONNX Runtime providers: {providers}", flush=True)
    if config.use_cuda and "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "Piper CUDA was requested, but onnxruntime does not expose CUDAExecutionProvider. "
            "Rebuild the piper image with the CUDA ONNX Runtime package."
        )

    print(f"Using Piper preset voice: {config.voice_name}", flush=True)
    print("Piper does not use the local reference WAV or transcript for zero-shot cloning.", flush=True)
    print(f"Downloading/loading {config.model_file}...", flush=True)
    loaded_at = time.perf_counter()
    model_path = Path(
        hf_hub_download(
            repo_id=config.model_repo,
            filename=config.model_file,
            revision=config.model_revision,
            cache_dir=str(config.model_dir),
        )
    )
    config_path = Path(
        hf_hub_download(
            repo_id=config.model_repo,
            filename=config.config_file,
            revision=config.model_revision,
            cache_dir=str(config.model_dir),
        )
    )

    from piper import PiperVoice

    voice = PiperVoice.load(str(model_path), config_path=str(config_path), use_cuda=config.use_cuda)
    print(f"Voice loaded in {time.perf_counter() - loaded_at:.1f}s.", flush=True)
    return PiperRuntime(
        voice=voice,
        model_path=model_path,
        config_path=config_path,
        providers=providers,
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        visible_gpus=visible_gpus,
    )


def interactive_loop(runtime: PiperRuntime, config: RuntimeConfig) -> int:
    print("READY: Piper TTS is listening. Type text, :paste, :file /workspace/prompt.txt, or :q.", flush=True)

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


def synthesize(runtime: PiperRuntime, config: RuntimeConfig, text: str) -> int:
    chunks = split_piper_text(text, max_chars=config.chunk_chars)
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
        write_piper_wav(runtime, config, chunk, part_path)
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


def write_piper_wav(runtime: PiperRuntime, config: RuntimeConfig, text: str, output_path: Path) -> None:
    from piper.config import SynthesisConfig

    syn_config = SynthesisConfig(
        length_scale=config.length_scale,
        noise_scale=config.noise_scale,
        noise_w_scale=config.noise_w,
    )
    silence_int16_bytes = bytes(int(runtime.voice.config.sample_rate * config.sentence_silence * 2))

    with wave.open(str(output_path), "wb") as wav_file:
        wav_params_set = False
        for index, audio_chunk in enumerate(runtime.voice.synthesize(text, syn_config)):
            if not wav_params_set:
                wav_file.setframerate(audio_chunk.sample_rate)
                wav_file.setsampwidth(audio_chunk.sample_width)
                wav_file.setnchannels(audio_chunk.sample_channels)
                wav_params_set = True

            if index > 0 and silence_int16_bytes:
                wav_file.writeframes(silence_int16_bytes)

            wav_file.writeframes(audio_chunk.audio_int16_bytes)


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
    runtime: PiperRuntime,
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
        "chunk_chars": config.chunk_chars,
        "chunk_timings": chunk_timings,
        "config_file": config.config_file,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "length_scale": config.length_scale,
        "model_file": config.model_file,
        "model_repo": config.model_repo,
        "model_revision": config.model_revision,
        "noise_scale": config.noise_scale,
        "noise_w": config.noise_w,
        "onnxruntime_providers": runtime.providers,
        "pause_ms": config.pause_ms,
        "realtime_factor": round(rtf, 4),
        "sentence_silence": config.sentence_silence,
        "source_text": source_text,
        "torch_cuda": runtime.torch_cuda,
        "torch_version": runtime.torch_version,
        "use_cuda": config.use_cuda,
        "visible_gpus": runtime.visible_gpus,
        "voice_name": config.voice_name,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def split_piper_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    if max_chars < 80:
        raise ValueError("max_chars must be at least 80")

    units = dialogue_units(text)
    chunks: list[str] = []
    for unit in units:
        unit_chunks: list[str] = []
        for sentence_chunk in split_long_text(unit, max_chars=max_chars, language="en"):
            unit_chunks.extend(split_piper_clause_chunk(sentence_chunk, max_chars=max_chars))
        chunks.extend(pack_short_chunks(unit_chunks, max_chars=max_chars))
    return chunks


def dialogue_units(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    non_empty = [line for line in lines if line]
    if len(non_empty) > 1:
        return [turn for line in non_empty for turn in inline_dialogue_turns(line)]

    compact_text = re.sub(r"\s+", " ", normalized).strip()
    return inline_dialogue_turns(compact_text) if compact_text else []


def inline_dialogue_turns(text: str) -> list[str]:
    parts = re.split(r"\s+(?=[A-Z][A-Za-z0-9 .'-]{0,32}:\s+)", text.strip())
    return [part.strip() for part in parts if part.strip()]


def split_piper_clause_chunk(text: str, max_chars: int) -> list[str]:
    soft_chars = max(95, int(max_chars * 0.62))
    if len(text) <= soft_chars:
        return [text]

    pieces = clause_pieces(text)
    if len(pieces) == 1:
        pieces = conjunction_pieces(text)
    if len(pieces) == 1:
        return split_words(text, max_chars=max_chars)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > soft_chars:
            chunks.extend(split_words(current, max_chars=max_chars))
            current = piece
        else:
            current = candidate

    if current:
        chunks.extend(split_words(current, max_chars=max_chars))
    return chunks


def clause_pieces(text: str) -> list[str]:
    pieces = [match.group(0).strip() for match in re.finditer(r"[^,;:]+[,;:]?", text)]
    return [piece for piece in pieces if piece]


def conjunction_pieces(text: str) -> list[str]:
    parts = re.split(r"\s+(?=(?:and|but|because|while|then|so|which|that)\b)", text, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


def split_words(text: str, max_chars: int) -> list[str]:
    remaining = text.strip()
    chunks: list[str] = []
    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars + 1)
        if cut < max_chars // 2:
            cut = max_chars
        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def pack_short_chunks(chunks: list[str], max_chars: int) -> list[str]:
    if not chunks:
        return []

    packed: list[str] = []
    current = ""
    target_chars = max(110, int(max_chars * 0.72))
    for chunk in chunks:
        candidate = f"{current} {chunk}".strip()
        if current and len(candidate) > target_chars:
            packed.append(current)
            current = chunk
        else:
            current = candidate

    if current:
        packed.append(current)
    return packed


def output_name(text: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    words = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    slug = words[:48].strip("-") or "piper"
    return f"{timestamp}-{slug}"


if __name__ == "__main__":
    raise SystemExit(main())
