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


DEFAULT_MODEL_ID = "samuel-vitorino/sopro"
DEFAULT_VOICE_MODE = "clone"
DEFAULT_VOICE_NAME = "Reference voice clone"
DEFAULT_CHUNK_CHARS = 220
DEFAULT_PAUSE_MS = 220
DEFAULT_MAX_FRAMES = 400
DEFAULT_TOP_P = 0.9
DEFAULT_TEMPERATURE = 1.05
DEFAULT_STYLE_STRENGTH = 1.2
DEFAULT_REF_SECONDS = 12.0
PROMPT = "soprotts> "


@dataclass
class RuntimeConfig:
    model_id: str
    revision: str | None
    cache_dir: Path
    voice_mode: str
    voice_name: str
    output_dir: Path
    ref_audio: Path
    chunk_chars: int
    pause_ms: int
    keep_parts: bool
    use_cuda: bool
    max_frames: int
    top_p: float
    temperature: float
    anti_loop: bool
    style_strength: float
    ref_seconds: float | None
    min_gen_frames: int | None


@dataclass
class SoproRuntime:
    tts: object
    reference: object
    device: str
    sample_rate: int
    torch_version: str
    torch_cuda: str
    visible_gpus: list[str]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        voice_mode = normalize_voice_mode(args.voice_mode)
        config = RuntimeConfig(
            model_id=args.model_id,
            revision=args.revision or None,
            cache_dir=Path(args.cache_dir),
            voice_mode=voice_mode,
            voice_name=args.voice_name,
            output_dir=Path(args.output_dir),
            ref_audio=required_path(args.ref_audio, "reference audio"),
            chunk_chars=args.chunk_chars,
            pause_ms=args.pause_ms,
            keep_parts=args.keep_parts,
            use_cuda=args.cuda,
            max_frames=args.max_frames,
            top_p=args.top_p,
            temperature=args.temperature,
            anti_loop=not args.no_anti_loop,
            style_strength=args.style_strength,
            ref_seconds=args.ref_seconds,
            min_gen_frames=args.min_gen_frames,
        )
    except Exception as exc:
        print(f"SoproTTS configuration failed: {exc}", file=sys.stderr, flush=True)
        return 2

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.interactive:
        print(
            "SoproTTS is loading. Wait for the READY line and soprotts> prompt before typing.",
            flush=True,
        )

    try:
        runtime = load_runtime(config)
    except Exception as exc:
        print(f"SoproTTS startup failed: {exc}", file=sys.stderr, flush=True)
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
    parser = argparse.ArgumentParser(description="Interactive SoproTTS voice-clone generator.")
    parser.add_argument("--interactive", action="store_true", help="Read text from stdin repeatedly.")
    parser.add_argument("--text", help="Synthesize one text string and exit.")
    parser.add_argument("--prompt-file", help="Read one UTF-8 text file and exit.")
    parser.add_argument("--model-id", default=os.getenv("SOPRO_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--revision", default=os.getenv("SOPRO_MODEL_REVISION", ""))
    parser.add_argument("--cache-dir", default=os.getenv("SOPRO_CACHE_DIR", "/models/sopro"))
    parser.add_argument("--voice-mode", default=os.getenv("SOPRO_VOICE_MODE", DEFAULT_VOICE_MODE))
    parser.add_argument("--voice-name", default=os.getenv("SOPRO_VOICE_NAME", DEFAULT_VOICE_NAME))
    parser.add_argument("--output-dir", default=os.getenv("SOPRO_OUTPUT_DIR", "/app/output/sopro"))
    parser.add_argument("--ref-audio", default=os.getenv("SOPRO_REF_AUDIO", "/workspace/reference.wav"))
    parser.add_argument("--chunk-chars", type=int, default=int(os.getenv("SOPRO_CHUNK_CHARS", str(DEFAULT_CHUNK_CHARS))))
    parser.add_argument("--pause-ms", type=int, default=int(os.getenv("SOPRO_PAUSE_MS", str(DEFAULT_PAUSE_MS))))
    parser.add_argument("--max-frames", type=int, default=int(os.getenv("SOPRO_MAX_FRAMES", str(DEFAULT_MAX_FRAMES))))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("SOPRO_TOP_P", str(DEFAULT_TOP_P))))
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("SOPRO_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
    )
    parser.add_argument(
        "--style-strength",
        type=float,
        default=float(os.getenv("SOPRO_STYLE_STRENGTH", str(DEFAULT_STYLE_STRENGTH))),
    )
    parser.add_argument(
        "--ref-seconds",
        type=optional_float,
        default=optional_env_float("SOPRO_REF_SECONDS", DEFAULT_REF_SECONDS),
    )
    parser.add_argument(
        "--min-gen-frames",
        type=optional_int,
        default=optional_env_int("SOPRO_MIN_GEN_FRAMES"),
    )
    parser.add_argument("--no-anti-loop", action="store_true", default=env_bool("SOPRO_NO_ANTI_LOOP", False))
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=env_bool("SOPRO_USE_CUDA", True))
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args(argv)


def normalize_voice_mode(value: str) -> str:
    normalized = (value or DEFAULT_VOICE_MODE).strip().lower().replace("_", "-")
    if normalized in {"reference", "reference-clone", "clone", "voice-clone"}:
        return "clone"
    raise ValueError("SoproTTS currently supports reference-clone only in this launcher")


def optional_float(value: str) -> float | None:
    if value.strip() == "":
        return None
    return float(value)


def optional_int(value: str) -> int | None:
    if value.strip() == "":
        return None
    return int(value)


def optional_env_float(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    if value is None:
        return default
    if value.strip() == "":
        return None
    return float(value)


def optional_env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return default
    if value.strip() == "":
        return None
    return int(value)


def required_path(value: str, label: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{label} was not found: {path}")
    return path


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_runtime(config: RuntimeConfig) -> SoproRuntime:
    import torch
    from sopro import SoproTTS
    from sopro.constants import TARGET_SR

    device = "cuda" if config.use_cuda else "cpu"
    torch_version = str(torch.__version__)
    torch_cuda = str(torch.version.cuda)
    visible_gpus: list[str] = []
    if config.use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside the container. Check Docker GPU support and the GPU mask.")
        visible_gpus = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        torch.empty((1,), device="cuda")
        print(f"CUDA ready: torch {torch_version}, CUDA {torch_cuda}, visible GPUs: {visible_gpus}", flush=True)
    else:
        print(f"Using CPU: torch {torch_version}", flush=True)

    print(f"Loading SoproTTS 135M on {device}...", flush=True)
    loaded_at = time.perf_counter()
    tts = SoproTTS.from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=str(config.cache_dir),
        device=device,
    )
    print(f"Model loaded in {time.perf_counter() - loaded_at:.1f}s.", flush=True)

    print(f"Preparing SoproTTS reference voice: {config.ref_audio}", flush=True)
    prepared_at = time.perf_counter()
    reference = tts.prepare_reference(ref_audio_path=str(config.ref_audio), ref_seconds=config.ref_seconds)
    print(f"Reference voice prepared in {time.perf_counter() - prepared_at:.1f}s.", flush=True)

    return SoproRuntime(
        tts=tts,
        reference=reference,
        device=device,
        sample_rate=int(TARGET_SR),
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        visible_gpus=visible_gpus,
    )


def interactive_loop(runtime: SoproRuntime, config: RuntimeConfig) -> int:
    print("READY: SoproTTS is listening. Type text, :paste, :file /workspace/prompt.txt, or :q.", flush=True)

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


def synthesize(runtime: SoproRuntime, config: RuntimeConfig, text: str) -> int:
    chunks = split_sopro_text(text, max_chars=config.chunk_chars)
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
        write_sopro_wav(runtime, config, chunk, part_path)
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


def write_sopro_wav(runtime: SoproRuntime, config: RuntimeConfig, text: str, output_path: Path) -> None:
    wav = runtime.tts.synthesize(
        text,
        ref=runtime.reference,
        max_frames=config.max_frames,
        top_p=config.top_p,
        temperature=config.temperature,
        anti_loop=config.anti_loop,
        style_strength=config.style_strength,
        min_gen_frames=config.min_gen_frames,
    )
    runtime.tts.save_wav(str(output_path), wav)


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
    runtime: SoproRuntime,
    source_text: str,
    chunks: list[str],
    chunk_timings: list[dict[str, float | int]],
    audio_seconds: float,
    elapsed_seconds: float,
    rtf: float,
) -> None:
    metadata = {
        "anti_loop": config.anti_loop,
        "audio_seconds": round(audio_seconds, 3),
        "chunk_chars": config.chunk_chars,
        "chunk_count": len(chunks),
        "chunk_timings": chunk_timings,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "max_frames": config.max_frames,
        "model_id": config.model_id,
        "pause_ms": config.pause_ms,
        "realtime_factor": round(rtf, 4),
        "ref_audio": str(config.ref_audio),
        "ref_seconds": config.ref_seconds,
        "sample_rate": runtime.sample_rate,
        "source_text": source_text,
        "style_strength": config.style_strength,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "torch_cuda": runtime.torch_cuda,
        "torch_version": runtime.torch_version,
        "use_cuda": config.use_cuda,
        "visible_gpus": runtime.visible_gpus,
        "voice_mode": config.voice_mode,
        "voice_name": config.voice_name,
    }
    if config.min_gen_frames is not None:
        metadata["min_gen_frames"] = config.min_gen_frames
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def split_sopro_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    if max_chars < 80:
        raise ValueError("max_chars must be at least 80")

    units = dialogue_units(text)
    chunks: list[str] = []
    for unit in units:
        unit_chunks: list[str] = []
        for sentence_chunk in split_long_text(unit, max_chars=max_chars, language="en"):
            unit_chunks.extend(split_clause_chunk(sentence_chunk, max_chars=max_chars))
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


def split_clause_chunk(text: str, max_chars: int) -> list[str]:
    soft_chars = max(115, int(max_chars * 0.62))
    if len(text) <= soft_chars:
        return [text]

    pieces = [match.group(0).strip() for match in re.finditer(r"[^,;:]+[,;:]?", text)]
    pieces = [piece for piece in pieces if piece]
    if len(pieces) == 1:
        pieces = re.split(r"\s+(?=(?:and|but|because|while|then|so|which|that)\b)", text, flags=re.IGNORECASE)
        pieces = [piece.strip() for piece in pieces if piece.strip()]
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
    target_chars = max(125, int(max_chars * 0.72))
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
    slug = words[:48].strip("-") or "sopro"
    return f"{timestamp}-{slug}"


if __name__ == "__main__":
    raise SystemExit(main())
