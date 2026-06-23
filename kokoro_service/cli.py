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


DEFAULT_MODEL_ID = "hexgrad/Kokoro-82M"
DEFAULT_VOICE_ID = "af_heart"
DEFAULT_VOICE_NAME = "American female Heart"
DEFAULT_LANG_CODE = "a"
DEFAULT_CHUNK_CHARS = 220
DEFAULT_PAUSE_MS = 220
DEFAULT_SPEED = 0.95
SAMPLE_RATE = 24000
PROMPT = "kokoro82m> "


@dataclass
class RuntimeConfig:
    model_id: str
    output_dir: Path
    voice_id: str
    voice_name: str
    lang_code: str
    chunk_chars: int
    pause_ms: int
    keep_parts: bool
    speed: float
    use_cuda: bool


@dataclass
class KokoroRuntime:
    torch: object
    pipeline: object
    device: str
    torch_version: str
    torch_cuda: str
    visible_gpus: list[str]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = RuntimeConfig(
        model_id=args.model_id,
        output_dir=Path(args.output_dir),
        voice_id=args.voice_id,
        voice_name=args.voice_name,
        lang_code=args.lang_code,
        chunk_chars=args.chunk_chars,
        pause_ms=args.pause_ms,
        keep_parts=args.keep_parts,
        speed=args.speed,
        use_cuda=args.cuda,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.interactive:
        print(
            "Kokoro-82M is loading. Wait for the READY line and kokoro82m> prompt before typing.",
            flush=True,
        )

    try:
        runtime = load_runtime(config)
    except Exception as exc:
        print(f"Kokoro startup failed: {exc}", file=sys.stderr, flush=True)
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
    parser = argparse.ArgumentParser(description="Interactive Kokoro-82M preset-voice generator.")
    parser.add_argument("--interactive", action="store_true", help="Read text from stdin repeatedly.")
    parser.add_argument("--text", help="Synthesize one text string and exit.")
    parser.add_argument("--prompt-file", help="Read one UTF-8 text file and exit.")
    parser.add_argument("--model-id", default=os.getenv("KOKORO_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--output-dir", default=os.getenv("KOKORO_OUTPUT_DIR", "/app/output/kokoro"))
    parser.add_argument("--voice-id", default=os.getenv("KOKORO_VOICE_ID", DEFAULT_VOICE_ID))
    parser.add_argument("--voice-name", default=os.getenv("KOKORO_VOICE_NAME", DEFAULT_VOICE_NAME))
    parser.add_argument("--lang-code", default=os.getenv("KOKORO_LANG_CODE", DEFAULT_LANG_CODE))
    parser.add_argument("--chunk-chars", type=int, default=int(os.getenv("KOKORO_CHUNK_CHARS", str(DEFAULT_CHUNK_CHARS))))
    parser.add_argument("--pause-ms", type=int, default=int(os.getenv("KOKORO_PAUSE_MS", str(DEFAULT_PAUSE_MS))))
    parser.add_argument("--speed", type=float, default=float(os.getenv("KOKORO_SPEED", str(DEFAULT_SPEED))))
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=env_bool("KOKORO_USE_CUDA", True))
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args(argv)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_runtime(config: RuntimeConfig) -> KokoroRuntime:
    import torch
    from kokoro import KPipeline

    device = "cuda" if config.use_cuda else "cpu"
    torch_version = str(torch.__version__)
    torch_cuda = str(torch.version.cuda)
    visible_gpus: list[str] = []
    if config.use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside the container. Check Docker GPU support and the GPU mask.")
        visible_gpus = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        print(f"CUDA ready: torch {torch_version}, CUDA {torch_cuda}, visible GPUs: {visible_gpus}", flush=True)
        torch.empty((1,), device="cuda")
    else:
        print(f"Using CPU: torch {torch_version}", flush=True)

    print(f"Using Kokoro voice: {config.voice_name} ({config.voice_id})", flush=True)
    print(f"Loading {config.model_id} with lang_code='{config.lang_code}' on {device}...", flush=True)
    loaded_at = time.perf_counter()
    pipeline = KPipeline(lang_code=config.lang_code, repo_id=config.model_id, device=device)
    pipeline.load_voice(config.voice_id)
    print(f"Model and voice loaded in {time.perf_counter() - loaded_at:.1f}s.", flush=True)
    return KokoroRuntime(
        torch=torch,
        pipeline=pipeline,
        device=device,
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        visible_gpus=visible_gpus,
    )


def interactive_loop(runtime: KokoroRuntime, config: RuntimeConfig) -> int:
    print("READY: Kokoro-82M is listening. Type text, :paste, :file /workspace/prompt.txt, or :q.", flush=True)

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


def synthesize(runtime: KokoroRuntime, config: RuntimeConfig, text: str) -> int:
    chunks = split_kokoro_text(text, max_chars=config.chunk_chars)
    if not chunks:
        print("Nothing to synthesize.", flush=True)
        return 0

    started = time.perf_counter()
    base_name = output_name(text)
    parts_dir = config.output_dir / ".parts" / base_name
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    chunk_timings: list[dict[str, float | int | str]] = []

    print(f"Synthesizing {len(text)} chars in {len(chunks)} chunk(s)...", flush=True)
    for index, chunk in enumerate(chunks, start=1):
        part_path = parts_dir / f"{index:04d}.wav"
        chunk_started = time.perf_counter()
        grapheme_preview = write_kokoro_wav(runtime, config, chunk, part_path)
        part_paths.append(part_path)
        elapsed = time.perf_counter() - chunk_started
        chunk_timings.append(
            {
                "chunk_index": index,
                "elapsed_seconds": round(elapsed, 3),
                "text": grapheme_preview,
            }
        )
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


def write_kokoro_wav(runtime: KokoroRuntime, config: RuntimeConfig, text: str, output_path: Path) -> str:
    import soundfile as sf

    audio_segments = []
    graphemes = []
    with runtime.torch.inference_mode():
        generator = runtime.pipeline(
            text,
            voice=config.voice_id,
            speed=config.speed,
            split_pattern=None,
        )
        for result in generator:
            gs, _ps, audio = result
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().float().cpu().numpy()
            audio_segments.append(audio)
            if gs:
                graphemes.append(gs)

    if not audio_segments:
        raise RuntimeError("Kokoro did not generate any audio for this chunk.")

    import numpy as np

    audio_out = audio_segments[0] if len(audio_segments) == 1 else np.concatenate(audio_segments)
    sf.write(output_path, audio_out, SAMPLE_RATE, subtype="PCM_16")
    return " ".join(graphemes).strip() or text


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
    runtime: KokoroRuntime,
    source_text: str,
    chunks: list[str],
    chunk_timings: list[dict[str, float | int | str]],
    audio_seconds: float,
    elapsed_seconds: float,
    rtf: float,
) -> None:
    metadata = {
        "audio_seconds": round(audio_seconds, 3),
        "chunk_chars": config.chunk_chars,
        "chunk_count": len(chunks),
        "chunk_timings": chunk_timings,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "lang_code": config.lang_code,
        "model_id": config.model_id,
        "pause_ms": config.pause_ms,
        "realtime_factor": round(rtf, 4),
        "sample_rate": SAMPLE_RATE,
        "source_text": source_text,
        "speed": config.speed,
        "torch_cuda": runtime.torch_cuda,
        "torch_version": runtime.torch_version,
        "use_cuda": config.use_cuda,
        "visible_gpus": runtime.visible_gpus,
        "voice_id": config.voice_id,
        "voice_name": config.voice_name,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def split_kokoro_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
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
    soft_chars = max(105, int(max_chars * 0.62))
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
    target_chars = max(120, int(max_chars * 0.72))
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
    slug = words[:48].strip("-") or "kokoro"
    return f"{timestamp}-{slug}"


if __name__ == "__main__":
    raise SystemExit(main())
