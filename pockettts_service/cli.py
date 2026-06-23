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


DEFAULT_LANGUAGE = "english"
DEFAULT_VOICE_MODE = "clone"
DEFAULT_VOICE_ID = "reference-clone"
DEFAULT_VOICE_NAME = "Reference voice clone"
DEFAULT_CHUNK_CHARS = 220
DEFAULT_PAUSE_MS = 220
DEFAULT_MAX_TOKENS = 320
DEFAULT_FRAMES_AFTER_EOS = 24
DEFAULT_TRUNCATE_REF = True
PROMPT = "pockettts> "
POCKETTTS_CLONE_AUTH_HELP = (
    "PocketTTS voice cloning needs Kyutai's gated weights from https://huggingface.co/kyutai/pocket-tts. "
    "Accept the terms on Hugging Face, then expose a token to Docker with HF_TOKEN or log in locally with "
    "`huggingface-cli login` / `uvx hf auth login`. Built-in voices work without these gated clone weights."
)


@dataclass
class RuntimeConfig:
    language: str
    voice_mode: str
    voice_id: str
    voice_name: str
    output_dir: Path
    ref_audio: Path | None
    chunk_chars: int
    pause_ms: int
    keep_parts: bool
    max_tokens: int
    frames_after_eos: int | None
    truncate_ref: bool
    quantize: bool


@dataclass
class PocketRuntime:
    model: object
    voice_state: object
    sample_rate: int
    torch_version: str
    torch_cuda: str
    cuda_available: bool
    prepared_voice_source: str


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        voice_mode = normalize_voice_mode(args.voice_mode)
        config = RuntimeConfig(
            language=args.language,
            voice_mode=voice_mode,
            voice_id=args.voice_id,
            voice_name=args.voice_name,
            output_dir=Path(args.output_dir),
            ref_audio=required_path(args.ref_audio, "reference audio") if voice_mode == "clone" else optional_path(args.ref_audio),
            chunk_chars=args.chunk_chars,
            pause_ms=args.pause_ms,
            keep_parts=args.keep_parts,
            max_tokens=args.max_tokens,
            frames_after_eos=args.frames_after_eos,
            truncate_ref=args.truncate_ref,
            quantize=args.quantize,
        )
    except Exception as exc:
        print(f"PocketTTS configuration failed: {exc}", file=sys.stderr, flush=True)
        return 2

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.interactive:
        print(
            "PocketTTS is loading. Wait for the READY line and pockettts> prompt before typing.",
            flush=True,
        )

    try:
        runtime = load_runtime(config)
    except Exception as exc:
        print(f"PocketTTS startup failed: {friendly_startup_error(exc)}", file=sys.stderr, flush=True)
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
    parser = argparse.ArgumentParser(description="Interactive PocketTTS voice-clone and preset generator.")
    parser.add_argument("--interactive", action="store_true", help="Read text from stdin repeatedly.")
    parser.add_argument("--text", help="Synthesize one text string and exit.")
    parser.add_argument("--prompt-file", help="Read one UTF-8 text file and exit.")
    parser.add_argument("--language", default=os.getenv("POCKETTTS_LANGUAGE", DEFAULT_LANGUAGE))
    parser.add_argument("--voice-mode", default=os.getenv("POCKETTTS_VOICE_MODE", DEFAULT_VOICE_MODE))
    parser.add_argument("--voice-id", default=os.getenv("POCKETTTS_VOICE_ID", DEFAULT_VOICE_ID))
    parser.add_argument("--voice-name", default=os.getenv("POCKETTTS_VOICE_NAME", DEFAULT_VOICE_NAME))
    parser.add_argument("--output-dir", default=os.getenv("POCKETTTS_OUTPUT_DIR", "/app/output/pockettts"))
    parser.add_argument("--ref-audio", default=os.getenv("POCKETTTS_REF_AUDIO", "/workspace/reference.wav"))
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=int(os.getenv("POCKETTTS_CHUNK_CHARS", str(DEFAULT_CHUNK_CHARS))),
    )
    parser.add_argument("--pause-ms", type=int, default=int(os.getenv("POCKETTTS_PAUSE_MS", str(DEFAULT_PAUSE_MS))))
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("POCKETTTS_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
    )
    parser.add_argument(
        "--frames-after-eos",
        type=optional_int,
        default=optional_env_int("POCKETTTS_FRAMES_AFTER_EOS", DEFAULT_FRAMES_AFTER_EOS),
    )
    parser.add_argument(
        "--truncate-ref",
        action=argparse.BooleanOptionalAction,
        default=env_bool("POCKETTTS_TRUNCATE_REF", DEFAULT_TRUNCATE_REF),
    )
    parser.add_argument("--quantize", action="store_true", default=env_bool("POCKETTTS_QUANTIZE", False))
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args(argv)


def normalize_voice_mode(value: str) -> str:
    normalized = (value or DEFAULT_VOICE_MODE).strip().lower().replace("_", "-")
    if normalized in {"reference", "reference-clone", "clone", "voice-clone"}:
        return "clone"
    if normalized in {"preset", "builtin", "built-in", "voice", "predefined"}:
        return "preset"
    raise ValueError("voice mode must be reference-clone or preset")


def optional_int(value: str) -> int | None:
    if value.strip() == "":
        return None
    return int(value)


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


def optional_path(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def friendly_startup_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    gated_markers = (
        "we could not download the weights for the model with voice cloning",
        "kyutai/pocket-tts",
        "voice cloning",
        "accept the terms",
    )
    if any(marker in lowered for marker in gated_markers) and "without voice cloning" in lowered:
        return POCKETTTS_CLONE_AUTH_HELP
    if "repository not found" in lowered and "kyutai/pocket-tts" in lowered:
        return POCKETTTS_CLONE_AUTH_HELP
    if "401" in lowered and "kyutai/pocket-tts" in lowered:
        return POCKETTTS_CLONE_AUTH_HELP
    if "403" in lowered and "kyutai/pocket-tts" in lowered:
        return POCKETTTS_CLONE_AUTH_HELP
    return message


def load_runtime(config: RuntimeConfig) -> PocketRuntime:
    import torch
    from pocket_tts import TTSModel

    torch_version = str(torch.__version__)
    torch_cuda = str(torch.version.cuda)
    cuda_available = bool(torch.cuda.is_available())
    print(
        f"Using PocketTTS CPU path: torch {torch_version}, CUDA build {torch_cuda}, CUDA available {cuda_available}",
        flush=True,
    )
    print("PocketTTS is CPU-first upstream; this container does not reserve the RTX 5070.", flush=True)

    print(f"Loading PocketTTS language model: {config.language}...", flush=True)
    loaded_at = time.perf_counter()
    model = TTSModel.load_model(language=config.language, quantize=config.quantize)
    print(f"Model loaded in {time.perf_counter() - loaded_at:.1f}s.", flush=True)

    if config.voice_mode == "clone":
        if config.ref_audio is None:
            raise RuntimeError("PocketTTS voice cloning requires a reference WAV.")
        voice_source = str(config.ref_audio)
        print(f"Preparing PocketTTS reference voice: {voice_source}", flush=True)
        print("Reference text is not used by PocketTTS.", flush=True)
        prepared_at = time.perf_counter()
        voice_state = model.get_state_for_audio_prompt(voice_source, truncate=config.truncate_ref)
        print(f"Reference voice prepared in {time.perf_counter() - prepared_at:.1f}s.", flush=True)
    else:
        voice_source = config.voice_id
        print(f"Using PocketTTS built-in voice: {voice_source}", flush=True)
        prepared_at = time.perf_counter()
        voice_state = model.get_state_for_audio_prompt(voice_source)
        print(f"Built-in voice loaded in {time.perf_counter() - prepared_at:.1f}s.", flush=True)

    return PocketRuntime(
        model=model,
        voice_state=voice_state,
        sample_rate=int(model.sample_rate),
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        cuda_available=cuda_available,
        prepared_voice_source=voice_source,
    )


def interactive_loop(runtime: PocketRuntime, config: RuntimeConfig) -> int:
    print("READY: PocketTTS is listening. Type text, :paste, :file /workspace/prompt.txt, or :q.", flush=True)

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


def synthesize(runtime: PocketRuntime, config: RuntimeConfig, text: str) -> int:
    chunks = split_pocket_text(text, max_chars=config.chunk_chars)
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
        write_pocket_wav(runtime, config, chunk, part_path)
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


def write_pocket_wav(runtime: PocketRuntime, config: RuntimeConfig, text: str, output_path: Path) -> None:
    import soundfile as sf

    audio = runtime.model.generate_audio(
        runtime.voice_state,
        text,
        max_tokens=config.max_tokens,
        frames_after_eos=config.frames_after_eos,
        copy_state=True,
    )
    audio_np = audio.detach().cpu().numpy()
    sf.write(output_path, audio_np, runtime.sample_rate, subtype="PCM_16")


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
    runtime: PocketRuntime,
    source_text: str,
    chunks: list[str],
    chunk_timings: list[dict[str, float | int]],
    audio_seconds: float,
    elapsed_seconds: float,
    rtf: float,
) -> None:
    metadata = {
        "audio_seconds": round(audio_seconds, 3),
        "chunk_chars": config.chunk_chars,
        "chunk_count": len(chunks),
        "chunk_timings": chunk_timings,
        "cuda_available": runtime.cuda_available,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "frames_after_eos": config.frames_after_eos,
        "language": config.language,
        "max_tokens": config.max_tokens,
        "pause_ms": config.pause_ms,
        "prepared_voice_source": runtime.prepared_voice_source,
        "quantize": config.quantize,
        "realtime_factor": round(rtf, 4),
        "sample_rate": runtime.sample_rate,
        "source_text": source_text,
        "torch_cuda": runtime.torch_cuda,
        "torch_version": runtime.torch_version,
        "truncate_ref": config.truncate_ref,
        "voice_id": config.voice_id,
        "voice_mode": config.voice_mode,
        "voice_name": config.voice_name,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def split_pocket_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
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
    slug = words[:48].strip("-") or "pockettts"
    return f"{timestamp}-{slug}"


if __name__ == "__main__":
    raise SystemExit(main())
