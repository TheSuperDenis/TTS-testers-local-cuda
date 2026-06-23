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


DEFAULT_VOICE_MODE = "clone"
DEFAULT_VOICE_NAME = "Reference voice clone"
DEFAULT_CHUNK_CHARS = 220
DEFAULT_PAUSE_MS = 220
DEFAULT_EXAGGERATION = 0.45
DEFAULT_CFG_WEIGHT = 0.35
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.05
DEFAULT_REPETITION_PENALTY = 1.2
PROMPT = "chatterbox> "


@dataclass
class RuntimeConfig:
    voice_mode: str
    voice_name: str
    output_dir: Path
    ref_audio: Path | None
    chunk_chars: int
    pause_ms: int
    keep_parts: bool
    use_cuda: bool
    exaggeration: float
    cfg_weight: float
    temperature: float
    top_p: float
    min_p: float
    repetition_penalty: float


@dataclass
class ChatterboxRuntime:
    model: object
    device: str
    sample_rate: int
    torch_version: str
    torch_cuda: str
    visible_gpus: list[str]
    prepared_voice_mode: str


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        voice_mode = normalize_voice_mode(args.voice_mode)
        config = RuntimeConfig(
            voice_mode=voice_mode,
            voice_name=args.voice_name,
            output_dir=Path(args.output_dir),
            ref_audio=required_path(args.ref_audio, "reference audio") if voice_mode == "clone" else optional_path(args.ref_audio),
            chunk_chars=args.chunk_chars,
            pause_ms=args.pause_ms,
            keep_parts=args.keep_parts,
            use_cuda=args.cuda,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
        )
    except Exception as exc:
        print(f"Chatterbox configuration failed: {exc}", file=sys.stderr, flush=True)
        return 2

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.interactive:
        print(
            "Chatterbox is loading. Wait for the READY line and chatterbox> prompt before typing.",
            flush=True,
        )

    try:
        runtime = load_runtime(config)
    except Exception as exc:
        print(f"Chatterbox startup failed: {exc}", file=sys.stderr, flush=True)
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
    parser = argparse.ArgumentParser(description="Interactive Chatterbox voice-clone generator.")
    parser.add_argument("--interactive", action="store_true", help="Read text from stdin repeatedly.")
    parser.add_argument("--text", help="Synthesize one text string and exit.")
    parser.add_argument("--prompt-file", help="Read one UTF-8 text file and exit.")
    parser.add_argument("--voice-mode", default=os.getenv("CHATTERBOX_VOICE_MODE", DEFAULT_VOICE_MODE))
    parser.add_argument("--voice-name", default=os.getenv("CHATTERBOX_VOICE_NAME", DEFAULT_VOICE_NAME))
    parser.add_argument("--output-dir", default=os.getenv("CHATTERBOX_OUTPUT_DIR", "/app/output/chatterbox"))
    parser.add_argument("--ref-audio", default=os.getenv("CHATTERBOX_REF_AUDIO", "/workspace/reference.wav"))
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=int(os.getenv("CHATTERBOX_CHUNK_CHARS", str(DEFAULT_CHUNK_CHARS))),
    )
    parser.add_argument("--pause-ms", type=int, default=int(os.getenv("CHATTERBOX_PAUSE_MS", str(DEFAULT_PAUSE_MS))))
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=float(os.getenv("CHATTERBOX_EXAGGERATION", str(DEFAULT_EXAGGERATION))),
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=float(os.getenv("CHATTERBOX_CFG_WEIGHT", str(DEFAULT_CFG_WEIGHT))),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("CHATTERBOX_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
    )
    parser.add_argument("--top-p", type=float, default=float(os.getenv("CHATTERBOX_TOP_P", str(DEFAULT_TOP_P))))
    parser.add_argument("--min-p", type=float, default=float(os.getenv("CHATTERBOX_MIN_P", str(DEFAULT_MIN_P))))
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=float(os.getenv("CHATTERBOX_REPETITION_PENALTY", str(DEFAULT_REPETITION_PENALTY))),
    )
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=env_bool("CHATTERBOX_USE_CUDA", True))
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args(argv)


def normalize_voice_mode(value: str) -> str:
    normalized = (value or DEFAULT_VOICE_MODE).strip().lower().replace("_", "-")
    if normalized in {"reference", "reference-clone", "clone", "voice-clone"}:
        return "clone"
    if normalized in {"default", "builtin", "built-in", "builtin-default", "preset"}:
        return "builtin"
    raise ValueError("voice mode must be reference-clone or builtin-default")


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


def load_runtime(config: RuntimeConfig) -> ChatterboxRuntime:
    import torch
    from chatterbox.tts import ChatterboxTTS

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

    print(f"Loading Chatterbox 500M on {device}...", flush=True)
    loaded_at = time.perf_counter()
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"Model loaded in {time.perf_counter() - loaded_at:.1f}s.", flush=True)

    if config.voice_mode == "clone":
        if config.ref_audio is None:
            raise RuntimeError("Chatterbox voice cloning requires a reference WAV.")
        print(f"Preparing Chatterbox reference voice: {config.ref_audio}", flush=True)
        prepared_at = time.perf_counter()
        model.prepare_conditionals(str(config.ref_audio), exaggeration=config.exaggeration)
        print(f"Reference voice prepared in {time.perf_counter() - prepared_at:.1f}s.", flush=True)
    else:
        if getattr(model, "conds", None) is None:
            raise RuntimeError("The Chatterbox checkpoint did not include built-in voice conditioning.")
        print("Using Chatterbox built-in default voice. No local reference WAV will be used.", flush=True)

    return ChatterboxRuntime(
        model=model,
        device=device,
        sample_rate=int(model.sr),
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        visible_gpus=visible_gpus,
        prepared_voice_mode=config.voice_mode,
    )


def interactive_loop(runtime: ChatterboxRuntime, config: RuntimeConfig) -> int:
    print("READY: Chatterbox is listening. Type text, :paste, :file /workspace/prompt.txt, or :q.", flush=True)

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


def synthesize(runtime: ChatterboxRuntime, config: RuntimeConfig, text: str) -> int:
    chunks = split_chatterbox_text(text, max_chars=config.chunk_chars)
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
        write_chatterbox_wav(runtime, config, chunk, part_path)
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


def write_chatterbox_wav(runtime: ChatterboxRuntime, config: RuntimeConfig, text: str, output_path: Path) -> None:
    import numpy as np
    import soundfile as sf

    wav = runtime.model.generate(
        text,
        repetition_penalty=config.repetition_penalty,
        min_p=config.min_p,
        top_p=config.top_p,
        exaggeration=config.exaggeration,
        cfg_weight=config.cfg_weight,
        temperature=config.temperature,
    )
    audio = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    if audio.size == 0:
        raise RuntimeError("Chatterbox did not generate any audio for this chunk.")
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
    runtime: ChatterboxRuntime,
    source_text: str,
    chunks: list[str],
    chunk_timings: list[dict[str, float | int]],
    audio_seconds: float,
    elapsed_seconds: float,
    rtf: float,
) -> None:
    metadata = {
        "audio_seconds": round(audio_seconds, 3),
        "cfg_weight": config.cfg_weight,
        "chunk_chars": config.chunk_chars,
        "chunk_count": len(chunks),
        "chunk_timings": chunk_timings,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "exaggeration": config.exaggeration,
        "min_p": config.min_p,
        "model": "ResembleAI/chatterbox",
        "pause_ms": config.pause_ms,
        "realtime_factor": round(rtf, 4),
        "repetition_penalty": config.repetition_penalty,
        "sample_rate": runtime.sample_rate,
        "source_text": source_text,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "torch_cuda": runtime.torch_cuda,
        "torch_version": runtime.torch_version,
        "use_cuda": config.use_cuda,
        "visible_gpus": runtime.visible_gpus,
        "voice_mode": config.voice_mode,
        "voice_name": config.voice_name,
    }
    if config.ref_audio:
        metadata["ref_audio"] = str(config.ref_audio)
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def split_chatterbox_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
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
    slug = words[:48].strip("-") or "chatterbox"
    return f"{timestamp}-{slug}"


if __name__ == "__main__":
    raise SystemExit(main())
