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


DEFAULT_MODEL = "F5TTS_v1_Base"
DEFAULT_VOICE_MODE = "clone"
DEFAULT_VOICE_NAME = "Reference voice clone"
DEFAULT_CHUNK_CHARS = 260
DEFAULT_PAUSE_MS = 180
DEFAULT_NFE_STEP = 32
DEFAULT_CFG_STRENGTH = 2.0
DEFAULT_SWAY_SAMPLING_COEF = -1.0
DEFAULT_SPEED = 1.0
DEFAULT_TARGET_RMS = 0.1
DEFAULT_CROSS_FADE_DURATION = 0.15
PROMPT = "f5tts> "


@dataclass
class RuntimeConfig:
    model_name: str
    voice_mode: str
    voice_name: str
    output_dir: Path
    ref_audio: Path
    ref_text_file: Path
    ref_text: str
    chunk_chars: int
    pause_ms: int
    keep_parts: bool
    use_cuda: bool
    hf_cache_dir: str
    nfe_step: int
    cfg_strength: float
    sway_sampling_coef: float
    speed: float
    target_rms: float
    cross_fade_duration: float
    remove_silence: bool


@dataclass
class F5Runtime:
    model: object
    device: str
    torch_version: str
    torch_cuda: str
    visible_gpus: list[str]
    sample_rate: int


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        voice_mode = normalize_voice_mode(args.voice_mode)
        ref_text_file = required_path(args.ref_text_file, "reference transcript")
        ref_text = ref_text_file.read_text(encoding="utf-8").strip()
        if not ref_text:
            raise ValueError(f"reference transcript was empty: {ref_text_file}")
        config = RuntimeConfig(
            model_name=args.model_name,
            voice_mode=voice_mode,
            voice_name=args.voice_name,
            output_dir=Path(args.output_dir),
            ref_audio=required_path(args.ref_audio, "reference audio"),
            ref_text_file=ref_text_file,
            ref_text=ref_text,
            chunk_chars=args.chunk_chars,
            pause_ms=args.pause_ms,
            keep_parts=args.keep_parts,
            use_cuda=args.cuda,
            hf_cache_dir=args.hf_cache_dir,
            nfe_step=args.nfe_step,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            speed=args.speed,
            target_rms=args.target_rms,
            cross_fade_duration=args.cross_fade_duration,
            remove_silence=args.remove_silence,
        )
    except Exception as exc:
        print(f"F5-TTS configuration failed: {exc}", file=sys.stderr, flush=True)
        return 2

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.interactive:
        print("F5-TTS is loading. Wait for the READY line and f5tts> prompt before typing.", flush=True)

    try:
        runtime = load_runtime(config)
    except Exception as exc:
        print(f"F5-TTS startup failed: {exc}", file=sys.stderr, flush=True)
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
    parser = argparse.ArgumentParser(description="Interactive F5-TTS reference voice-clone generator.")
    parser.add_argument("--interactive", action="store_true", help="Read text from stdin repeatedly.")
    parser.add_argument("--text", help="Synthesize one text string and exit.")
    parser.add_argument("--prompt-file", help="Read one UTF-8 text file and exit.")
    parser.add_argument("--model-name", default=os.getenv("F5TTS_MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--voice-mode", default=os.getenv("F5TTS_VOICE_MODE", DEFAULT_VOICE_MODE))
    parser.add_argument("--voice-name", default=os.getenv("F5TTS_VOICE_NAME", DEFAULT_VOICE_NAME))
    parser.add_argument("--output-dir", default=os.getenv("F5TTS_OUTPUT_DIR", "/app/output/f5tts"))
    parser.add_argument("--ref-audio", default=os.getenv("F5TTS_REF_AUDIO", "/workspace/reference.wav"))
    parser.add_argument("--ref-text-file", default=os.getenv("F5TTS_REF_TEXT_FILE", "/workspace/reference.txt"))
    parser.add_argument("--chunk-chars", type=int, default=int(os.getenv("F5TTS_CHUNK_CHARS", str(DEFAULT_CHUNK_CHARS))))
    parser.add_argument("--pause-ms", type=int, default=int(os.getenv("F5TTS_PAUSE_MS", str(DEFAULT_PAUSE_MS))))
    parser.add_argument("--hf-cache-dir", default=os.getenv("F5TTS_HF_CACHE_DIR", "/models/f5tts"))
    parser.add_argument("--nfe-step", type=int, default=int(os.getenv("F5TTS_NFE_STEP", str(DEFAULT_NFE_STEP))))
    parser.add_argument("--cfg-strength", type=float, default=float(os.getenv("F5TTS_CFG_STRENGTH", str(DEFAULT_CFG_STRENGTH))))
    parser.add_argument(
        "--sway-sampling-coef",
        type=float,
        default=float(os.getenv("F5TTS_SWAY_SAMPLING_COEF", str(DEFAULT_SWAY_SAMPLING_COEF))),
    )
    parser.add_argument("--speed", type=float, default=float(os.getenv("F5TTS_SPEED", str(DEFAULT_SPEED))))
    parser.add_argument("--target-rms", type=float, default=float(os.getenv("F5TTS_TARGET_RMS", str(DEFAULT_TARGET_RMS))))
    parser.add_argument(
        "--cross-fade-duration",
        type=float,
        default=float(os.getenv("F5TTS_CROSS_FADE_DURATION", str(DEFAULT_CROSS_FADE_DURATION))),
    )
    parser.add_argument("--remove-silence", action=argparse.BooleanOptionalAction, default=env_bool("F5TTS_REMOVE_SILENCE", False))
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=env_bool("F5TTS_USE_CUDA", True))
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args(argv)


def normalize_voice_mode(value: str) -> str:
    normalized = (value or DEFAULT_VOICE_MODE).strip().lower().replace("_", "-")
    if normalized in {"reference", "reference-clone", "clone", "voice-clone"}:
        return "clone"
    raise ValueError("F5-TTS only exposes reference-clone in this launcher; no official preset catalog is bundled.")


def required_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    raise FileNotFoundError(f"{label} was not found: {path}")


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_runtime(config: RuntimeConfig) -> F5Runtime:
    import torch
    from f5_tts.api import F5TTS

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

    print(f"Loading F5-TTS model {config.model_name} on {device}...", flush=True)
    loaded_at = time.perf_counter()
    model = F5TTS(model=config.model_name, device=device, hf_cache_dir=config.hf_cache_dir)
    print(f"Model loaded in {time.perf_counter() - loaded_at:.1f}s.", flush=True)
    print(f"Reference voice: {config.ref_audio}", flush=True)
    print(f"Reference transcript: {config.ref_text_file}", flush=True)

    return F5Runtime(
        model=model,
        device=device,
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        visible_gpus=visible_gpus,
        sample_rate=int(getattr(model, "target_sample_rate", 24000)),
    )


def interactive_loop(runtime: F5Runtime, config: RuntimeConfig) -> int:
    print("READY: F5-TTS is listening. Type text, :paste, :file /workspace/prompt.txt, or :q.", flush=True)

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


def synthesize(runtime: F5Runtime, config: RuntimeConfig, text: str) -> int:
    chunks = split_f5tts_text(text, max_chars=config.chunk_chars)
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
        write_f5tts_wav(runtime, config, chunk, part_path)
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


def write_f5tts_wav(runtime: F5Runtime, config: RuntimeConfig, text: str, output_path: Path) -> None:
    wav, _sample_rate, _spec = runtime.model.infer(
        ref_file=str(config.ref_audio),
        ref_text=config.ref_text,
        gen_text=text,
        show_info=lambda message: print(message, flush=True),
        progress=None,
        target_rms=config.target_rms,
        cross_fade_duration=config.cross_fade_duration,
        sway_sampling_coef=config.sway_sampling_coef,
        cfg_strength=config.cfg_strength,
        nfe_step=config.nfe_step,
        speed=config.speed,
        remove_silence=config.remove_silence,
        file_wave=str(output_path),
    )
    if wav is None or not output_path.exists():
        raise RuntimeError("F5-TTS did not generate audio for this chunk.")


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
    runtime: F5Runtime,
    source_text: str,
    chunks: list[str],
    chunk_timings: list[dict[str, float | int]],
    audio_seconds: float,
    elapsed_seconds: float,
    rtf: float,
) -> None:
    metadata = {
        "audio_seconds": round(audio_seconds, 3),
        "cfg_strength": config.cfg_strength,
        "chunk_chars": config.chunk_chars,
        "chunk_count": len(chunks),
        "chunk_timings": chunk_timings,
        "cross_fade_duration": config.cross_fade_duration,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "model_name": config.model_name,
        "nfe_step": config.nfe_step,
        "pause_ms": config.pause_ms,
        "realtime_factor": round(rtf, 4),
        "ref_audio": str(config.ref_audio),
        "ref_text_file": str(config.ref_text_file),
        "remove_silence": config.remove_silence,
        "sample_rate": runtime.sample_rate,
        "source_text": source_text,
        "speed": config.speed,
        "sway_sampling_coef": config.sway_sampling_coef,
        "target_rms": config.target_rms,
        "torch_cuda": runtime.torch_cuda,
        "torch_version": runtime.torch_version,
        "use_cuda": config.use_cuda,
        "visible_gpus": runtime.visible_gpus,
        "voice_mode": config.voice_mode,
        "voice_name": config.voice_name,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def split_f5tts_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
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
    soft_chars = max(130, int(max_chars * 0.66))
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
    target_chars = max(145, int(max_chars * 0.75))
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
    slug = words[:48].strip("-") or "f5tts"
    return f"{timestamp}-{slug}"


if __name__ == "__main__":
    raise SystemExit(main())
