from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from xtts_service.text_splitter import split_long_text


DEFAULT_MODEL_DIR = "/opt/moss-tts-nano/models"
DEFAULT_VOICE_MODE = "clone"
DEFAULT_VOICE_ID = "reference-clone"
DEFAULT_VOICE_NAME = "Reference voice clone"
DEFAULT_CPU_THREADS = 4
DEFAULT_EXECUTION_PROVIDER = "cuda"
DEFAULT_MAX_NEW_FRAMES = 375
DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS = 75
DEFAULT_SAMPLE_MODE = "fixed"
DEFAULT_CHUNK_CHARS = 260
DEFAULT_REALTIME_STREAMING_DECODE = True
PROMPT = "mossttsnano> "


@dataclass
class RuntimeConfig:
    model_dir: Path | None
    voice_mode: str
    voice_id: str
    voice_name: str
    output_dir: Path
    ref_audio: Path | None
    cpu_threads: int
    execution_provider: str
    max_new_frames: int
    voice_clone_max_text_tokens: int
    sample_mode: str
    realtime_streaming_decode: bool
    enable_wetext: bool
    normalize_text: bool
    chunk_chars: int
    seed: int | None
    text_temperature: float
    text_top_p: float
    text_top_k: int
    audio_temperature: float
    audio_top_p: float
    audio_top_k: int
    audio_repetition_penalty: float


@dataclass
class MossRuntime:
    engine: object
    prompt_audio_codes: list[list[int]]
    sample_rate: int
    channels: int
    ort_available_providers: list[str]
    ort_session_providers: list[str]
    prepared_voice_source: str


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        voice_mode = normalize_voice_mode(args.voice_mode)
        config = RuntimeConfig(
            model_dir=optional_path(args.model_dir),
            voice_mode=voice_mode,
            voice_id=args.voice_id,
            voice_name=args.voice_name,
            output_dir=Path(args.output_dir),
            ref_audio=required_path(args.ref_audio, "reference audio") if voice_mode == "clone" else optional_path(args.ref_audio),
            cpu_threads=args.cpu_threads,
            execution_provider=normalize_execution_provider(args.execution_provider),
            max_new_frames=args.max_new_frames,
            voice_clone_max_text_tokens=args.voice_clone_max_text_tokens,
            sample_mode=normalize_sample_mode(args.sample_mode),
            realtime_streaming_decode=args.realtime_streaming_decode,
            enable_wetext=args.enable_wetext,
            normalize_text=args.normalize_text,
            chunk_chars=args.chunk_chars,
            seed=args.seed,
            text_temperature=args.text_temperature,
            text_top_p=args.text_top_p,
            text_top_k=args.text_top_k,
            audio_temperature=args.audio_temperature,
            audio_top_p=args.audio_top_p,
            audio_top_k=args.audio_top_k,
            audio_repetition_penalty=args.audio_repetition_penalty,
        )
    except Exception as exc:
        print(f"MOSS-TTS-Nano configuration failed: {exc}", file=sys.stderr, flush=True)
        return 2

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.interactive:
        print(
            "MOSS-TTS-Nano is loading. Wait for the READY line and mossttsnano> prompt before typing.",
            flush=True,
        )

    try:
        runtime = load_runtime(config)
    except Exception as exc:
        print(f"MOSS-TTS-Nano startup failed: {exc}", file=sys.stderr, flush=True)
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
    parser = argparse.ArgumentParser(description="Interactive MOSS-TTS-Nano ONNX voice-clone and preset generator.")
    parser.add_argument("--interactive", action="store_true", help="Read text from stdin repeatedly.")
    parser.add_argument("--text", help="Synthesize one text string and exit.")
    parser.add_argument("--prompt-file", help="Read one UTF-8 text file and exit.")
    parser.add_argument("--model-dir", default=os.getenv("MOSS_MODEL_DIR", DEFAULT_MODEL_DIR))
    parser.add_argument("--voice-mode", default=os.getenv("MOSS_VOICE_MODE", DEFAULT_VOICE_MODE))
    parser.add_argument("--voice-id", default=os.getenv("MOSS_VOICE_ID", DEFAULT_VOICE_ID))
    parser.add_argument("--voice-name", default=os.getenv("MOSS_VOICE_NAME", DEFAULT_VOICE_NAME))
    parser.add_argument("--output-dir", default=os.getenv("MOSS_OUTPUT_DIR", "/app/output/moss"))
    parser.add_argument("--ref-audio", default=os.getenv("MOSS_REF_AUDIO", "/workspace/reference.wav"))
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=int(os.getenv("MOSS_CPU_THREADS", str(DEFAULT_CPU_THREADS))),
    )
    parser.add_argument(
        "--execution-provider",
        default=os.getenv("MOSS_EXECUTION_PROVIDER", DEFAULT_EXECUTION_PROVIDER),
        choices=("cpu", "cuda"),
    )
    parser.add_argument(
        "--max-new-frames",
        type=int,
        default=int(os.getenv("MOSS_MAX_NEW_FRAMES", str(DEFAULT_MAX_NEW_FRAMES))),
    )
    parser.add_argument(
        "--voice-clone-max-text-tokens",
        type=int,
        default=int(os.getenv("MOSS_VOICE_CLONE_MAX_TEXT_TOKENS", str(DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS))),
    )
    parser.add_argument(
        "--sample-mode",
        default=os.getenv("MOSS_SAMPLE_MODE", DEFAULT_SAMPLE_MODE),
        choices=("greedy", "fixed", "full"),
    )
    parser.add_argument(
        "--realtime-streaming-decode",
        action=argparse.BooleanOptionalAction,
        default=env_bool("MOSS_REALTIME_STREAMING_DECODE", DEFAULT_REALTIME_STREAMING_DECODE),
    )
    parser.add_argument("--enable-wetext", action="store_true", default=env_bool("MOSS_ENABLE_WETEXT", False))
    parser.add_argument(
        "--normalize-text",
        action=argparse.BooleanOptionalAction,
        default=env_bool("MOSS_NORMALIZE_TEXT", True),
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=int(os.getenv("MOSS_CHUNK_CHARS", str(DEFAULT_CHUNK_CHARS))),
        help="Helper splitter limit used by tests and future chunk diagnostics. MOSS also chunks by token budget internally.",
    )
    parser.add_argument("--seed", type=optional_int, default=optional_env_int("MOSS_SEED"))
    parser.add_argument("--text-temperature", type=float, default=float(os.getenv("MOSS_TEXT_TEMPERATURE", "1.0")))
    parser.add_argument("--text-top-p", type=float, default=float(os.getenv("MOSS_TEXT_TOP_P", "1.0")))
    parser.add_argument("--text-top-k", type=int, default=int(os.getenv("MOSS_TEXT_TOP_K", "50")))
    parser.add_argument("--audio-temperature", type=float, default=float(os.getenv("MOSS_AUDIO_TEMPERATURE", "0.8")))
    parser.add_argument("--audio-top-p", type=float, default=float(os.getenv("MOSS_AUDIO_TOP_P", "0.95")))
    parser.add_argument("--audio-top-k", type=int, default=int(os.getenv("MOSS_AUDIO_TOP_K", "25")))
    parser.add_argument(
        "--audio-repetition-penalty",
        type=float,
        default=float(os.getenv("MOSS_AUDIO_REPETITION_PENALTY", "1.2")),
    )
    return parser.parse_args(argv)


def normalize_voice_mode(value: str) -> str:
    normalized = (value or DEFAULT_VOICE_MODE).strip().lower().replace("_", "-")
    if normalized in {"reference", "reference-clone", "clone", "voice-clone"}:
        return "clone"
    if normalized in {"preset", "builtin", "built-in", "voice", "predefined"}:
        return "preset"
    raise ValueError("voice mode must be reference-clone or preset")


def normalize_execution_provider(value: str) -> str:
    normalized = (value or DEFAULT_EXECUTION_PROVIDER).strip().lower()
    if normalized in {"cpu", "cpuexecutionprovider"}:
        return "cpu"
    if normalized in {"cuda", "gpu", "cudaexecutionprovider"}:
        return "cuda"
    raise ValueError("execution provider must be cpu or cuda")


def normalize_sample_mode(value: str) -> str:
    normalized = (value or DEFAULT_SAMPLE_MODE).strip().lower()
    if normalized in {"greedy", "fixed", "full"}:
        return normalized
    raise ValueError("sample mode must be greedy, fixed, or full")


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


def optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_runtime(config: RuntimeConfig) -> MossRuntime:
    import onnxruntime as ort
    from onnx_tts_runtime import OnnxTtsRuntime

    print(f"ONNX Runtime providers available: {ort.get_available_providers()}", flush=True)
    print(
        "Loading MOSS-TTS-Nano ONNX "
        f"with provider={config.execution_provider}, cpu_threads={config.cpu_threads}...",
        flush=True,
    )
    loaded_at = time.perf_counter()
    engine = OnnxTtsRuntime(
        model_dir=config.model_dir,
        thread_count=config.cpu_threads,
        max_new_frames=config.max_new_frames,
        do_sample=config.sample_mode != "greedy",
        sample_mode=config.sample_mode,
        execution_provider=config.execution_provider,
        output_dir=config.output_dir,
    )
    apply_generation_defaults(engine, config)
    print(f"Model loaded in {time.perf_counter() - loaded_at:.1f}s.", flush=True)

    session_providers = sorted({provider for session in engine.sessions.values() for provider in session.get_providers()})
    sample_rate = int(engine.codec_meta["codec_config"]["sample_rate"])
    channels = int(engine.codec_meta["codec_config"]["channels"])

    if config.voice_mode == "clone":
        if config.ref_audio is None:
            raise RuntimeError("MOSS-TTS-Nano voice cloning requires a reference WAV.")
        voice_source = "reference-clone"
        print(f"Preparing MOSS reference voice: {config.ref_audio}", flush=True)
        print("Reference text is not used by MOSS-TTS-Nano ONNX.", flush=True)
        prepared_at = time.perf_counter()
        prompt_audio_codes = engine.resolve_prompt_audio_codes(voice=None, prompt_audio_path=config.ref_audio)
        print(f"Reference voice prepared in {time.perf_counter() - prepared_at:.1f}s.", flush=True)
    else:
        voice_source = config.voice_id
        voice_names = {str(item.get("voice", "")) for item in engine.list_builtin_voices()}
        if config.voice_id not in voice_names:
            raise RuntimeError(f"MOSS built-in voice '{config.voice_id}' was not found in the model manifest.")
        print(f"Preparing MOSS built-in voice: {config.voice_id}", flush=True)
        prepared_at = time.perf_counter()
        prompt_audio_codes = engine.resolve_prompt_audio_codes(voice=config.voice_id, prompt_audio_path=None)
        print(f"Built-in voice prepared in {time.perf_counter() - prepared_at:.1f}s.", flush=True)

    return MossRuntime(
        engine=engine,
        prompt_audio_codes=prompt_audio_codes,
        sample_rate=sample_rate,
        channels=channels,
        ort_available_providers=list(ort.get_available_providers()),
        ort_session_providers=session_providers,
        prepared_voice_source=voice_source,
    )


def apply_generation_defaults(engine: object, config: RuntimeConfig) -> None:
    defaults = engine.manifest["generation_defaults"]
    defaults["max_new_frames"] = int(config.max_new_frames)
    defaults["sample_mode"] = config.sample_mode
    defaults["do_sample"] = config.sample_mode != "greedy"
    defaults["text_temperature"] = float(config.text_temperature)
    defaults["text_top_p"] = float(config.text_top_p)
    defaults["text_top_k"] = int(config.text_top_k)
    defaults["audio_temperature"] = float(config.audio_temperature)
    defaults["audio_top_p"] = float(config.audio_top_p)
    defaults["audio_top_k"] = int(config.audio_top_k)
    defaults["audio_repetition_penalty"] = float(config.audio_repetition_penalty)


def interactive_loop(runtime: MossRuntime, config: RuntimeConfig) -> int:
    print("READY: MOSS-TTS-Nano is listening. Type text, :paste, :file /workspace/prompt.txt, or :q.", flush=True)

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


def synthesize(runtime: MossRuntime, config: RuntimeConfig, text: str) -> int:
    text = str(text or "").strip()
    if not text:
        print("Nothing to synthesize.", flush=True)
        return 0

    started = time.perf_counter()
    base_name = output_name(text)
    output_path = config.output_dir / f"{base_name}.wav"

    if config.seed is not None:
        import numpy as np

        runtime.engine.rng = np.random.default_rng(int(config.seed))

    prepared_texts = runtime.engine.prepare_synthesis_text(
        text=text,
        voice=config.voice_id if config.voice_mode == "preset" else "",
        enable_wetext=config.enable_wetext,
        enable_normalize_tts_text=config.normalize_text,
    )
    prepared_text = str(prepared_texts["text"])
    text_chunks = runtime.engine.split_voice_clone_text(
        prepared_text,
        max_tokens=int(config.voice_clone_max_text_tokens),
    )
    if not text_chunks:
        print("Nothing to synthesize after normalization.", flush=True)
        return 0

    print(f"Synthesizing {len(text)} chars in {len(text_chunks)} MOSS token chunk(s)...", flush=True)
    waveform = synthesize_chunks(runtime, text_chunks, streaming=config.realtime_streaming_decode)
    write_wav(output_path, waveform, runtime.sample_rate)

    audio_seconds = audio_duration_seconds(waveform, runtime.sample_rate)
    elapsed = time.perf_counter() - started
    rtf = elapsed / audio_seconds if audio_seconds > 0 else 0.0
    write_metadata(
        output_path,
        config=config,
        runtime=runtime,
        source_text=text,
        prepared_texts=prepared_texts,
        chunks=text_chunks,
        audio_seconds=audio_seconds,
        elapsed_seconds=elapsed,
        rtf=rtf,
    )
    print(
        f"Done: {output_path} | audio {audio_seconds:.1f}s | elapsed {elapsed:.1f}s | RTF {rtf:.2f}x",
        flush=True,
    )
    return 0


def synthesize_chunks(runtime: MossRuntime, chunks: list[str], *, streaming: bool) -> Any:
    import numpy as np

    waveforms: list[Any] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_started = time.perf_counter()
        result = runtime.engine.synthesize_single_chunk(
            text=chunk,
            prompt_audio_codes=runtime.prompt_audio_codes,
            streaming=streaming,
        )
        chunk_waveform = np.asarray(result["waveform"], dtype=np.float32)
        waveforms.append(chunk_waveform)
        print(f"  chunk {index}/{len(chunks)}: {time.perf_counter() - chunk_started:.2f}s", flush=True)
        if index < len(chunks):
            pause_seconds = runtime.engine.estimate_voice_clone_inter_chunk_pause_seconds(chunk)
            pause_samples = max(0, int(round(runtime.sample_rate * pause_seconds)))
            if pause_samples:
                waveforms.append(np.zeros((pause_samples, runtime.channels), dtype=np.float32))

    if not waveforms:
        return np.zeros((0, runtime.channels), dtype=np.float32)
    return np.concatenate(waveforms, axis=0)


def write_wav(output_path: Path, waveform: Any, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    sf.write(output_path, audio, int(sample_rate), subtype="PCM_16")


def audio_duration_seconds(waveform: Any, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return float(getattr(waveform, "shape", [0])[0]) / float(sample_rate)


def write_metadata(
    output_path: Path,
    *,
    config: RuntimeConfig,
    runtime: MossRuntime,
    source_text: str,
    prepared_texts: dict[str, object],
    chunks: list[str],
    audio_seconds: float,
    elapsed_seconds: float,
    rtf: float,
) -> None:
    metadata = {
        "audio_repetition_penalty": config.audio_repetition_penalty,
        "audio_seconds": round(audio_seconds, 3),
        "audio_temperature": config.audio_temperature,
        "audio_top_k": config.audio_top_k,
        "audio_top_p": config.audio_top_p,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "cpu_threads": config.cpu_threads,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "enable_wetext": config.enable_wetext,
        "execution_provider": config.execution_provider,
        "max_new_frames": config.max_new_frames,
        "model_dir": str(config.model_dir) if config.model_dir else "",
        "normalize_text": config.normalize_text,
        "ort_available_providers": runtime.ort_available_providers,
        "ort_session_providers": runtime.ort_session_providers,
        "prepared_texts": prepared_texts,
        "prepared_voice_source": runtime.prepared_voice_source,
        "realtime_factor": round(rtf, 4),
        "realtime_streaming_decode": config.realtime_streaming_decode,
        "sample_mode": config.sample_mode,
        "sample_rate": runtime.sample_rate,
        "source_text": source_text,
        "text_temperature": config.text_temperature,
        "text_top_k": config.text_top_k,
        "text_top_p": config.text_top_p,
        "voice_clone_max_text_tokens": config.voice_clone_max_text_tokens,
        "voice_id": config.voice_id,
        "voice_mode": config.voice_mode,
        "voice_name": config.voice_name,
    }
    if config.seed is not None:
        metadata["seed"] = config.seed
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def split_moss_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
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
    soft_chars = max(130, int(max_chars * 0.62))
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
    target_chars = max(150, int(max_chars * 0.72))
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
    slug = words[:48].strip("-") or "moss"
    return f"{timestamp}-{slug}"


if __name__ == "__main__":
    raise SystemExit(main())
