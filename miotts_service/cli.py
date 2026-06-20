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


DEFAULT_MODEL = "Aratako/MioTTS-0.1B"
DEFAULT_CODEC_MODEL = "Aratako/MioCodec-25Hz-24kHz"
SPEECH_TOKEN_PATTERN = re.compile(r"<\|s_(\d+)\|>")
PROMPT = "miotts0.1b> "


@dataclass
class RuntimeConfig:
    model_id: str
    codec_model_id: str
    output_dir: Path
    ref_audio: Path
    ref_text_file: Path | None
    chunk_chars: int
    pause_ms: int
    keep_parts: bool
    temperature: float
    top_p: float
    max_new_tokens: int
    max_token_overhead: int
    max_tokens_per_char: float
    no_repeat_ngram_size: int
    repetition_penalty: float
    reference_max_seconds: float


@dataclass
class MioRuntime:
    torch: object
    tokenizer: object
    model: object
    codec: object
    device: str
    global_embedding: object
    sample_rate: int
    reference_text: str


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = RuntimeConfig(
            model_id=args.model_id,
            codec_model_id=args.codec_model_id,
            output_dir=Path(args.output_dir),
            ref_audio=required_path(args.ref_audio, "reference audio"),
            ref_text_file=optional_path(args.ref_text_file, "reference transcript"),
            chunk_chars=args.chunk_chars,
            pause_ms=args.pause_ms,
            keep_parts=args.keep_parts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            max_token_overhead=args.max_token_overhead,
            max_tokens_per_char=args.max_tokens_per_char,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
            reference_max_seconds=args.reference_max_seconds,
        )
    except ValueError as exc:
        print(f"MioTTS startup failed: {exc}", file=sys.stderr, flush=True)
        return 1

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.interactive:
        print(
            "MioTTS 0.1B is loading. Wait for the READY line and miotts0.1b> prompt before typing.",
            flush=True,
        )

    try:
        runtime = load_runtime(config)
    except Exception as exc:
        print(f"MioTTS startup failed: {exc}", file=sys.stderr, flush=True)
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
    parser = argparse.ArgumentParser(description="Interactive MioTTS 0.1B voice-clone generator.")
    parser.add_argument("--interactive", action="store_true", help="Read text from stdin repeatedly.")
    parser.add_argument("--text", help="Synthesize one text string and exit.")
    parser.add_argument("--prompt-file", help="Read one UTF-8 text file and exit.")
    parser.add_argument("--model-id", default=os.getenv("MIOTTS_MODEL_ID", DEFAULT_MODEL))
    parser.add_argument("--codec-model-id", default=os.getenv("MIOTTS_CODEC_MODEL", DEFAULT_CODEC_MODEL))
    parser.add_argument("--output-dir", default=os.getenv("MIOTTS_OUTPUT_DIR", "/app/output/miotts"))
    parser.add_argument("--ref-audio", default=os.getenv("MIOTTS_REF_AUDIO", "/workspace/reference.wav"))
    parser.add_argument("--ref-text-file", default=os.getenv("MIOTTS_REF_TEXT_FILE", "/workspace/reference.txt"))
    parser.add_argument("--chunk-chars", type=int, default=int(os.getenv("MIOTTS_CHUNK_CHARS", "160")))
    parser.add_argument("--pause-ms", type=int, default=int(os.getenv("MIOTTS_PAUSE_MS", "180")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("MIOTTS_TEMPERATURE", "0.55")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("MIOTTS_TOP_P", "0.85")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("MIOTTS_MAX_NEW_TOKENS", "520")))
    parser.add_argument("--max-token-overhead", type=int, default=int(os.getenv("MIOTTS_MAX_TOKEN_OVERHEAD", "80")))
    parser.add_argument("--max-tokens-per-char", type=float, default=float(os.getenv("MIOTTS_MAX_TOKENS_PER_CHAR", "3.0")))
    parser.add_argument("--no-repeat-ngram-size", type=int, default=int(os.getenv("MIOTTS_NO_REPEAT_NGRAM_SIZE", "6")))
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=float(os.getenv("MIOTTS_REPETITION_PENALTY", "1.15")),
    )
    parser.add_argument(
        "--reference-max-seconds",
        type=float,
        default=float(os.getenv("MIOTTS_REFERENCE_MAX_SECONDS", "20.0")),
    )
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args(argv)


def required_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    raise ValueError(f"{label} not found at {path}")


def optional_path(value: str, label: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    raise ValueError(f"{label} not found at {path}")


def load_runtime(config: RuntimeConfig) -> MioRuntime:
    import torch
    from miocodec import MioCodecModel
    from miocodec.util import load_audio
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the container. Check Docker GPU support and the GPU mask.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    visible = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    print(f"CUDA ready: torch {torch.__version__}, CUDA {torch.version.cuda}, visible GPUs: {visible}", flush=True)

    device = "cuda:0"
    print(f"Loading {config.model_id}...", flush=True)
    loaded_at = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"LLM loaded in {time.perf_counter() - loaded_at:.1f}s.", flush=True)

    print(f"Loading {config.codec_model_id}...", flush=True)
    codec_started = time.perf_counter()
    codec = MioCodecModel.from_pretrained(config.codec_model_id).eval().to(device)
    sample_rate = int(codec.config.sample_rate)
    print(f"Codec loaded in {time.perf_counter() - codec_started:.1f}s at {sample_rate} Hz.", flush=True)

    print(f"Voice clone sample: {config.ref_audio}", flush=True)
    reference_started = time.perf_counter()
    reference_waveform = load_audio(str(config.ref_audio), sample_rate=sample_rate)
    reference_waveform = trim_reference(reference_waveform, sample_rate, config.reference_max_seconds)
    reference_waveform = reference_waveform.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        ref_features = codec.encode(reference_waveform, return_content=False, return_global=True)
    global_embedding = ref_features.global_embedding
    print(f"Voice clone embedding prepared in {time.perf_counter() - reference_started:.1f}s.", flush=True)

    reference_text = ""
    if config.ref_text_file:
        reference_text = config.ref_text_file.read_text(encoding="utf-8").strip()
        print(f"Voice reference transcript: {config.ref_text_file}", flush=True)

    return MioRuntime(
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        codec=codec,
        device=device,
        global_embedding=global_embedding,
        sample_rate=sample_rate,
        reference_text=reference_text,
    )


def trim_reference(waveform, sample_rate: int, max_seconds: float):
    if max_seconds <= 0:
        return waveform
    max_samples = int(sample_rate * max_seconds)
    if waveform.numel() > max_samples:
        original_seconds = waveform.numel() / sample_rate
        print(f"Reference audio trimmed: {original_seconds:.1f}s -> {max_seconds:.1f}s", flush=True)
        return waveform[:max_samples]
    return waveform


def interactive_loop(runtime: MioRuntime, config: RuntimeConfig) -> int:
    print("READY: MioTTS 0.1B is listening. Type text, :paste, :file /workspace/prompt.txt, or :q.", flush=True)

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


def synthesize(runtime: MioRuntime, config: RuntimeConfig, text: str) -> int:
    chunks = split_long_text(text, max_chars=config.chunk_chars, language="en")
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
        llm_started = time.perf_counter()
        llm_text = generate_speech_token_text(runtime, config, chunk)
        tokens = parse_speech_tokens(llm_text)
        llm_seconds = time.perf_counter() - llm_started

        codec_started = time.perf_counter()
        audio = decode_tokens(runtime, tokens)
        codec_seconds = time.perf_counter() - codec_started

        write_wav(part_path, audio, runtime.sample_rate)
        part_paths.append(part_path)
        elapsed = time.perf_counter() - chunk_started
        chunk_timings.append(
            {
                "chunk_index": index,
                "codec_seconds": round(codec_seconds, 3),
                "elapsed_seconds": round(elapsed, 3),
                "llm_seconds": round(llm_seconds, 3),
                "token_count": len(tokens),
            }
        )
        print(
            f"  chunk {index}/{len(chunks)}: {elapsed:.1f}s "
            f"(llm {llm_seconds:.1f}s, codec {codec_seconds:.1f}s, tokens {len(tokens)})",
            flush=True,
        )

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


def generate_speech_token_text(runtime: MioRuntime, config: RuntimeConfig, text: str) -> str:
    torch = runtime.torch
    tokenizer = runtime.tokenizer
    model = runtime.model
    messages = [{"role": "user", "content": text.strip()}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(runtime.device)
    attention_mask = torch.ones_like(input_ids)

    effective_max_tokens = effective_miotts_max_tokens(config, text)
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": effective_max_tokens,
        "do_sample": config.temperature > 0,
        "top_p": config.top_p,
        "repetition_penalty": config.repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if config.no_repeat_ngram_size > 0:
        generation_kwargs["no_repeat_ngram_size"] = config.no_repeat_ngram_size
    if config.temperature > 0:
        generation_kwargs["temperature"] = config.temperature

    print(f"    MioTTS token cap: {effective_max_tokens}", flush=True)
    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)

    prompt_length = input_ids.shape[-1]
    generated_ids = output_ids[0, prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    if SPEECH_TOKEN_PATTERN.search(generated_text):
        return generated_text

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    if SPEECH_TOKEN_PATTERN.search(full_text):
        return full_text

    preview = generated_text.replace("\n", " ")[:240]
    raise RuntimeError(f"No speech tokens found in MioTTS LLM output. Preview: {preview}")


def effective_miotts_max_tokens(config: RuntimeConfig, text: str) -> int:
    dynamic_cap = int(len(text.strip()) * config.max_tokens_per_char) + config.max_token_overhead
    return max(80, min(config.max_new_tokens, dynamic_cap))


def parse_speech_tokens(text: str) -> list[int]:
    tokens = [int(value) for value in SPEECH_TOKEN_PATTERN.findall(text)]
    if not tokens:
        raise RuntimeError("No speech tokens found in MioTTS LLM output.")
    return tokens


def decode_tokens(runtime: MioRuntime, tokens: list[int]):
    torch = runtime.torch
    token_tensor = torch.tensor(tokens, dtype=torch.long, device=runtime.device)
    with torch.inference_mode():
        return runtime.codec.decode(
            global_embedding=runtime.global_embedding,
            content_token_indices=token_tensor,
        )


def write_wav(path: Path, waveform, sample_rate: int) -> None:
    import soundfile as sf

    if hasattr(waveform, "detach"):
        waveform = waveform.detach().float().cpu()
    if hasattr(waveform, "dim") and waveform.dim() == 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    if hasattr(waveform, "numpy"):
        waveform = waveform.numpy()
    sf.write(path, waveform, sample_rate, subtype="PCM_16")


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
    runtime: MioRuntime,
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
        "chunk_timings": chunk_timings,
        "codec_model_id": config.codec_model_id,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "max_new_tokens": config.max_new_tokens,
        "max_token_overhead": config.max_token_overhead,
        "max_tokens_per_char": config.max_tokens_per_char,
        "model_id": config.model_id,
        "no_repeat_ngram_size": config.no_repeat_ngram_size,
        "pause_ms": config.pause_ms,
        "realtime_factor": round(rtf, 4),
        "reference_max_seconds": config.reference_max_seconds,
        "sample_rate": runtime.sample_rate,
        "source_text": source_text,
        "speaker_reference_text": runtime.reference_text,
        "speaker_reference_text_path": str(config.ref_text_file) if config.ref_text_file else "",
        "speaker_wav": str(config.ref_audio),
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def output_name(text: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    words = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    slug = words[:48].strip("-") or "miotts"
    return f"{timestamp}-{slug}"


if __name__ == "__main__":
    raise SystemExit(main())
