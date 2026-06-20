from __future__ import annotations

import re
from typing import Iterable


def split_long_text(text: str, max_chars: int = 250, language: str = "en") -> list[str]:
    """Split long text into synthesis-sized chunks without changing the words."""
    if max_chars < 80:
        raise ValueError("max_chars must be at least 80")

    normalized = _normalize_text(text)
    if not normalized:
        return []

    chunks: list[str] = []
    for paragraph in _paragraphs(normalized):
        sentences = _sentences(paragraph, language)
        chunks.extend(_pack_sentences(sentences, max_chars))
    return chunks


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _paragraphs(text: str) -> Iterable[str]:
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            yield paragraph


def _sentences(text: str, language: str) -> list[str]:
    try:
        import pysbd

        segmenter = pysbd.Segmenter(language=language, clean=False)
        sentences = [item.strip() for item in segmenter.segment(text)]
        return [item for item in sentences if item]
    except Exception:
        return _fallback_sentences(text)


def _fallback_sentences(text: str) -> list[str]:
    starts = [0]
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        starts.append(match.end())

    sentences: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def _pack_sentences(sentences: Iterable[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        for part in _split_oversized(sentence, max_chars):
            if not current:
                current = part
                continue

            candidate = f"{current} {part}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                chunks.append(current)
                current = part

    if current:
        chunks.append(current)
    return chunks


def _split_oversized(text: str, max_chars: int) -> list[str]:
    remaining = text.strip()
    parts: list[str] = []

    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars + 1)
        if cut < max_chars // 2:
            cut = max_chars

        part = remaining[:cut].strip()
        if part:
            parts.append(part)
        remaining = remaining[cut:].strip()

    if remaining:
        parts.append(remaining)
    return parts
