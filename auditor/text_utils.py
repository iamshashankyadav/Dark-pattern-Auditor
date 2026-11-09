"""Small text helpers shared across phases."""

import re

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Split text into sentences, returning ``(start, end, sentence)`` character offsets."""
    sentences = []
    pos = 0
    for piece in _SENT_BOUNDARY.split(text):
        if not piece:
            continue
        start = text.find(piece, pos)
        end = start + len(piece)
        pos = end
        stripped = piece.strip()
        if stripped:
            offset = piece.find(stripped)
            sentences.append((start + offset, start + offset + len(stripped), stripped))
    return sentences


def merge_spans(spans: list[dict]) -> list[dict]:
    """Merge overlapping spans that share a category, keeping the highest score."""
    merged: list[dict] = []
    for span in sorted(spans, key=lambda s: (s["category"], s["start"], -s["end"])):
        last = merged[-1] if merged else None
        if last and last["category"] == span["category"] and span["start"] <= last["end"]:
            last["end"] = max(last["end"], span["end"])
            last["score"] = max(last["score"], span["score"])
        else:
            merged.append(dict(span))
    return sorted(merged, key=lambda s: s["start"])
