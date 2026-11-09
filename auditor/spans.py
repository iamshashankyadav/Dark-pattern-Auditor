"""Inline span markup used for annotating training/test data.

Annotated text marks a manipulative phrase as ``[[TAG|phrase]]``, e.g.::

    "Hurry, [[URGENCY|only 3 left in stock]]! Free returns."

``parse_markup`` turns that into clean text plus character-offset spans.
"""

import re

_MARKUP = re.compile(r"\[\[(\w+)\|(.+?)\]\]")


def parse_markup(marked: str) -> dict:
    text_parts, spans = [], []
    cursor, out_len = 0, 0
    for m in _MARKUP.finditer(marked):
        before = marked[cursor : m.start()]
        text_parts.append(before)
        out_len += len(before)
        tag, phrase = m.group(1), m.group(2)
        spans.append({"start": out_len, "end": out_len + len(phrase), "label": tag})
        text_parts.append(phrase)
        out_len += len(phrase)
        cursor = m.end()
    text_parts.append(marked[cursor:])
    return {"text": "".join(text_parts), "spans": spans}


def char_spans_to_token_labels(offsets: list[tuple[int, int]], spans: list[dict]) -> list[str]:
    """Assign BIO labels to tokens given their ``(start, end)`` character offsets.

    Special tokens (offset ``(0, 0)``) get the label ``"IGNORE"`` so callers can mask them.
    """
    labels = []
    for start, end in offsets:
        if start == end:
            labels.append("IGNORE")
            continue
        label = "O"
        for span in spans:
            if start >= span["start"] and end <= span["end"]:
                prefix = "B" if start == span["start"] else "I"
                label = f"{prefix}-{span['label']}"
                break
            if start < span["end"] and end > span["start"]:  # partial overlap
                label = f"I-{span['label']}" if start > span["start"] else f"B-{span['label']}"
                break
        labels.append(label)
    # A span whose first token does not start exactly at the span start still needs a B-.
    for i, lab in enumerate(labels):
        if lab.startswith("I-"):
            prev = labels[i - 1] if i else "O"
            if prev[2:] != lab[2:]:
                labels[i] = "B-" + lab[2:]
    return labels
