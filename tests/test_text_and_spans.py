from auditor.metrics import bio_to_entities, entity_prf
from auditor.spans import char_spans_to_token_labels, parse_markup
from auditor.text_utils import merge_spans, split_sentences


def test_parse_markup_offsets():
    doc = parse_markup("Hurry, [[URGENCY|only 3 left]]! [[CLAIM|Cures acne]].")
    assert doc["text"] == "Hurry, only 3 left! Cures acne."
    got = [(doc["text"][s["start"]:s["end"]], s["label"]) for s in doc["spans"]]
    assert got == [("only 3 left", "URGENCY"), ("Cures acne", "CLAIM")]


def test_token_labels_bio():
    # offsets for tokens: [CLS] hurry , only 3 left ! [SEP]
    offsets = [(0, 0), (0, 5), (5, 6), (7, 11), (12, 13), (14, 18), (18, 19), (0, 0)]
    spans = [{"start": 7, "end": 18, "label": "URGENCY"}]
    assert char_spans_to_token_labels(offsets, spans) == [
        "IGNORE", "O", "O", "B-URGENCY", "I-URGENCY", "I-URGENCY", "O", "IGNORE",
    ]


def test_split_sentences_offsets():
    text = "First one. Second one!  Third?"
    sents = split_sentences(text)
    assert [s[2] for s in sents] == ["First one.", "Second one!", "Third?"]
    assert all(text[s:e] == t for s, e, t in sents)


def test_merge_spans():
    spans = [
        {"start": 0, "end": 5, "category": "a", "score": 0.5},
        {"start": 3, "end": 9, "category": "a", "score": 0.9},
        {"start": 4, "end": 6, "category": "b", "score": 0.1},
    ]
    merged = merge_spans(spans)
    assert {(m["start"], m["end"], m["category"], m["score"]) for m in merged} == {
        (0, 9, "a", 0.9), (4, 6, "b", 0.1)
    }


def test_bio_entities_and_prf():
    gold = [["B-X", "I-X", "O", "B-Y"]]
    pred = [["B-X", "I-X", "O", "O"]]
    assert bio_to_entities(gold[0]) == {("X", 0, 2), ("Y", 3, 4)}
    report = entity_prf(gold, pred)
    assert report["micro"]["precision"] == 1.0
    assert report["micro"]["recall"] == 0.5
    assert report["Y"]["f1"] == 0.0


def test_lenient_matching_counts_overlap():
    gold = [["B-X", "I-X", "I-X", "O"]]
    pred = [["O", "B-X", "I-X", "O"]]  # shorter span, same type
    assert entity_prf(gold, pred, "strict")["micro"]["f1"] == 0.0
    assert entity_prf(gold, pred, "lenient")["micro"]["f1"] == 1.0


def test_join_fragments():
    from auditor.extractor import _join_fragments

    text = "It balances hormones and prevents illness."
    spans = [
        {"start": 3, "end": 20, "category": "c", "score": 0.9},
        {"start": 25, "end": 41, "category": "c", "score": 0.7},
    ]
    [joined] = _join_fragments(text, spans)
    assert text[joined["start"]:joined["end"]] == "balances hormones and prevents illness"
