"""Tokenizer and wikitext extraction tests (require optional parser installation)."""

import pytest

pytest.importorskip("mwparserfromhell")
from temporal_hnsw.preprocess import chunk_text, clean_wikitext


def test_wikitext_removes_template_and_keeps_prose():
    clean = clean_wikitext("''Bold'' [[Trondheim]] is in [[Norway|Norway]]. {{Infobox|x=private}} <ref>source</ref>")
    assert "Bold" in clean
    assert "Trondheim" in clean
    assert "Norway" in clean
    assert "private" not in clean
    assert "source" not in clean


def test_sentence_chunking_keeps_sentence_intact():
    text = "Trondheim has several districts. Oslo is the capital of Norway. Norway is in northern Europe."
    chunks = chunk_text(text, target_words=10, min_words=2)
    assert len(chunks) >= 2
    assert chunks[0].endswith("Norway.")
    assert " ".join(chunks) == text


def test_long_sentence_splits_to_target_size():
    chunks = chunk_text(" ".join(["word"] * 251), target_words=100, min_words=1)
    assert [len(x.split()) for x in chunks] == [100, 100, 51]


# Run: python -m pytest -q tests/test_preprocess.py


@pytest.mark.parametrize('length', [101, 107, 201, 207, 301])
def test_short_final_tail_is_merged_without_loss_or_duplication(length):
    text = ' '.join(str(i) for i in range(length))
    chunks = chunk_text(text, target_words=100, min_words=8)
    assert ' '.join(chunks) == text
    assert len(chunks[-1].split()) >= 8
