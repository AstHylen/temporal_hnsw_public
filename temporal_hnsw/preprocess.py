"""Deterministic Wikipedia wikitext parsing and sentence-aware chunking."""

from __future__ import annotations

import hashlib
import html
import re
import time
from pathlib import Path

import mwparserfromhell

from .common import DATA, LOGS, append_csv, log_event, read_csv, read_json, stable_json


_WS = re.compile(r"\s+")
_SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(])")


def normalize_spaces(text: str) -> str:
    return _WS.sub(" ", text).strip()


EXTRACTOR_VERSION = "bounded-cleanup-v2"
SELECTION_EXTRACTOR_VERSION = "original-strip-code-v1"


def clean_wikitext_selection_v1(wikitext: str) -> str:
    # This intentionally removes templates and most markup. Infobox-only facts
    # are therefore not guaranteed to be represented and must not be annotated.
    code = mwparserfromhell.parse(wikitext)
    for tag in list(code.filter_tags(recursive=True)):
        if str(tag.tag).lower() in {"ref", "gallery", "math"}:
            try:
                code.remove(tag)
            except ValueError:
                pass
    text = code.strip_code(normalize=True, collapse=True)
    text = html.unescape(text)
    # Remove lightweight residual citation artifacts and normalize whitespace.
    text = re.sub(r"\[\s*(?:citation needed|edit)\s*\]", "", text, flags=re.I)
    return normalize_spaces(text)


def clean_wikitext(raw: str) -> str:
    """Approved bounded cleanup; preserve unparsed text rather than deleting facts."""
    code = mwparserfromhell.parse(raw)
    def transform(code):
        for node in list(code.nodes):
            for child in node.__children__():
                transform(child)
        for link in list(code.filter_wikilinks(recursive=False)):
            title = str(link.title).strip()
            # Leading-colon links are ordinary article links, not category assignments.
            namespace = title.split(':', 1)[0].lower()
            replacement = None
            if namespace == 'category':
                replacement = ''
            elif namespace in ('file', 'image'):
                # Split only top-level pipes: nested caption links/templates stay intact.
                parts = []
                current = ''
                for node in (link.text.nodes if link.text is not None else []):
                    if isinstance(node, mwparserfromhell.nodes.Text):
                        pieces = str(node).split('|')
                        current += pieces[0]
                        for piece in pieces[1:]:
                            parts.append(current); current = piece
                    else:
                        current += str(node)
                parts.append(current)
                captions = [p.strip() for p in parts if p.strip() and not re.fullmatch(
                    r'(?:thumb(?:nail)?|frame|frameless|border|left|right|center|none|baseline|sub|super|top|text-top|middle|bottom|text-bottom|(?:\d+)?x?\d+px|upright(?:\s*=.*)?|(?:alt|link|lang|page|class|start|end|thumbtime)\s*=.*)',p.strip(),re.I)]
                # MediaWiki uses the last caption-like option; do not index filename/options.
                replacement = (' ' + captions[-1] + ' ') if captions else ' '
            if replacement is not None:
                try: code.replace(link, replacement, recursive=False)
                except ValueError: pass  # Parent removal already removed this nested node.
        # Preserve recognized table content and separate cells/rows; do not guess headers.
        for tag in reversed(list(code.filter_tags(recursive=False))):
            name = str(tag.tag).strip().lower()
            if name in {'td','th','tr','caption','table'}:
                replacement = ' ' + str(tag.contents) + (' ; ' if name in {'td','th'} else ' . ')
                try: code.replace(tag, replacement, recursive=False)
                except ValueError: pass
    transform(code)
    return clean_wikitext_selection_v1(str(code))


def chunk_text(text: str, target_words: int = 100, min_words: int = 8) -> list[str]:
    """Join sentences until a roughly target-sized chunk; split very long sentences.

    This parser is intentionally simple, fixed, and documented. It is not a
    linguistically complete sentence segmenter for every Wikipedia language.
    """
    if not text.strip():
        return []
    sentences = _SENTENCES.split(normalize_spaces(text))
    chunks = []
    pending = []

    def flush() -> None:
        nonlocal pending
        if pending:
            chunks.append(" ".join(pending))
            pending = []

    for sentence in sentences:
        words = sentence.split()
        if len(words) > target_words:
            flush()
            for start in range(0, len(words), target_words):
                part = words[start:start + target_words]
                if part:
                    chunks.append(" ".join(part))
            continue
        if pending and len(" ".join(pending).split()) + len(words) > target_words:
            flush()
        pending.append(sentence)
    flush()
    # Avoid losing short, factually useful tails; merge a tiny final tail where possible.
    if len(chunks) >= 2 and len(chunks[-1].split()) < min_words:
        tail = chunks.pop()
        chunks[-1] += " " + tail
    return chunks


def process(config: dict) -> None:
    if (DATA / "annotations" / "ground_truth.lock.json").exists():
        raise RuntimeError("Ground truth is frozen; refusing to overwrite evidence inputs.")
    selected = DATA / "selected_articles.csv"
    if not selected.exists():
        raise FileNotFoundError("First run sample and fetch.")
    article_ids = [int(row["page_id"]) for row in read_csv(selected)]
    counts = []
    for snapshot in config["snapshots"]:
        label = snapshot["label"]
        start = time.perf_counter()
        path = DATA / "processed" / label / "chunks.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        num_chunks = 0
        with path.open("w", encoding="utf-8") as out:
            for page_id in article_ids:
                raw = read_json(DATA / "raw" / label / f"{page_id}.json")
                text = clean_wikitext(raw["wikitext"])
                parts = chunk_text(text, int(config["preprocessing"]["target_words_per_chunk"]),
                                   int(config["preprocessing"]["min_chunk_words"]))
                for j, part in enumerate(parts):
                    row = {
                        "vector_id": num_chunks, "page_id": page_id, "title": raw["title"],
                        "snapshot": label, "revision_id": raw["revision_id"], "chunk_ordinal": j,
                        "text": part,
                    }
                    out.write(stable_json(row) + "\n")
                    num_chunks += 1
        elapsed = time.perf_counter() - start
        counts.append({"snapshot": label, "num_articles": len(article_ids), "num_chunks": num_chunks,
                       "preprocessing_seconds": round(elapsed, 6)})
        log_event("preprocessing_complete", **counts[-1])
    from .common import write_csv, write_json, file_sha256
    write_csv(LOGS / "preprocessing.csv", counts, list(counts[0]))
    provenance = {"extractor_version": EXTRACTOR_VERSION, "extractor_source_sha256": file_sha256(Path(__file__)),
                  "selection_extractor_version": SELECTION_EXTRACTOR_VERSION, "config": config,
                  "selection_sha256": file_sha256(selected),
                  "revision_manifest_sha256": file_sha256(DATA / "revision_manifest.csv"),
                  "processed_sha256": {s["label"]: file_sha256(DATA / "processed" / s["label"] / "chunks.jsonl") for s in config["snapshots"]}}
    write_json(LOGS / "preprocessing_provenance.json", provenance)
    log_event("preprocessing_provenance_recorded", **provenance)


def iter_chunks(snapshot: str):
    path = DATA / "processed" / snapshot / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run prepare.")
    import json
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def show_article(snapshot: str, page_id: int, limit: int = 12) -> str:
    entries = [row for row in iter_chunks(snapshot) if row["page_id"] == page_id]
    if not entries:
        raise ValueError(f"No chunks for page_id={page_id} in {snapshot}.")
    lines = []
    for row in entries[:limit]:
        lines.append(f"[{snapshot} | {row['title']} | rev={row['revision_id']} | chunk={row['chunk_ordinal']}]\n{row['text']}\n")
    return "\n".join(lines)


def diff_article(first: str, second: str, page_id: int, max_lines: int = 120) -> str:
    """Show a sentence-level preview of one page's historical text edits.

    This is for manual annotation support; a textual diff is not a semantic fact label.
    """
    import difflib
    old = [c["text"] for c in iter_chunks(first) if c["page_id"] == page_id]
    new = [c["text"] for c in iter_chunks(second) if c["page_id"] == page_id]
    if not old or not new:
        raise ValueError(f"Page ID {page_id} is not present in both snapshots.")
    old_sentences = _SENTENCES.split(" ".join(old))
    new_sentences = _SENTENCES.split(" ".join(new))
    diff_lines = list(difflib.unified_diff(old_sentences, new_sentences,
                                           fromfile=first, tofile=second, lineterm=""))
    return "\n".join(diff_lines[:max_lines]) if diff_lines else "No textual differences detected."


