"""Create and validate a fixed question set and snapshot-specific ground truth.

Manual annotations record the reviewed scientific evidence.
"""

from __future__ import annotations

import difflib
import hashlib
from collections import Counter

from .common import ANNOTATIONS, DATA, LOGS, file_sha256, log_event, read_csv, read_json, write_csv, write_json
from .preprocess import iter_chunks, normalize_spaces


QUESTION_COLUMNS = ["query_id", "group", "source_page_id", "question"]
GT_COLUMNS = ["query_id", "snapshot", "expected_answer", "supporting_excerpt", "source_revision_id"]


def templates(config: dict) -> None:
    qp = ANNOTATIONS / "questions.csv"
    gp = ANNOTATIONS / "ground_truth.csv"
    if qp.exists() or gp.exists():
        raise FileExistsError("Annotation CSVs already exist; refusing to overwrite researcher edits.")
    questions = []
    groups = (("changed", config["annotations"]["changed_queries"]),
              ("stable", config["annotations"]["stable_queries"]))
    for group, count in groups:
        for _ in range(int(count)):
            questions.append({"query_id": f"q{len(questions) + 1:03d}", "group": group,
                              "source_page_id": "", "question": ""})
    truth = [{"query_id": q["query_id"], "snapshot": snap["label"],
              "expected_answer": "", "supporting_excerpt": "", "source_revision_id": ""}
             for q in questions for snap in config["snapshots"]]
    write_csv(qp, questions, QUESTION_COLUMNS)
    write_csv(gp, truth, GT_COLUMNS)
    log_event("annotation_templates_created", questions=len(questions), ground_truth_rows=len(truth))


def validate(config: dict, config_path, freeze: bool = True) -> dict:
    selected = read_csv(DATA / "selected_articles.csv")
    allowed_pages = {int(p["page_id"]) for p in selected}
    questions = read_csv(ANNOTATIONS / "questions.csv")
    gt = read_csv(ANNOTATIONS / "ground_truth.csv")
    expected_count = int(config["annotations"]["changed_queries"]) + int(config["annotations"]["stable_queries"])
    if len(questions) != expected_count:
        raise ValueError(f"Expected {expected_count} questions, received {len(questions)}.")
    if len({q["query_id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate query_id.")
    group_counts = Counter(q["group"] for q in questions)
    if group_counts != {"changed": int(config["annotations"]["changed_queries"]),
                        "stable": int(config["annotations"]["stable_queries"])}:
        raise ValueError(f"Wrong changed/stable group sizes: {group_counts}")
    for q in questions:
        if not q["question"].strip():
            raise ValueError(f"Empty question: {q['query_id']}")
        if not q["source_page_id"].isdigit() or int(q["source_page_id"]) not in allowed_pages:
            raise ValueError(f"Invalid anchor page ID for {q['query_id']}")
    expected_keys = {(q["query_id"], s["label"]) for q in questions for s in config["snapshots"]}
    actual_keys = {(row["query_id"], row["snapshot"]) for row in gt}
    if len(gt) != len(expected_keys) or actual_keys != expected_keys:
        raise ValueError("Exactly one ground-truth row per query per snapshot is required.")
    qmap = {q["query_id"]: q for q in questions}
    row_map = {(r["query_id"], r["snapshot"]): r for r in gt}
    for snap in config["snapshots"]:
        label = snap["label"]
        by_page = {}
        for chunk in iter_chunks(label):
            by_page.setdefault(chunk["page_id"], []).append(chunk)
        for q in questions:
            row = row_map[(q["query_id"], label)]
            ans = normalize_spaces(row["expected_answer"])
            excerpt = normalize_spaces(row["supporting_excerpt"])
            if not ans or not excerpt or not row["source_revision_id"].isdigit():
                raise ValueError(f"Missing manual ground truth for {q['query_id']} at {label}.")
            chunks = by_page.get(int(q["source_page_id"]), [])
            if not chunks:
                raise ValueError(f"Source article has no chunks: {q['query_id']} at {label}")
            if int(row["source_revision_id"]) != chunks[0]["revision_id"]:
                raise ValueError(f"Wrong source_revision_id for {q['query_id']} at {label}")
            # Literal excerpt checks provenance, not its factual interpretation.
            if not any(excerpt in normalize_spaces(c["text"]) for c in chunks):
                raise ValueError(
                    f"Supporting excerpt for {q['query_id']} at {label} must appear in one "
                    "processed chunk of the annotated source article. Use 'show-page' to inspect chunks."
                )
    for q in questions:
        answers = {normalize_spaces(row_map[(q["query_id"], s["label"])]["expected_answer"]).casefold()
                   for s in config["snapshots"]}
        if q["group"] == "changed" and len(answers) < 2:
            raise ValueError(f"Changed query never changes answer: {q['query_id']}")
        if q["group"] == "stable" and len(answers) != 1:
            raise ValueError(f"Stable query changes answer: {q['query_id']}")
    sources = {
        "config_sha256": config_path,
        "questions_sha256": ANNOTATIONS / "questions.csv",
        "ground_truth_sha256": ANNOTATIONS / "ground_truth.csv",
        "selection_sha256": DATA / "selected_articles.csv",
        "revision_manifest_sha256": DATA / "revision_manifest.csv",
    }
    lock = {name: file_sha256(path) for name, path in sources.items()}
    lock["processed_chunks_sha256"] = {
        s["label"]: file_sha256(DATA / "processed" / s["label"] / "chunks.jsonl")
        for s in config["snapshots"]
    }
    lock["num_questions"] = len(questions)
    lock["num_ground_truth_rows"] = len(gt)
    lock["snapshot_labels"] = [s["label"] for s in config["snapshots"]]
    if freeze:
        write_json(ANNOTATIONS / "ground_truth.lock.json", lock)
    log_event("ground_truth_validated", questions=len(questions), rows=len(gt), frozen=freeze)
    return lock


def change_candidates(config: dict) -> None:
    """Article-level lexical change summaries to guide manual curation.

    Counter token differences are a coarse screening heuristic, NOT semantic edits.
    """
    snapshots = [s["label"] for s in config["snapshots"]]
    article_rows = read_csv(DATA / "selected_articles.csv")
    texts_by_snap = {}
    for label in snapshots:
        parts = {}
        for c in iter_chunks(label):
            parts.setdefault(c["page_id"], []).append(c["text"])
        texts_by_snap[label] = {k: " ".join(v) for k, v in parts.items()}
    results = []
    for entry in article_rows:
        pid = int(entry["page_id"])
        hashes = [hashlib.sha256(texts_by_snap[s][pid].encode()).hexdigest() for s in snapshots]
        total_changes = 0
        changes_by_transition = []
        for prev, current in zip(snapshots, snapshots[1:]):
            a = Counter(texts_by_snap[prev][pid].lower().split())
            b = Counter(texts_by_snap[current][pid].lower().split())
            distance = sum((a - b).values()) + sum((b - a).values())
            total_changes += distance
            changes_by_transition.append(str(distance))
        results.append({"page_id": pid, "title": entry["title"],
                        "distinct_snapshots_by_cleaned_text": len(set(hashes)),
                        "sum_word_multiset_difference": total_changes,
                        "differences_per_transition": "|".join(changes_by_transition)})
    results.sort(key=lambda r: (-r["sum_word_multiset_difference"], r["page_id"]))
    write_csv(ANNOTATIONS / "change_candidates.csv", results, list(results[0]))
    log_event("change_candidates_created", articles=len(results),
              note="Lexical change is a screening heuristic; verify factual changes manually.")


