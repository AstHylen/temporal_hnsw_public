"""Run matched stale/rebuilt retrieval and prepare a blinded manual review sheet.

No generated language-model answers are evaluated. Manual judgments are required
for the paper's primary Temporal Evidence Hit@5 metric.
"""

from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path

import numpy as np

from .annotations import GT_COLUMNS
from .common import ANNOTATIONS, DATA, LOGS, check_frozen_inputs, file_sha256, log_event, read_csv, write_csv, write_json
from .indexing import Encoder, load_index

HIT_COLUMNS = ["snapshot", "query_id", "group", "arm", "rank", "vector_id", "distance", "page_id",
               "title", "chunk_ordinal", "text", "review_id", "source_article_hit"]
REVIEW_COLUMNS = ["review_id", "snapshot", "query_id", "question", "expected_answer", "passage_title",
                  "passage_text", "auto_suggestion_only", "decision", "notes"]
ANN_COLUMNS = ["snapshot", "arm", "ann_recall_at_5", "num_questions", "num_index_vectors"]
ANN_PER_QUERY_COLUMNS = ["snapshot", "arm", "query_id", "group", "ann_recall_at_5"]


def exact_topk(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Compute exact cosine top-k within the SAME active vector collection.

    Vectors are L2-normalized by the sentence encoder; exact cosine similarity
    is therefore the dot product. Tiny floating-point ties use vector ID order.
    """
    # Use a vector-by-vector scan so we never rely on ANNs for ground truth.
    scores = np.asarray(vectors @ query, dtype=np.float64)
    if not 1 <= k <= len(scores):
        raise ValueError("k must be between 1 and the vector count")
    # Sort all IDs so ties crossing the kth boundary are deterministic too.
    return np.lexsort((np.arange(len(scores)), -scores))[:k]


def suggested_evidence(text: str, expected_answer: str) -> str:
    """Only a manual-review hint, never a score or an automated judgment."""
    from .preprocess import normalize_spaces
    needle = normalize_spaces(expected_answer).casefold()
    body = normalize_spaces(text).casefold()
    return "possible" if needle and needle in body else "unknown"


def retrieve(config: dict, config_path: Path) -> None:
    check_frozen_inputs(config_path)
    queries = read_csv(ANNOTATIONS / "questions.csv")
    ground_truth = {(r["query_id"], r["snapshot"]): r for r in read_csv(ANNOTATIONS / "ground_truth.csv")}
    # Ensure indexes correspond to the files used for the frozen experiment.
    timing_rows = read_csv(LOGS / "build_timings.csv")
    timing_by_snapshot = {r["snapshot"]: r for r in timing_rows}
    for s in config["snapshots"]:
        name = s["label"]
        provenance = timing_by_snapshot[name]
        if provenance.get("config_sha256") != file_sha256(config_path) or provenance.get("processed_chunks_sha256") != file_sha256(DATA / "processed" / name / "chunks.jsonl"):
            raise RuntimeError(f"Index {name} was not built from the frozen configuration and chunks.")
        for filename, hash_key in (("vectors.npy", "vector_file_sha256"),
                                   ("chunks.jsonl", "chunks_file_sha256"),
                                   ("index.bin", "index_file_sha256")):
            path = DATA / "indexes" / name / filename
            if not path.exists() or file_sha256(path) != timing_by_snapshot[name][hash_key]:
                raise RuntimeError(f"Index artifact changed since construction: {path}")
    encoder = Encoder(config)
    query_vectors = encoder.encode([q["question"] for q in queries])
    np.save(LOGS / "query_vectors.npy", query_vectors, allow_pickle=False)
    write_json(LOGS / "query_embedding_provenance.json", {
        "query_ids_in_order": [q["query_id"] for q in queries],
        "questions_sha256": file_sha256(ANNOTATIONS / "questions.csv"),
        "vectors_sha256": file_sha256(LOGS / "query_vectors.npy"),
        "config_sha256": file_sha256(config_path),
        "device": str(encoder.model.device),
        "model_revision": config["embeddings"]["model_revision"],
        "shared_across_all_arms_and_snapshots": True,
    })
    top_k = int(config["experiment"]["top_k"])
    first = config["snapshots"][0]["label"]
    all_hits = []
    ann_rows = []
    ann_per_query = []
    reviews = {}
    for snap in config["snapshots"]:
        label = snap["label"]
        for arm, dataset in (("stale", first), ("rebuilt", label)):
            idx, vectors, chunks = load_index(dataset, config)
            if len(vectors) < top_k:
                raise RuntimeError(f"Insufficient chunks in index {dataset}")
            labels, distances = idx.knn_query(query_vectors, k=top_k, num_threads=1)
            recall_by_query = []
            for qidx, q in enumerate(queries):
                exact = exact_topk(vectors, query_vectors[qidx], top_k)
                recall = len(set(int(x) for x in labels[qidx]) & set(map(int, exact))) / top_k
                recall_by_query.append(recall)
                ann_per_query.append({"snapshot": label, "arm": arm, "query_id": q["query_id"],
                                      "group": q["group"], "ann_recall_at_5": recall})
                gt = ground_truth[(q["query_id"], label)]
                for rank, (vid, distance) in enumerate(zip(labels[qidx], distances[qidx]), start=1):
                    chunk = chunks[int(vid)]
                    # Deduplicate the same displayed evidence from both arms, without exposing arm or rank.
                    signature = f"{label}\0{q['query_id']}\0{chunk['page_id']}\0{chunk['text']}"
                    review_id = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:24]
                    if review_id not in reviews:
                        reviews[review_id] = {
                            "review_id": review_id, "snapshot": label, "query_id": q["query_id"],
                            "question": q["question"], "expected_answer": gt["expected_answer"],
                            "passage_title": chunk["title"], "passage_text": chunk["text"],
                            "auto_suggestion_only": suggested_evidence(chunk["text"], gt["expected_answer"]),
                            "decision": "", "notes": "",
                        }
                    all_hits.append({
                        "snapshot": label, "query_id": q["query_id"], "group": q["group"],
                        "arm": arm, "rank": rank, "vector_id": int(vid),
                        "distance": round(float(distance), 8), "page_id": chunk["page_id"],
                        "title": chunk["title"], "chunk_ordinal": chunk["chunk_ordinal"],
                        "text": chunk["text"], "review_id": review_id,
                        "source_article_hit": int(int(q["source_page_id"]) == int(chunk["page_id"])),
                    })
            row = {"snapshot": label, "arm": arm, "ann_recall_at_5": float(np.mean(recall_by_query)),
                   "num_questions": len(queries), "num_index_vectors": len(vectors)}
            ann_rows.append(row)
            log_event("retrieval_arm_complete", **row)
    sheet = LOGS / "manual_review.csv"
    if sheet.exists():
        old = {x["review_id"]: x for x in read_csv(sheet)}
        old_ids, new_ids = set(old), set(reviews)
        if old_ids != new_ids:
            raise RuntimeError(
                "A review sheet exists but retrieved passages changed. Move it to a backup first, "
                "or restore the pinned inputs. Refusing to overwrite human judgments."
            )
        for key, row in reviews.items():
            if any(old[key][field] != row[field] for field in ("question", "expected_answer", "passage_title", "passage_text")):
                raise RuntimeError("Review context changed; archive the existing judgments before rerunning.")
            row["decision"] = old[key]["decision"]
            row["notes"] = old[key]["notes"]
    write_csv(LOGS / "retrieval_hits.csv", all_hits, HIT_COLUMNS)
    write_csv(LOGS / "ann_recall.csv", ann_rows, ANN_COLUMNS)
    write_csv(LOGS / "ann_recall_by_query.csv", ann_per_query, ANN_PER_QUERY_COLUMNS)
    shuffled = list(reviews.values())
    random.Random(int(config["experiment"]["seed"])).shuffle(shuffled)
    write_csv(sheet, shuffled, REVIEW_COLUMNS)
    log_event("retrieval_complete", total_hits=len(all_hits), unique_review_rows=len(shuffled),
              review_sheet=str(sheet), input_sha256=file_sha256(config_path))


