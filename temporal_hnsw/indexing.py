"""Build independent HNSW indexes at each historical snapshot and log full costs.

The D0 index is loaded unmodified for all future stale-index searches.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path

import hnswlib
import numpy as np

from .common import DATA, LOGS, file_sha256, log_event, read_csv, stable_json, write_csv, write_json
from .preprocess import iter_chunks


class Encoder:
    def __init__(self, config: dict):
        """Load the frozen sentence embedding model once per invocation."""
        from sentence_transformers import SentenceTransformer
        options = config["embeddings"]
        kwargs = {}
        if options.get("model_revision"):
            kwargs["revision"] = options["model_revision"]
        self.model = SentenceTransformer(options["model_name"], **kwargs)
        self.batch_size = int(options["batch_size"])
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.name = options["model_name"]

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                                    normalize_embeddings=True, convert_to_numpy=True)
        return np.ascontiguousarray(vectors, dtype=np.float32)


def create_index(embeddings: np.ndarray, config: dict) -> hnswlib.Index:
    """One deterministic HNSW construction; ANN still uses an approximate search."""
    n, dim = embeddings.shape
    if n < config["experiment"]["top_k"]:
        raise ValueError(f"Only {n} chunks; at least top_k required.")
    opts = config["hnsw"]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, M=int(opts["M"]), ef_construction=int(opts["ef_construction"]),
                     random_seed=int(opts["random_seed"]))
    ids = np.arange(n, dtype=np.int64)
    index.add_items(embeddings, ids, num_threads=int(opts["num_threads"]))
    index.set_ef(int(opts["ef_search"]))
    index.set_num_threads(int(opts["num_threads"]))
    return index


def build(config: dict, config_path: Path) -> None:
    from .common import check_frozen_inputs
    check_frozen_inputs(config_path)
    from .common import read_csv
    encoder = Encoder(config)
    logs = []
    preprocessing = {x["snapshot"]: x for x in read_csv(LOGS / "preprocessing.csv")}
    for snap in config["snapshots"]:
        label = snap["label"]
        output = DATA / "indexes" / label
        output.mkdir(parents=True, exist_ok=True)
        chunks = list(iter_chunks(label))
        if [r["vector_id"] for r in chunks] != list(range(len(chunks))):
            raise ValueError(f"Nonsequential vector IDs in {label}")
        texts = [r["text"] for r in chunks]
        start = time.perf_counter()
        vectors = encoder.encode(texts)
        embed_seconds = time.perf_counter() - start
        if vectors.shape != (len(chunks), encoder.dimension):
            raise ValueError("Embedding shape mismatch")
        if not np.isfinite(vectors).all():
            raise ValueError("Non-finite embedding values detected")
        np.save(output / "vectors.npy", vectors, allow_pickle=False)
        chunk_meta = [{key: r[key] for key in ("vector_id", "page_id", "title", "revision_id", "chunk_ordinal", "text")}
                      for r in chunks]
        with (output / "chunks.jsonl").open("w", encoding="utf-8") as out:
            for row in chunk_meta:
                out.write(stable_json(row) + "\n")
        timings = []
        for repeat in range(int(config["hnsw"]["repetitions"])):
            t0 = time.perf_counter()
            idx = create_index(vectors, config)
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)
            if repeat == 0:
                idx.save_index(str(output / "index.bin"))
            del idx
        rec = {
            "snapshot": label, "num_chunks": len(chunks),
            "config_sha256": file_sha256(config_path),
            "processed_chunks_sha256": file_sha256(DATA / "processed" / label / "chunks.jsonl"), "embedding_dimensions": encoder.dimension,
            "preprocess_seconds": float(preprocessing[label]["preprocessing_seconds"]),
            "embedding_seconds": round(embed_seconds, 6),
            "hnsw_build_seconds_median": round(float(np.median(timings)), 6),
            "hnsw_build_seconds_trials": "|".join(f"{s:.6f}" for s in timings),
            "sum_stage_seconds_approx": round(float(preprocessing[label]["preprocessing_seconds"]) +
                                              embed_seconds + float(np.median(timings)), 6),
            "index_disk_bytes": (output / "index.bin").stat().st_size,
            "vector_file_sha256": file_sha256(output / "vectors.npy"),
            "chunks_file_sha256": file_sha256(output / "chunks.jsonl"),
            "index_file_sha256": file_sha256(output / "index.bin"),
            "model": encoder.name,
            "model_revision_configured": config["embeddings"].get("model_revision") or "UNPINNED",
        }
        logs.append(rec)
        log_event("index_built", **rec)
    write_csv(LOGS / "build_timings.csv", logs, list(logs[0]))
    if all(row["model_revision_configured"] == "UNPINNED" for row in logs):
        log_event("model_revision_warning", note="Pin model_revision to an immutable Hugging Face commit SHA for a published run.")


def load_index(snapshot: str, config: dict):
    folder = DATA / "indexes" / snapshot
    vectors = np.load(folder / "vectors.npy", allow_pickle=False, mmap_mode="r")
    idx = hnswlib.Index(space="cosine", dim=int(vectors.shape[1]))
    idx.load_index(str(folder / "index.bin"))
    idx.set_ef(int(config["hnsw"]["ef_search"]))
    idx.set_num_threads(int(config["hnsw"]["num_threads"]))
    import json
    with (folder / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    if len(chunks) != len(vectors) or idx.get_current_count() != len(vectors):
        raise RuntimeError("Mismatch between index, vectors and chunk metadata.")
    return idx, vectors, chunks


