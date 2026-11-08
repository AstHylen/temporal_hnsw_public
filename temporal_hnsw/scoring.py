"""Score manual evidence judgments and compute paired query-level uncertainty.

Primary metrics require resolved top-k outcomes, not labels for every passage.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import ANNOTATIONS, LOGS, check_frozen_inputs, file_sha256, log_event, read_csv, write_csv, write_json

METRIC_COLUMNS = ["snapshot", "group", "arm", "num_queries", "temporal_evidence_hit_at_5",
                  "source_article_hit_at_5", "ann_recall_at_5"]
DIFF_COLUMNS = ["snapshot", "group", "num_paired_queries", "rebuilt_minus_stale_hit_at_5",
                "bootstrap_ci_95_low", "bootstrap_ci_95_high"]


def paired_ci(stale: np.ndarray, rebuilt: np.ndarray, *, seed: int = 42, n_boot: int = 2000) -> tuple[float, float]:
    """Percentile CI using the QUERY as the resampling unit (paired by query ID)."""
    if stale.shape != rebuilt.shape or stale.size == 0:
        raise ValueError("Paired samples must have equal nonzero length.")
    differences = rebuilt.astype(float) - stale.astype(float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(differences), size=(n_boot, len(differences)))
    values = differences[indices].mean(axis=1)
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def hit_outcome(retrieved, review_map, top_k=5):
    """Return 1, 0, or None; blank labels never imply non-relevance."""
    if len(retrieved) != top_k or sorted(int(r["rank"]) for r in retrieved) != list(range(1, top_k + 1)):
        raise ValueError("Wrong result count/ranks for hit outcome")
    labels = [review_map[r["review_id"]]["decision"].strip() for r in retrieved]
    if any(label not in {"", "0", "1"} for label in labels):
        raise ValueError("Invalid manual review decision; expected blank, 0 or 1")
    if "1" in labels:
        return 1
    return 0 if all(label == "0" for label in labels) else None


def next_review(retrieved, review_map, top_k=5):
    """Only the earliest unresolved rank of an unknown outcome is needed now."""
    if hit_outcome(retrieved, review_map, top_k) is not None:
        return None
    return next(r["review_id"] for r in sorted(retrieved, key=lambda r: int(r["rank"]))
                if not review_map[r["review_id"]]["decision"].strip())


def score(config: dict, config_path: Path) -> None:
    lock = check_frozen_inputs(config_path)
    hits = read_csv(LOGS / "retrieval_hits.csv")
    reviews = read_csv(LOGS / "manual_review.csv")
    review_map = {r["review_id"]: r for r in reviews}
    if len(review_map) != len(reviews):
        raise ValueError("Duplicate review_id in manual_review.csv.")
    used_ids = {x["review_id"] for x in hits}
    if used_ids != set(review_map):
        raise ValueError("Review sheet does not match retrieval results.")
    questions = read_csv(ANNOTATIONS / "questions.csv")
    qmap = {r["query_id"]: r for r in questions}
    outcomes = defaultdict(list)
    for row in hits:
        outcomes[(row["snapshot"], row["group"], row["arm"], row["query_id"])].append(row)
    explicit = {}
    outcome_path = LOGS / "outcome_review.csv"
    if outcome_path.exists():
        from .outcome_review import validate_outcomes
        gt = {(r["query_id"], r["snapshot"]): r for r in read_csv(ANNOTATIONS / "ground_truth.csv")}
        contexts = {(snap, arm, qid): (qmap[qid]["question"], gt[qid, snap]["expected_answer"], hs)
                    for (snap, group, arm, qid), hs in outcomes.items()}
        explicit = validate_outcomes(read_csv(outcome_path), contexts, review_map)
    top_k = int(config["experiment"]["top_k"])
    snapshots = [s["label"] for s in config["snapshots"]]
    groups = ("changed", "stable", "all")
    arms = ("stale", "rebuilt")
    per_query = {}
    unresolved = []
    for snap in snapshots:
        for q in questions:
            for arm in arms:
                key = (snap, q["group"], arm, q["query_id"])
                retrieved = outcomes.get(key, [])
                if len(retrieved) != top_k or sorted(int(x["rank"]) for x in retrieved) != list(range(1, top_k + 1)):
                    raise ValueError(f"Wrong result count/ranks for {key}")
                relevant = hit_outcome(retrieved, review_map, top_k)
                outcome_key = (snap, arm, q["query_id"])
                if outcome_key in explicit:
                    if relevant is not None and relevant != explicit[outcome_key]:
                        raise ValueError("Conflicting passage and outcome judgments")
                    relevant = explicit[outcome_key]
                if relevant is None:
                    unresolved.append(key)
                source = any(int(r["source_article_hit"]) == 1 for r in retrieved)
                per_query[(snap, q["query_id"], arm)] = (relevant, int(source))
    if unresolved:
        log_event("scoring_incomplete", unresolved_outcomes=len(unresolved), protocol="hit_level_early_stopping")
        raise ValueError(f"{len(unresolved)} top-k outcomes need manual review; unlabeled passages in positive outcomes are not required.")
    ann_rows = read_csv(LOGS / "ann_recall.csv")
    ann_lookup = {(r["snapshot"], r["arm"]): float(r["ann_recall_at_5"]) for r in ann_rows}
    ann_by_query_file = LOGS / "ann_recall_by_query.csv"
    ann_per_query = {
        (r["snapshot"], r["arm"], r["query_id"]): float(r["ann_recall_at_5"])
        for r in read_csv(ann_by_query_file)
    } if ann_by_query_file.exists() else {}
    metrics = []
    differences = []
    for snap in snapshots:
        for group in groups:
            ids = [q["query_id"] for q in questions if group == "all" or q["group"] == group]
            for arm in arms:
                values = [per_query[(snap, qid, arm)] for qid in ids]
                metrics.append({
                    "snapshot": snap, "group": group, "arm": arm, "num_queries": len(ids),
                    "temporal_evidence_hit_at_5": round(float(np.mean([v[0] for v in values])), 6),
                    "source_article_hit_at_5": round(float(np.mean([v[1] for v in values])), 6),
                    "ann_recall_at_5": round(float(np.mean([
                        ann_per_query[(snap, arm, qid)] for qid in ids
                    ])), 6) if ann_per_query else (round(ann_lookup[(snap, arm)], 6) if group == "all" else ""),
                })
            s = np.array([per_query[(snap, qid, "stale")][0] for qid in ids])
            r = np.array([per_query[(snap, qid, "rebuilt")][0] for qid in ids])
            low, high = paired_ci(s, r, seed=config["experiment"]["seed"],
                                  n_boot=config["experiment"]["bootstrap_replicates"])
            differences.append({
                "snapshot": snap, "group": group, "num_paired_queries": len(ids),
                "rebuilt_minus_stale_hit_at_5": round(float(np.mean(r - s)), 6),
                "bootstrap_ci_95_low": round(low, 6), "bootstrap_ci_95_high": round(high, 6),
            })
    write_csv(LOGS / "retrieval_metrics.csv", metrics, METRIC_COLUMNS)
    write_csv(LOGS / "paired_differences.csv", differences, DIFF_COLUMNS)
    write_json(LOGS / "scoring_provenance.json", {
        "all_hit_outcomes_resolved": True,
        "explicit_human_outcome_count": len(explicit),
        "outcome_review_sha256": file_sha256(outcome_path) if outcome_path.exists() else None,
        "full_manual_review": all(r["decision"].strip() in {"0", "1"} for r in reviews),
        "adjudication_protocol": "human-authorized hit-level early stopping; positive on confirmed passage or reviewed outcome with supporting ranks; negative on all confirmed 0 or reviewed complete-set negative",
        "existing_judgment_provenance": "manual_review.csv notes preserve human decisions versus exact-evidence reuse; no new automated relevance labels", "review_sha256": file_sha256(LOGS / "manual_review.csv"),
        "retrieval_hits_sha256": file_sha256(LOGS / "retrieval_hits.csv"),
        "frozen_annotation_hashes": lock,
        "metric_definition": "1 if any manually judged top-5 retrieved passage supports the current documented answer",
        "annotation_policy": "0 = no supporting evidence; 1 = explicit, current and relevant supporting evidence",
        "uncertainty": "95% paired bootstrap percentile interval over query IDs, within snapshot and group",
        "generalization_warning": "Changed/stable queries are purposively balanced; pooled results do not estimate natural user traffic.",
    })
    for row in metrics:
        log_event("measured_result", **row)
    for row in differences:
        log_event("paired_result", **row)


def plot_results(config: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_csv(LOGS / "retrieval_metrics.csv")
    snaps = [s["label"] for s in config["snapshots"]]
    for group in ("changed", "stable", "all"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for arm in ("stale", "rebuilt"):
            values = [float(next(row["temporal_evidence_hit_at_5"] for row in rows
                                 if row["snapshot"] == s and row["group"] == group and row["arm"] == arm))
                      for s in snaps]
            ax.plot(snaps, values, marker="o", label=arm)
        ax.set(xlabel="Historical snapshot", ylabel="Temporal Evidence Hit@5",
               title=f"Retrieval effectiveness: {group} queries", ylim=(-0.05, 1.05))
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(LOGS / f"figure_{group}.png", dpi=200)
        plt.close(fig)
    log_event("figures_written", directory=str(LOGS))


