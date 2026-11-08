#!/usr/bin/env python3
"""Scoring and full-pipeline reproducibility workflows.

Manual judgments transfer only to identical reviewed contexts.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def unpack(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            if (
                not m.isfile()
                or Path(m.name).is_absolute()
                or ".." in Path(m.name).parts
            ):
                raise ValueError("Unsafe archive member")
        tar.extractall(destination, filter="data")


def signature(row):
    return tuple(
        row[k]
        for k in [
            "query_id",
            "snapshot",
            "question",
            "expected_answer",
            "passage_title",
            "passage_text",
        ]
    )


def hit_sets(rows):
    result = {}
    for r in rows:
        result.setdefault((r["snapshot"], r["arm"], r["query_id"]), []).append(r)
    return {k: sorted(v, key=lambda r: int(r["rank"])) for k, v in result.items()}


def content_set(rows):
    return [
        (r["rank"], r["page_id"], r["chunk_ordinal"], r["title"], r["text"])
        for r in rows
    ]


def make_figures(output):
    """Publication layouts from the same scored CSVs; no scientific recomputation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = read(output / "logs/retrieval_metrics.csv")
    diffs = read(output / "logs/paired_differences.csv")
    times = read(output / "logs/build_timings.csv")
    dest = output / "figures"
    dest.mkdir(exist_ok=True)
    snaps = [f"D{i}" for i in range(6)]
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for ax, group in zip(axes, ["changed", "stable"]):
        for arm, style in [("stale", "--"), ("rebuilt", "-")]:
            vals = [
                100
                * float(
                    next(
                        r["temporal_evidence_hit_at_5"]
                        for r in rows
                        if (r["snapshot"], r["group"], r["arm"]) == (snap, group, arm)
                    )
                )
                for snap in snaps
            ]
            ax.plot(snaps, vals, style, marker="o", label=arm)
        ax.set(
            title=f"{group.capitalize()} questions (n=20)",
            xlabel="Snapshot",
            ylabel="Temporal Evidence Hit@5 (%)",
            ylim=(-4, 104),
        )
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(dest / f"temporal_hit.{ext}", dpi=250)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    for group, offset in [("changed", -0.06), ("stable", 0.06)]:
        rs = [
            next(r for r in diffs if r["snapshot"] == snap and r["group"] == group)
            for snap in snaps
        ]
        y = np.array([100 * float(r["rebuilt_minus_stale_hit_at_5"]) for r in rs])
        lo = np.array([100 * float(r["bootstrap_ci_95_low"]) for r in rs])
        hi = np.array([100 * float(r["bootstrap_ci_95_high"]) for r in rs])
        ax.errorbar(
            np.arange(6) + offset,
            y,
            yerr=[y - lo, hi - y],
            fmt="o-",
            capsize=3,
            label=group,
        )
    ax.set(
        xticks=np.arange(6),
        xticklabels=snaps,
        xlabel="Snapshot",
        ylabel="Hit@5 improvement (pp)",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(dest / f"paired_difference.{ext}", dpi=250)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    bottom = np.zeros(6)
    for key, label in [
        ("preprocess_seconds", "Preprocessing"),
        ("embedding_seconds", "Embedding"),
        ("hnsw_build_seconds_median", "HNSW median"),
    ]:
        vals = np.array([float(r[key]) for r in times])
        ax.bar(snaps, vals, bottom=bottom, label=label)
        bottom += vals
    ax.set(
        xlabel="Snapshot", ylabel="Seconds", title="Sum of separately measured stages"
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(dest / f"reconstruction_costs.{ext}", dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["rescore", "full"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "frozen/manifest.json").read_text())
    for p, h in manifest.items():
        if sha(ROOT / p) != h:
            raise RuntimeError(f"Frozen input changed: {p}")
    output = (args.output or ROOT / "outputs" / args.mode).resolve()
    if output.exists():
        raise RuntimeError(
            "Output exists. Preserve it and choose a new --output directory."
        )
    if ROOT not in output.parents:
        raise ValueError("Output must be inside this repository")
    output.mkdir(parents=True)
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_CACHE"] = str(ROOT / ".cache/huggingface/hub")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    from temporal_hnsw import common as c

    c.DATA = output / "data"
    c.LOGS = output / "logs"
    c.ANNOTATIONS = c.DATA / "annotations"
    c.ensure_dirs()
    config_path = ROOT / "config/experiment.yaml"
    config = c.load_config(config_path)
    for name in ["selected_articles.csv", "revision_manifest.csv"]:
        shutil.copy2(ROOT / "frozen" / name, c.DATA / name)
    for name in ["questions.csv", "ground_truth.csv"]:
        shutil.copy2(ROOT / "frozen/annotations" / name, c.ANNOTATIONS / name)
    from temporal_hnsw.scoring import score, hit_outcome

    # Scoring does not import the encoder or require model weights.
    if args.mode == "rescore":
        unpack(ROOT / "frozen/processed_chunks.tar.gz", c.DATA / "processed")
        shutil.copy2(
            ROOT / "frozen/annotations/ground_truth.lock.json",
            c.ANNOTATIONS / "ground_truth.lock.json",
        )
        for name in [
            "retrieval_hits.csv",
            "manual_review.csv",
            "outcome_review.csv",
            "ann_recall.csv",
            "ann_recall_by_query.csv",
            "build_timings.csv",
        ]:
            shutil.copy2(ROOT / "reference" / name, c.LOGS / name)
    else:
        unpack(ROOT / "frozen/raw_revisions.tar.gz", c.DATA / "raw")
        revs = read(c.DATA / "revision_manifest.csv")
        for r in revs:
            raw = json.loads(
                (c.DATA / "raw" / r["snapshot"] / f"{r['page_id']}.json").read_text()
            )
            if (
                str(raw["revision_id"]) != r["revision_id"]
                or str(raw["page_id"]) != r["page_id"]
            ):
                raise ValueError(
                    f"Revision identity mismatch: {r['snapshot']}/{r['page_id']}"
                )
            if (
                hashlib.sha256(raw["wikitext"].encode()).hexdigest()
                != r["wikitext_sha256"]
            ):
                raise ValueError(
                    f"Raw content hash mismatch: {r['snapshot']}/{r['page_id']}"
                )
            if raw["revision_timestamp"] != r["revision_timestamp"]:
                raise ValueError(
                    f"Revision timestamp mismatch: {r['snapshot']}/{r['page_id']}"
                )
            cutoff = next(
                s["timestamp"]
                for s in config["snapshots"]
                if s["label"] == r["snapshot"]
            )
            if raw["revision_timestamp"] > cutoff:
                raise ValueError(
                    f"Revision exceeds snapshot cutoff: {r['snapshot']}/{r['page_id']}"
                )
        from temporal_hnsw.preprocess import process

        process(config)
        original_lock = json.loads(
            (ROOT / "frozen/annotations/ground_truth.lock.json").read_text()
        )
        for s, h in original_lock["processed_chunks_sha256"].items():
            if sha(c.DATA / "processed" / s / "chunks.jsonl") != h:
                raise RuntimeError(f"Chunk reproduction mismatch at {s}")
        from temporal_hnsw.annotations import validate

        validate(config, config_path)
        from temporal_hnsw.indexing import build
        from temporal_hnsw.retrieval import retrieve

        c.environment_report(config_path)
        build(config, config_path)
        retrieve(config, config_path)
        # Only identical previously reviewed full passages inherit passage labels.
        old_reviews = {
            signature(r): r for r in read(ROOT / "reference/manual_review.csv")
        }
        new_reviews = read(c.LOGS / "manual_review.csv")
        for r in new_reviews:
            old = old_reviews.get(signature(r))
            if old:
                r["decision"] = old["decision"]
                r["notes"] = old["notes"]
        write(c.LOGS / "manual_review.csv", new_reviews)
        original_sets = hit_sets(read(ROOT / "reference/retrieval_hits.csv"))
        new_sets = hit_sets(read(c.LOGS / "retrieval_hits.csv"))
        diffs = [
            {"snapshot": k[0], "arm": k[1], "query_id": k[2]}
            for k, v in new_sets.items()
            if content_set(v) != content_set(original_sets[k])
        ]
        # Complete-set judgments bind all ranks to full texts; never extend a negative
        # complete-set decision to a changed top-five set.
        from temporal_hnsw.outcome_review import context_hash

        qs = {q["query_id"]: q for q in read(c.ANNOTATIONS / "questions.csv")}
        gt = {
            (r["query_id"], r["snapshot"]): r
            for r in read(c.ANNOTATIONS / "ground_truth.csv")
        }
        reused = []
        for r in read(ROOT / "reference/outcome_review.csv"):
            k = (r["snapshot"], r["arm"], r["query_id"])
            if content_set(original_sets[k]) == content_set(new_sets[k]):
                r = dict(r)
                r["context_sha256"] = context_hash(
                    qs[k[2]]["question"], gt[k[2], k[0]]["expected_answer"], new_sets[k]
                )
                reused.append(r)
        if reused:
            write(c.LOGS / "outcome_review.csv", reused)
        dump(
            output / "retrieval_comparison.json",
            dict(
                changed_ranked_sets=diffs,
                original_sets=len(original_sets),
                reproduced_sets=len(new_sets),
                byte_identical_hits=sha(c.LOGS / "retrieval_hits.csv")
                == sha(ROOT / "reference/retrieval_hits.csv"),
                reused_complete_set_judgments=len(reused),
                policy="Exact ranked content and frozen question/answer identity only; distances may differ without changing evidence. Original import/context hashes are preserved in reference/.",
            ),
        )
        unresolved = []
        rm = {r["review_id"]: r for r in new_reviews}
        explicit = {
            (r["snapshot"], r["arm"], r["query_id"]): int(r["decision"])
            for r in reused
            if r["decision"] in ["0", "1"]
        }
        for k, hs in new_sets.items():
            if k not in explicit and hit_outcome(hs, rm) is None:
                for h in hs:
                    unresolved.append(
                        dict(
                            **h,
                            question=qs[k[2]]["question"],
                            expected_answer=gt[k[2], k[0]]["expected_answer"],
                        )
                    )
        if unresolved:
            write(output / "needs_human_review.csv", unresolved)
            dump(
                output / "verification.json",
                dict(
                    status="BLOCKED_ADJUDICATION",
                    unresolved_outcomes=len(
                        {(r["snapshot"], r["arm"], r["query_id"]) for r in unresolved}
                    ),
                    note="No unknown passage or outcome was assigned a negative label.",
                ),
            )
            print(
                "Retrieval completed; changed evidence requires human review. See needs_human_review.csv."
            )
            raise SystemExit(2)
    score(config, config_path)
    make_figures(output)
    metrics = read(c.LOGS / "retrieval_metrics.csv")
    diffs = read(c.LOGS / "paired_differences.csv")
    comparisons = {
        name: read(c.LOGS / name) == read(ROOT / "reference" / name)
        for name in ["retrieval_metrics.csv", "paired_differences.csv"]
    }
    if not all(comparisons.values()):
        dump(
            output / "verification.json",
            dict(status="FAIL_NUMERICAL_COMPARISON", comparisons=comparisons),
        )
        raise RuntimeError(
            "Reproduced metrics differ; outputs retained for investigation"
        )
    from temporal_hnsw.outcome_review import validate_outcomes

    qs = {q["query_id"]: q for q in read(c.ANNOTATIONS / "questions.csv")}
    gt = {
        (r["query_id"], r["snapshot"]): r
        for r in read(c.ANNOTATIONS / "ground_truth.csv")
    }
    groups = hit_sets(read(c.LOGS / "retrieval_hits.csv"))
    rm = {r["review_id"]: r for r in read(c.LOGS / "manual_review.csv")}
    explicit = validate_outcomes(
        read(c.LOGS / "outcome_review.csv"),
        {
            k: (qs[k[2]]["question"], gt[k[2], k[0]]["expected_answer"], v)
            for k, v in groups.items()
        },
        rm,
    )
    vals = [explicit.get(k, hit_outcome(v, rm)) for k, v in groups.items()]
    if len(vals) != 480 or vals.count(1) != 392 or vals.count(0) != 88:
        raise RuntimeError(
            "Outcome verification failed: expected 480 resolved outcomes, 392 hits and 88 misses."
        )
    dump(
        output / "verification.json",
        dict(
            status="PASS",
            mode=args.mode,
            all_metric_rows_exact=True,
            all_confidence_intervals_exact=True,
            resolved=480,
            hits=392,
            misses=88,
        ),
    )
    lines = [
        "# Reproduced results",
        "",
        f"Mode: {args.mode}. All 480 outcomes resolved: 392 hits, 88 misses. All published metric rows and paired confidence intervals match exactly.",
        "",
        "| Snapshot | Group | Stale (%) | Rebuilt (%) | Difference (pp) | 95% CI (pp) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in diffs:
        rates = [
            100
            * float(
                next(
                    m["temporal_evidence_hit_at_5"]
                    for m in metrics
                    if m["snapshot"] == r["snapshot"]
                    and m["group"] == r["group"]
                    and m["arm"] == a
                )
            )
            for a in ["stale", "rebuilt"]
        ]
        lines.append(
            f"| {r['snapshot']} | {r['group']} | {rates[0]:.1f} | {rates[1]:.1f} | {100*float(r['rebuilt_minus_stale_hit_at_5']):.1f} | [{100*float(r['bootstrap_ci_95_low']):.1f}, {100*float(r['bootstrap_ci_95_high']):.1f}] |"
        )
    lines += [
        "",
        "Figures: [temporal hit](figures/temporal_hit.pdf), [paired differences](figures/paired_difference.pdf), [costs](figures/reconstruction_costs.pdf).",
        "",
        "Timing measurements: logs/build_timings.csv. Full mode remeasures preprocessing, encoding and three HNSW construction trials per snapshot; rescore uses original measurements. Sum-stage cost is approximate, not end-to-end latency.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(
        f"PASS: published metrics and confidence intervals reproduced. See {output.relative_to(ROOT)}/RESULTS.md"
    )


if __name__ == "__main__":
    main()
