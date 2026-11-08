# Temporal retrieval in a fixed versus rebuilt HNSW index

This experiment compares an unchanged D0 index with indexes rebuilt from six English Wikipedia snapshots (January 2024–July 2026). It studies temporal evidence retrieval, not incremental graph degradation or LLM answers.

The corpus is fixed: 500 article IDs, exact historical revisions, sentence-aware non-overlapping ~100-word chunks, and 40 human-reviewed questions (20 changed, 20 stable). Embeddings use `all-MiniLM-L6-v2` at commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. HNSW uses cosine, M=16, ef_construction=200, ef_search=100, top-5, seed 42, one construction thread.

Temporal Evidence Hit@5 is positive if any retrieved passage supports the snapshot answer; a negative requires human review of the entire top-five set. Source Article Hit@5 and ANN Recall@5 are secondary metrics. Paired query-bootstrap intervals use 2,000 replicates and seed 42. Exact-reuse evidence provenance is retained separately from new human judgments.

## Install

The original used Python 3.12.8; release verification uses Python 3.14.2. Python 3.12+ is supported (see `reference/environment.json` and `verification/REPORT.md`). For scoring only:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-rescore.txt
.venv/bin/python reproduce.py rescore
```

For the full pipeline, install the complete pinned environment:

```sh
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python reproduce.py full
```

The repository includes compressed exact raw revisions and processed chunks, so no Wikipedia API sampling or live revision selection is required. Full mode loads the included raw data, verifies revision identities/content hashes, reconstructs chunks, downloads the pinned model from Hugging Face into `.cache/huggingface`, embeds all six snapshots, builds indexes, retrieves, safely reuses applicable judgments, and scores. It never uses a cache outside this repository. Model download needs network access; subsequent runs can use the documented local cache.

Each command creates a separate `outputs/rescore/` or `outputs/full/`; existing outputs are never overwritten. Use `--output outputs/another_run` to repeat. Outputs include `RESULTS.md`, numerical CSVs, PNG figures, provenance, and `verification.json`. Full runs also include timings, vectors, indexes and `retrieval_comparison.json`. If ranked evidence differs, unreviewed outcomes are exported to `needs_human_review.csv` and primary scoring stops with exit status 2 (`BLOCKED_ADJUDICATION`); the verification report and unresolved review data are preserved. Human judgments are never extrapolated to different text or different top-five sets.

## Published results

All 480 outcomes were resolved: 392 hits and 88 misses. Changed-question stale Hit@5 declines from 90% at D0 to 0% at D5; rebuilt is 90–95%, ending at 95%. D5 improvement is 95 percentage points (95% CI 85–100). Stable questions achieve 100% throughout. Both secondary metrics are 100% throughout: finding the right article or nearest neighbors does not establish temporal correctness.

Scoring-only mode aggregates the preserved per-query ANN validation records; full mode recomputes exact cosine top-five comparisons.

Original sum-stage reconstruction costs were 33.51–35.29 seconds; embedding dominated. These sums combine previously measured preprocessing, encoding and median-of-three HNSW construction times, excluding model loading, downloads, serialization and retrieval. They are not end-to-end latency. Full mode remeasures these stages; timing equality is not expected.

## Contents

- `temporal_hnsw/`: retained scientific implementation.
- `reproduce.py`: scoring/full-pipeline runner.
- `config/`: unchanged experiment settings.
- `frozen/`: fixed manifests, source archives, annotations and human-review provenance.
- `reference/`: original retrieval, labels, metrics, timings and environment.
- `results/`: published figures and summary.
- `tests/`: scientific unit/regression tests.
- `verification/`: release checks and cleanup report.

Run tests with `.venv/bin/python -m pytest -q`. Tests importing HNSW/parser components need the full requirements.

## Limitations and attribution

Selection-time eligibility used the original extractor; the approved cleanup retains all 500 articles, including ten below 200 D0 words afterward. No filtering or resampling occurs during reproduction.

The experiments were conducted on macOS 26.5.2 (Apple Silicon, arm64), with 10 logical CPU cores and 16 GB RAM, using Python 3.12.8. Software dependencies and the embedding model revision are pinned in `requirements.lock.txt` and `config/experiment.yaml`, respectively.

Questions are purposively balanced and small; pooled results do not represent natural search traffic. Wikipedia provides source-documented evidence, not independently established truth. Snapshot intervals are correlated; bootstrap intervals do not correct multiple comparisons. No independent second-reviewer agreement is claimed.

Wikipedia content is included with article IDs, titles and exact revision URLs in the evidence/manifest files. See `DATA_ATTRIBUTION.md`. See `AI_USE.md` for the AI assistance disclosure. See `verification/REPORT.md` for executed verification and any blockers.
