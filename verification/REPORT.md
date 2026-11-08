# Release verification

## Original release verification

The full reproduction, environment details and commands below describe the original release verification, before the subsequent cleanup.

- **Re-scoring: PASS.** All 36 metric rows and 18 paired-confidence-interval rows match the original CSVs exactly. 480 outcomes, 392 hits, 88 misses. Verified both in a fresh full environment and a separate fresh four-package scoring-only environment.
- **Full reproduction: PASS for data, rankings and numerical scientific results.** Loaded all 3,000 exact bundled revisions; checked page/revision identities, raw hashes and cutoffs; reconstructed all six processed chunk files byte-for-byte. Downloaded the pinned model into a new repository-local cache. Embedded, ran 18 HNSW construction trials, all 12 retrieval arms and exact-cosine checks. All 480 ranked evidence sets and all final metric/CI rows match. All 89 complete-set judgments remain applicable; no additional adjudication was needed.
- **Original release: 38 tests passed**, including release-specific tests preventing judgment extrapolation.
- Original dataset, configuration, queries and labels were not altered. A separate read-only preservation check found all **4,022 original research-file hashes unchanged** and no added/removed research files. Original environment/cache/Git files were neither copied nor used for verification.

### Differences, not hidden

Verification ran on Python 3.14.2; the original used 3.12.8. All pinned package versions match. Both runs used macOS arm64. CPU inference was confirmed for the original query embeddings, but not independently documented for all embedding stages. All six embedding and index files have different hashes; query-vector hashes also differ. **2,065 of 2,400 stored cosine distances differ**, with maximum absolute difference **7.1e-7**. All article/chunk/rank identities and every scored outcome are unchanged. Runtime/wheel numerical variation is a plausible explanation, not an independently isolated cause. This is numerical result/ranking reproduction, not byte-identical vector/index reproduction.

New sum-stage costs are **17.34–18.25 seconds** versus original **33.51–35.29 seconds**. Timing equality is not expected. Exact per-stage/trial measurements are in reproduced_build_timings.csv; historical originals remain in ../reference/build_timings.csv. Both use the same original timing procedure; totals are approximate sums, not end-to-end costs.

The first full attempt stopped after successful preprocessing because the original environment logger expected the retired CLI filename. Its metadata-only path was corrected to reproduce.py. The complete subsequent run passed. A scoring-only import dependency on the parser was also removed before testing the minimal environment. No scientific formula, extractor, chunker, encoder, index parameter or annotation was changed to obtain agreement.

### Commands actually verified

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements.lock.txt
.venv/bin/python reproduce.py full --output outputs/full_verified
.venv/bin/python -m pytest -q
python3 -m venv .venv-rescore
.venv-rescore/bin/python -m pip install --no-cache-dir -r requirements-rescore.txt
.venv-rescore/bin/python reproduce.py rescore --output outputs/rescore_final
```

Publication figure generation was checked from these scored CSVs. Generated environments, downloads and heavy reproduction outputs are excluded from the public release; the small verification records here preserve the results. Re-run the documented commands to reconstruct them. No Git remote, push or publication was performed.

## Subsequent cleanup verification

**50 tests passed** after the code and documentation cleanup, including blocked-adjudication exit/report preservation, revision integrity, outcome totals and configuration checks under optimized Python. Re-scoring preserved all 480 individual outcomes (392 hits, 88 misses); all 36 metric rows and 18 confidence-interval rows matched exactly. Full embedding/index reconstruction was not rerun for the cleanup.

### Editorial review metadata

Only the `notes` field changed in the following files. No other CSV fields or row order changed: decisions, supporting ranks, reviewer fields, questions, reference answers, passage text, context hashes and identifiers were preserved. Specific evidence rationales were retained. The frozen manifest was updated for these two files only; original reference records and the original research repository were not modified. The detailed before/after comparison is retained in a private backup outside the public repository.

| File | Notes changed | Original SHA-256 | Updated SHA-256 |
|---|---:|---|---|
| `frozen/review/top5_decisions.csv` | 89 | `06ac18ac9f81e12b4582d6d8940d22fb2ef6703af04dc78bb487f3c41a5c7a75` | `f8f928e7f3dd66d0d4fbcbc371cee8bd3a4f5eb6297d238d624e74809e858f7b` |
| `frozen/review/passage_decisions.csv` | 16 | `89b2c80aaec9bda242668daee5026b754b7683047d6229652c3d2f2759fc0645` | `ec603d5c43a4c4e1045ca6ed8c6a60f7c6bb974526f466dad8b45b64856ba02a` |

### Historical verification artifacts

[release_inventory.json](release_inventory.json) and [source_comparison.json](source_comparison.json) are historical snapshots of the original release, before the cleanup. Their file counts, hashes and source-equality flags do not describe the current repository. Their original measurements are retained with explicit historical labels. [final_cleanup_integrity.json](final_cleanup_integrity.json) records the subsequent code/metadata cleanup checks, before this documentation-only consolidation.

### Documentation consolidation check

After consolidating these reports, all 50 tests passed again. Re-scoring with `.venv/bin/python reproduce.py rescore --output outputs/rescore_documentation_cleanup` preserved every outcome (480 total, 392 hits, 88 misses) and produced byte-identical metric and confidence-interval CSVs. All 26 frozen-manifest entries verified; all 53 implementation, configuration, scientific data, reference, test and result files checked against the pre-consolidation inventory were unchanged. Local Markdown links were checked with no broken targets. No push or publication was performed.

## Remaining limitations

Pinned dependencies/model downloads still require their upstream services for fresh installation/full inference. Original raw revisions and processed chunks are bundled, so Wikipedia availability is not a prerequisite. Cross-platform numerical variations can change rankings in future runs; the runner exports unresolved outcomes rather than inventing labels. Small purposive query groups, source-document truth and lack of independent dual-reviewer agreement remain scientific limitations. The bundled Wikipedia material remains subject to its original licensing terms, as described in `DATA_ATTRIBUTION.md`.
