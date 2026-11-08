# Published results

480 resolved outcomes: 392 hits, 88 misses. All source-article and ANN recall metrics are 100%.

| Snapshot | Group | Stale (%) | Rebuilt (%) | Difference (pp) | 95% CI (pp) |
|---|---|---:|---:|---:|---|
| D0 | changed | 90.0 | 90.0 | 0.0 | [0.0, 0.0] |
| D0 | stable | 100.0 | 100.0 | 0.0 | [0.0, 0.0] |
| D0 | all | 95.0 | 95.0 | 0.0 | [0.0, 0.0] |
| D1 | changed | 65.0 | 95.0 | 30.0 | [10.0, 50.0] |
| D1 | stable | 100.0 | 100.0 | 0.0 | [0.0, 0.0] |
| D1 | all | 82.5 | 97.5 | 15.0 | [5.0, 27.5] |
| D2 | changed | 30.0 | 95.0 | 65.0 | [40.0, 85.0] |
| D2 | stable | 100.0 | 100.0 | 0.0 | [0.0, 0.0] |
| D2 | all | 65.0 | 97.5 | 32.5 | [17.5, 47.5] |
| D3 | changed | 15.0 | 90.0 | 75.0 | [55.0, 95.0] |
| D3 | stable | 100.0 | 100.0 | 0.0 | [0.0, 0.0] |
| D3 | all | 57.5 | 95.0 | 37.5 | [22.5, 52.5] |
| D4 | changed | 5.0 | 90.0 | 85.0 | [70.0, 100.0] |
| D4 | stable | 100.0 | 100.0 | 0.0 | [0.0, 0.0] |
| D4 | all | 52.5 | 95.0 | 42.5 | [27.5, 57.5] |
| D5 | changed | 0.0 | 95.0 | 95.0 | [85.0, 100.0] |
| D5 | stable | 100.0 | 100.0 | 0.0 | [0.0, 0.0] |
| D5 | all | 50.0 | 97.5 | 47.5 | [32.5, 62.5] |

Figures: [temporal hit](temporal_hit.pdf), [paired differences](paired_difference.pdf), [reconstruction costs](reconstruction_costs.pdf).

Paired bootstrap: 2,000 resamples, seed 42. These results concern a small purposively balanced query set, not natural search traffic. Stable queries remain at 100%; changed queries degrade in the unchanged stale index while rebuilding retains 90–95% hit rates. Source retrieval and nearest-neighbor accuracy do not imply current evidence.

Original costs and environment are preserved under ../reference/. Sum-stage reconstruction cost is approximate, not end-to-end downtime.
