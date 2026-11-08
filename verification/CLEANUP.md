# Release scope and exclusions

| Original category | Retained public purpose |
|---|---|
| Experimental source | Original preprocessing, annotations, encoding/HNSW, retrieval, scoring and outcome-validation modules. The shared environment logger names the new CLI; scientific functions are retained. |
| Configuration/dependencies | Unchanged experiment.yaml; original exact package lock; four-package scoring-only requirements; original and verification environments. |
| Frozen data | 500 selected IDs; 3,000 exact revision records; compact raw/processed archives; 40 questions; 240 answers/evidence; original freeze hashes. |
| Human relevance/provenance | Final passage and complete-set judgments, ranks, approved reference evidence, original candidate decisions and exact-reuse mappings. Earlier blank proposals are not substituted for final labels. |
| Reproduction evidence | All original retrieval hits, per-query ANN records, timings, scoring provenance and final metric/CI tables. |
| Final figures | Temporal hit, paired confidence intervals and cost figures in PDF plus PNG previews. |
| Tests | Relevant scientific tests plus exact-context reuse tests. Obsolete prototype/interaction tests removed. |
| Development history | Not copied: Git history, virtualenvs, caches, exploratory scripts, repeated review cards/rounds, temporary logs, superseded preprocessing, 64 unselected cached revisions and earlier result versions. |

The original Wikipedia sampling/download module and its tests are omitted: the public experiment loads bundled exact revisions and cannot select a new corpus. Unused CLI helpers inside retained scientific modules were not aggressively refactored, to avoid unnecessary scientific-code risk. The public runner replaces many session-specific scripts with two commands.

Raw source archives are included (~14 MB compressed), rather than depending on later availability of historical pages. Processed chunks are also included (~7 MB compressed) to make scoring-only reproduction parser-free and to verify reconstruction byte-for-byte. Source and processed archives serve different reproduction modes; they are not accidental duplication.

The original project contained 47,835 files including installed packages and Git/cache files; 4,022 were research files. The original release file count and size are recorded in the historical snapshot `release_inventory.json`; they do not describe the current repository. Local verification environments, model downloads and generated outputs were removed after successful tests, retaining only concise numerical verification records.

The README and runner use relative project paths. Personal checkout paths and original dirty-Git listings are excluded. AI assistance disclosure is preserved. No original .git directory was copied; the public .git is newly initialized and has no remote.
