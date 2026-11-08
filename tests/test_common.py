"""Offline tests for configuration and paired-bootstrap implementation."""

from pathlib import Path

import numpy as np
import pytest

from temporal_hnsw.common import ROOT, file_sha256, load_config
from temporal_hnsw.scoring import paired_ci


def test_six_snapshots_and_expected_experimental_conditions():
    cfg = load_config(ROOT / "config" / "experiment.yaml")
    assert len(cfg["snapshots"]) == 6
    assert cfg["experiment"]["target_articles"] == 500
    assert cfg["experiment"]["top_k"] == 5
    assert cfg["annotations"]["changed_queries"] == 20
    assert cfg["annotations"]["stable_queries"] == 20
    assert cfg["hnsw"]["space"] == "cosine"
    assert cfg["preprocessing"]["overlap_words"] == 0


def test_paired_ci_identical_conditions_returns_zero():
    a = np.array([1, 0, 1, 1, 0])
    lo, hi = paired_ci(a, a)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)


def test_paired_ci_perfect_rebuilt_improvement():
    stale = np.zeros(20)
    rebuilt = np.ones(20)
    lo, hi = paired_ci(stale, rebuilt)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_hash_matches_independent_hashlib(tmp_path: Path):
    import hashlib
    example = tmp_path / "sample"
    example.write_bytes(b"frozen input evidence")
    assert file_sha256(example) == hashlib.sha256(example.read_bytes()).hexdigest()


# Run: python -m pytest -q tests/test_common.py


@pytest.mark.parametrize('case,message', [
    ('not_mapping', 'Expected a YAML object'),
    ('duplicate_snapshots', 'unique labels'),
    ('single_snapshot', 'at least two'),
    ('zero_top_k', 'top_k must be at least 1'),
    ('overlap', 'Overlap not in the fixed design'),
])
def test_config_integrity_checks_survive_optimized_python(tmp_path, case, message):
    import subprocess
    import sys
    import yaml

    config = load_config(ROOT / 'config/experiment.yaml')
    if case == 'not_mapping':
        config = []
    elif case == 'duplicate_snapshots':
        config['snapshots'][1]['label'] = config['snapshots'][0]['label']
    elif case == 'single_snapshot':
        config['snapshots'] = config['snapshots'][:1]
    elif case == 'zero_top_k':
        config['experiment']['top_k'] = 0
    else:
        config['preprocessing']['overlap_words'] = 1
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [sys.executable, '-O', '-c',
         'from pathlib import Path; import sys; '
         'from temporal_hnsw.common import load_config; load_config(Path(sys.argv[1]))',
         str(path)], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'ValueError' in result.stderr
    assert message in result.stderr
