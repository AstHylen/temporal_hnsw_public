"""Evidence protection regression tests."""
from pathlib import Path
import pytest
from temporal_hnsw import preprocess, indexing


def test_frozen_prepare_refuses_before_writing(tmp_path, monkeypatch):
    annotations = tmp_path / 'annotations'
    annotations.mkdir()
    (annotations / 'ground_truth.lock.json').write_text('{}')
    monkeypatch.setattr(preprocess, 'DATA', tmp_path)
    with pytest.raises(RuntimeError, match='frozen'):
        preprocess.process({})
    assert list(tmp_path.iterdir()) == [annotations]


def test_build_requires_frozen_inputs_before_loading_model(tmp_path, monkeypatch):
    import temporal_hnsw.common as common
    monkeypatch.setattr(common, 'ANNOTATIONS', tmp_path)
    monkeypatch.setattr(indexing, 'Encoder', lambda cfg: pytest.fail('Model must not load before freeze check'))
    with pytest.raises(RuntimeError, match='Missing ground_truth'):
        indexing.build({}, tmp_path / 'unused.yaml')
