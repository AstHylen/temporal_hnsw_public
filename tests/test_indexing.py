"""Real HNSW approximate-nearest-neighbor checks (require hnswlib)."""

import pytest
import numpy as np

pytest.importorskip("hnswlib")
from temporal_hnsw.indexing import create_index
from temporal_hnsw.retrieval import exact_topk


def test_exact_topk_and_hnsw_consistency_for_easy_vectors():
    vectors = np.eye(16, dtype=np.float32)
    config = {"experiment": {"top_k": 5},
              "hnsw": {"M": 8, "ef_construction": 100, "ef_search": 32,
                       "num_threads": 1, "random_seed": 42}}
    index = create_index(vectors, config)
    query = vectors[3]
    found, _ = index.knn_query(query, k=5, num_threads=1)
    nearest = exact_topk(vectors, query, 5)
    assert 3 in found[0]
    assert 3 in nearest
    assert set(found[0]).intersection(set(nearest))


def test_exact_topk_identifies_vector_with_highest_cosine():
    a = np.array([[1., 0.], [0., 1.], [0.8, 0.6]], dtype=np.float32)
    q = np.array([1., 0.], dtype=np.float32)
    assert exact_topk(a, q, 1).tolist() == [0]


# Run: python -m pytest -q tests/test_indexing.py


def test_exact_topk_breaks_boundary_ties_by_vector_id():
    vectors = np.ones((40, 2), dtype=np.float32)
    assert exact_topk(vectors, np.array([1., 0.]), 5).tolist() == list(range(5))
