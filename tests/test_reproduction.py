"""Release runner must not extrapolate complete-set judgments to changed evidence."""

import hashlib
import json
import sys

import pytest

import reproduce
from temporal_hnsw import annotations, common, indexing, preprocess, retrieval, scoring

from reproduce import content_set, signature


def test_ranked_content_identity_ignores_only_distance_metadata():
    row = dict(
        rank="1",
        page_id="9",
        chunk_ordinal="0",
        title="Article",
        text="Full text",
        distance="0.1",
    )
    assert content_set([row]) == content_set([dict(row, distance="0.10000001")])
    for field, value in [
        ("rank", "2"),
        ("page_id", "10"),
        ("chunk_ordinal", "1"),
        ("title", "Other"),
        ("text", "Different text"),
    ]:
        assert content_set([row]) != content_set([dict(row, **{field: value})])


def test_passage_reuse_requires_identical_question_answer_and_snapshot():
    row = dict(
        query_id="Q",
        snapshot="D0",
        question="Which?",
        expected_answer="A",
        passage_title="Title",
        passage_text="A is current.",
    )
    for field, value in [
        ("snapshot", "D1"),
        ("expected_answer", "B"),
        ("question", "When?"),
        ("passage_text", "A was current."),
    ]:
        assert signature(row) != signature(dict(row, **{field: value}))


@pytest.fixture
def full_run_fixture(tmp_path, monkeypatch):
    """Small full-run inputs with changed evidence and no reusable decision."""
    root = tmp_path / "public"
    (root / "config").mkdir(parents=True)
    (root / "config/experiment.yaml").write_bytes(
        (common.ROOT / "config/experiment.yaml").read_bytes()
    )
    (root / "frozen/annotations").mkdir(parents=True)
    (root / "reference").mkdir()
    reproduce.dump(root / "frozen/manifest.json", {})
    reproduce.dump(
        root / "frozen/annotations/ground_truth.lock.json",
        {"processed_chunks_sha256": {}},
    )
    reproduce.write(root / "frozen/selected_articles.csv", [{"page_id": "9"}])
    raw = dict(
        page_id=9,
        revision_id=10,
        revision_timestamp="2023-12-01T00:00:00Z",
        wikitext="Original source",
    )
    revision = dict(
        snapshot="D0",
        page_id="9",
        revision_id="10",
        revision_timestamp=raw["revision_timestamp"],
        wikitext_sha256=hashlib.sha256(raw["wikitext"].encode()).hexdigest(),
    )
    reproduce.write(root / "frozen/revision_manifest.csv", [revision])
    reproduce.write(
        root / "frozen/annotations/questions.csv",
        [dict(query_id="Q", question="Which?")],
    )
    reproduce.write(
        root / "frozen/annotations/ground_truth.csv",
        [dict(query_id="Q", snapshot="D0", expected_answer="A")],
    )
    hits = [
        dict(
            snapshot="D0",
            arm="stale",
            query_id="Q",
            rank=str(i),
            page_id="9",
            chunk_ordinal=str(i),
            title="Title",
            text=f"Changed passage {i}",
            review_id=str(i),
        )
        for i in range(1, 6)
    ]
    reviews = [
        dict(
            review_id=h["review_id"],
            query_id="Q",
            snapshot="D0",
            question="Which?",
            expected_answer="A",
            passage_title="Title",
            passage_text=h["text"],
            decision="",
            notes="",
        )
        for h in hits
    ]
    reproduce.write(
        root / "reference/retrieval_hits.csv",
        [dict(h, text="Original passage") for h in hits],
    )
    reproduce.write(
        root / "reference/manual_review.csv",
        [dict(r, passage_text="Original passage", decision="0") for r in reviews],
    )
    reproduce.write(
        root / "reference/outcome_review.csv",
        [dict(snapshot="D0", arm="stale", query_id="Q", decision="0")],
    )

    def unpack_raw(archive, destination):
        (destination / "D0").mkdir(parents=True)
        reproduce.dump(destination / "D0/9.json", raw)

    def retrieve_changed(*args):
        reproduce.write(common.LOGS / "retrieval_hits.csv", hits)
        reproduce.write(common.LOGS / "manual_review.csv", reviews)

    monkeypatch.setattr(reproduce, "ROOT", root)
    monkeypatch.setattr(reproduce, "unpack", unpack_raw)
    for module, name in [
        (preprocess, "process"),
        (annotations, "validate"),
        (indexing, "build"),
        (common, "environment_report"),
    ]:
        monkeypatch.setattr(module, name, lambda *args: None)
    monkeypatch.setattr(retrieval, "retrieve", retrieve_changed)
    monkeypatch.setattr(
        scoring, "score", lambda *args: pytest.fail("Blocked run must not score")
    )
    for name in ["DATA", "LOGS", "ANNOTATIONS"]:
        monkeypatch.setattr(common, name, getattr(common, name))
    for name in ["HF_HOME", "HF_HUB_CACHE", "MPLCONFIGDIR"]:
        monkeypatch.setenv(name, str(tmp_path / "cache"))
    monkeypatch.setattr(sys, "argv", ["reproduce.py", "full"])
    return root, raw, revision, hits


def test_blocked_adjudication_exits_nonzero_and_preserves_review(full_run_fixture):
    root, _, _, hits = full_run_fixture
    with pytest.raises(SystemExit) as exc:
        reproduce.main()
    assert exc.value.code == 2
    output = root / "outputs/full"
    report = json.loads((output / "verification.json").read_text())
    assert report["status"] == "BLOCKED_ADJUDICATION"
    assert report["unresolved_outcomes"] == 1
    assert reproduce.read(output / "needs_human_review.csv") == [
        dict(h, question="Which?", expected_answer="A") for h in hits
    ]
    assert all(
        r["decision"] == "" for r in reproduce.read(output / "logs/manual_review.csv")
    )
    assert not (output / "RESULTS.md").exists()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("page_id", 99, "identity mismatch"),
        ("revision_id", 99, "identity mismatch"),
        ("wikitext", "Altered source", "content hash mismatch"),
        ("revision_timestamp", "2023-12-02T00:00:00Z", "timestamp mismatch"),
        ("cutoff", "2024-01-02T00:00:00Z", "exceeds snapshot cutoff"),
    ],
)
def test_full_run_rejects_invalid_revision(full_run_fixture, field, value, message):
    root, raw, revision, _ = full_run_fixture
    if field == "cutoff":
        raw["revision_timestamp"] = value
        revision["revision_timestamp"] = value
        reproduce.write(root / "frozen/revision_manifest.csv", [revision])
    else:
        raw[field] = value
    with pytest.raises(ValueError, match=message):
        reproduce.main()
    assert not (root / "outputs/full/verification.json").exists()


def test_runner_rejects_unexpected_outcome_totals(full_run_fixture, monkeypatch):
    root, _, _, _ = full_run_fixture
    for name in ["retrieval_metrics.csv", "paired_differences.csv"]:
        (root / "reference" / name).write_text("snapshot\n")

    def score_stub(*args):
        for name in ["retrieval_metrics.csv", "paired_differences.csv"]:
            (common.LOGS / name).write_text("snapshot\n")
        (common.LOGS / "outcome_review.csv").write_text("snapshot\n")

    monkeypatch.setattr(scoring, "score", score_stub)
    monkeypatch.setattr(scoring, "hit_outcome", lambda *args: 1)
    monkeypatch.setattr(reproduce, "make_figures", lambda *args: None)
    with pytest.raises(RuntimeError, match="expected 480 resolved outcomes"):
        reproduce.main()
    assert not (root / "outputs/full/verification.json").exists()
