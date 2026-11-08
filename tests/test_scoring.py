"""Offline end-to-end scoring tests using synthetic (NOT research) retrieval logs."""

from pathlib import Path

import pytest

import temporal_hnsw.scoring as sc
from temporal_hnsw.common import read_csv, write_csv


def make_scoring_fixture(tmp_path: Path, monkeypatch, *, incomplete: bool = False):
    ann, log = tmp_path / "annotations", tmp_path / "logs"
    ann.mkdir()
    log.mkdir()
    monkeypatch.setattr(sc, "ANNOTATIONS", ann)
    monkeypatch.setattr(sc, "LOGS", log)
    monkeypatch.setattr(sc, "check_frozen_inputs", lambda path: {"fake": "frozen"})
    monkeypatch.setattr(sc, "log_event", lambda *args, **kwargs: None)
    qs = [{"query_id": "q001", "group": "changed"}, {"query_id": "q002", "group": "stable"}]
    write_csv(ann / "questions.csv", qs, ["query_id", "group"])
    hits, reviews, annrec = [], [], []
    for s in ["D0", "D1"]:
        for arm in ("stale", "rebuilt"):
            annrec.append({"snapshot": s, "arm": arm, "ann_recall_at_5": 1.0})
            for q in qs:
                for rank in range(1, 6):
                    rid = f"{s}_{arm}_{q['query_id']}_{rank}"
                    label = "1" if rank == 1 and (arm == "rebuilt" or s == "D0" or q["group"] == "stable") else "0"
                    if incomplete and rid == "D1_stale_q001_1":
                        label = ""
                    hits.append({"snapshot": s, "group": q["group"], "arm": arm,
                                 "query_id": q["query_id"], "rank": rank,
                                 "source_article_hit": 1 if rank == 1 else 0, "review_id": rid})
                    reviews.append({"review_id": rid, "decision": label})
    write_csv(log / "retrieval_hits.csv", hits, list(hits[0]))
    write_csv(log / "manual_review.csv", reviews, list(reviews[0]))
    write_csv(log / "ann_recall.csv", annrec, list(annrec[0]))
    return ann, log


def test_scoring_computes_paired_difference(tmp_path, monkeypatch):
    _, log = make_scoring_fixture(tmp_path, monkeypatch)
    sc.score({"snapshots": [{"label": "D0"}, {"label": "D1"}],
              "experiment": {"top_k": 5, "seed": 42, "bootstrap_replicates": 100}}, tmp_path / "unused.yaml")
    rows = read_csv(log / "paired_differences.csv")
    changed_d1 = next(row for row in rows if row["snapshot"] == "D1" and row["group"] == "changed")
    stable_d1 = next(row for row in rows if row["snapshot"] == "D1" and row["group"] == "stable")
    assert float(changed_d1["rebuilt_minus_stale_hit_at_5"]) == 1.0
    assert float(stable_d1["rebuilt_minus_stale_hit_at_5"]) == 0.0
    assert (log / "scoring_provenance.json").is_file()


def test_scoring_refuses_incomplete_manual_judgments(tmp_path, monkeypatch):
    _, log = make_scoring_fixture(tmp_path, monkeypatch, incomplete=True)
    with pytest.raises(ValueError, match="manual review"):
        sc.score({"snapshots": [{"label": "D0"}, {"label": "D1"}],
                  "experiment": {"top_k": 5, "seed": 42, "bootstrap_replicates": 100}}, tmp_path / "unused.yaml")
    assert not (log / "retrieval_metrics.csv").exists()


# Run: python -m pytest -q tests/test_scoring.py


def test_scoring_handles_whitespace_in_manual_decisions(tmp_path, monkeypatch):
    _, log = make_scoring_fixture(tmp_path, monkeypatch)
    rows = read_csv(log / "manual_review.csv")
    for row in rows:
        row['decision'] = ' ' + row['decision'] + ' '
    write_csv(log / "manual_review.csv", rows, list(rows[0]))
    sc.score({'snapshots': [{'label': 'D0'}, {'label': 'D1'}],
              'experiment': {'top_k': 5, 'seed': 42, 'bootstrap_replicates': 100}}, tmp_path / 'unused.yaml')
    metrics = read_csv(log / 'retrieval_metrics.csv')
    assert all(float(r['temporal_evidence_hit_at_5']) == 1 for r in metrics if r['arm'] == 'rebuilt')


def test_early_stopping_accepts_blank_passages_after_positive(tmp_path, monkeypatch):
    _, log = make_scoring_fixture(tmp_path, monkeypatch)
    rows = read_csv(log / 'manual_review.csv')
    # Preserve the single known all-negative outcome; clear unneeded labels elsewhere.
    for r in rows:
        if not r['review_id'].startswith('D1_stale_q001_') and r['decision'] == '0':
            r['decision'] = ''
    write_csv(log / 'manual_review.csv', rows, list(rows[0]))
    sc.score({'snapshots': [{'label': 'D0'}, {'label': 'D1'}],
              'experiment': {'top_k': 5, 'seed': 42, 'bootstrap_replicates': 100}}, tmp_path / 'unused.yaml')
    changed = next(r for r in read_csv(log / 'paired_differences.csv') if r['snapshot']=='D1' and r['group']=='changed')
    assert float(changed['rebuilt_minus_stale_hit_at_5']) == 1
    assert float(changed['bootstrap_ci_95_low']) == float(changed['bootstrap_ci_95_high']) == 1


@pytest.mark.parametrize('labels,outcome,next_id', [
    (['','','','','1'],1,None),
    (['0','0','0','0','0'],0,None),
    (['0','','0','',''],None,'2'),
    (['','','','',''],None,'1'),
])
def test_hit_outcome_and_rank_order(labels,outcome,next_id):
    hits=[{'rank':i,'review_id':str(i)} for i in range(5,0,-1)]
    reviews={str(i):{'decision':label} for i,label in enumerate(labels,1)}
    assert sc.hit_outcome(hits,reviews)==outcome
    assert sc.next_review(hits,reviews)==next_id


def test_invalid_label_is_not_silently_ignored_even_with_positive():
    hits=[{'rank':i,'review_id':str(i)} for i in range(1,6)]
    reviews={str(i):{'decision':label} for i,label in enumerate(['1','maybe','','',''],1)}
    with pytest.raises(ValueError,match='Invalid manual review'):
        sc.hit_outcome(hits,reviews)


def test_scoring_accepts_reviewed_top5_negative_without_passage_negatives(tmp_path, monkeypatch):
    from temporal_hnsw.outcome_review import context_hash
    ann, log = make_scoring_fixture(tmp_path, monkeypatch,incomplete=True)
    questions=read_csv(ann/'questions.csv')
    for q in questions:q['question']='Question '+q['query_id']
    write_csv(ann/'questions.csv',questions,list(questions[0]))
    gt=[dict(query_id=q['query_id'],snapshot=s,expected_answer='Answer') for q in questions for s in ['D0','D1']]
    write_csv(ann/'ground_truth.csv',gt,list(gt[0]))
    hs=[h for h in read_csv(log/'retrieval_hits.csv') if h['snapshot']=='D1' and h['arm']=='stale' and h['query_id']=='q001']
    row=dict(snapshot='D1',arm='stale',query_id='q001',question='Question q001',expected_answer='Answer',context_sha256=context_hash('Question q001','Answer',hs),decision='0',supporting_ranks='',reviewed_all_five='yes',reviewer='human')
    write_csv(log/'outcome_review.csv',[row],list(row))
    sc.score({'snapshots':[{'label':'D0'},{'label':'D1'}],'experiment':{'top_k':5,'seed':42,'bootstrap_replicates':100}},tmp_path/'unused.yaml')
    assert next(r for r in read_csv(log/'paired_differences.csv') if r['snapshot']=='D1' and r['group']=='changed')['rebuilt_minus_stale_hit_at_5']=='1.0'
    assert next(r for r in read_csv(log/'manual_review.csv') if r['review_id']=='D1_stale_q001_1')['decision']==''
