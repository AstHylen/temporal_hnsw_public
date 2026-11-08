"""Reviewed top-five outcomes must be bound to exact evidence and existing labels."""
import pytest
from temporal_hnsw.outcome_review import context_hash,validate_outcomes

def fixture():
 hits=[dict(rank=str(i),review_id=str(i),text=f'passage {i}') for i in range(1,6)]
 ctx={('D0','stale','Q'):('Question','Answer',hits)}
 reviews={str(i):dict(decision='') for i in range(1,6)}
 row=dict(snapshot='D0',arm='stale',query_id='Q',question='Question',expected_answer='Answer',context_sha256=context_hash('Question','Answer',hits),decision='1',supporting_ranks='2;4',reviewed_all_five='',reviewer='human')
 return row,ctx,reviews

def test_positive_accepts_ranked_support_without_labeling_others():
 r,c,p=fixture();assert validate_outcomes([r],c,p)=={('D0','stale','Q'):1}
 assert all(x['decision']=='' for x in p.values())

def test_negative_requires_explicit_all_five_review():
 r,c,p=fixture();r.update(decision='0',supporting_ranks='')
 with pytest.raises(ValueError,match='all five'):validate_outcomes([r],c,p)
 r['reviewed_all_five']='yes';assert validate_outcomes([r],c,p)=={('D0','stale','Q'):0}

def test_uncertain_remains_unknown():
 r,c,p=fixture();r.update(decision='uncertain',supporting_ranks='');assert validate_outcomes([r],c,p)=={}

@pytest.mark.parametrize('change',[{'supporting_ranks':''},{'supporting_ranks':'6'},{'reviewer':''},{'context_sha256':'wrong'}])
def test_rejects_invalid_or_unbound_positive(change):
 r,c,p=fixture();r.update(change)
 with pytest.raises(ValueError):validate_outcomes([r],c,p)

def test_conflict_with_verified_negative():
 r,c,p=fixture();p['2']['decision']='0'
 with pytest.raises(ValueError,match='Conflict'):validate_outcomes([r],c,p)

def test_negative_conflicts_with_existing_positive():
 r,c,p=fixture();r.update(decision='0',supporting_ranks='',reviewed_all_five='yes');p['5']['decision']='1'
 with pytest.raises(ValueError,match='Conflict'):validate_outcomes([r],c,p)
