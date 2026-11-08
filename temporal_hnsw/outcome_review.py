"""Top-five relevance judgments and evidence validation."""
import hashlib
from .common import stable_json

def context_hash(question, answer, hits):
    payload={'question':question,'answer':answer,'hits':sorted(hits,key=lambda h:int(h['rank']))}
    return hashlib.sha256(stable_json(payload).encode()).hexdigest()

def validate_outcomes(rows, contexts, passage_reviews):
    """Validate reviewed judgments, frozen context and shared-passage consistency."""
    result={}; seen=set(); implied={}
    for r in rows:
        key=(r['snapshot'],r['arm'],r['query_id'])
        if key in seen:raise ValueError('Duplicate outcome')
        seen.add(key)
        if key not in contexts:raise ValueError('Unknown outcome')
        question,answer,hits=contexts[key]
        if r['context_sha256']!=context_hash(question,answer,hits):raise ValueError('Outcome context changed')
        if r['question']!=question or r['expected_answer']!=answer:raise ValueError('Question/answer changed')
        if len(hits)!=5 or sorted(int(h['rank']) for h in hits)!=[1,2,3,4,5]:raise ValueError('Expected complete top-five set')
        decision=r['decision'].strip()
        if decision not in {'','uncertain','0','1'}:raise ValueError('Invalid outcome decision')
        if not decision:
            if r['supporting_ranks'].strip():raise ValueError('Ranks without a positive judgment')
            continue
        if not r['reviewer'].strip():raise ValueError('Human reviewer required')
        ranks=[]
        if r['supporting_ranks'].strip():
            try:ranks=[int(x.strip()) for x in r['supporting_ranks'].split(';')]
            except ValueError:raise ValueError('Use semicolon-separated supporting ranks')
        if decision=='1':
            if not ranks or len(set(ranks))!=len(ranks) or any(n not in range(1,6) for n in ranks):raise ValueError('Positive requires valid unique supporting ranks')
        elif ranks:raise ValueError('Only positive outcomes may specify ranks')
        if decision=='0' and r['reviewed_all_five'].strip().lower()!='yes':raise ValueError('Negative requires explicit review of all five')
        if decision=='uncertain':continue
        value=int(decision)
        for h in hits:
            rank=int(h['rank']);rid=h['review_id'];existing=passage_reviews[rid]['decision'].strip()
            if existing not in {'','0','1'}:raise ValueError('Invalid existing passage judgment')
            label='0' if value==0 else ('1' if rank in ranks else None)
            if label is None:continue
            if existing and existing!=label:raise ValueError('Conflict with verified passage judgment')
            # Stored outcome judgments may imply labels, but these are never written
            # to the passage sheet or represented as separately reviewed passages.
            if rid in implied and implied[rid]!=label:raise ValueError('Conflicting human outcomes for the same passage context')
            implied[rid]=label
        result[key]=value
    return result
