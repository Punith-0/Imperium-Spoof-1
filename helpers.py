"""Input normalization, exact quote verification, scoring and exports."""
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from schemas import Conversation, Turn, Criterion, Analysis, QualityCheck

MAX_BYTES = 10 * 1024 * 1024
MAX_CALLS = 500
MAX_TURNS = 200
MAX_CHARS = 40000
ROLE = {'agent':'agent','assistant':'agent','support':'agent','representative':'agent',
        'customer':'customer','user':'customer','client':'customer','action':'system',
        'system':'system','unknown':'unknown'}
MISSING = re.compile(r'\[(?:[^\]]*(?:missing|unavailable|inaudible|recording ended|truncated)[^\]]*)\]', re.I)
LINE = re.compile(r'^\s*(?:\[([^\]]+)\]\s*)?(agent|customer|system|action|user|assistant|support|unknown)\s*:\s*(.*)$', re.I)

def parse_text(text):
    turns = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = LINE.match(line)
        if match:
            _, role, content = match.groups()
            turns.append({'speaker': role.lower(), 'text': content})
        elif turns and not MISSING.search(line):
            turns[-1]['text'] += '\n' + line
        else:
            turns.append({'speaker': 'system' if MISSING.search(line) else 'unknown', 'text': line})
    return turns

def normalize_record(record, index=1):
    if not isinstance(record, dict):
        raise ValueError(f'Record {index} must be an object.')
    warnings = []
    raw = record.get('turns')
    if raw is None:
        raw = record.get('original')
    if raw is None and isinstance(record.get('transcript'), str):
        raw = parse_text(record['transcript'])
    if not isinstance(raw, list) or not raw:
        raise ValueError(f'Record {index}: expected non-empty turns, original, or transcript.')
    if len(raw) > MAX_TURNS:
        raise ValueError(f'Record {index}: maximum {MAX_TURNS} turns per conversation; nothing was truncated.')
    turns, gaps = [], False
    for i, t in enumerate(raw, 1):
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            speaker, text = t[:2]
        elif isinstance(t, dict):
            speaker = t.get('speaker', t.get('speaker_role', t.get('role')))
            text = t.get('text', t.get('utterance', t.get('content')))
        else:
            raise ValueError(f'Record {index}, turn {i}: invalid turn format.')
        speaker = ROLE.get(str(speaker).strip().lower(), 'unknown')
        if speaker == 'unknown':
            warnings.append(f't{i:03}: speaker unavailable; no role was guessed.')
        if text is None or (isinstance(text, str) and not text.strip()):
            text = '[MISSING TURN TEXT]'
            gaps = True
            warnings.append(f't{i:03}: missing text retained as a gap marker.')
        if not isinstance(text, str):
            raise ValueError(f'Record {index}, turn {i}: text must be a string or null.')
        gaps |= bool(MISSING.search(text))
        turns.append(Turn(turn_id=f't{i:03}', speaker=speaker, text=text))
    if sum(len(t.text) for t in turns) > MAX_CHARS:
        raise ValueError(f'Record {index}: maximum {MAX_CHARS:,} characters; nothing was truncated.')
    if not any(t.speaker in ('agent','customer','unknown') and not MISSING.search(t.text) for t in turns):
        raise ValueError(f'Record {index}: no usable conversation text.')
    coverage = str(record.get('coverage', 'unknown'))
    coverage = 'incomplete' if gaps or coverage in ('incomplete','known_incomplete') else ('complete' if coverage == 'complete' else 'unknown')
    if gaps:
        warnings.append('Transcript has missing or unavailable content.')
    if not any(t.speaker == 'customer' for t in turns):
        warnings.append('No identified customer turns; customer sentiment cannot be assessed.')
    return Conversation(conversation_id=str(record.get('conversation_id', record.get('convo_id', f'call_{index:03}'))),
                        turns=turns, coverage=coverage, warnings=warnings,
                        source=str(record.get('source','upload')), record_type=str(record.get('record_type','source')))

def load_conversations(data, filename):
    if len(data) > MAX_BYTES:
        raise ValueError('Upload must be 10 MB or smaller.')
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError as e:
        raise ValueError('Save the file using UTF-8 encoding.') from e
    ext = Path(filename).suffix.lower()
    if ext == '.json':
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON near line {e.lineno}: {e.msg}') from e
        if isinstance(raw, dict) and 'conversations' in raw:
            raw = raw['conversations']
        elif isinstance(raw, dict):
            raw = [raw]
    elif ext == '.csv':
        try:
            raw = list(csv.DictReader(io.StringIO(text)))
            for i, row in enumerate(raw, 1):
                if row.get('turns_json'):
                    row['turns'] = json.loads(row['turns_json'])
        except (csv.Error, json.JSONDecodeError) as e:
            raise ValueError('Invalid CSV or turns_json field.') from e
    elif ext == '.txt':
        raw = [{'conversation_id':Path(filename).stem, 'transcript':text}]
    else:
        raise ValueError('Use JSON, CSV or TXT.')
    if not isinstance(raw, list) or not raw:
        raise ValueError('The file contains no conversation records.')
    if len(raw) > MAX_CALLS:
        raise ValueError(f'Maximum {MAX_CALLS} conversations per upload.')
    calls = [normalize_record(r, i) for i, r in enumerate(raw, 1)]
    ids = [c.conversation_id for c in calls]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate conversation IDs. Give each record a unique ID.')
    return calls

def load_criteria(text):
    raw = json.loads(text)
    if not isinstance(raw,list) or not 1 <= len(raw) <= 20:
        raise ValueError('Use a JSON list of 1–20 criteria.')
    criteria = [Criterion.model_validate(r) for r in raw]
    if len({c.id for c in criteria}) != len(criteria):
        raise ValueError('Criterion IDs must be unique.')
    for c in criteria:
        for pattern in c.demo_patterns:
            re.compile(pattern)
    if not any(c.enabled for c in criteria):
        raise ValueError('Enable at least one criterion.')
    return criteria

def fingerprint(call, criteria, mode, model):
    data = {'call':call.model_dump(), 'criteria':[c.model_dump() for c in criteria], 'mode':mode, 'model':model}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

def verify_analysis(raw, call, criteria):
    """Reject unsupported citations, reconcile checklist, guard absence claims.

    Exact text existence is checked; semantic entailment still needs human review.
    """
    result = raw.model_copy(deep=True)
    lookup = {t.turn_id:t for t in call.turns}
    warnings, rejected = [], []
    def valid(e):
        t = lookup.get(e.turn_id)
        return bool(t and e.quote.strip() and e.quote in t.text and not MISSING.search(e.quote))
    def supported(item):
        return bool(item.evidence) and all(valid(e) for e in item.evidence)
    result.issues = []
    for issue in raw.issues:
        if supported(issue):
            result.issues.append(issue)
        else:
            rejected.append({'type':'issue','title':issue.title,'reason':'Missing or invalid exact quote evidence.'})
    result.risk_flags = []
    for risk in raw.risk_flags:
        if supported(risk):
            result.risk_flags.append(risk)
        else:
            rejected.append({'type':'risk','title':risk.title,'reason':'Missing or invalid exact quote evidence.'})
    result.sentiment = []
    seen = set()
    for s in raw.sentiment:
        t = lookup.get(s.turn_id)
        if t and t.speaker == 'customer' and s.turn_id not in seen and supported(s) and all(e.turn_id == s.turn_id for e in s.evidence):
            result.sentiment.append(s); seen.add(s.turn_id)
        else:
            rejected.append({'type':'sentiment','title':s.turn_id,'reason':'Invalid, duplicate or non-customer evidence.'})
    result.sentiment.sort(key=lambda s:list(lookup).index(s.turn_id))
    expected = {c.id:c for c in criteria if c.enabled}
    grouped = {}
    for q in raw.quality_checks:
        if q.criterion_id in expected:
            grouped.setdefault(q.criterion_id,[]).append(q)
        else:
            warnings.append(f'Unknown/disabled criterion ignored: {q.criterion_id}')
    result.quality_checks = []
    for cid in expected:
        qs = grouped.get(cid,[])
        if len(qs) != 1:
            q = QualityCheck(criterion_id=cid,status='unknown',explanation='Missing or duplicate model checklist result.',support_type='insufficient_evidence',evidence=[])
        else:
            q = qs[0].model_copy(deep=True)
            bad = any(not valid(e) for e in q.evidence)
            if bad:
                rejected.append({'type':'check','title':cid,'reason':'Invalid exact quote evidence.'})
            if q.status in ('pass','fail'):
                if q.support_type == 'absence' and q.status == 'fail' and not bad:
                    if call.coverage != 'complete':
                        q.status='unknown'; q.support_type='insufficient_evidence'
                        q.explanation='Absence cannot establish failure because transcript completeness is not confirmed.'
                    else:
                        q.explanation='Not observed in the reviewer-confirmed complete transcript. ' + q.explanation
                elif q.support_type != 'quote' or not supported(q):
                    q.status='unknown'; q.support_type='insufficient_evidence'
                    q.explanation='Finding withheld: it lacks valid exact quote evidence.'
            if bad:
                q.evidence = [e for e in q.evidence if valid(e)]
            if q.status == 'unknown':
                q.support_type='insufficient_evidence'
        result.quality_checks.append(q)
    n = sum(t.speaker=='customer' and not MISSING.search(t.text) for t in call.turns)
    if len(result.sentiment) < n:
        warnings.append(f'Sentiment evidence covers {len(result.sentiment)} of {n} customer turns; other turns are unassessed.')
    return result, warnings, rejected

def score_checks(analysis, criteria):
    weights = {c.id:c.weight for c in criteria if c.enabled}
    assessed = [q for q in analysis.quality_checks if q.status in ('pass','fail')]
    denom = sum(weights[q.criterion_id] for q in assessed)
    score = round(100 * sum(weights[q.criterion_id] for q in assessed if q.status=='pass') / denom,1) if denom else None
    eligible = [q for q in analysis.quality_checks if q.status!='not_applicable']
    coverage = round(100*len(assessed)/len(eligible),1) if eligible else None
    return {'score':score, 'assessment_coverage_percent':coverage, 'assessed':len(assessed), 'eligible':len(eligible)}

def evidence_offsets(analysis, call):
    lookup = {t.turn_id:t.text for t in call.turns}; spans=[]
    for group in ('issues','quality_checks','sentiment','risk_flags'):
        for idx, finding in enumerate(getattr(analysis,group)):
            for e in finding.evidence:
                start=lookup[e.turn_id].find(e.quote)
                if start>=0:
                    spans.append({'finding':f'{group}:{idx}','turn_id':e.turn_id,'quote':e.quote,'start':start,'end':start+len(e.quote)})
    return spans

def report_csv(report):
    """One finding per row; quote all fields and neutralize spreadsheet formulas."""
    buf=io.StringIO(); writer=csv.writer(buf)
    writer.writerow(['conversation_id','mode','finding_id','type','title','status','explanation','evidence_json','review_decision','review_notes'])
    a=report['analysis']; reviews=report.get('reviews',{})
    for kind in ('issues','quality_checks','risk_flags','sentiment'):
        for i,item in enumerate(a[kind]):
            fid=f'{kind}:{i}'; rev=reviews.get(fid,{})
            row=[report['conversation_id'],report['mode'],fid,kind,item.get('title',item.get('criterion_id',item.get('turn_id',''))),item.get('outcome',item.get('status',item.get('severity',item.get('label','')))),item['explanation'],json.dumps(item['evidence'],ensure_ascii=False),rev.get('decision','pending'),rev.get('notes','')]
            writer.writerow([("'"+str(v)) if str(v).lstrip().startswith(('=','+','-','@')) else str(v) for v in row])
    return ('\ufeff'+buf.getvalue()).encode('utf-8')
