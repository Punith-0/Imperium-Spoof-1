"""Run: python -m streamlit run app.py"""
import base64
import hashlib
import html
import io
import json
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from analyzer import analyze, AnalysisError
from helpers import load_conversations, load_criteria, fingerprint, report_csv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
API_KEY = os.getenv('OPENAI_API_KEY', '').strip()  # read from .env only; never shown in the UI
SAMPLES = ROOT / 'samples' / 'mixed_dataset.json'
STORE = ROOT / 'reports'  # saved reviews + done-status (already in .gitignore)
STATUS_FILE = STORE / 'review_status.json'
FINDING_KINDS = ('issues', 'quality_checks', 'risk_flags', 'sentiment')


# ───────────────────────── logo ─────────────────────────
@lru_cache(maxsize=1)
def load_logo():
    """Return (base64 PNG, favicon). Background is made transparent so the logo sits on the dark header."""
    try:
        rgb = np.asarray(Image.open(ROOT / 'logo.jpeg').convert('RGB')).astype('float32')
        peak = rgb.max(axis=2)
        ys, xs = np.where(peak > 110)
        pad = 14
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, rgb.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, rgb.shape[1])
        rgb, peak = rgb[y0:y1, x0:x1], peak[y0:y1, x0:x1]
        alpha = np.clip((peak - 60) / 70, 0, 1) * 255
        logo = Image.fromarray(np.dstack([rgb, alpha]).astype('uint8'))  # (H, W, 4) uint8 → RGBA
        # favicon = the "D" speech-bubble mark (left ~31% of the logo) on the brand navy
        mark = logo.crop((0, 0, int(logo.width * 0.31), logo.height))
        side = max(mark.size) + 30
        icon = Image.new('RGBA', (side, side), (11, 26, 46, 255))
        icon.alpha_composite(mark, ((side - mark.width) // 2, (side - mark.height) // 2))
        # header logo: shrink to ~1.7x display size so the page stays light
        logo = logo.resize((round(logo.width * 160 / logo.height), 160), Image.LANCZOS)
        buf = io.BytesIO()
        logo.save(buf, 'PNG', optimize=True)
        return base64.b64encode(buf.getvalue()).decode(), icon.convert('RGB').resize((128, 128), Image.LANCZOS)
    except Exception:
        return None, None


LOGO_B64, FAVICON = load_logo()
st.set_page_config(page_title='DynamicIQ · Transcript Intelligence', page_icon=FAVICON or '🎧', layout='wide')

# ───────────────────────── styles ─────────────────────────
st.markdown('''<style>
.stApp {background:#f3f6fb;color:#14213d}
.block-container {padding-top:2.6rem;padding-bottom:3rem;max-width:1560px}
footer {visibility:hidden}
[data-testid="stAppDeployButton"], .stDeployButton {display:none}
h1,h2,h3 {letter-spacing:-.025em}
[data-testid="stSidebar"] {background:#eaf0f9;border-right:1px solid #d6e0ee}
.side-title {font-size:.78rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#4a5f80;margin:.4rem 0 .2rem}

/* header banner */
.hero {display:flex;align-items:center;justify-content:space-between;gap:24px;padding:18px 30px;border-radius:18px;margin-bottom:18px;
  background:radial-gradient(900px 260px at 12% 0%,#1d3f6b 0%,#0e2038 55%,#0a1626 100%);box-shadow:0 10px 28px rgba(10,22,38,.28)}
.hero img {height:92px;display:block}
.hero-name {font-size:2rem;font-weight:800;color:#fff}
.hero-meta {min-width:290px;text-align:right;color:#c9daf2;font-size:.95rem}
.hero-meta .tag {font-size:.72rem;letter-spacing:.14em;color:#7fa0cc;font-weight:700;margin-bottom:4px}
.hero-meta b {font-size:1.7rem;color:#fff}
.hero-bar {height:8px;border-radius:99px;background:rgba(255,255,255,.14);overflow:hidden;margin-top:8px}
.hero-bar i {display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#4f7bff,#4fe0c8)}

/* cards */
.card {background:#fff;border:1px solid #dce4ef;border-radius:14px;padding:16px 20px;margin-bottom:14px;box-shadow:0 1px 2px rgba(20,33,61,.04)}
.head {display:flex;align-items:center;justify-content:space-between;gap:16px}
.eyebrow {font-size:.72rem;font-weight:700;letter-spacing:.12em;color:#7a8ca8}
.head .title {font-size:1.7rem;font-weight:800;letter-spacing:-.03em;line-height:1.15}
.head .meta {color:#5b6f8c;font-size:.9rem;margin-top:2px}
.pill {padding:5px 14px;border-radius:99px;font-size:.82rem;font-weight:700;white-space:nowrap}
.pill.done {background:#dcf5e8;color:#0d7a4d}
.pill.review {background:#e3ebff;color:#2b4fc4}
.pill.new {background:#eceff5;color:#5b6f8c}
.panel-title {font-size:1.15rem;font-weight:800;letter-spacing:-.02em;margin-bottom:-.2rem}
.welcome {text-align:center;padding:48px 20px;color:#41557a}
.welcome h3 {margin-bottom:.4rem}

/* controls */
[data-testid="stMetric"] {background:#fff;border:1px solid #dce4ef;border-radius:12px;padding:14px 16px}
[data-testid="stMetricValue"] {font-size:1.6rem}
[data-testid="stExpander"] {background:#fff;border:1px solid #dce4ef;border-radius:12px}
[data-testid="stForm"] {background:#f6f9fd;border:1px solid #e1e8f3;border-radius:10px;padding:.7rem .9rem}
.stTabs [data-baseweb="tab"] {font-weight:600;padding:.6rem 1rem}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {background:linear-gradient(90deg,#3b6cf0,#16b5a5);border:none;color:#fff;font-weight:600}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {filter:brightness(1.08);color:#fff}
.st-key-mark_done button {background:linear-gradient(90deg,#10b981,#0d9f6e) !important;color:#fff !important;border:none !important;font-weight:700}

/* conversation list (right panel) */
div[class*="st-key-cv_"] button {justify-content:flex-start;text-align:left;font-weight:600;border-radius:10px;min-height:2.5rem}
div[class*="st-key-cv_done_"] button {background:#e7f7ef;border:1px solid #b7e5cd;color:#0d6e47}
div[class*="st-key-cv_done_"] button:hover {background:#d8f1e5;border-color:#8fd5b3;color:#0d6e47}

/* transcript turns */
.turn {border:1px solid #dce4ef;border-radius:10px;padding:12px 16px;margin:8px 0;background:#fff;color:#14213d}
.turn.customer {border-left:4px solid #2563eb}
.turn.agent {border-left:4px solid #10a58a}
.turn.system {border-left:4px solid #e5a334}
.turn.unknown {border-left:4px solid #94a3b8}
.turn small {color:#536782;font-weight:600;letter-spacing:.04em}
mark {background:#ffe69a;color:#1d2939;padding:1px 2px;border-radius:3px}
</style>''', unsafe_allow_html=True)


# ───────────────────────── persistence (done status) ─────────────────────────
def read_status():
    try:
        return json.loads(STATUS_FILE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def write_status(data):
    STORE.mkdir(exist_ok=True)
    tmp = STATUS_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(STATUS_FILE)


def load_saved_report(uid):
    try:
        return json.loads((STORE / f'{uid}.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def convo_uid(c):
    """Stable ID from the transcript itself, so a done-mark follows the conversation across reloads."""
    raw = c.conversation_id + '\x1f' + '\x1e'.join(f'{t.speaker}:{t.text}' for t in c.turns)
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]


def store_report(run_key, report):
    reports = st.session_state.reports
    if run_key not in reports and len(reports) >= 10:  # bounded per-session cache
        del reports[next(iter(reports))]
    reports[run_key] = report


# ───────────────────────── callbacks ─────────────────────────
def select(i):
    if i is not None:
        st.session_state.selected = i


def mark_done(uid, run_key, label, next_idx):
    report = st.session_state.reports.get(run_key)
    if report is None:
        return
    STORE.mkdir(exist_ok=True)
    (STORE / f'{uid}.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    status = read_status()
    status[uid] = {'conversation_id': report['conversation_id'], 'done_at': datetime.now().astimezone().isoformat(timespec='seconds')}
    write_status(status)
    if next_idx is not None:
        st.session_state.selected = next_idx
    st.toast(f'{label} marked as done', icon='✅')


def reopen(uid, run_key, label):
    status = read_status()
    if status.pop(uid, None) is not None:
        write_status(status)
    saved = load_saved_report(uid)
    if saved:
        store_report(run_key, saved)
    st.toast(f'{label} reopened for review', icon='↩️')


# ───────────────────────── small UI helpers ─────────────────────────
def hero(done, total):
    logo = f'<img src="data:image/png;base64,{LOGO_B64}" alt="DynamicIQ">' if LOGO_B64 else '<span class="hero-name">DynamicIQ</span>'
    if total:
        meta = (f'<div class="hero-meta"><div class="tag">TEAM IMPERIUM · REVIEW PROGRESS</div>'
                f'<div><b>{done}</b> / {total} conversations done</div>'
                f'<div class="hero-bar"><i style="width:{round(100 * done / total)}%"></i></div></div>')
    else:
        meta = '<div class="hero-meta"><div class="tag">TEAM IMPERIUM</div><div>Call quality review</div></div>'
    st.markdown(f'<div class="hero">{logo}{meta}</div>', unsafe_allow_html=True)


def turn_html(t, spans=()):
    intervals = sorted((s['start'], s['end']) for s in spans if s['turn_id'] == t.turn_id)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    pieces, cursor = [], 0
    for start, end in merged:
        pieces.extend([html.escape(t.text[cursor:start]), '<mark>' + html.escape(t.text[start:end]) + '</mark>'])
        cursor = end
    pieces.append(html.escape(t.text[cursor:]))
    body = ''.join(pieces).replace('\r', '').replace('\n', '<br>').replace('$', '&#36;')
    return f'<div class="turn {t.speaker}"><small>{t.turn_id} · {t.speaker.upper()}</small><br>{body}</div>'


DECISIONS = {'pending': '⏳ Pending', 'accept': '✅ Accept', 'reject': '❌ Reject', 'needs_review': '🔎 Needs review'}


# ═════════════════════════ sidebar: setup ═════════════════════════
calls, load_error = None, None
with st.sidebar:
    st.markdown('<div class="side-title">1 · Transcripts</div>', unsafe_allow_html=True)
    sources = (['Included samples'] if SAMPLES.exists() else []) + ['Upload JSON / CSV / TXT', 'Paste transcript']
    source = st.radio('Transcript source', sources, label_visibility='collapsed')
    try:
        if source == 'Included samples':
            calls = load_conversations(SAMPLES.read_bytes(), 'mixed_dataset.json')
        elif source == 'Upload JSON / CSV / TXT':
            uploaded = st.file_uploader('Choose a transcript or dataset', type=['json', 'csv', 'txt'])
            if uploaded is not None:
                calls = load_conversations(uploaded.getvalue(), uploaded.name)
        else:
            pasted = st.text_area('Use Agent: and Customer: labels', height=160, placeholder='Agent: Hello, how can I help?\nCustomer: My refund has not arrived.')
            if pasted.strip():
                calls = load_conversations(pasted.encode(), 'pasted_call.txt')
    except Exception as e:
        load_error = f'Cannot read input: {e}'

    st.markdown('<div class="side-title">2 · Analysis</div>', unsafe_allow_html=True)
    mode_label = st.radio('Mode', ['Offline demo', 'AI analysis'], horizontal=True, label_visibility='collapsed')
    mode = 'demo' if mode_label == 'Offline demo' else 'ai'
    if mode == 'ai':
        model = st.text_input('Model', value=os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'))
        if API_KEY:
            st.success('API key loaded from .env', icon=':material/key:')
        else:
            st.error('No API key found. Add OPENAI_API_KEY to your .env file and restart the app.', icon=':material/key_off:')
        st.caption('Analyze sends the selected transcript and criteria to the API. Usage may incur charges.')
    else:
        model = 'offline_keyword_rules'
        st.info('Offline mode uses keyword rules. It is a UI demo, not an AI quality assessment.')

    st.markdown('<div class="side-title">3 · Review</div>', unsafe_allow_html=True)
    with st.expander('Review checklist'):
        st.caption('Review checklist only; these ticks do not alter QA scores.')
        for label in ['Issues and outcomes', 'Quality checklist', 'Sentiment changes', 'Risk flags', 'Quote evidence', 'Missing sections']:
            st.checkbox(label, key='req_' + label)
    st.caption('Limits: 10 MB/file · 500 calls/file · 200 turns and 40,000 characters/call. One call is analyzed at a time.')

st.session_state.setdefault('reports', {})

# ═════════════════════════ nothing loaded yet ═════════════════════════
if calls is None:
    hero(0, 0)
    if load_error:
        st.error(load_error)
    else:
        st.markdown('<div class="card welcome"><h3>Load a transcript to get started</h3>'
                    '<div>Use the sidebar to upload a JSON / CSV / TXT dataset or paste a speaker-labelled transcript.<br>'
                    'Every conversation will then appear on the right as Convo 1, Convo 2, …</div></div>', unsafe_allow_html=True)
    st.stop()

# ═════════════════════════ conversation list state ═════════════════════════
uids = [convo_uid(c) for c in calls]
signature = hashlib.sha1('|'.join(uids).encode()).hexdigest()
if st.session_state.get('dataset_signature') != signature:  # new dataset → start at the top
    st.session_state.dataset_signature = signature
    st.session_state.selected = 0
sel = min(max(st.session_state.get('selected', 0), 0), len(calls) - 1)
status = read_status()
done_count = sum(u in status for u in uids)


def next_pending(after):
    for step in range(1, len(uids)):
        j = (after + step) % len(uids)
        if uids[j] not in status:
            return j
    return None


hero(done_count, len(calls))
work, side = st.columns([3.3, 1], gap='large')

# ═════════════════════════ right: all conversations ═════════════════════════
with side:
    st.markdown('<div class="panel-title">Conversations</div>', unsafe_allow_html=True)
    st.caption(f'{len(calls) - done_count} pending · {done_count} done')
    view = st.radio('Show', ['All', 'Pending', 'Done'], horizontal=True, label_visibility='collapsed', key='list_filter')
    nxt = next_pending(sel)
    st.button('Next pending conversation', on_click=select, args=(nxt,), disabled=nxt is None, icon=':material/skip_next:', use_container_width=True, key='next_pending')
    with st.container(height=560, border=True):
        shown = [i for i, u in enumerate(uids) if view == 'All' or (view == 'Done') == (u in status)]
        if not shown:
            st.caption('Nothing to show here.')
        for i in shown:
            is_done = uids[i] in status
            state = 'sel' if i == sel else ('done' if is_done else 'open')
            st.button(f'Convo {i + 1}', key=f'cv_{state}_{i}', on_click=select, args=(i,), use_container_width=True,
                      type='primary' if i == sel else 'secondary',
                      icon=':material/check_circle:' if is_done else ':material/radio_button_unchecked:',
                      help=f'{calls[i].conversation_id} · {len(calls[i].turns)} turns')
    if done_count == len(calls):
        st.success('All conversations are done.')

# ═════════════════════════ left: the review workspace ═════════════════════════
with work:
    head_slot = st.container()  # filled once the report state is known, so the header card stays on top
    with st.expander('⚙ Quality criteria (advanced)'):
        criteria_text = st.text_area('Edit checklist JSON', value=(ROOT / 'criteria.json').read_text(), height=260)
        st.caption('Weights apply only to assessed pass/fail checks. Unknown and not-applicable results are excluded. Changes here apply to this session.')
        st.download_button('Download criteria', criteria_text, 'criteria.json', 'application/json')
    try:
        criteria = load_criteria(criteria_text)
    except Exception as e:
        st.error(f'Invalid criteria: {e}')
        st.stop()

    call = calls[sel].model_copy(deep=True)
    uid = uids[sel]
    label = f'Convo {sel + 1}'
    is_done = uid in status

    # Key includes full input so changing an upload cannot carry a stale completeness confirmation.
    input_key = fingerprint(call, criteria, 'input', '')[:16]
    complete = st.checkbox('I confirm this transcript contains the full conversation.', value=call.coverage == 'complete',
                           disabled=call.coverage == 'incomplete' or is_done, key='coverage_' + input_key)
    if complete and call.coverage != 'incomplete':
        call.coverage = 'complete'
    run_key = fingerprint(call, criteria, mode, model)
    report = load_saved_report(uid) if is_done else st.session_state.reports.get(run_key)

    pill = '<span class="pill done">✓ Done</span>' if is_done else ('<span class="pill review">In review</span>' if report else '<span class="pill new">Not analyzed</span>')
    head_slot.markdown(f'<div class="card head"><div><div class="eyebrow">NOW REVIEWING</div><div class="title">{label}</div>'
                       f'<div class="meta">{html.escape(call.conversation_id)} · {html.escape(call.source)} · {html.escape(call.record_type)} · {len(call.turns)} turns</div></div>{pill}</div>',
                       unsafe_allow_html=True)
    if call.coverage == 'incomplete':
        st.warning('Missing transcript content detected. Absence-based checks remain unknown.')
    for w in call.warnings:
        st.caption(w)

    # ── action row ──
    b1, b2, _ = st.columns([1.2, 1.2, 1.6])
    analyze_clicked = b1.button('Analyze conversation', type='primary', disabled=is_done, use_container_width=True, icon=':material/bolt:')
    if is_done:
        b2.button('Reopen for review', on_click=reopen, args=(uid, run_key, label), use_container_width=True, icon=':material/undo:', key='reopen')
    elif report is not None:
        pending_left = sum(len(report['analysis'][k]) for k in FINDING_KINDS) - sum(1 for r in report['reviews'].values() if r.get('decision', 'pending') != 'pending')
        b2.button('Mark as done', on_click=mark_done, args=(uid, run_key, label, next_pending(sel)), use_container_width=True,
                  icon=':material/check_circle:', key='mark_done',
                  help='Saves this review and locks the conversation so it is not revisited.' + (f' {pending_left} finding(s) are still pending.' if pending_left > 0 else ''))

    if analyze_clicked:
        if mode == 'ai' and not API_KEY:
            st.error('No API key found. Add OPENAI_API_KEY to your .env file and restart the app.')
        else:
            ok = False
            with st.spinner('Checking transcript and verifying quote evidence…'):
                try:
                    store_report(run_key, analyze(call, criteria, mode, API_KEY, model))
                    ok = True
                except AnalysisError as e:
                    st.error(str(e))
                except Exception:
                    st.error('Analysis failed unexpectedly. Check your configuration and retry. No incomplete report was saved.')
            if ok:
                st.rerun()

    if is_done:
        when = datetime.fromisoformat(status[uid]['done_at']).strftime('%d %b %Y, %H:%M') if status.get(uid, {}).get('done_at') else ''
        st.success(f'{label} was marked done {("on " + when) if when else ""}. It is locked, so the reviewer does not need to revisit it. Use “Reopen for review” to change it.')
        if report is None:
            st.warning('The saved report file was not found in the reports/ folder.')

    # ── nothing analyzed yet: show the transcript ──
    if report is None:
        st.caption('Click Analyze conversation to start. Changing criteria, mode or completeness requires a new analysis.')
        st.markdown('#### Transcript preview')
        with st.container(height=520, border=False):
            st.markdown(''.join(turn_html(t) for t in call.turns), unsafe_allow_html=True)
        st.stop()

    # ═════════════ review dashboard ═════════════
    locked = is_done
    a, m = report['analysis'], report['metrics']
    names = {c.id: c.name for c in criteria}
    turn_lookup = {t.turn_id: t for t in call.turns}

    st.subheader('Review dashboard')
    st.caption(f'Generated {report["generated_at"]} · {report["model"]}')
    for w in report['warnings']:
        st.warning(w)
    cols = st.columns(4)
    cols[0].metric('QA score', f'{m["score"]} / 100' if m['score'] is not None else 'Not assessed')
    cols[1].metric('Checks assessed', f'{m["assessed"]} / {m["eligible"]}')
    cols[2].metric('Issues / candidates', len(a['issues']))
    cols[3].metric('Risk flags', len(a['risk_flags']))
    st.caption('Score covers assessed checks only; unknown and not-applicable checks are excluded. Reviewer decisions are recorded separately and do not silently change the score.')
    total_findings = sum(len(a[k]) for k in FINDING_KINDS)
    reviewed = sum(1 for r in report['reviews'].values() if r.get('decision', 'pending') != 'pending')
    if total_findings:
        st.progress(min(reviewed / total_findings, 1.0), text=f'Review progress: {reviewed} of {total_findings} findings have a decision')

    def evidence_view(evidence):
        if not evidence:
            st.caption('No quoted evidence. Read the explanation and completeness status.')
        for e in evidence:
            st.caption(f'Exact text match · {e["turn_id"]} · {turn_lookup[e["turn_id"]].speaker}')
            st.text(e['quote'])

    def review_control(kind, i):
        """Decision + remark form. Nothing is stored until the Save button (or Enter) is used."""
        fid = f'{kind}:{i}'
        # Generated timestamp prevents widget values leaking into a later rerun's new findings.
        widget = f'{run_key}:{report["generated_at"]}:{fid}'
        saved = report['reviews'].get(fid)
        options = list(DECISIONS)
        with st.form(widget + ':form', border=False):
            c1, c2, c3 = st.columns([1.1, 2.4, 0.8])
            decision = c1.selectbox('Reviewer decision', options, index=options.index(saved['decision']) if saved else 0,
                                    format_func=DECISIONS.get, key=widget + ':decision', label_visibility='collapsed', disabled=locked)
            note = c2.text_input('Reviewer remark', value=saved['notes'] if saved else '', key=widget + ':note', label_visibility='collapsed',
                                 placeholder='Add a remark, then press Enter or click Save', disabled=locked)
            if c3.form_submit_button('Save', use_container_width=True, disabled=locked):
                report['reviews'][fid] = {'decision': decision, 'notes': note}
                saved = report['reviews'][fid]
        if saved:
            st.caption(f'✓ Saved · {DECISIONS[saved["decision"]]}' + (f' · “{saved["notes"]}”' if saved['notes'] else ''))

    issues_tab, qa_tab, sent_tab, risk_tab, transcript_tab, export_tab = st.tabs(
        ['Issues & outcomes', 'Quality checks', 'Sentiment', 'Risk flags', 'Transcript & evidence', 'Review & export'])

    with issues_tab:
        if not a['issues']:
            st.info('No evidence-backed issues were returned. This does not prove the call is issue-free.')
        for i, item in enumerate(a['issues']):
            with st.expander(f'{item["title"]} · {item["outcome"].upper()}', expanded=True):
                st.write(item['explanation'])
                evidence_view(item['evidence'])
                review_control('issues', i)

    with qa_tab:
        for i, item in enumerate(a['quality_checks']):
            with st.expander(f'{names.get(item["criterion_id"], item["criterion_id"])} · {item["status"].upper()}', expanded=True):
                st.write(item['explanation'])
                st.caption('Support: ' + item['support_type'])
                evidence_view(item['evidence'])
                review_control('quality_checks', i)
        st.subheader('Coaching prompts')
        failing = [q for q in a['quality_checks'] if q['status'] == 'fail']
        if failing:
            for q in failing:
                st.write('• Review ' + names.get(q['criterion_id'], q['criterion_id']) + ': ' + q['explanation'])
        else:
            st.caption('No assessed failing checks. Review unknowns before deciding whether coaching is needed.')

    with sent_tab:
        st.caption('Customer turns only. -1 = negative, 0 = neutral, +1 = positive. These are categorical positions, not measured emotional intensity.')
        if a['sentiment']:
            values = {'negative': -1, 'neutral': 0, 'positive': 1}
            points = {s['turn_id']: s for s in a['sentiment']}
            # Null values retain unassessed gaps instead of drawing a false continuous series.
            frame = pd.DataFrame([{'turn': int(t.turn_id[1:]), 'sentiment': values[points[t.turn_id]['label']] if t.turn_id in points else None}
                                  for t in call.turns if t.speaker == 'customer']).set_index('turn')
            st.line_chart(frame)
            st.dataframe(pd.DataFrame([{'turn_id': s['turn_id'], 'label': s['label'], 'explanation': s['explanation']} for s in a['sentiment']]),
                         width='stretch', hide_index=True)
            for i, s in enumerate(a['sentiment']):
                with st.expander(f'{s["turn_id"]} · {s["label"]}'):
                    evidence_view(s['evidence'])
                    review_control('sentiment', i)
        else:
            st.info('No valid customer sentiment findings.')

    with risk_tab:
        if not a['risk_flags']:
            st.info('No evidence-backed risk flags returned. This is not a compliance certificate.')
        for i, item in enumerate(a['risk_flags']):
            with st.expander(f'{item["severity"].upper()} · {item["title"]}', expanded=True):
                st.write(item['explanation'])
                evidence_view(item['evidence'])
                review_control('risk_flags', i)

    with transcript_tab:
        st.caption(report['verification_note'])
        options = ['All findings'] + [f'{kind}:{i}' for kind in FINDING_KINDS for i in range(len(a[kind]))]
        highlight = st.selectbox('Highlight evidence for', options)
        spans = [s for s in report['evidence_offsets'] if highlight == 'All findings' or s['finding'] == highlight]
        st.markdown(''.join(turn_html(t, spans) for t in call.turns), unsafe_allow_html=True)

    with export_tab:
        with st.form(run_key + report['generated_at'] + ':overall_form', border=False):
            overall = st.text_area('Overall reviewer notes', value=report.get('review_notes', ''), disabled=locked)
            if st.form_submit_button('Save overall notes', disabled=locked):
                report['review_notes'] = overall
        if report.get('review_notes'):
            st.caption('✓ Overall notes saved')
        st.caption('Session results are kept while this browser session is active. Marking a conversation done saves it to the reports/ folder. Only the last 10 analyses are held in the session.')
        if report['rejected_findings']:
            st.warning(f'{len(report["rejected_findings"])} unsupported findings were withheld.')
            st.dataframe(pd.DataFrame(report['rejected_findings']), width='stretch', hide_index=True)
        safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', call.conversation_id)[:80]
        st.download_button('Download full JSON report', json.dumps(report, indent=2, ensure_ascii=False), f'{safe_name}_report.json', 'application/json')
        st.download_button('Download findings CSV', report_csv(report), f'{safe_name}_findings.csv', 'text/csv')
        st.caption('JSON includes transcript, criteria, score, evidence offsets and reviewer decisions. CSV contains one finding per row. Neither export contains your API key.')