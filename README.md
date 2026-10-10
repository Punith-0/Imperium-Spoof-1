# DynamicIQ — Team IMPERIUM

A compact call-transcript quality analyzer for your hackathon. Five Python files implement the app. Includes the 85-conversation mixed dataset and a no-key offline demo.

## Start on Windows with VS Code + Git Bash

Use Python 3.11 or 3.12. Extract the ZIP, open the DynamicIQ folder, then:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
python -m streamlit run app.py
```

If `python` is unavailable, use `py` to create the virtual environment. On macOS/Linux activate with `source .venv/bin/activate` instead. Open http://localhost:8501. The offline demo runs immediately without credentials. `cp -n` preserves an existing .env.

## Enable actual AI analysis

Edit `.env`:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Restart the app after editing environment settings. Select **AI analysis**, choose a conversation and click **Analyze conversation**. You can also enter a key in the sidebar for the current session. Do not commit `.env`. API requests go directly to OpenAI; account access and API billing are required. A model must support Chat Completions with strict JSON Schema outputs. No paid live API request was made during package verification; the HTTP contract, validation and error paths were tested using mocks.

Official API references used:
- https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat
- https://developers.openai.com/api/docs/models/gpt-4.1-mini

## Files

- `app.py`: Streamlit dashboard, configuration, uploads, review controls, evidence highlights and downloads.
- `analyzer.py`: structured AI request, safe errors, offline rules and analysis orchestration.
- `helpers.py`: parsers, exact-quote checks, completeness guards, scoring and CSV export.
- `schemas.py`: Pydantic input/output contracts.
- `test_project.py`: regression checks; no paid requests.
- `criteria.json`: editable QA criteria and weights. This is a team-defined checklist, not a claim of legal compliance.
- `.env.example`: credential configuration template.
- `samples/mixed_dataset.json`: 50 ABCD + 20 NatCS + 15 modified cases, including your original 3 ABCD records.
- `samples/normal.json`: small authored sample with an explicit resolution.
- `samples/edge_cases.json`: the 15 marked modified cases for convenient upload.

## Five-minute demo

1. Start in **Offline demo** and analyze an included conversation. Its keyword candidates and approximate sentiment are labelled as demo output; QA scores and issue outcomes remain unassessed.
2. Select **AI analysis** with a valid key for actual issue/outcome extraction, configurable checklist results, contextual sentiment and risk analysis.
3. Choose an `edge_missing_section_*` or `edge_abrupt_ending_*` conversation. The app detects gaps and does not score absent actions as failures.
4. Open **Transcript & evidence**. Select a finding to highlight its exact quote spans.
5. Accept/reject findings, add notes and download JSON/CSV from **Review & export**.

## Input formats

JSON supports:
- The delivered wrapper: `{"metadata": {...}, "conversations": [...]}`.
- A list of call records or one call object.
- Original ABCD records using `convo_id` and `original: [["agent", "..."], ...]`.
- Normalized records using `conversation_id` and `turns: [{"speaker":"customer", "text":"..."}]`.

CSV: one call per row, with `conversation_id` and either `turns_json` or `transcript`. The previously delivered mixed CSV works directly. TXT/paste: use `Agent:` / `Customer:` lines; unlabelled continuations stay with the previous turn. A wholly unlabelled transcript keeps speaker `unknown`; no customer sentiment is guessed from it.

Turn IDs are normalized to t001, t002, etc. Missing/null text becomes a visible gap marker. Source metadata, expected edge-case checks and source intent labels are never sent to the model. Action events are kept as `system`, not customer speech. There is no automatic audio transcription or PDF ingestion.

Limits: 10 MB upload, 500 records per file, 200 turns and 40,000 characters per call. Oversized inputs are rejected, never silently truncated. One selected call is analyzed per click; batch AI analysis is not implemented.

## Evidence, scoring and limitations

Each issue/risk must contain exact substrings from the specified transcript turns. Invalid evidence is withheld and listed in the report. Sentiment is restricted to customer turns. Exact text matching verifies citation existence, not whether the quoted text logically supports the interpretation; review is still needed.

Checklist outcomes: pass, fail, unknown, not_applicable. Missing or duplicate checklist results become unknown. Absence-based failures require a reviewer-confirmed complete transcript. Known gaps override that confirmation. Score = 100 × passed weight / (passed + failed weight). Unknown and not_applicable checks are excluded, and assessment coverage is shown alongside the score. A high score with low coverage is not a full-call assessment.

Offline demo uses regex topics and a basic positive/negative word list. It does not evaluate full QA semantics, infer resolution, or claim model accuracy. It intentionally leaves all criteria unknown, while showing evidence candidates and approximate sentiment.

Reviewer decisions and notes are stored in the current browser session and included in report downloads. Decisions do not mutate the original generated score. Only the last 10 analyses are retained in session; restart/reconnect may clear them. Download to persist. No authentication, production database or multi-team access control is included in this hackathon app.

Original dataset QA outcomes are not gold-labelled. The 15 modified cases contain authored test expectations, not independent validation. Keep parent/modified records together using the dataset's group_id if training or splitting. Preserve dataset licenses in `samples/` when redistributing. Source text is not automatically anonymized.

## Test

```bash
python -m unittest -v test_project.py
```

Tests check all 85 samples, exact-quote rejection, incomplete evidence, unknown speakers, malformed input, scoring, CSV round-trips, API error handling and prevention of evaluation-label leakage. Tests mock AI responses rather than claiming a live API test.
