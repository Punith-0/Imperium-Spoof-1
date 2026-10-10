"""Validated input and AI output contracts. No UI dependencies."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Turn(StrictModel):
    turn_id: str
    speaker: Literal['agent', 'customer', 'system', 'unknown']
    text: str

class Conversation(StrictModel):
    conversation_id: str
    turns: list[Turn]
    coverage: Literal['unknown', 'complete', 'incomplete'] = 'unknown'
    warnings: list[str] = Field(default_factory=list)
    source: str = 'upload'
    record_type: str = 'source'

class Criterion(StrictModel):
    id: str
    name: str
    description: str
    weight: float = Field(default=1, gt=0, le=100)
    enabled: bool = True
    demo_patterns: list[str] = Field(default_factory=list)

class Evidence(StrictModel):
    turn_id: str
    quote: str

class Issue(StrictModel):
    title: str
    outcome: Literal['resolved', 'unresolved', 'pending', 'unknown']
    explanation: str
    evidence: list[Evidence]

class QualityCheck(StrictModel):
    criterion_id: str
    status: Literal['pass', 'fail', 'unknown', 'not_applicable']
    explanation: str
    support_type: Literal['quote', 'absence', 'insufficient_evidence', 'not_applicable']
    evidence: list[Evidence]

class Sentiment(StrictModel):
    turn_id: str
    label: Literal['negative', 'neutral', 'positive']
    explanation: str
    evidence: list[Evidence]

class Risk(StrictModel):
    title: str
    severity: Literal['low', 'medium', 'high']
    explanation: str
    evidence: list[Evidence]

class Analysis(StrictModel):
    issues: list[Issue]
    quality_checks: list[QualityCheck]
    sentiment: list[Sentiment]
    risk_flags: list[Risk]
