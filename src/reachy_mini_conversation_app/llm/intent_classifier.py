"""Pattern-based intent classifier for LLM routing.

Classifies user messages into intent categories to decide which
LLM configuration to use.  Uses keyword/pattern matching (no ML
overhead).  Upgrade to a small classifier model later if needed.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import NamedTuple


class Intent(Enum):
    CASUAL = "casual"       # greetings, small talk, acknowledgments
    COMMAND = "command"      # tool calls, robot control
    REASONING = "reasoning"  # complex questions, explanations
    CREATIVE = "creative"   # stories, jokes, creative tasks


class Classification(NamedTuple):
    intent: Intent
    confidence: float  # 0.0–1.0


# Pattern definitions: (compiled regex, intent, confidence)
_CASUAL_PATTERNS = [
    re.compile(r"^\s*(hi|hey|hello|yo|sup|what'?s up|hola|bonjour)\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(thanks|thank you|cheers|thx|ty)\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(yeah|yep|yup|nah|nope|ok|okay|sure|cool|nice|great|good)\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(bye|goodbye|see ya|later|ciao)\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*how are you\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*what'?s your name\s*[.!?]*\s*$", re.I),
    re.compile(r"^\s*(good morning|good night|good evening|good afternoon)\s*[.!?]*\s*$", re.I),
]

_COMMAND_PATTERNS = [
    re.compile(r"\b(dance|move|look|turn|stop|camera|photo|picture|snapshot)\b", re.I),
    re.compile(r"\b(play|show|express|emotion|feel|happy|sad|angry|surprised)\b", re.I),
    re.compile(r"\b(remember|forget|recall|store|memory)\b", re.I),
    re.compile(r"\b(head|nod)\b", re.I),
    re.compile(r"\b(look|move|turn)\s+(left|right|up|down)\b", re.I),
]

_REASONING_PATTERNS = [
    re.compile(r"\b(explain|describe|analyze|compare|why|how does|what is|define|difference between)\b", re.I),
    re.compile(r"\b(calculate|compute|solve|prove|derive)\b", re.I),
    re.compile(r"\b(think about|consider|evaluate|assess)\b", re.I),
]

_CREATIVE_PATTERNS = [
    re.compile(r"\b(story|joke|poem|song|write|compose|imagine|pretend|roleplay)\b", re.I),
    re.compile(r"\b(creative|invent|make up|fictional)\b", re.I),
]


def classify(text: str) -> Classification:
    """Classify user text into an intent category.

    Returns the highest-confidence match, defaulting to REASONING
    for ambiguous or long messages.
    """
    text = text.strip()
    if not text:
        return Classification(Intent.CASUAL, 0.5)

    # Short messages → likely casual
    word_count = len(text.split())

    # Check casual patterns first (exact match on short messages)
    for pattern in _CASUAL_PATTERNS:
        if pattern.match(text):
            return Classification(Intent.CASUAL, 0.9)

    # Very short non-greeting → check all categories before defaulting
    if word_count <= 3:
        for pattern in _COMMAND_PATTERNS:
            if pattern.search(text):
                return Classification(Intent.COMMAND, 0.8)
        for pattern in _CREATIVE_PATTERNS:
            if pattern.search(text):
                return Classification(Intent.CREATIVE, 0.8)
        for pattern in _REASONING_PATTERNS:
            if pattern.search(text):
                return Classification(Intent.REASONING, 0.8)
        return Classification(Intent.CASUAL, 0.7)

    # Check command patterns
    command_hits = sum(1 for p in _COMMAND_PATTERNS if p.search(text))
    if command_hits >= 1:
        return Classification(Intent.COMMAND, min(0.6 + command_hits * 0.1, 0.9))

    # Check creative patterns
    for pattern in _CREATIVE_PATTERNS:
        if pattern.search(text):
            return Classification(Intent.CREATIVE, 0.8)

    # Check reasoning patterns
    for pattern in _REASONING_PATTERNS:
        if pattern.search(text):
            return Classification(Intent.REASONING, 0.8)

    # Long messages default to reasoning
    if word_count > 10:
        return Classification(Intent.REASONING, 0.6)

    # Medium-length, no strong signal → casual
    return Classification(Intent.CASUAL, 0.5)
