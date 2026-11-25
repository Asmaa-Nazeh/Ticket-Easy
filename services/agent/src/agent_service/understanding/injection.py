"""Detect instruction hierarchy attacks and attempts to extract private prompts or tool details.

Nothing is blocked here: the checks in front of every action do not depend on what the customer or the model believes.
This only lets the reply say plainly, in a fixed sentence, that the rules cannot be changed, so a customer is not left
thinking the agent agreed. The words are matched on the normalized text, in English, Egyptian Arabic and Arabizi.
"""

import re

from agent_service.understanding.text import normalize

_VERB_EN = r"(?:ignore|disregard|forget|override|bypass|skip|drop|break)"
_THING_EN = r"(?:rules?|instructions?|polic(?:y|ies)|guidelines?|restrictions?|limits?|prompt|programming)"
_VERB_AR = r"(?:تجاهل|تجاهلي|انسي|انسى|اهمل|اهملي|خالف|اكسر|تخطي)"
_THING_AR = r"(?:القواعد|قواعدك|التعليمات|تعليماتك|القوانين|قوانينك|السياسات|السياسه|سياستك|الحدود)"
_VERB_FRANCO = r"(?:tagahal|tegahel|egahel|ensa|nsa|ensy|ehmel|ihmel|khaleff)"
_THING_FRANCO = r"(?:2awa3ed|qawa3ed|el 2awa3ed|ta3limat|el ta3limat|2awanin|siyasat|el siyasa|rules|instructions)"

_PATTERNS = tuple(
    re.compile(p)
    for p in (
        rf"\b{_VERB_EN}\b.{{0,30}}\b{_THING_EN}\b",
        r"\byou are now\b",
        r"\b(?:system|developer) (?:prompt|message|mode)\b",
        r"\bpretend (?:you|to be|that)\b",
        r"\bjailbreak\b|\bdan mode\b",
        r"\b(?:you are now|act as|become|switch to)\s+(?:dan|developer|administrator|admin|system|unrestricted|unfiltered)\b",
        r"\b(?:system|developer|admin(?:istrator)?)\s+(?:override|message|instruction|mode|prompt)\b",
        r"\b(?:reveal|show|print|repeat|dump|leak|extract|translate|summari[sz]e)\b.{0,40}\b(?:system|developer|hidden|initial)\s+(?:prompt|instructions?|message)\b",
        r"\b(?:quote|repeat|print|show|tell me|give me|translate|encode)\b.{0,50}\b(?:everything|all|previous|prior|hidden|initial|system|developer)\b.{0,25}\b(?:above|prompt|instructions?|messages?|text)\b",
        r"\b(?:list|show|expose|describe|dump|extract)\b.{0,35}\b(?:tools?|functions?|mcp|connectors?|credentials?|secrets?|api keys?)\b",
        r"\b(?:what|which|name|enumerate|describe|tell me)\b.{0,35}\b(?:tools?|functions?|mcp)\b.{0,35}\b(?:you have access to|you can call|available to you|your)\b",
        r"\b(?:ignore|bypass|skip|disable|override)\b.{0,40}\b(?:identity|verification|authentication|ownership|authorization|policy gate|safety check)\b",
        r"\b(?:i am|this is)\s+(?:the\s+)?(?:system administrator|administrator|admin|developer|operator)\b",
        r"\b(?:for the next|from now on)\b.{0,35}\b(?:ignore|forget|override|bypass|disable)\b",
        rf"{_VERB_AR}.{{0,25}}{_THING_AR}",
        r"انت دلوقتي (?:مش|بقيت|حر)",
        rf"\b(?:{_VERB_FRANCO}|{_VERB_EN})\b.{{0,25}}\b(?:{_THING_FRANCO}|{_THING_EN})\b",
        r"\b(?:enta|inta)\s+(?:(?:delwa2ty|el)\s+)?(?:dan|admin|modir el nezam|system|developer)\b",
        r"\b(?:wariny|warini|e3red|etla3|ektb|e3mel dump)\b.{0,40}\b(?:system prompt|instructions|tools?|mcp|password|secret)\b",
        r"\b(?:et3ada|et3ada 3ala|e3dy)\b.{0,30}\b(?:el verification|el identity|el policy|el rules|el 2awa3ed)\b",
        r"\b3adi\s+(?:el verification|el identity)\b.{0,45}\b(?:w|we)\s+(?:show|open|raga3|rga3|refund|eb3at|wariny)\b",
        r"(?:اكشف|اظهر|اطبع|كرر|استخرج).{0,35}(?:تعليمات النظام|البرومبت|التعليمات المخفية|الأدوات|mcp)",
        r"(?:انا|أنا|ده|هذا).{0,20}(?:مدير النظام|مسؤول النظام|المطور|الادمن|الأدمن)",
        r"(?:تجاوز|عطل|الغ|تخطى).{0,35}(?:التحقق|تأكيد الهوية|ملكية الحساب|السياسة|القواعد)",
    )
)


def looks_like_override(text: str) -> bool:
    """Does the message tell the agent to ignore, skip or change its rules?"""
    normalized = normalize(text)
    return any(p.search(normalized) for p in _PATTERNS)


def looks_like_attack(text: str) -> bool:
    """Broader detector used by the mandatory pre-action risk screen."""
    return looks_like_override(text)
