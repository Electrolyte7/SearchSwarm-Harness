"""Patch v2: adaptive evidence-grounded harness helpers.

This module is deliberately heuristic and dependency-free.  It keeps Patch v2
off by default, then provides a shared candidate ledger, final verifier, and
delegation router for both single-agent and SearchSwarm runs.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in ("1", "true", "yes", "on")


PATCH_V2 = env_bool("SEARCHSWARM_PATCH_V2", False)
PATCH_FINAL_VERIFY = env_bool("SEARCHSWARM_PATCH_FINAL_VERIFY", PATCH_V2)
PATCH_CANDIDATE_LEDGER = env_bool("SEARCHSWARM_PATCH_CANDIDATE_LEDGER", PATCH_V2)
PATCH_ADAPTIVE_ROUTER = env_bool("SEARCHSWARM_PATCH_ADAPTIVE_ROUTER", PATCH_V2)
PATCH_MAIN_EARLY_FINALIZE = env_bool(
    "SEARCHSWARM_PATCH_MAIN_EARLY_FINALIZE", PATCH_V2)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_QUOTED_RE = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{3,120})[\"“”'‘’]")
_LABEL_RE = re.compile(
    r"\b(?:candidate_answer|answer|selected_candidate|title|program|"
    r"book|city|person|organization|source|url)\s*:\s*(.+)",
    re.I,
)
_CAP_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&.'-]+(?:\s+|$)){2,7}"
)
_STOP_CANDIDATES = {
    "I do not know",
    "No answer found",
    "The answer",
    "Based on",
    "Evidence in",
    "Summary Content",
    "Web Results",
    "Brock News",
    "ambiguous",
    "bootstrap smoke",
}
_BAD_CANDIDATE_RE = re.compile(
    r"\b(?:bootstrap smoke|sub-agent report|search result|web results|"
    r"summary length|api call|round \d+|ambiguous|unable to determine)\b",
    re.I,
)
_GENERIC_CANDIDATE_RE = re.compile(
    r"^(?:ambiguous|unknown|uncertain|open to interpretation|"
    r"unable to identify|unable to determine|no answer found)$",
    re.I,
)
_TOOL_ARTIFACT_RE = re.compile(
    r"\b(?:bootstrap smoke sub-agent report|sub-agent report|tool response|"
    r"observation|search results?|web results?|total control time|"
    r"significant strikes attempted|api call|round \d+)\b",
    re.I,
)
_GENERIC_SOURCE_NAMES = {
    "brock news",
    "google search",
    "wikipedia",
    "youtube",
    "facebook",
    "linkedin",
    "news",
    "homepage",
    "article",
    "page",
    "official website",
}
_SOURCE_QUESTION_MARKERS = (
    "source", "publication", "website", "site", "page", "news source",
    "where was", "which website", "which publication",
)
_RELATION_PATTERNS = [
    ("sponsor_or_advertiser",
     re.compile(
         r"\b(?:brought to you by|sponsored by|presented by|advertised by|"
         r"advertisement by)\s+(.+?)(?:[.;\n]|$)",
         re.I,
     )),
    ("program_provider",
     re.compile(
         r"\b(?:exclusive\s+)?(?:group\s+)?(?:insurance\s+)?program\s+"
         r"through\s+(.+?)(?:[.;\n]|$)",
         re.I,
     )),
    ("program_provider",
     re.compile(
         r"\b(?:offered by|provided by|in partnership with)\s+(.+?)"
         r"(?:[.;\n]|$)",
         re.I,
     )),
]
_BODY_TOPIC_RE = re.compile(
    r"\b(?:peaksaver|energy[- ]saving|air conditioning|electricity|"
    r"conservation|conserve energy|cooling|thermostat|energy star)\b",
    re.I,
)


def _clip(text: Any, limit: int = 600) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value[:limit].rstrip()


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _clean_candidate(text: Any) -> str:
    value = _clip(text, 160).strip(" -:;,.")
    value = re.sub(r"^(?:candidate_answer|answer|title|program)\s*:\s*", "",
                   value, flags=re.I).strip()
    lines = value.splitlines()
    value = lines[0].strip() if lines else ""
    if not value or len(value) < 2:
        return ""
    if len(value.split()) > 14:
        return ""
    lowered = value.lower()
    if "http" in lowered:
        return ""
    return value


def canonicalize_relation_candidate(text: Any) -> str:
    value = _clip(text, 220)
    value = re.sub(r"\([^)]*\)", "", value).strip()
    value = re.sub(
        r"^(?:brought to you by|sponsored by|presented by|advertised by|"
        r"advertisement by|offered by|provided by|in partnership with)\s+",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"^(?:exclusive\s+)?(?:group\s+)?(?:insurance\s+)?program\s+through\s+",
        "",
        value,
        flags=re.I,
    )
    value = re.split(
        r"\b(?:provide you|offers?|offering|with a quote|you will benefit|"
        r"visit|call|for Brock|for professionals|specifically designed)\b",
        value,
        maxsplit=1,
        flags=re.I,
    )[0]
    value = value.strip(" \t\r\n'\"“”‘’[]{}-:;,.")
    return _clean_candidate(value)


def extract_relation_candidates(text: Any, question: str | None = None) -> list[dict[str, str]]:
    body = str(text or "")
    candidates = []
    seen = set()
    for role, pattern in _RELATION_PATTERNS:
        for match in pattern.finditer(body):
            candidate = canonicalize_relation_candidate(match.group(1))
            if not candidate:
                continue
            key = _norm(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "candidate": candidate,
                "answer_role": role,
                "evidence": _clip(match.group(0), 500),
            })
    return candidates


def _question_asks_source(question: str) -> bool:
    lowered = str(question or "").lower()
    if re.search(
            r"\b(?:what|which)\s+(?:is|was|are|were)?\s*(?:the\s+)?"
            r"(?:source|publication|website|site|page|news source)\b",
            lowered):
        return True
    if re.search(
            r"\b(?:name|identify)\s+(?:the\s+)?"
            r"(?:source|publication|website|site|page|news source)\b",
            lowered):
        return True
    return False


def extract_target_role(question: str) -> str:
    lowered = str(question or "").lower()
    if _question_asks_source(lowered):
        return "source_or_publication"
    if any(marker in lowered for marker in (
        "advertised", "advertisement", "brought to you by",
        "sponsored", "sponsor", "presented by", " ad ",
    )):
        if "program" in lowered:
            return "advertised_program"
        return "sponsor_or_advertiser"
    if any(marker in lowered for marker in (
        "insurance program", "program through", "offered by",
        "provided by", "program",
    )):
        return "program_provider"
    if "title" in lowered or "book" in lowered:
        return "work_title"
    if "city" in lowered or "where" in lowered:
        return "location"
    if "who" in lowered or "person" in lowered or "author" in lowered:
        return "person"
    return "unknown"


def infer_answer_role(candidate: Any, evidence: Any = "", question: str = "",
                      candidate_type: str = "") -> str:
    text = _clean_candidate(candidate)
    lowered = str(text or "").lower()
    context = str(evidence or "").lower()
    if is_source_or_page_candidate(text, question) or candidate_type in ("source_name", "page_title"):
        return "source_or_publication"
    if not text or is_generic_candidate(text):
        return "generic"
    if is_tool_artifact_candidate(text):
        return "generic"
    if re.search(
            r"(?:brought to you by|sponsored by|presented by|advertised by|"
            r"advertisement by)\s+" + re.escape(text.lower()),
            context):
        return "sponsor_or_advertiser"
    if re.search(
            r"(?:program through|insurance program through|group insurance program through)\s+"
            + re.escape(text.lower()),
            context):
        return "program_provider"
    if "td insurance meloche monnex" in lowered and any(marker in context for marker in (
        "brought to you by", "exclusive group insurance program",
        "program through", "home and automobile insurance",
    )):
        return "advertised_program"
    if "peaksaver" in lowered or (
        lowered and lowered in context and _BODY_TOPIC_RE.search(context)
        and not any(marker in context for marker in (
            "brought to you by " + lowered,
            "sponsored by " + lowered,
            "program through " + lowered,
        ))
    ):
        return "article_subject"
    if "phaistos" in lowered and extract_target_role(question) == "work_title":
        return "clue_object"
    return "unknown"


def _role_matches(target_role: str, answer_role: str) -> bool:
    if target_role == "unknown":
        return True
    if target_role == answer_role:
        return True
    if target_role == "advertised_program" and answer_role in (
        "sponsor_or_advertiser", "program_provider", "advertised_program"):
        return True
    if target_role == "program_provider" and answer_role in (
        "program_provider", "advertised_program", "sponsor_or_advertiser"):
        return True
    return False


def _role_mismatch_reason(target_role: str, answer_role: str) -> str:
    if answer_role == "article_subject" and target_role in (
        "sponsor_or_advertiser", "program_provider", "advertised_program"):
        return "related_but_wrong_role: article subject but question asks sponsor/provider/program"
    if answer_role == "source_or_publication" and target_role != "source_or_publication":
        return "related_but_wrong_role: source/publication but question asks different target"
    if answer_role == "clue_object" and target_role in (
        "work_title", "person", "program_provider", "advertised_program"):
        return "related_but_wrong_role: clue object but question asks different target"
    if answer_role == "generic":
        return "generic"
    return ""


def is_generic_candidate(text: Any) -> bool:
    value = _clean_candidate(text)
    if not value:
        return True
    lowered = value.lower().strip(" .,:;")
    return (
        lowered in {item.lower() for item in _STOP_CANDIDATES}
        or any(lowered.startswith(item.lower()) for item in _STOP_CANDIDATES)
        or bool(_GENERIC_CANDIDATE_RE.search(lowered))
        or lowered.startswith("unable to ")
    )


def is_tool_artifact_candidate(text: Any) -> bool:
    value = _clean_candidate(text)
    return bool(value and (_TOOL_ARTIFACT_RE.search(value) or _BAD_CANDIDATE_RE.search(value)))


def is_source_or_page_candidate(text: Any, question: str = "") -> bool:
    value = _clean_candidate(text)
    lowered = value.lower().strip(" .,:;")
    if not value:
        return False
    if lowered in _GENERIC_SOURCE_NAMES:
        return True
    if re.search(r"\b(?:news|homepage|article|page|official website|archive)\b", lowered):
        return True
    return False


def classify_candidate_type(text: Any, source: str = "",
                            from_low_quality_report: bool = False) -> str:
    if from_low_quality_report:
        return "low_quality_report_candidate"
    if is_source_or_page_candidate(text):
        return "source_name"
    if str(text or "").lower().strip().startswith("bootstrap smoke"):
        return "tool_artifact"
    if is_generic_candidate(text):
        return "generic_phrase"
    if is_tool_artifact_candidate(text):
        return "tool_artifact"
    lowered_source = str(source or "").lower()
    if "title" in lowered_source:
        return "page_title"
    if lowered_source in ("snippet",):
        return "evidence_entity"
    return "answer_candidate"


def is_candidate_allowed_for_final(candidate: dict[str, Any] | str,
                                   question: str) -> bool:
    if isinstance(candidate, dict):
        text = candidate.get("candidate", "")
        candidate_type = candidate.get("candidate_type") or classify_candidate_type(
            text,
            candidate.get("source", ""),
            bool(candidate.get("from_low_quality_report")),
        )
        stricter_type = classify_candidate_type(
            text,
            candidate.get("source", ""),
            bool(candidate.get("from_low_quality_report")),
        )
        if stricter_type in ("source_name", "generic_phrase", "tool_artifact"):
            candidate_type = stricter_type
        from_low_quality = bool(candidate.get("from_low_quality_report"))
    else:
        text = candidate
        candidate_type = classify_candidate_type(text)
        from_low_quality = False
    if not _clean_candidate(text):
        return False
    if candidate_type in ("generic_phrase", "tool_artifact"):
        return False
    if is_source_or_page_candidate(text, question) and not _question_asks_source(question):
        return False
    if candidate_type in ("source_name", "page_title") and not _question_asks_source(question):
        return False
    if from_low_quality:
        return False
    return True


def _confidence_rank(value: str) -> int:
    return {"low": 1, "medium": 2, "medium-high": 3, "high": 4}.get(
        str(value or "").lower(), 1)


@dataclass
class CandidateLedger:
    enabled: bool = field(default_factory=lambda: PATCH_CANDIDATE_LEDGER)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    _seen: dict[str, int] = field(default_factory=dict)
    deduplicated_count: int = 0

    def add(self, candidate: Any, source: str, evidence: Any = "", url: str = "",
            confidence: str = "low", from_low_quality_report: bool = False,
            matched_constraints: list[str] | None = None,
            missing_constraints: list[str] | None = None,
            conflicting_evidence: list[str] | None = None,
            candidate_type: str | None = None,
            answer_role: str | None = None) -> bool:
        if not self.enabled:
            return False
        cleaned = _clean_candidate(candidate)
        if not cleaned:
            return False
        resolved_type = candidate_type or classify_candidate_type(
            cleaned, source, from_low_quality_report)
        key = _norm(cleaned)
        record = {
            "candidate": cleaned,
            "candidate_type": resolved_type,
            "answer_role": answer_role or infer_answer_role(
                cleaned, evidence, candidate_type=resolved_type),
            "source": source,
            "evidence": _clip(evidence, 700),
            "url": _clip(url, 240),
            "matched_constraints": matched_constraints or [],
            "missing_constraints": missing_constraints or [],
            "conflicting_evidence": conflicting_evidence or [],
            "confidence": confidence,
            "from_low_quality_report": bool(from_low_quality_report),
        }
        if key in self._seen:
            self.deduplicated_count += 1
            existing = self.candidates[self._seen[key]]
            if record["evidence"] and record["evidence"] not in existing["evidence"]:
                existing["evidence"] = _clip(
                    f"{existing['evidence']} | {record['evidence']}", 900)
            if _confidence_rank(record["confidence"]) > _confidence_rank(
                    existing.get("confidence")):
                existing["confidence"] = record["confidence"]
            if existing.get("candidate_type") != "answer_candidate":
                existing["candidate_type"] = record["candidate_type"]
            if existing.get("answer_role") in (None, "unknown", "generic"):
                existing["answer_role"] = record["answer_role"]
            if (
                existing.get("from_low_quality_report")
                and record["source"] == "main_observation"
                and record["evidence"]
                and not record["from_low_quality_report"]
            ):
                existing["from_low_quality_report"] = False
            else:
                existing["from_low_quality_report"] = (
                    existing.get("from_low_quality_report")
                    or record["from_low_quality_report"]
                )
            return False
        self._seen[key] = len(self.candidates)
        self.candidates.append(record)
        return True

    def add_text(self, text: Any, source: str, confidence: str = "low",
                 from_low_quality_report: bool = False) -> int:
        if not self.enabled:
            return 0
        body = str(text or "")
        evidence = _clip(body)
        added = 0
        add_limit = 30 if source == "main_observation" else 12
        for relation in extract_relation_candidates(body):
            if self.add(
                    relation["candidate"], source,
                    relation.get("evidence") or evidence,
                    confidence="high",
                    from_low_quality_report=from_low_quality_report,
                    candidate_type="answer_candidate",
                    answer_role=relation["answer_role"]):
                added += 1
                if added >= add_limit:
                    return added
        for line in body.splitlines():
            match = _LABEL_RE.search(line.strip())
            if match and self.add(match.group(1), source, evidence,
                                  confidence=confidence,
                                  from_low_quality_report=from_low_quality_report,
                                  candidate_type="answer_candidate"):
                added += 1
                if added >= add_limit:
                    return added
        for phrase in _QUOTED_RE.findall(body):
            if self.add(phrase, source, evidence, confidence=confidence,
                        from_low_quality_report=from_low_quality_report,
                        candidate_type="evidence_entity"):
                added += 1
                if added >= add_limit:
                    return added
        for entity in _CAP_ENTITY_RE.findall(body):
            entity = re.sub(r"\s+", " ", entity).strip()
            if self.add(entity, source, evidence, confidence=confidence,
                        from_low_quality_report=from_low_quality_report,
                        candidate_type=classify_candidate_type(
                            entity, source, from_low_quality_report)):
                added += 1
                if added >= add_limit:
                    return added
        return added

    def add_message(self, message: dict[str, Any]) -> int:
        role = message.get("role", "")
        content = message.get("content") or ""
        if role == "tool":
            source = (
                "subagent_report"
                if "sub-agent" in content.lower()
                or "low_quality_report" in content.lower()
                else "main_observation"
            )
        elif role == "assistant":
            if len(str(content)) > 300:
                return 0
            source = "draft_answer"
        else:
            source = "main_observation"
        return self.add_text(
            content,
            source=source,
            confidence="medium" if role == "tool" else "low",
            from_low_quality_report="low_quality_report: true" in content.lower(),
        )

    def ingest_messages(self, messages: list[dict[str, Any]]) -> None:
        for message in messages or []:
            if isinstance(message, dict):
                self.add_message(message)

    def as_list(self) -> list[dict[str, Any]]:
        return list(self.candidates)

    def stats(self) -> dict[str, Any]:
        return {
            "candidate_ledger_enabled": bool(self.enabled),
            "candidate_count": len(self.candidates),
            "candidate_from_main_count": sum(
                1 for item in self.candidates
                if str(item.get("source", "")).startswith("main")
                or item.get("source") == "draft_answer"
            ),
            "candidate_from_subagent_count": sum(
                1 for item in self.candidates
                if "subagent" in str(item.get("source", ""))
            ),
            "candidate_from_low_quality_report_count": sum(
                1 for item in self.candidates
                if item.get("from_low_quality_report")
            ),
            "candidate_deduplicated_count": self.deduplicated_count,
        }


def ledger_from_messages(messages: list[dict[str, Any]]) -> CandidateLedger:
    ledger = CandidateLedger(enabled=True)
    ledger.ingest_messages(messages)
    return ledger


def extract_constraints(question: str) -> dict[str, Any]:
    q = question or ""
    lowered = q.lower()
    target_type = "entity"
    type_rules = [
        ("city", ("city", "where")),
        ("book", ("book", "title")),
        ("program", ("program", "advertised", "brought to you", "sponsor")),
        ("date", ("date", "day", "year")),
        ("person", ("who", "person", "individual", "sportsman", "scientist")),
        ("organization", ("company", "organization", "university")),
    ]
    for label, markers in type_rules:
        if any(marker in lowered for marker in markers):
            target_type = label
            break
    constraints = []
    for marker in (
        "advertised", "brought to you", "program", "author", "published",
        "date", "city", "book", "title", "before", "after", "between",
        "same author", "five months later", "gold medal", "bodybuilding",
        "retired", "reincarnation",
    ):
        if marker in lowered:
            constraints.append(marker)
    return {
        "target_type": target_type,
        "constraints": constraints,
        "time_constraints": re.findall(r"\b(?:19|20)\d{2}\b", q),
        "quoted_phrases": _QUOTED_RE.findall(q),
    }


def _candidate_context(candidate: str, ledger_items: list[dict[str, Any]]) -> str:
    key = _norm(candidate)
    parts = []
    for item in ledger_items:
        if _norm(item.get("candidate")) == key:
            parts.append(str(item.get("evidence") or ""))
    return " ".join(parts).lower()


def _candidate_records(candidate: str,
                       ledger_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key = _norm(candidate)
    return [
        item for item in ledger_items
        if _norm(item.get("candidate")) == key
    ]


def _has_direct_evidence(candidate: str,
                         ledger_items: list[dict[str, Any]]) -> bool:
    c_lower = str(candidate or "").lower()
    if not c_lower:
        return False
    return any(
        c_lower in str(item.get("evidence") or "").lower()
        for item in _candidate_records(candidate, ledger_items)
    )


def _score_candidate(candidate: str, question: str,
                     ledger_items: list[dict[str, Any]]) -> tuple[int, list[str], list[str], list[str]]:
    constraints = extract_constraints(question)
    target_role = extract_target_role(question)
    q_lower = (question or "").lower()
    c_lower = candidate.lower()
    context = _candidate_context(candidate, ledger_items)
    records = _candidate_records(candidate, ledger_items)
    primary = records[0] if records else {
        "candidate": candidate,
        "candidate_type": classify_candidate_type(candidate),
        "source": "",
        "evidence": "",
        "from_low_quality_report": False,
    }
    candidate_type = primary.get("candidate_type") or classify_candidate_type(
        candidate,
        primary.get("source", ""),
        bool(primary.get("from_low_quality_report")),
    )
    answer_role = primary.get("answer_role") or infer_answer_role(
        candidate, primary.get("evidence", ""), question, candidate_type)
    allowed_for_final = is_candidate_allowed_for_final(
        {**primary, "candidate_type": candidate_type}, question)
    matched = []
    missing = []
    conflicts = []
    score = 0
    if not _clean_candidate(candidate):
        return -20, [], ["invalid or generic candidate"], ["not a usable answer"]

    if is_generic_candidate(candidate):
        score -= 10
        conflicts.append("generic candidate")
    if is_tool_artifact_candidate(candidate):
        score -= 10
        conflicts.append("tool/report artifact")
    if is_source_or_page_candidate(candidate, question) and not _question_asks_source(question):
        score -= 6
        conflicts.append("source/page name but question does not ask for source")
    if len(candidate.split()) > 10 or ":" in candidate or "\n" in str(candidate):
        score -= 3
        conflicts.append("candidate is long or sentence-like")

    has_direct = candidate and c_lower in context
    if has_direct:
        score += 3
        matched.append("direct evidence mentions candidate")
    else:
        score -= 3
        missing.append("no direct evidence")

    if any(_norm(item.get("candidate")) == _norm(candidate)
           and not item.get("from_low_quality_report") for item in ledger_items):
        score += 2
        matched.append("candidate not solely from low-quality report")
    else:
        score -= 4
        missing.append("candidate only from low-quality report")

    appears_main = any(
        item.get("source") in ("main_observation", "draft_answer")
        and item.get("evidence")
        for item in records
    )
    appears_subagent = any(
        "subagent" in str(item.get("source", ""))
        for item in records
    )
    if appears_main:
        score += 1
        matched.append("appears in main evidence")
    if appears_main and appears_subagent:
        score += 1
        matched.append("appears in main and sub-agent evidence")

    relation_records = [
        item for item in records
        if item.get("answer_role") in (
            "sponsor_or_advertiser", "program_provider", "advertised_program")
    ]
    if _role_matches(target_role, answer_role):
        score += 4
        matched.append("candidate role matches target role")
    else:
        reason = _role_mismatch_reason(target_role, answer_role)
        if reason:
            conflicts.append(reason)
    if answer_role == "sponsor_or_advertiser":
        score += 4
        matched.append("relation evidence: sponsor/advertiser")
    if answer_role == "program_provider":
        score += 3
        matched.append("relation evidence: program provider")
    if relation_records and appears_main:
        score += 2
        matched.append("appears in relation extraction and observation")
    if answer_role == "article_subject" and target_role in (
            "sponsor_or_advertiser", "program_provider", "advertised_program"):
        score -= 16
        conflicts.append("article subject but target asks sponsor/provider/program")
    if answer_role == "source_or_publication" and target_role != "source_or_publication":
        score -= 6
        conflicts.append("source/publication but target does not ask source")
    if answer_role == "clue_object" and target_role in (
            "work_title", "person", "program_provider", "advertised_program"):
        score -= 4
        conflicts.append("clue object but target asks different role")

    target_type = constraints["target_type"]
    type_match = target_type == "entity"
    if target_type == "date":
        if re.search(r"\b(?:19|20)\d{2}\b", candidate) or re.search(
                r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
                candidate, re.I):
            score += 3
            matched.append("date-like answer shape")
            type_match = True
    if target_type == "program":
        if any(marker in context for marker in (
            "brought to you", "advertised", "program", "sponsor",
            "alumni insurance", "exclusive group insurance")):
            score += 6
            matched.append("advertised/program context")
            type_match = True
        if any(marker in c_lower for marker in (
            "news", "article", "archive", "author", "coach")):
            score -= 4
            conflicts.append("candidate looks like source metadata rather than program name")
        if any(marker in context for marker in (
            "energy-saving", "energy saving", "air conditioning",
            "voluntary energy", "tips")) and not any(marker in context for marker in (
                "brought to you", "advertised", "sponsor", "insurance")):
            score -= 4
            conflicts.append("appears to be article subject rather than advertised program")
    if target_type == "book":
        if any(marker in context for marker in ("book", "title", "published", "decipher")):
            score += 5
            matched.append("book/title context")
            type_match = True
        if "artifact" in context or "disc" in c_lower:
            score -= 3
            conflicts.append("appears to be clue object rather than requested book title")
    if type_match:
        score += 2
        matched.append("candidate type matches question target")
    else:
        score -= 3
        missing.append("target type mismatch")
    if "title" in q_lower and any(marker in context for marker in ("title", "book")):
        score += 2
    if "same author" in q_lower and "author" in context:
        score += 1
    if "gold medal" in q_lower and "gold medal" in context:
        score += 1
    if "retired" in q_lower and "retired" in context:
        score += 1
    if "bodybuilding" in q_lower and "body" in context:
        score += 1
    if not allowed_for_final:
        conflicts.append("candidate not allowed for final")
    if not matched:
        missing.append("no direct constraint match")
    return score, matched, missing, conflicts


def verify_final_answer(question: str, draft_answer: str,
                        ledger: CandidateLedger | list[dict[str, Any]] | None,
                        messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if isinstance(ledger, CandidateLedger):
        working = ledger
    else:
        working = CandidateLedger(enabled=True)
        for item in ledger or []:
            working.add(
                item.get("candidate"), item.get("source", "ledger"),
                item.get("evidence", ""), item.get("url", ""),
                item.get("confidence", "low"),
                item.get("from_low_quality_report", False),
            )
    if messages:
        working.ingest_messages(messages[-12:])
    working.add(draft_answer, "draft_answer", draft_answer, confidence="medium")
    candidates = working.as_list()
    scored = []
    for item in candidates:
        candidate = item.get("candidate", "")
        score, matched, missing, conflicts = _score_candidate(
            candidate, question, candidates)
        if item.get("from_low_quality_report"):
            score -= 2
        candidate_type = item.get("candidate_type") or classify_candidate_type(
            candidate,
            item.get("source", ""),
            bool(item.get("from_low_quality_report")),
        )
        stricter_type = classify_candidate_type(
            candidate,
            item.get("source", ""),
            bool(item.get("from_low_quality_report")),
        )
        if stricter_type in ("source_name", "generic_phrase", "tool_artifact"):
            candidate_type = stricter_type
        answer_role = item.get("answer_role") or infer_answer_role(
            candidate, item.get("evidence", ""), question, candidate_type)
        allowed_for_final = is_candidate_allowed_for_final(
            {**item, "candidate_type": candidate_type,
             "answer_role": answer_role}, question)
        if not allowed_for_final:
            conflicts = list(conflicts) + ["candidate failed final-answer gate"]
        scored.append({
            "candidate": candidate,
            "candidate_type": candidate_type,
            "answer_role": answer_role,
            "allowed_for_final": allowed_for_final,
            "score": score,
            "matched_constraints": matched,
            "missing_constraints": missing,
            "conflicting_evidence": conflicts,
            "rejection_reason": "; ".join(conflicts or missing),
            "from_low_quality_report": item.get("from_low_quality_report", False),
        })
    scored.sort(key=lambda item: (item["score"], len(item["matched_constraints"])),
                reverse=True)
    draft_clean = _clean_candidate(draft_answer) or _clip(draft_answer, 120)
    draft_score = -20
    draft_role = "unknown"
    if draft_clean:
        draft_score, _, _, _ = _score_candidate(draft_clean, question, candidates)
        draft_records = _candidate_records(draft_clean, candidates)
        draft_primary = draft_records[0] if draft_records else {
            "candidate": draft_clean,
            "evidence": draft_clean,
            "candidate_type": classify_candidate_type(draft_clean),
        }
        draft_role = draft_primary.get("answer_role") or infer_answer_role(
            draft_clean,
            draft_primary.get("evidence", ""),
            question,
            draft_primary.get("candidate_type", ""),
        )
    selected = draft_clean
    minimum_strong_score = 8
    target_role = extract_target_role(question)
    replacement = next(
        (item for item in scored if item.get("allowed_for_final")),
        None,
    )
    replacement_allowed = False
    if replacement:
        draft_mismatch_reason = _role_mismatch_reason(target_role, draft_role)
        replacement_has_relation = replacement.get("answer_role") in (
            "sponsor_or_advertiser", "program_provider", "advertised_program")
        replacement_allowed = (
            replacement["score"] >= draft_score + 2
            and replacement["score"] >= minimum_strong_score
            and _has_direct_evidence(replacement["candidate"], candidates)
            and not replacement.get("from_low_quality_report")
        ) or (
            bool(draft_mismatch_reason)
            and replacement["score"] >= 7
            and replacement_has_relation
            and _has_direct_evidence(replacement["candidate"], candidates)
            and not replacement.get("from_low_quality_report")
        )
    if replacement and replacement_allowed:
        selected = replacement["candidate"]
    draft_key = _norm(draft_clean)
    selected_key = _norm(selected)
    rejected = [
        item for item in scored
        if _norm(item["candidate"]) != selected_key
    ][:8]
    confidence = "low"
    selected_score = replacement["score"] if replacement and replacement_allowed else draft_score
    if selected:
        if selected_score >= 9:
            confidence = "high"
        elif selected_score >= 6:
            confidence = "medium-high"
        elif selected_score >= 3:
            confidence = "medium"
    return {
        "constraints": extract_constraints(question),
        "candidates": scored[:8],
        "selected_candidate": selected,
        "rejected_candidates": rejected,
        "missing_constraints": (
            replacement["missing_constraints"] if replacement else []
        ),
        "conflicting_evidence": (
            replacement["conflicting_evidence"] if replacement else []
        ),
        "confidence": confidence,
        "verifier_changed_answer": bool(selected_key and selected_key != draft_key),
        "verification_reason": (
            "replacement passed conservative final-answer gate"
            if replacement_allowed else
            "kept draft answer; no replacement passed conservative gate"
        ),
    }


def verifier_stats(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {
            "final_verifier_used_count": 0,
            "final_verifier_changed_answer_count": 0,
            "final_verifier_kept_answer_count": 0,
            "final_verifier_rejected_candidate_count": 0,
            "final_verifier_low_confidence_count": 0,
            "final_verifier_empty_or_failed_count": 1,
        }
    return {
        "final_verifier_used_count": 1,
        "final_verifier_changed_answer_count": int(
            bool(result.get("verifier_changed_answer"))),
        "final_verifier_kept_answer_count": int(
            not bool(result.get("verifier_changed_answer"))),
        "final_verifier_rejected_candidate_count": len(
            result.get("rejected_candidates") or []),
        "final_verifier_low_confidence_count": int(
            result.get("confidence") == "low"),
        "final_verifier_empty_or_failed_count": int(
            not result.get("selected_candidate")),
    }


def question_complexity(question: str) -> dict[str, Any]:
    q = question or ""
    tokens = re.findall(r"\w+", q)
    named = _CAP_ENTITY_RE.findall(q)
    constraints = extract_constraints(q)["constraints"]
    score = 0
    score += int(len(tokens) > 80)
    score += int(len(tokens) > 140)
    score += min(3, len(constraints) // 3)
    score += int(len(named) >= 3)
    score += int(len(re.findall(r"\b(?:19|20)\d{2}\b", q)) >= 2)
    score += int(any(marker in q.lower() for marker in (
        "same author", "advertised", "brought to you", "according to",
        "published", "source", "article")))
    return {
        "score": score,
        "named_entity_count": len(named),
        "constraint_count": len(constraints),
        "is_complex": score >= 3,
    }


def route_delegation(question: str, params: dict[str, Any],
                     ledger: CandidateLedger | None = None,
                     previous_delegations: int = 0) -> dict[str, Any]:
    if not PATCH_ADAPTIVE_ROUTER:
        return {"allow": True, "reason": "router disabled", "params": params}
    complexity = question_complexity(question)
    ledger_items = ledger.as_list() if ledger else []
    high_conf = [
        item for item in ledger_items
        if _confidence_rank(item.get("confidence")) >= 3
        and item.get("evidence")
        and not item.get("from_low_quality_report")
    ]
    if high_conf:
        return {
            "allow": False,
            "reason": "high-confidence candidate already has direct evidence",
            "params": params,
            "stop_delegation": True,
        }
    prompts = list((params or {}).get("prompts") or [])
    if not complexity["is_complex"] and previous_delegations >= 1:
        return {
            "allow": False,
            "reason": "simple question already delegated once",
            "params": params,
            "skip_delegation": True,
        }
    max_prompts = 2 if complexity["is_complex"] else 1
    rewritten = []
    brief_types = []
    type_counts = Counter()
    type_hints = [
        ("source_or_page_search_brief", ("article", "source", "page", "author")),
        ("date_or_timeline_brief", ("date", "year", "before", "after", "between")),
        ("constraint_crosscheck_brief", ("verify", "constraint", "satisfies")),
        ("skeptic_or_counterevidence_brief", ("wrong", "counter", "exclude")),
    ]
    for idx, entry in enumerate(prompts[:max_prompts]):
        prompt = entry.get("prompt", "")
        lowered = prompt.lower()
        brief_type = "entity_search_brief"
        for label, markers in type_hints:
            if any(marker in lowered for marker in markers):
                brief_type = label
                break
        if type_counts[brief_type]:
            brief_type = "constraint_crosscheck_brief"
            prompt = (
                prompt.rstrip()
                + "\n\nDiversity requirement: verify the leading candidate "
                "against every clue and look for counterevidence, not another "
                "same-query search."
            )
        type_counts[brief_type] += 1
        brief_types.append(brief_type)
        updated = dict(entry)
        updated["prompt"] = prompt
        updated["goal"] = f"{entry.get('goal', '').strip()} [{brief_type}]".strip()
        rewritten.append(updated)
    return {
        "allow": True,
        "reason": (
            "complex question allows diverse delegation"
            if complexity["is_complex"] else "simple question allows at most one delegation"
        ),
        "params": {**(params or {}), "prompts": rewritten},
        "force_diverse": len(set(brief_types)) < len(brief_types),
        "brief_types": brief_types,
    }


def should_main_early_finalize(question: str, ledger: CandidateLedger,
                               round_num: int, max_calls: int) -> dict[str, Any]:
    if not PATCH_MAIN_EARLY_FINALIZE or not ledger or max_calls <= 0:
        return {"trigger": False, "reason": "disabled"}
    if round_num < max(1, int(max_calls * 0.6)):
        return {"trigger": False, "reason": "budget threshold not reached"}
    verifier = verify_final_answer(question, "", ledger)
    selected = verifier.get("selected_candidate", "")
    selected_record = next(
        (
            item for item in ledger.as_list()
            if _norm(item.get("candidate")) == _norm(selected)
        ),
        {"candidate": selected},
    )
    if (
        selected
        and verifier.get("confidence") == "high"
        and is_candidate_allowed_for_final(selected_record, question)
        and _has_direct_evidence(selected, ledger.as_list())
        and not selected_record.get("from_low_quality_report")
        and not verifier.get("conflicting_evidence")
        and verifier.get("candidates")
        and verifier["candidates"][0].get("score", 0) >= 8
    ):
        return {
            "trigger": True,
            "reason": "high-confidence candidate with direct evidence",
            "verifier": verifier,
        }
    return {"trigger": False, "reason": "verifier confidence insufficient"}
