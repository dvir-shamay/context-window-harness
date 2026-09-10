"""Score LLM extraction results against human ground truth.

Scoring dimensions:
  - Relevance accuracy: model's relevance score vs human (exact=1.0, off-by-1=0.5)
  - Fact type recall: what fraction of human-labelled types did the model find?
  - Fact type precision: what fraction of model's types are correct?
  - Partial credit: similar type pairs (e.g. DECISION↔ACTION) get 0.5.
"""
import json
import re

# Canonical fact types
FACT_TYPES = {"DECISION", "ACTION", "BLOCKER", "STATUS", "INFO", "NONE"}

# Map common label variations to canonical types
LABEL_MAP = {
    "decisions": "DECISION",
    "decision": "DECISION",
    "actions": "ACTION",
    "action": "ACTION",
    "blockers": "BLOCKER",
    "blocker": "BLOCKER",
    "status": "STATUS",
    "info": "INFO",
    "none": "NONE",
    "decsions": "DECISION",  # common typo
}

# Partial credit for semantically similar types
TYPE_SIMILARITY = {
    ("DECISION", "ACTION"): 0.5,
    ("ACTION", "DECISION"): 0.5,
    ("STATUS", "INFO"): 0.5,
    ("INFO", "STATUS"): 0.5,
    ("BLOCKER", "STATUS"): 0.3,
    ("STATUS", "BLOCKER"): 0.3,
}


def parse_human_fact_types(fact_types_str: str) -> set[str]:
    """Parse the human Fact Types column into canonical types."""
    if not fact_types_str or fact_types_str.strip().lower() == "none":
        return {"NONE"}
    parts = re.split(r"[,\s]+and\s+|,\s*", fact_types_str.strip().lower())
    result = set()
    for p in parts:
        p = p.strip()
        if p in LABEL_MAP:
            result.add(LABEL_MAP[p])
    return result or {"NONE"}


def parse_llm_extraction(llm_output: str) -> dict:
    """Parse the LLM's JSON extraction response."""
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        pass

    # Try markdown-fenced JSON
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", llm_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try any JSON object
    brace_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def score_extraction(
    llm_output: str,
    human_score: int,
    human_fact_types: str,
) -> dict:
    """Score an LLM extraction against human ground truth.

    Returns a dict with:
    - relevance_score: model's score (0-3)
    - relevance_accuracy: 1 if exact match, partial credit for close
    - fact_type_recall: fraction of human fact types the model found
    - fact_type_precision: fraction of model fact types that are correct
    - fact_count: number of individual facts extracted
    - parse_success: whether the LLM output was valid JSON
    - raw_output: the original LLM text (truncated)
    """
    parsed = parse_llm_extraction(llm_output)
    human_types = parse_human_fact_types(human_fact_types)

    result = {
        "parse_success": bool(parsed),
        "raw_output": llm_output[:500],
        "human_score": human_score,
        "human_fact_types": sorted(human_types),
    }

    if not parsed:
        result.update({
            "relevance_score": -1,
            "relevance_accuracy": 0.0,
            "fact_type_recall": 0.0,
            "fact_type_precision": 0.0,
            "fact_count": 0,
            "model_fact_types": [],
        })
        return result

    # Extract relevance score
    model_score = parsed.get("relevance_score", parsed.get("score", -1))
    if isinstance(model_score, str):
        try:
            model_score = int(model_score)
        except ValueError:
            model_score = -1

    # Relevance accuracy: exact match = 1.0, off by 1 = 0.5, off by 2+ = 0.0
    if model_score == human_score:
        rel_accuracy = 1.0
    elif abs(model_score - human_score) == 1:
        rel_accuracy = 0.5
    else:
        rel_accuracy = 0.0

    # Extract fact types from model output
    facts = parsed.get("facts", parsed.get("key_facts", []))
    if isinstance(facts, list):
        model_types = set()
        for fact in facts:
            if isinstance(fact, dict):
                ft = fact.get("type", fact.get("fact_type", "")).upper().strip()
                if ft in FACT_TYPES:
                    model_types.add(ft)
                elif ft in ("UPDATE", "STATUS_UPDATE", "PROGRESS"):
                    model_types.add("STATUS")
                elif ft in ("TASK", "TODO", "ACTION_ITEM", "FOLLOW_UP", "FOLLOWUP"):
                    model_types.add("ACTION")
                elif ft in ("ISSUE", "RISK", "CONCERN", "BLOCK"):
                    model_types.add("BLOCKER")
                elif ft in ("CONCLUSION", "AGREEMENT", "RESOLUTION"):
                    model_types.add("DECISION")
                elif ft in ("DETAIL", "FACT", "DATA", "NOTE", "CONTEXT", "INFORMATION"):
                    model_types.add("INFO")
            elif isinstance(fact, str):
                for ft in FACT_TYPES:
                    if fact.upper().startswith(ft):
                        model_types.add(ft)
                        break
        fact_count = len(facts)
    else:
        model_types = set()
        fact_count = 0

    if not model_types and human_types == {"NONE"}:
        model_types = {"NONE"}

    # Fact type recall with partial credit for similar types
    if human_types and human_types != {"NONE"}:
        recall_score = 0.0
        for ht in human_types:
            if ht in model_types:
                recall_score += 1.0
            else:
                best_sim = max(
                    (TYPE_SIMILARITY.get((ht, mt), 0.0) for mt in model_types),
                    default=0.0
                )
                recall_score += best_sim
        recall = recall_score / len(human_types)
    elif human_types == {"NONE"} and (model_types == {"NONE"} or fact_count == 0):
        recall = 1.0
    else:
        recall = 0.0

    # Fact type precision with partial credit
    if model_types and model_types != {"NONE"}:
        prec_score = 0.0
        for mt in model_types:
            if mt in human_types:
                prec_score += 1.0
            else:
                best_sim = max(
                    (TYPE_SIMILARITY.get((mt, ht), 0.0) for ht in human_types),
                    default=0.0
                )
                prec_score += best_sim
        precision = prec_score / len(model_types)
    elif model_types == {"NONE"} and human_types == {"NONE"}:
        precision = 1.0
    elif not model_types and human_types == {"NONE"}:
        precision = 1.0
    else:
        precision = 0.0

    result.update({
        "relevance_score": model_score,
        "relevance_accuracy": rel_accuracy,
        "fact_type_recall": recall,
        "fact_type_precision": precision,
        "fact_count": fact_count,
        "model_fact_types": sorted(model_types),
    })
    return result
