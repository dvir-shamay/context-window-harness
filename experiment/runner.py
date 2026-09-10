"""Context Window Experiment Runner

Orchestrates context window composition experiments:
  - Builds context windows with various signal:noise ratios and positions
  - Calls an LLM to extract structured facts from the window
  - Scores the extraction against human ground truth

Experiment modes:
  - quick: 1 trial per config (smoke test)
  - pilot: 5 trials per config (directional signal)
  - full: 20 trials per config (publishable)
  - factorial: matched-pool confirmatory design (paired comparisons)
  - dry-run: validate prompt construction without API calls

Usage:
  python -m experiment.runner quick azure-ai DeepSeek-V3.2
  python -m experiment.runner factorial azure-openai gpt-4o
  python -m experiment.runner dry-run dry-run any-model
"""
import csv
import json
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from experiment.data_loader import DataItem, load_all, summary
from experiment.llm_client import call_llm, LLMResponse
from experiment.scorer import score_extraction

RESULTS_DIR = Path(__file__).parent / "results"

# --- Prompt templates ---

EXTRACTION_SYSTEM = """You are a work-item extraction system. Given communication content from a work week, extract structured facts.

For each item provided, return a JSON object with:
- "item_id": the ID provided
- "relevance_score": 0-3 (0=irrelevant, 1=low, 2=medium, 3=high relevance to active engineering work)
- "facts": array of objects, each with "type" (DECISION|ACTION|BLOCKER|STATUS|INFO) and "text" (the fact)
- If no relevant facts, set "facts": [] and "relevance_score": 0

Return a JSON array of these objects. Return ONLY valid JSON, no markdown fencing."""

EXTRACTION_USER = """Extract work-relevant facts from these {item_count} communication items:

{items_text}

Return a JSON array with one object per item. Each object: {{"item_id": "...", "relevance_score": 0-3, "facts": [{{"type": "DECISION|ACTION|BLOCKER|STATUS|INFO", "text": "..."}}]}}"""


def format_item_for_prompt(item: DataItem, max_content_chars: int = 3000) -> str:
    """Format a single data item for inclusion in a prompt."""
    content = item.content[:max_content_chars]
    if len(item.content) > max_content_chars:
        content += "\n[...truncated...]"
    return f"""--- Item {item.item_id} ({item.item_type}) ---
Subject: {item.subject}
Content:
{content}
---"""


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    description: str
    provider: str = "azure-ai"
    model: str = "DeepSeek-V3.2"
    max_content_chars: int = 3000
    temperature: float = 0.1
    max_tokens: int = 4000


@dataclass
class TrialResult:
    """Result of a single trial (one prompt → one LLM call)."""
    config_name: str
    trial_id: int
    items_sent: int
    target_items: int
    noise_items: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    success: bool
    error: str
    scores: list[dict]
    avg_relevance_accuracy: float
    avg_fact_type_recall: float
    avg_fact_type_precision: float
    parse_success_rate: float
    model: str
    provider: str
    timestamp: str
    trial_block: int = 0
    signal_item_ids: list[str] = None
    noise_item_ids: list[str] = None
    presented_item_ids: list[str] = None


@dataclass
class TrialPool:
    """Shared item pool for one trial block in the factorial experiment.

    All configs within a trial block derive their items from nested subsets
    of the same signal and noise pools (matched-pool design for paired
    comparisons).
    """
    block_id: int
    signal_pool: list[DataItem]
    noise_pool: list[DataItem]

    @property
    def signal_ids(self) -> list[str]:
        return [i.item_id for i in self.signal_pool]

    @property
    def noise_ids(self) -> list[str]:
        return [i.item_id for i in self.noise_pool]


def sample_trial_pool(
    all_items: list[DataItem],
    block_id: int,
    rng: random.Random,
) -> TrialPool:
    """Sample one matched pool for a trial block."""
    signal = [i for i in all_items if i.manual_score >= 2 and i.content]
    noise = [i for i in all_items if i.manual_score == 0 and i.content]
    return TrialPool(
        block_id=block_id,
        signal_pool=rng.sample(signal, min(10, len(signal))),
        noise_pool=rng.sample(noise, min(9, len(noise))),
    )


def build_factorial_window(
    pool: TrialPool,
    arrangement: str,
    rng: random.Random,
) -> list[DataItem]:
    """Build a fixed-count (10 item) context window from a shared trial pool.

    Ratio configs: nested signal prefixes + nested noise prefixes, shuffled.
    Position configs: same 5 signal + 5 noise, only order varies.
    """
    sig = pool.signal_pool
    noi = pool.noise_pool

    # --- Ratio configs (nested subsets, shuffled order) ---
    if arrangement == "ratio_100pct":
        items = list(sig[:10])
        rng.shuffle(items)
        return items
    elif arrangement == "ratio_75pct":
        items = list(sig[:8]) + list(noi[:2])
        rng.shuffle(items)
        return items
    elif arrangement == "ratio_50pct":
        items = list(sig[:5]) + list(noi[:5])
        rng.shuffle(items)
        return items
    elif arrangement == "ratio_25pct":
        items = list(sig[:3]) + list(noi[:7])
        rng.shuffle(items)
        return items
    elif arrangement == "ratio_10pct":
        items = list(sig[:1]) + list(noi[:9])
        rng.shuffle(items)
        return items

    # --- Position configs (fixed 50:50, same items, only order varies) ---
    s = list(sig[:5])
    n = list(noi[:5])

    if arrangement == "position_start":
        return s + n
    elif arrangement == "position_middle":
        return n[:2] + s + n[2:]
    elif arrangement == "position_end":
        return n + s
    elif arrangement == "position_scattered":
        combined = []
        for i in range(5):
            combined.append(n[i])
            combined.append(s[i])
        return combined
    elif arrangement == "position_scattered_front":
        return [s[0], s[1], n[0], s[2], n[1], s[3], n[2], n[3], s[4], n[4]]
    elif arrangement == "position_scattered_back":
        return [n[0], s[0], n[1], n[2], s[1], n[3], s[2], n[4], s[3], s[4]]

    raise ValueError(f"Unknown factorial arrangement: {arrangement}")


def build_context_window(
    all_items: list[DataItem],
    arrangement: str,
    rng: random.Random,
) -> list[DataItem]:
    """Build a context window with a specific arrangement of items.

    Arrangements:
    - signal_only: only score 2-3 items
    - signal_plus_noise_low: signal + equal noise
    - signal_plus_noise_high: signal + 5x noise
    - signal_buried: signal buried in middle of noise
    - all_data: random sample of all items (up to 20)
    - noise_ratio_N: N noise items per signal item
    - stratified_meetings/emails/chats: single media type
    - ratio_*/position_*: factorial configs (uses ad-hoc pool)
    """
    signal = [i for i in all_items if i.manual_score >= 2 and i.content]
    noise = [i for i in all_items if i.manual_score == 0 and i.content]

    signal_sample = rng.sample(signal, min(5, len(signal)))

    if arrangement == "signal_only":
        return signal_sample

    elif arrangement == "signal_plus_noise_low":
        noise_sample = rng.sample(noise, min(len(signal_sample), len(noise)))
        combined = signal_sample + noise_sample
        rng.shuffle(combined)
        return combined

    elif arrangement == "signal_plus_noise_high":
        noise_count = min(len(signal_sample) * 5, len(noise))
        noise_sample = rng.sample(noise, noise_count)
        combined = signal_sample + noise_sample
        rng.shuffle(combined)
        return combined

    elif arrangement == "signal_buried":
        noise_count = min(len(signal_sample) * 5, len(noise))
        noise_sample = rng.sample(noise, noise_count)
        mid = len(noise_sample) // 2
        combined = noise_sample[:mid] + signal_sample + noise_sample[mid:]
        return combined

    elif arrangement == "all_data":
        sample = rng.sample(all_items, min(20, len(all_items)))
        return sample

    elif arrangement.startswith("noise_ratio_"):
        ratio = float(arrangement.replace("noise_ratio_", ""))
        noise_count = max(1, int(len(signal_sample) * ratio))
        noise_count = min(noise_count, len(noise))
        noise_sample = rng.sample(noise, noise_count)
        combined = signal_sample + noise_sample
        rng.shuffle(combined)
        return combined

    elif arrangement.startswith("stratified_"):
        item_type = arrangement.replace("stratified_", "")
        type_map = {"meetings": "meeting", "emails": "email", "chats": "chat"}
        item_type = type_map.get(item_type, item_type)
        typed = [i for i in all_items if i.item_type == item_type and i.content]
        typed_signal = [i for i in typed if i.manual_score >= 2]
        typed_other = [i for i in typed if i.manual_score < 2]
        sample = typed_signal[:5] + rng.sample(typed_other, min(5, len(typed_other)))
        rng.shuffle(sample)
        return sample

    elif arrangement.startswith("ratio_") or arrangement.startswith("position_"):
        pool = sample_trial_pool(all_items, block_id=0, rng=rng)
        return build_factorial_window(pool, arrangement, rng)

    else:
        raise ValueError(f"Unknown arrangement: {arrangement}")


def run_trial(
    items: list[DataItem],
    config: ExperimentConfig,
    trial_id: int,
) -> TrialResult:
    """Run a single trial: format prompt, call LLM, score results."""
    items_text = "\n\n".join(
        format_item_for_prompt(i, config.max_content_chars) for i in items
    )
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": EXTRACTION_USER.format(
            item_count=len(items), items_text=items_text
        )},
    ]

    if not items:
        return TrialResult(
            config_name=config.name, trial_id=trial_id, items_sent=0,
            target_items=0, noise_items=0, prompt_tokens=0,
            completion_tokens=0, total_tokens=0, latency_ms=0,
            success=False, error="No items in context window",
            scores=[], avg_relevance_accuracy=0, avg_fact_type_recall=0,
            avg_fact_type_precision=0, parse_success_rate=0,
            model=config.model, provider=config.provider,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # Call LLM
    resp = call_llm(
        messages,
        provider=config.provider,
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    if not resp.success:
        return TrialResult(
            config_name=config.name, trial_id=trial_id,
            items_sent=len(items),
            target_items=sum(1 for i in items if i.manual_score >= 2),
            noise_items=sum(1 for i in items if i.manual_score <= 1),
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            total_tokens=resp.total_tokens,
            latency_ms=resp.latency_ms,
            success=False, error=resp.error,
            scores=[], avg_relevance_accuracy=0,
            avg_fact_type_recall=0, avg_fact_type_precision=0,
            parse_success_rate=0,
            model=config.model, provider=config.provider,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # Parse LLM response
    import re as _re
    extractions = []
    raw = resp.content.strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            extractions = parsed
        elif isinstance(parsed, dict):
            extractions = [parsed]
    except json.JSONDecodeError:
        pass

    if not extractions:
        fenced = _re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, _re.DOTALL)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                extractions = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                pass

    if not extractions:
        arr_match = _re.search(r"\[.*\]", raw, _re.DOTALL)
        if arr_match:
            try:
                extractions = json.loads(arr_match.group(0))
            except json.JSONDecodeError:
                pass

    # Build lookup by item_id
    extraction_map = {}
    for ext in extractions:
        if isinstance(ext, dict):
            eid = ext.get("item_id") or ext.get("id") or ext.get("Item_ID") or ""
            extraction_map[str(eid).strip()] = ext

    # Score each item
    scores = []
    for item in items:
        ext = extraction_map.get(item.item_id, {})
        ext_json = json.dumps(ext) if ext else ""
        score_result = score_extraction(ext_json, item.manual_score, item.fact_types)
        score_result["item_id"] = item.item_id
        score_result["item_type"] = item.item_type
        scores.append(score_result)

    # Aggregate metrics (only for target items — score 2-3)
    target_scores = [s for s in scores if s["human_score"] >= 2]
    if target_scores:
        avg_rel = sum(s["relevance_accuracy"] for s in target_scores) / len(target_scores)
        avg_recall = sum(s["fact_type_recall"] for s in target_scores) / len(target_scores)
        avg_prec = sum(s["fact_type_precision"] for s in target_scores) / len(target_scores)
    else:
        avg_rel = avg_recall = avg_prec = 0.0

    parse_rate = sum(1 for s in scores if s["parse_success"]) / len(scores) if scores else 0

    return TrialResult(
        config_name=config.name, trial_id=trial_id,
        items_sent=len(items),
        target_items=sum(1 for i in items if i.manual_score >= 2),
        noise_items=sum(1 for i in items if i.manual_score <= 1),
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        total_tokens=resp.total_tokens,
        latency_ms=resp.latency_ms,
        success=True, error="",
        scores=scores,
        avg_relevance_accuracy=avg_rel,
        avg_fact_type_recall=avg_recall,
        avg_fact_type_precision=avg_prec,
        parse_success_rate=parse_rate,
        model=config.model, provider=config.provider,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_experiment(
    configs: list[tuple[str, str]],
    provider: str = "azure-ai",
    model: str = "DeepSeek-V3.2",
    trials_per_config: int = 5,
    seed: int = 42,
) -> list[TrialResult]:
    """Run a full experiment across multiple configurations."""
    print("Loading data...")
    all_items = load_all()
    s = summary(all_items)
    print(f"  {s['total']} items, {s['target_items']} target (score 2-3), {s['noise_items']} noise (score 0)")

    rng = random.Random(seed)
    all_results = []
    total_trials = len(configs) * trials_per_config

    print(f"\nRunning {total_trials} trials ({len(configs)} configs x {trials_per_config} trials each)")
    print(f"Model: {model} ({provider})\n")

    for config_name, description in configs:
        config = ExperimentConfig(
            name=config_name, description=description,
            provider=provider, model=model,
        )

        print(f"  Config: {config_name}")
        for t in range(trials_per_config):
            items = build_context_window(all_items, config_name, rng)
            result = run_trial(items, config, trial_id=t + 1)

            status = "OK" if result.success else f"FAIL: {result.error[:50]}"
            print(f"    Trial {t+1}/{trials_per_config}: "
                  f"{result.items_sent} items, "
                  f"{result.prompt_tokens} prompt_tokens, "
                  f"rel_acc={result.avg_relevance_accuracy:.2f}, "
                  f"recall={result.avg_fact_type_recall:.2f}, "
                  f"{result.latency_ms:.0f}ms — {status}")
            all_results.append(result)
            time.sleep(1)

    return all_results


def run_factorial_experiment(
    configs: list[tuple[str, str]],
    provider: str = "azure-ai",
    model: str = "DeepSeek-V3.2",
    trials_per_block: int = 20,
    seed: int = 42,
) -> list[TrialResult]:
    """Run the confirmatory factorial experiment with matched trial pools.

    All configs within a trial block share the same signal/noise pools
    (nested subsets), creating paired comparisons.
    """
    print("Loading data...")
    all_items = load_all()
    s = summary(all_items)
    print(f"  {s['total']} items, {s['target_items']} target (score 2-3), {s['noise_items']} noise (score 0)")

    rng = random.Random(seed)
    all_results = []
    total_trials = len(configs) * trials_per_block

    print(f"\nRunning {total_trials} trials ({trials_per_block} blocks x {len(configs)} configs each)")
    print(f"Model: {model} ({provider})")
    print(f"Design: nested matched pools (all configs share same items per block)\n")

    for block in range(trials_per_block):
        pool = sample_trial_pool(all_items, block_id=block + 1, rng=rng)
        print(f"  Block {block+1}/{trials_per_block} — signal: {pool.signal_ids[:3]}... noise: {pool.noise_ids[:3]}...")

        for config_name, description in configs:
            config = ExperimentConfig(
                name=config_name, description=description,
                provider=provider, model=model,
            )

            items = build_factorial_window(pool, config_name, rng)
            try:
                result = run_trial(items, config, trial_id=block + 1)
            except Exception as exc:
                print(f"    {config_name}: EXCEPTION — {exc}")
                result = TrialResult(
                    config_name=config_name, trial_id=block + 1,
                    items_sent=len(items),
                    target_items=sum(1 for i in items if i.manual_score >= 2),
                    noise_items=sum(1 for i in items if i.manual_score <= 1),
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    latency_ms=0, success=False, error=str(exc)[:200],
                    scores=[], avg_relevance_accuracy=0,
                    avg_fact_type_recall=0, avg_fact_type_precision=0,
                    parse_success_rate=0, model=model, provider=provider,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

            result.trial_block = pool.block_id
            result.signal_item_ids = pool.signal_ids
            result.noise_item_ids = pool.noise_ids
            result.presented_item_ids = [i.item_id for i in items]

            status = "OK" if result.success else f"FAIL: {result.error[:50]}"
            print(f"    {config_name}: "
                  f"{result.items_sent} items, "
                  f"{result.prompt_tokens} tok, "
                  f"rel={result.avg_relevance_accuracy:.2f}, "
                  f"rec={result.avg_fact_type_recall:.2f}, "
                  f"{result.latency_ms:.0f}ms — {status}")
            all_results.append(result)
            time.sleep(1)

        # Checkpoint after each block
        if all_results:
            _save_checkpoint(all_results, model)

    return all_results


def _save_checkpoint(results: list[TrialResult], model: str):
    """Save incremental checkpoint so progress isn't lost on interruption."""
    RESULTS_DIR.mkdir(exist_ok=True)
    slug = model.replace(".", "_")
    path = RESULTS_DIR / f"factorial_{slug}_checkpoint.json"
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_trials": len(results),
        "is_checkpoint": True,
        "results": [asdict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_results(results: list[TrialResult], filename: str = None):
    """Save results to a JSON file."""
    RESULTS_DIR.mkdir(exist_ok=True)
    if not filename:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{ts}.json"

    path = RESULTS_DIR / filename
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_trials": len(results),
        "results": [asdict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {path}")
    return path


# --- Experiment presets ---

PILOT_CONFIGS = [
    ("signal_only", "Only score 2-3 items — pure signal baseline"),
    ("signal_plus_noise_low", "Signal + equal noise — light dilution"),
    ("signal_plus_noise_high", "Signal + 5x noise — heavy dilution"),
    ("signal_buried", "Signal buried in middle of noise — lost-in-the-middle test"),
    ("all_data", "Random sample of all items — naive baseline"),
    ("stratified_meetings", "Meetings only — type-stratified extraction"),
    ("stratified_emails", "Emails only — type-stratified extraction"),
    ("stratified_chats", "Chats only — type-stratified extraction"),
]

NOISE_CURVE_CONFIGS = [
    ("signal_only", "Ratio 1:0 — pure signal baseline"),
    ("noise_ratio_0.5", "Ratio 1:0.5 — half noise per signal item"),
    ("signal_plus_noise_low", "Ratio 1:1 — equal noise"),
    ("noise_ratio_2", "Ratio 1:2 — double noise per signal item"),
    ("noise_ratio_3", "Ratio 1:3 — triple noise per signal item"),
    ("signal_plus_noise_high", "Ratio 1:5 — 5x noise"),
    ("noise_ratio_7", "Ratio 1:7 — 7x noise per signal item"),
    ("noise_ratio_10", "Ratio 1:10 — 10x noise per signal item"),
    ("noise_ratio_15", "Ratio 1:15 — 15x noise per signal item"),
]

FACTORIAL_CONFIGS = [
    # Factor 1: Signal-to-noise ratio (all 10 items, shuffled order)
    ("ratio_100pct", "10 signal, 0 noise — pure signal at fixed count"),
    ("ratio_75pct", "8 signal, 2 noise — mostly signal"),
    ("ratio_50pct", "5 signal, 5 noise — balanced"),
    ("ratio_25pct", "3 signal, 7 noise — mostly noise"),
    ("ratio_10pct", "1 signal, 9 noise — nearly all noise"),
    # Factor 2: Signal position (all 5 signal + 5 noise = 10 items)
    ("position_start", "Signal items first, noise after"),
    ("position_middle", "Signal items in center, noise around"),
    ("position_end", "Noise first, signal items at end"),
    ("position_scattered", "Signal evenly interleaved with noise"),
    # Factor 2b: Non-contiguous position variants
    ("position_scattered_front", "Front-biased dispersed — centroid 4.4"),
    ("position_scattered_back", "Back-biased dispersed — centroid 6.6"),
]


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    provider = sys.argv[2] if len(sys.argv) > 2 else "azure-ai"
    model = sys.argv[3] if len(sys.argv) > 3 else "DeepSeek-V3.2"

    if mode == "pilot":
        results = run_experiment(PILOT_CONFIGS, provider=provider, model=model, trials_per_config=5)
    elif mode == "quick":
        results = run_experiment(PILOT_CONFIGS, provider=provider, model=model, trials_per_config=1)
    elif mode == "full":
        results = run_experiment(PILOT_CONFIGS, provider=provider, model=model, trials_per_config=20)
    elif mode == "noise_curve":
        results = run_experiment(NOISE_CURVE_CONFIGS, provider=provider, model=model, trials_per_config=20)
    elif mode == "noise_curve_quick":
        results = run_experiment(NOISE_CURVE_CONFIGS, provider=provider, model=model, trials_per_config=1)
    elif mode == "factorial":
        results = run_factorial_experiment(FACTORIAL_CONFIGS, provider=provider, model=model, trials_per_block=20)
    elif mode == "factorial_quick":
        results = run_factorial_experiment(FACTORIAL_CONFIGS, provider=provider, model=model, trials_per_block=1)
    elif mode == "dry-run":
        results = run_experiment(PILOT_CONFIGS, provider="dry-run", model="dry-run", trials_per_config=1)
    else:
        print(f"Unknown mode: {mode}. Use: pilot|quick|full|noise_curve|factorial|dry-run")
        sys.exit(1)

    save_results(results, f"{mode}_{model.replace('.', '_')}.json")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in results:
        by_config[r.config_name].append(r)

    print(f"{'Config':<30} {'Trials':>6} {'AvgTokens':>10} {'RelAcc':>8} {'Recall':>8} {'Parse%':>8} {'AvgMs':>8}")
    print("-" * 80)
    for name, trials in by_config.items():
        ok = [t for t in trials if t.success]
        if not ok:
            print(f"{name:<30} {len(trials):>6}  ALL FAILED")
            continue
        avg_tok = sum(t.total_tokens for t in ok) / len(ok)
        avg_rel = sum(t.avg_relevance_accuracy for t in ok) / len(ok)
        avg_rec = sum(t.avg_fact_type_recall for t in ok) / len(ok)
        avg_parse = sum(t.parse_success_rate for t in ok) / len(ok)
        avg_ms = sum(t.latency_ms for t in ok) / len(ok)
        print(f"{name:<30} {len(ok):>6} {avg_tok:>10.0f} {avg_rel:>8.2f} {avg_rec:>8.2f} {avg_parse:>8.2f} {avg_ms:>8.0f}")
