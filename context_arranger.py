"""
Context Window Arrangement Utility
===================================
Applies empirically-validated arrangement strategies to improve LLM extraction.

Key strategies (grounded in published research):
  1. Calibration context: 50:50 signal:noise ratio improves extraction quality
     over sending pure signal alone.
  2. Type routing: process meetings / emails / chats separately (+20-30pp).
  3. Position for long contexts (>15 items): signal items last (recency bias).
  4. Position for short contexts (<=15 items): order doesn't matter — ratio dominates.
  5. Don't over-filter: including calibration noise outperforms aggressive pre-filtering.

Python API
----------
    from context_arranger import ContextArranger, Item

    items = [
        Item(id="M1", content="...", item_type="meeting", relevance_hint=0.9),
        Item(id="E1", content="...", item_type="email",   relevance_hint=0.1),
        ...
    ]
    arranger = ContextArranger()
    plan = arranger.arrange(items)

    for batch in plan.batches:
        print(batch.summary())
        prompt_text = batch.render()   # ready to send to your LLM

CLI
---
    python context_arranger.py items.json           # arrange and print plan
    python context_arranger.py items.json --render  # print rendered prompts too
    python context_arranger.py --demo               # run on built-in demo items
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Item:
    """A single communication item to include in a context window.

    id           : unique identifier (any string)
    content      : the actual text of the item
    item_type    : "meeting" | "email" | "chat" | "other"
    subject      : optional title / subject line
    relevance_hint : float 0-1 (higher = more likely signal).
                   If you have a pre-computed relevance score (e.g. from
                   a retrieval system or a cheap LLM pre-filter), set this.
                   If None, the arranger uses keyword heuristics to estimate.
    manual_score : int 0-3 (ground-truth score if known; for research/eval)
    """
    id: str
    content: str
    item_type: str = "other"
    subject: str = ""
    relevance_hint: Optional[float] = None
    manual_score: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    @property
    def estimated_relevance(self) -> float:
        """Return relevance_hint if provided, else heuristic estimate."""
        if self.relevance_hint is not None:
            return self.relevance_hint
        return _heuristic_relevance(self.content, self.subject)

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def render(self, max_chars: int = 3000) -> str:
        content = self.content[:max_chars]
        if len(self.content) > max_chars:
            content += "\n[...truncated...]"
        lines = [f"--- Item {self.id} ({self.item_type}) ---"]
        if self.subject:
            lines.append(f"Subject: {self.subject}")
        lines += [f"Content:\n{content}", "---"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Arrangement plan
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    """One LLM call's worth of context window items, in presentation order."""
    batch_id: int
    item_type: str
    items: list[Item]
    signal_count: int
    noise_count: int
    strategy_notes: list[str]

    def summary(self) -> str:
        lines = [
            f"Batch {self.batch_id} [{self.item_type}]: {len(self.items)} items "
            f"(~{self.signal_count} signal, ~{self.noise_count} calibration noise)",
        ]
        for note in self.strategy_notes:
            lines.append(f"  * {note}")
        return "\n".join(lines)

    def render(self, max_chars_per_item: int = 3000) -> str:
        """Return a ready-to-send context string for this batch."""
        return "\n\n".join(item.render(max_chars_per_item) for item in self.items)

    def item_ids_in_order(self) -> list[str]:
        return [i.id for i in self.items]


@dataclass
class ArrangementPlan:
    """Full arrangement plan across all batches."""
    batches: list[Batch]
    total_items: int
    total_signal_est: int
    total_noise_est: int
    arrangement_strategy: str
    notes: list[str]

    def summary(self) -> str:
        lines = [
            "=== Context Arrangement Plan ===",
            f"Strategy  : {self.arrangement_strategy}",
            f"Items     : {self.total_items} total "
            f"(~{self.total_signal_est} signal, ~{self.total_noise_est} noise)",
            f"Batches   : {len(self.batches)}",
        ]
        for note in self.notes:
            lines.append(f"  * {note}")
        lines.append("")
        for batch in self.batches:
            lines.append(batch.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core arranger
# ---------------------------------------------------------------------------

_LONG_CONTEXT_THRESHOLD = 15
_TARGET_NOISE_RATIO     = 1.0
_SIGNAL_THRESHOLD       = 0.45
_MAX_BATCH_SIZE         = 20


class ContextArranger:
    """Arrange items for optimal LLM extraction.

    Parameters
    ----------
    type_routing : bool
        If True (default), split items by type and create one batch per type.
    target_noise_ratio : float
        Desired ratio of calibration-noise items to signal items (default 1.0).
    max_batch_size : int
        Maximum items per batch / LLM call.
    position_optimize : bool
        If True (default), for long batches, place signal items at the END.
    signal_threshold : float
        Relevance score (0-1) above which an item is classified as signal.
    """

    def __init__(
        self,
        type_routing: bool = True,
        target_noise_ratio: float = _TARGET_NOISE_RATIO,
        max_batch_size: int = _MAX_BATCH_SIZE,
        position_optimize: bool = True,
        signal_threshold: float = _SIGNAL_THRESHOLD,
    ):
        self.type_routing = type_routing
        self.target_noise_ratio = target_noise_ratio
        self.max_batch_size = max_batch_size
        self.position_optimize = position_optimize
        self.signal_threshold = signal_threshold

    def arrange(self, items: list[Item]) -> ArrangementPlan:
        """Arrange items into an optimal set of batches."""
        if not items:
            return ArrangementPlan(
                batches=[], total_items=0, total_signal_est=0,
                total_noise_est=0, arrangement_strategy="empty", notes=[],
            )

        classified = [(item, item.estimated_relevance >= self.signal_threshold)
                      for item in items]
        signal_items = [item for item, is_sig in classified if is_sig]
        noise_items  = [item for item, is_sig in classified if not is_sig]

        plan_notes = []
        hints_provided = sum(1 for i in items if i.relevance_hint is not None)
        if hints_provided == 0:
            plan_notes.append(
                "No relevance_hint provided — using keyword heuristics. "
                "For better accuracy, set relevance_hint=0.0-1.0 on each Item."
            )
        elif hints_provided < len(items):
            plan_notes.append(
                f"{hints_provided}/{len(items)} items have relevance_hint; "
                "heuristics used for the rest."
            )

        if self.type_routing:
            strategy = "type_routed"
            groups = _group_by_type(signal_items, noise_items)
        else:
            strategy = "mixed"
            groups = {"mixed": (signal_items, noise_items)}

        batches: list[Batch] = []
        batch_id = 1

        for type_label, (sig, noi) in groups.items():
            if not sig and not noi:
                continue
            type_batches = self._build_batches(type_label, sig, noi, batch_id_start=batch_id)
            batches.extend(type_batches)
            batch_id += len(type_batches)

        total_sig = sum(b.signal_count for b in batches)
        total_noi = sum(b.noise_count for b in batches)

        return ArrangementPlan(
            batches=batches,
            total_items=sum(len(b.items) for b in batches),
            total_signal_est=total_sig,
            total_noise_est=total_noi,
            arrangement_strategy=strategy,
            notes=plan_notes,
        )

    # ------------------------------------------------------------------

    def _build_batches(self, type_label, signal, noise, batch_id_start):
        if not signal:
            return []

        target_noise = min(
            int(math.ceil(len(signal) * self.target_noise_ratio)),
            len(noise),
            self.max_batch_size - 1,
        )
        selected_noise = noise[:target_noise]

        total = len(signal) + len(selected_noise)
        if total <= self.max_batch_size:
            return [self._make_batch(batch_id_start, type_label, signal, selected_noise)]

        chunks = []
        chunk_size = self.max_batch_size // 2
        sig_chunks = _chunk_list(signal, chunk_size)
        for k, sig_chunk in enumerate(sig_chunks):
            noise_for_chunk = int(math.ceil(len(sig_chunk) * self.target_noise_ratio))
            noise_chunk = noise[:noise_for_chunk]
            noise = noise[noise_for_chunk:] + noise[:noise_for_chunk]
            chunks.append(self._make_batch(batch_id_start + k, type_label, sig_chunk, noise_chunk))
        return chunks

    def _make_batch(self, batch_id, type_label, signal, noise):
        notes = []
        total_items = len(signal) + len(noise)

        actual_noise_ratio = (len(noise) / len(signal)) if signal else 0
        notes.append(
            f"Signal:noise = {len(signal)}:{len(noise)} "
            f"({100*len(signal)//max(total_items,1)}% signal)"
        )

        if actual_noise_ratio < 0.4:
            notes.append(
                "Warning: noise ratio below recommended 50:50 — consider adding "
                "more calibration items from your broader pool."
            )

        if self.position_optimize and total_items > _LONG_CONTEXT_THRESHOLD:
            ordered = noise + signal
            notes.append(
                f"Long context ({total_items} items) -> signal placed at END (recency bias)."
            )
        elif self.position_optimize and total_items <= _LONG_CONTEXT_THRESHOLD:
            import random
            combined = signal + noise
            random.shuffle(combined)
            ordered = combined
            notes.append(
                f"Short context ({total_items} items) -> random order (position effect null at <=15 items)."
            )
        else:
            ordered = signal + noise

        return Batch(
            batch_id=batch_id,
            item_type=type_label,
            items=ordered,
            signal_count=len(signal),
            noise_count=len(noise),
            strategy_notes=notes,
        )


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_SIGNAL_KEYWORDS = re.compile(
    r"\b("
    r"action|assigned|deadline|owner|must|need to|will be|decision|decided|"
    r"blocker|blocked|dependency|approve|approved|denied|shipped|released|"
    r"next step|follow.?up|todo|by [A-Z][a-z]+day|due|ETA|milestone|"
    r"risk|escalat|on track|off track|delayed|slip|P[0-9]|sev[0-9]|"
    r"commit|confirm|agreed|resolved|unresolved|pending"
    r")\b",
    re.IGNORECASE,
)

_NOISE_SIGNALS = re.compile(
    r"\b(OOO|out of office|thanks|thank you|noted|sounds good|"
    r"invite|accepted|declined|fyi|heads up|reminder)\b",
    re.IGNORECASE,
)


def _heuristic_relevance(content: str, subject: str = "") -> float:
    """Lightweight keyword heuristic to estimate item relevance (0-1).

    This is intentionally simple and fast. For production pipelines,
    replace with your retrieval similarity score or a cheap model call.
    """
    text = subject + " " + content
    word_count = len(content.split())

    if word_count < 10:
        return 0.1

    signal_hits = len(_SIGNAL_KEYWORDS.findall(text))
    noise_hits  = len(_NOISE_SIGNALS.findall(text))

    score = min(1.0, signal_hits * 0.15) - min(0.3, noise_hits * 0.1)

    has_numbers = bool(re.search(r"\b\d{1,3}%|\$[\d,]+|\d+ [a-z]+ \d{4}", text, re.I))
    if has_numbers:
        score += 0.1

    return max(0.0, min(1.0, score))


def _group_by_type(signal, noise):
    type_order = ["meeting", "email", "chat", "other"]
    all_types = set(i.item_type for i in signal + noise)
    ordered_types = [t for t in type_order if t in all_types] + \
                    [t for t in all_types if t not in type_order]

    groups = {}
    noise_by_type: dict[str, list[Item]] = {}
    for item in noise:
        noise_by_type.setdefault(item.item_type, []).append(item)

    for t in ordered_types:
        t_signal = [i for i in signal if i.item_type == t]
        if not t_signal:
            continue
        t_noise = list(noise_by_type.get(t, []))
        if len(t_noise) < len(t_signal):
            for other_type in ordered_types:
                if other_type == t:
                    continue
                t_noise.extend(noise_by_type.get(other_type, []))
        groups[t] = (t_signal, t_noise)

    return groups


def _chunk_list(lst: list, size: int) -> list[list]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_items_from_json(path: str) -> list[Item]:
    """Load items from a JSON file.

    Expected format: list of objects with at minimum "id" and "content".
    Optional: "item_type", "subject", "relevance_hint", "manual_score".
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    for obj in raw:
        items.append(Item(
            id=str(obj.get("id", obj.get("item_id", "?"))),
            content=obj.get("content", ""),
            item_type=obj.get("item_type", "other"),
            subject=obj.get("subject", ""),
            relevance_hint=obj.get("relevance_hint"),
            manual_score=obj.get("manual_score"),
            metadata={k: v for k, v in obj.items()
                      if k not in {"id", "item_id", "content", "item_type",
                                   "subject", "relevance_hint", "manual_score"}},
        ))
    return items


def _demo_items() -> list[Item]:
    """Small demo dataset for testing without a real file."""
    return [
        Item("M1", "Team agreed to delay v2.3 release to May 15. Owner: Alex.",
             "meeting", "Sprint Planning", relevance_hint=0.9),
        Item("M2", "Demo of new dashboard. No decisions made.",
             "meeting", "Demo Review", relevance_hint=0.2),
        Item("M3", "Blocker: storage quota exceeded in prod. Escalated to infra.",
             "meeting", "Ops Standup", relevance_hint=0.95),
        Item("E1", "Thanks for joining yesterday's call!",
             "email", "Re: call", relevance_hint=0.0),
        Item("E2", "FYI — office closed Friday.",
             "email", "Office Closure", relevance_hint=0.0),
        Item("E3", "Action required: approve the Q2 headcount plan by EOD Thursday.",
             "email", "Q2 Headcount", relevance_hint=0.85),
        Item("E4", "Quick note — the build pipeline is fixed now.",
             "email", "Build fix", relevance_hint=0.6),
        Item("E5", "Reminder: team lunch at noon.",
             "email", "Team Lunch", relevance_hint=0.0),
        Item("C1", "Sounds good, I'll take a look",
             "chat", "", relevance_hint=0.0),
        Item("C2", "The SLA breach ticket is assigned to me, due EOD.",
             "chat", "", relevance_hint=0.8),
        Item("C3", "noted, thx",
             "chat", "", relevance_hint=0.0),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Arrange context window items for optimal LLM extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python context_arranger.py items.json
  python context_arranger.py items.json --render
  python context_arranger.py --demo
  python context_arranger.py items.json --no-type-routing --noise-ratio 0
""",
    )
    parser.add_argument("input", nargs="?",
                        help="Path to JSON file containing items")
    parser.add_argument("--demo", action="store_true",
                        help="Run on built-in demo items instead of a file")
    parser.add_argument("--render", action="store_true",
                        help="Print the full rendered prompt text for each batch")
    parser.add_argument("--no-type-routing", dest="type_routing",
                        action="store_false", default=True,
                        help="Disable type-based routing")
    parser.add_argument("--noise-ratio", type=float, default=1.0,
                        help="Calibration noise items per signal item (default 1.0)")
    parser.add_argument("--max-batch", type=int, default=20,
                        help="Max items per LLM call (default 20)")
    parser.add_argument("--signal-threshold", type=float, default=0.45,
                        help="Relevance score threshold for signal (default 0.45)")
    args = parser.parse_args()

    if args.demo:
        items = _demo_items()
        print("Running on built-in demo items (11 items across meeting/email/chat).\n")
    elif args.input:
        items = _load_items_from_json(args.input)
        print(f"Loaded {len(items)} items from {args.input}\n")
    else:
        parser.print_help()
        sys.exit(0)

    arranger = ContextArranger(
        type_routing=args.type_routing,
        target_noise_ratio=args.noise_ratio,
        max_batch_size=args.max_batch,
        signal_threshold=args.signal_threshold,
    )

    plan = arranger.arrange(items)
    print(plan.summary())

    if args.render:
        print("\n" + "=" * 60)
        print("RENDERED PROMPTS")
        print("=" * 60)
        for batch in plan.batches:
            print(f"\n--- Batch {batch.batch_id} [{batch.item_type}] ---")
            print(batch.render())


if __name__ == "__main__":
    main()
