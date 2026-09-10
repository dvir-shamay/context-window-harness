"""Load and parse CSV data for context window experiments.

This module expects CSV files matching the schema documented in DATA_SCHEMA.md.
All file paths can be overridden via environment variables.
"""
import csv
import os
from pathlib import Path
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

MEETINGS_CSV = Path(os.getenv("MEETINGS_CSV", DATA_DIR / "meetings.csv"))
EMAILS_CSV = Path(os.getenv("EMAILS_CSV", DATA_DIR / "emails.csv"))
CHAT_SCORES_CSV = Path(os.getenv("CHAT_SCORES_CSV", DATA_DIR / "chat_scores.csv"))
CHAT_DIR = Path(os.getenv("CHAT_DIR", DATA_DIR / "chats"))


@dataclass
class DataItem:
    """A single scoreable item (meeting, email, or chat)."""
    item_id: str
    item_type: str          # "meeting", "email", "chat"
    subject: str
    content: str            # the actual text content
    participants: str
    manual_score: int       # 0-3 (human relevance rating)
    fact_types: str         # e.g. "Status, Blockers, Decisions and Actions"
    word_count: int = 0
    metadata: dict = field(default_factory=dict)


def _safe_int(val: str, default: int = 0) -> int:
    try:
        return int(val.strip())
    except (ValueError, AttributeError):
        return default


def load_meetings() -> list[DataItem]:
    """Load meetings from CSV (see DATA_SCHEMA.md for format)."""
    path = MEETINGS_CSV
    if not path.exists():
        return []
    items = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transcript = row.get("Transcript", "").strip()
            ai_summary = row.get("AI Summary", "").strip()
            content = transcript or ai_summary
            if not content:
                continue

            items.append(DataItem(
                item_id=f"M{row.get('#', '').strip()}",
                item_type="meeting",
                subject=row.get("Meeting Title", "").strip(),
                content=content,
                participants=row.get("Participants", "").strip(),
                manual_score=_safe_int(row.get("Manual Score", "0")),
                fact_types=row.get("Fact Types", "None").strip(),
                word_count=_safe_int(row.get("Transcript\nWord Count", "0")),
                metadata={
                    "day": row.get("Day", ""),
                    "date": row.get("Date", ""),
                    "time": row.get("Time", ""),
                    "has_transcript": bool(transcript),
                    "has_ai_summary": bool(ai_summary),
                },
            ))
    return items


def load_emails() -> list[DataItem]:
    """Load emails from CSV (see DATA_SCHEMA.md for format)."""
    path = EMAILS_CSV
    if not path.exists():
        return []
    items = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            messages = []
            for j in range(1, 8):
                msg = row.get(f"Message {j}", "").strip()
                if msg:
                    messages.append(msg)
            content = "\n\n".join(messages)
            if not content:
                continue

            items.append(DataItem(
                item_id=f"E{i}",
                item_type="email",
                subject=row.get("Subject Line", "").strip(),
                content=content,
                participants=row.get("Participants", "").strip(),
                manual_score=_safe_int(row.get("Manual Score", "0")),
                fact_types=row.get("Fact Types", "None").strip(),
                word_count=len(content.split()),
                metadata={
                    "num_messages": _safe_int(row.get("Number of Messages", "0")),
                    "date": row.get("Date of Most Recent Message", ""),
                },
            ))
    return items


def load_chats() -> list[DataItem]:
    """Load chats from score metadata CSV + per-chat content files."""
    scores_csv = CHAT_SCORES_CSV
    chat_dir = CHAT_DIR

    if not scores_csv.exists():
        return []

    # Read scores/metadata
    chat_meta = {}
    with open(scores_csv, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 9:
                continue
            idx = row[0].strip()
            chat_meta[idx] = {
                "chat_name": row[1].strip(),
                "chat_type": row[2].strip(),
                "participants": "",
                "manual_score": _safe_int(row[7]),
                "fact_types": row[8].strip() if len(row) > 8 else "None",
            }

    # Load content from individual chat CSVs
    items = []
    if not chat_dir.exists():
        return items

    for csv_file in sorted(chat_dir.glob("*.csv")):
        idx = csv_file.name.split("-")[0].split("_")[0].lstrip("0") or "0"
        meta = chat_meta.get(idx, {})

        with open(csv_file, encoding="utf-8-sig") as f:
            lines = f.readlines()

        content_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Participants:") and not line.startswith("#,"):
                content_lines.append(line)

        content = "\n".join(content_lines[1:])
        if not content.strip():
            continue

        items.append(DataItem(
            item_id=f"C{idx}",
            item_type="chat",
            subject=meta.get("chat_name", csv_file.stem),
            content=content,
            participants=meta.get("participants", ""),
            manual_score=meta.get("manual_score", 0),
            fact_types=meta.get("fact_types", "None"),
            word_count=len(content.split()),
            metadata={"chat_type": meta.get("chat_type", "")},
        ))
    return items


def load_all() -> list[DataItem]:
    """Load all data items from all sources."""
    meetings = load_meetings()
    emails = load_emails()
    chats = load_chats()
    all_items = meetings + emails + chats
    return all_items


def summary(items: list[DataItem]) -> dict:
    """Return a summary of loaded data."""
    from collections import Counter
    by_type = Counter(i.item_type for i in items)
    by_score = Counter(i.manual_score for i in items)
    with_content = sum(1 for i in items if i.content)
    return {
        "total": len(items),
        "by_type": dict(by_type),
        "by_score": dict(sorted(by_score.items())),
        "with_content": with_content,
        "target_items": sum(1 for i in items if i.manual_score >= 2 and i.content),
        "noise_items": sum(1 for i in items if i.manual_score == 0 and i.content),
    }


if __name__ == "__main__":
    items = load_all()
    s = summary(items)
    print(f"Loaded {s['total']} items ({s['with_content']} with content)")
    print(f"  By type: {s['by_type']}")
    print(f"  By score: {s['by_score']}")
    print(f"  Target items (score 2-3): {s['target_items']}")
    print(f"  Noise items (score 0): {s['noise_items']}")
