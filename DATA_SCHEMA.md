# Data Schema

This document describes the CSV formats expected by the experiment harness.
You supply your own data — the harness is data-agnostic.

---

## Directory Layout

```
data/
├── meetings.csv          # Meeting transcripts/summaries
├── emails.csv            # Email thread content
├── chat_scores.csv       # Chat metadata and scores
└── chats/                # Per-chat content files
    ├── 01-teamname.csv
    ├── 02-another.csv
    └── ...
```

All paths can be overridden via environment variables (see `.env.example`).

---

## meetings.csv

| Column | Type | Description |
|--------|------|-------------|
| `#` | int | Unique meeting index (becomes `M{#}` item ID) |
| `Day` | string | Day of week (e.g. "Monday") |
| `Date` | string | Date string (any format) |
| `Time` | string | Meeting time |
| `Meeting Title` | string | Subject/title of the meeting |
| `Participants` | string | Comma-separated participant names |
| `Transcript` | string | Full meeting transcript (primary content) |
| `AI Summary` | string | AI-generated summary (fallback if no transcript) |
| `Transcript\nWord Count` | int | Word count of transcript |
| `Manual Score` | int | Human relevance rating: 0=irrelevant, 1=low, 2=medium, 3=high |
| `Fact Types` | string | Comma-separated: "Decisions, Actions, Blockers, Status, Info, None" |

**Notes:**
- The harness uses `Transcript` if available, otherwise falls back to `AI Summary`.
- Rows without any content in both columns are skipped.
- `Manual Score` is the ground truth for scoring — items with score >= 2 are "signal".

---

## emails.csv

| Column | Type | Description |
|--------|------|-------------|
| `Subject Line` | string | Email thread subject |
| `Participants` | string | All participants in the thread |
| `Number of Messages` | int | Message count in thread |
| `Date of Most Recent Message` | string | Date string |
| `Message 1` | string | First message in thread |
| `Message 2` | string | Second message (if any) |
| `Message 3` | string | Third message (if any) |
| `Message 4` | string | Fourth message (if any) |
| `Message 5` | string | Fifth message (if any) |
| `Message 6` | string | Sixth message (if any) |
| `Message 7` | string | Seventh message (if any) |
| `Manual Score` | int | Human relevance rating: 0-3 |
| `Fact Types` | string | Comma-separated fact types |

**Notes:**
- All `Message N` columns are concatenated into one content block (separated by double newlines).
- Item IDs are assigned sequentially as `E1`, `E2`, `E3`, etc.
- Empty threads (no messages) are skipped.

---

## chat_scores.csv

A metadata/scoring file with one row per chat channel.

| Column Index | Description |
|---|---|
| 0 | Chat index (e.g. "1", "2", ...) |
| 1 | Chat name / channel title |
| 2 | Chat type (e.g. "group", "1:1") |
| 3-6 | (reserved / unused by harness) |
| 7 | `Manual Score` (int 0-3) |
| 8 | `Fact Types` (string) |

**Notes:**
- This is a positional CSV (column indices, not headers used internally).
- The chat index links to the corresponding content file in `chats/`.

---

## chats/ directory

Individual CSV files per chat channel.  Filename format:

```
{index}-{chatname}.csv
```

The leading numeric index (before the first `-`) is used to match against `chat_scores.csv`.

**Content format:** Plain text messages, one per line. The harness skips:
- Lines starting with `Participants:`
- Lines starting with `#,`
- The first content line (treated as a title/header)

Item IDs are assigned as `C{index}` (e.g. `C1`, `C2`).

---

## Scoring Scale

The `Manual Score` column uses a 0-3 relevance scale:

| Score | Meaning | Role in Experiments |
|-------|---------|---------------------|
| 0 | Irrelevant / noise | Used as "calibration noise" items |
| 1 | Low relevance | Treated as noise (score < 2) |
| 2 | Medium relevance | **Signal** — scored in evaluation |
| 3 | High relevance | **Signal** — scored in evaluation |

---

## Fact Types

The `Fact Types` column uses a comma-separated list from this vocabulary:

- `Decisions` — agreements, approvals, selections made
- `Actions` — tasks assigned, next steps, follow-ups
- `Blockers` — impediments, dependencies, escalations
- `Status` — progress updates, current state
- `Info` — context, background, FYI content
- `None` — no relevant facts

Example: `"Decisions, Actions and Blockers"`

---

## Minimum Viable Dataset

To run the harness, you need at minimum:
- **1 CSV file** (meetings OR emails — either is sufficient)
- At least **5 items with Manual Score >= 2** (signal)
- At least **5 items with Manual Score = 0** (noise)

The factorial experiment mode requires at least 10 signal and 9 noise items.

---

## Creating Your Own Dataset

1. Export your communications (meeting notes, emails, chat logs)
2. Create CSVs matching the schemas above
3. Score each item 0-3 for relevance to your domain
4. Label fact types (or set all to "None" — fact types are secondary metrics)
5. Place files in `data/` and run `python -m experiment.data_loader` to verify loading

The harness works with any domain — engineering, legal, medical, sales —
as long as items are scored on the 0-3 relevance scale.
