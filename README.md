# Context Window Experiment Harness

A reusable Python framework for measuring how **context window composition** (signal-to-noise ratio, item position, type routing) affects LLM extraction quality.

## What This Is

This harness lets you:
1. **Load** scored communication items (meetings, emails, chats) from CSVs
2. **Compose** context windows with controlled signal:noise ratios and positions
3. **Call** any LLM (Azure AI, Azure OpenAI, GitHub Models, or any local proxy)
4. **Score** extractions against human ground truth (relevance + fact types)
5. **Run** full factorial experiments with matched-pool paired comparisons

It also includes `context_arranger.py` — a standalone tool that applies empirically-validated arrangement strategies to any set of items before sending them to an LLM.

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env with your API keys

# 3. Prepare data (see DATA_SCHEMA.md for CSV formats)
mkdir data
# Place your CSVs in data/

# 4. Verify data loads
python -m experiment.data_loader

# 5. Run a dry-run (no API calls — validates prompt construction)
python -m experiment.runner dry-run dry-run any-model

# 6. Run a quick smoke test (1 trial per config)
python -m experiment.runner quick azure-ai DeepSeek-V3.2

# 7. Run the full factorial experiment (matched-pool design)
python -m experiment.runner factorial azure-openai gpt-4o
```

## Architecture

```
harness/
├── experiment/
│   ├── __init__.py
│   ├── data_loader.py    # Load CSVs → DataItem list
│   ├── llm_client.py     # Multi-provider LLM caller with retry
│   ├── runner.py          # Experiment orchestration + factorial design
│   └── scorer.py          # Score extractions vs ground truth
├── context_arranger.py    # Standalone arrangement utility (CLI + API)
├── DATA_SCHEMA.md         # CSV format documentation
├── .env.example           # Credential template
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Experiment Modes

| Mode | Trials | Purpose |
|------|--------|---------|
| `dry-run` | 8 | Validate prompt construction, no API calls |
| `quick` | 8 | 1 trial per config — smoke test |
| `pilot` | 40 | 5 trials per config — directional signal |
| `full` | 160 | 20 trials per config — publishable |
| `factorial` | 220 | 20 matched-pool blocks × 11 configs — confirmatory |
| `noise_curve` | 180 | 20 trials × 9 ratio points — dose-response |

## Supported Providers

| Provider | Env Vars Required | Models |
|----------|-------------------|--------|
| `azure-openai` | `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT` | GPT-4o, GPT-5.x |
| `azure-ai` | `AZURE_AI_KEY`, `AZURE_AI_ENDPOINT` | DeepSeek, Llama, Grok, Kimi |
| `github-models` | `GITHUB_TOKEN` | Claude, Gemini, any GitHub marketplace model |
| `local-bridge` | (optional) `LOCAL_BRIDGE_PORT` | Any OpenAI-compatible local proxy |
| `dry-run` | (none) | Prints prompts, returns empty JSON |

## Context Arranger (Standalone Tool)

Use `context_arranger.py` independently to arrange items before sending to any LLM:

```bash
# Demo with built-in sample items
python context_arranger.py --demo

# Arrange your own items (JSON format — see DATA_SCHEMA.md)
python context_arranger.py items.json --render
```

**Python API:**
```python
from context_arranger import ContextArranger, Item

items = [
    Item(id="M1", content="We decided to ship v2 on Friday.",
         item_type="meeting", relevance_hint=0.9),
    Item(id="E1", content="Thanks for the update!",
         item_type="email", relevance_hint=0.05),
    # ...
]

arranger = ContextArranger(type_routing=True, target_noise_ratio=1.0)
plan = arranger.arrange(items)

for batch in plan.batches:
    prompt = batch.render()  # send this to your LLM
```

## Scoring

The harness scores LLM outputs on three dimensions:

- **Relevance Accuracy** — does the model's relevance score (0-3) match the human label? (exact=1.0, off-by-1=0.5)
- **Fact Type Recall** — what fraction of human-labelled fact types did the model find? (with partial credit for similar types)
- **Fact Type Precision** — what fraction of the model's predicted types are correct?

Partial credit pairs: DECISION↔ACTION (0.5), STATUS↔INFO (0.5), BLOCKER↔STATUS (0.3).

## Bringing Your Own Data

See [DATA_SCHEMA.md](DATA_SCHEMA.md) for the complete CSV format specification. At minimum you need:
- 5+ items scored 2-3 (signal)
- 5+ items scored 0 (noise)
- Any domain works — the harness is content-agnostic

## License

MIT
