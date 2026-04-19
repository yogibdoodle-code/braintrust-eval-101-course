# Module 14: The Improvement Loop

## Assets

- **`eval_original.py`** — Runs the baseline eval (`module_14_original`) against the "Account And Login Issues" dataset using the original system prompt. Runs at `temperature=0` with `max_concurrency=1` to avoid API rate limits and sampling variance.
- **`eval_brand_fix.py`** — Runs the same eval with an updated system prompt that adds concrete company policy information (`module_14_brand_fix`). Compare to the baseline to see whether the fix moves Brand Alignment scores.
- **`requirements.txt`** — Python dependencies.

## Prerequisites

This module assumes you have:

1. A "Customer Support Chatbot" project in Braintrust with production logs from Modules 10 and 12.
2. An **"Account And Login Issues"** dataset — create this by filtering the Logs tab to `topic=account_and_login`, `Conversation Quality=N`, `Brand Alignment≤0.5`, and `metadata.total_turns=1`, then saving the selected traces as a new dataset. The module walkthrough explains the reasoning behind each filter.
3. A Brand Alignment scorer configured in the UI with slug `brand-alignment-99e2`.

## Setup

```bash
pip install -r requirements.txt
```

Set your API keys:

```bash
export BRAINTRUST_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

## Run

Run the baseline:

```bash
python3 eval_original.py
```

Run the fix:

```bash
python3 eval_brand_fix.py
```

Compare `module_14_original` vs `module_14_brand_fix` in the Braintrust UI.

Expected result: the policy-info prompt does **not** improve Brand Alignment on this slice — the baseline lands around 52.6% and the fix around 50.0%. That's the lesson of the module: sometimes a reasonable hypothesis doesn't pan out, and the chain-of-thought rationale on the failing rows points you toward the next iteration (often a scorer calibration issue rather than a prompt issue on account/login flows).

## Notes

**Why temperature=0?** At `temperature=1.0`, scoring variance between runs can exceed the actual signal from a prompt change. Running both evals at `temperature=0` makes the comparison deterministic.

**Why max_concurrency=1?** The Brand Alignment scorer is called via `init_function`, which hits the Braintrust BTQL API on each invocation. The org rate limit is 20 requests/60s — `max_concurrency=1` keeps well within that.
