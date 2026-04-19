# Module 7: Nondeterminism

This module demonstrates LLM nondeterminism and shows how `trial_count` can reduce scorer variance.

## Assets

- **`eval_with_trials.py`** — Runs the same polite vs. concise customer support eval as Module 6, but with `trial_count=3`. Each input is evaluated three times and the scores are averaged, producing a more stable measurement than a single run. Creates two experiments (`module_7_polite_persona`, `module_7_concise_persona`) for comparison against the Module 6 baselines.
- **`requirements.txt`** — Python dependencies.

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

```bash
python eval_with_trials.py
```

Compare `module_7_polite_persona` and `module_7_concise_persona` against `module_6_polite_persona` and `module_6_concise_persona` in the Braintrust UI. Rows with flipping grades across trials are borderline cases — that's a different signal than rows that consistently score B.
