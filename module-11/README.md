# Module 11: The Improvement Loop

## Assets

- **`eval_original.py`** — Runs the baseline eval (`module_10_original`) against the full dataset, including refund-related inputs sampled from production logs.
- **`eval_refund_fix.py`** — Runs the same eval with an updated system prompt that includes explicit refund-handling instructions (`module_10_refund_fix`). Compare to the baseline to verify the fix improved refund scores without regressing other inputs.
- **`requirements.txt`** — Python dependencies.

## Prerequisites

This module assumes you have:

1. A "Customer Support Chatbot" project in Braintrust.
2. A "Customer Support Messages" dataset in that project, expanded with refund conversations sampled from your production logs (as described in the module).

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

Run the baseline first:

```bash
python eval_original.py
```

Then run the fixed version:

```bash
python eval_refund_fix.py
```

Compare `module_10_original` vs `module_10_refund_fix` in the Braintrust UI. Refund inputs should score higher in the fix; everything else should stay the same.
