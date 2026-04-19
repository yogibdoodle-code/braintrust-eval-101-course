# Module 11: Analyzing Multi-Turn Traces

This module scores production conversation traces at two levels: **per-turn** and **per-trace**.

- **Per-turn (`Brand Alignment`)** — Scores each individual assistant response for helpfulness, tone, and policy compliance. Written to each turn span.
- **Per-trace (`Conversation Quality`)** — Scores the full conversation as a unit: was the issue resolved, or was there unnecessary back-and-forth? Written to the root span.

Both scorers use chain-of-thought reasoning (`use_cot=True`), and the rationale is written back as metadata so you can inspect *why* each score was given.

## Assets

- **`score_traces.py`** — Fetches all spans from your project logs, groups them by trace, runs both scorers, and writes scores + CoT rationale back to Braintrust via the insert API.
- **`requirements.txt`** — Python dependencies.

## Prerequisites

This module assumes you have:

1. A "Customer Support Chatbot" project in Braintrust with logs from Module 10.
2. Your project ID. Find it by running:

```bash
curl "https://api.braintrust.dev/v1/project?project_name=Customer+Support+Chatbot" \
  -H "Authorization: Bearer $BRAINTRUST_API_KEY"
```

## Setup

```bash
pip install -r requirements.txt
```

Set your API keys and project ID:

```bash
export BRAINTRUST_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export BRAINTRUST_PROJECT_ID="your-project-id"
```

## Run

```bash
python score_traces.py
```

The script prints each trace ID with per-turn Brand Alignment grades and a trace-level Conversation Quality grade (A / B / C). Open the Logs tab in your Braintrust project to see scores on each span and CoT rationale in the metadata.
