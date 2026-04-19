# Module 13: Analyzing Production Logs

## Assets

- **`generate_logs.py`** — Generates 250 unique customer support conversations across 5 task areas (shipping and delivery, refunds and returns, product questions, account and login, billing and payments — 50 per task). About 50% are single-turn; the rest are multi-turn (2–5 turns) with LLM-generated customer follow-ups to simulate realistic back-and-forth. Uses `gpt-4o-mini` to keep costs low. This volume gives Topics enough data to generate meaningful clusters.
- **`requirements.txt`** — Python dependencies.

## Prerequisites

This module assumes you have a "Customer Support Chatbot" project in Braintrust.

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
python generate_logs.py
```

Each conversation is logged to Braintrust with a `conversation` root span and child `turn_N` spans, plus metadata including the task area and total turn count. Once it finishes, go to **Topics** in your project to set up topic maps and inspect clusters across tasks, sentiments, and cross-cutting issue patterns.
