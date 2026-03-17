# Module 9: Discovering Patterns with Topics

## Assets

- **`generate_logs.py`** — Generates 200 single-turn customer support conversations across 5 topic areas (shipping delays, refund requests, product questions, account issues, order tracking). This meets the minimum threshold (200 traces) for Braintrust Topics to generate topic clusters. Uses `gpt-4o-mini` to keep costs low.
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

The script logs 200 conversations (80 unique messages, cycled to 200 and shuffled). Once it finishes, go to **Topics** in your project to set up topic maps and generate topic clusters.
