# Module 9: Online Scoring

## Assets

- **`generate_conversations.py`** — Generates 10 scripted customer support conversations (5 single-turn, 5 multi-turn) and logs them to Braintrust. Run this after configuring online scoring to see scores appear automatically.

## Prerequisites

1. A "Customer Support Chatbot" project in Braintrust.
2. Online scoring rules configured in the Braintrust UI (see the module walkthrough):
   - **Brand Alignment** on all spans (per-turn scoring)
   - **Conversation Quality** on root spans only (trace-level scoring)

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
python generate_conversations.py
```

The script generates 10 conversations across different complaint types (shipping delays, damaged products, wrong items, double charges, etc.) and logs them with the same span structure as the chat app from Module 6. Open the Logs tab to see online scores appear on each trace.
