# Module 10: Building a Multi-Turn Chat App

## Assets

- **`chat_app.py`** — Interactive CLI chat app instrumented with Braintrust logging. Every conversation turn is traced and logged.
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
python chat_app.py
```

Type customer support messages and chat with the bot. Type `quit` to exit. Every conversation turn is logged to the "Customer Support Chatbot" project in Braintrust.
