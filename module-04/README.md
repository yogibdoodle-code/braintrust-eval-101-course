# Module 4: Build a Simple Eval in Code

## Assets

- **`eval_customer_support.py`** — Complete eval script that runs two customer support chatbot personalities (Polite vs Concise) and scores them for helpfulness and tone.
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
python eval_customer_support.py
```

This will run two experiments ("Polite Personality" and "Concise Personality") and upload the results to the "Customer Support Chatbot" project in Braintrust. You can then compare them in the Braintrust UI.
