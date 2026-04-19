import os
import braintrust
from openai import OpenAI
from braintrust import Eval, init_function

braintrust.init(project="Customer Support Chatbot")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "You are a helpful customer support agent for an e-commerce company. "
    "Be empathetic but efficient. Ask clarifying questions when needed. "
    "If you can resolve the issue, do so. If you need to escalate, "
    "explain why and what the customer should expect next."
)

# References the Brand Alignment scorer defined in the Braintrust UI.
brand_alignment_scorer = init_function(
    project_name="Customer Support Chatbot",
    slug="brand-alignment-99e2",
)

def task(input):
    # Input may be a conversation history list (from logs) or a plain string.
    if isinstance(input, list):
        user_msg = next((m["content"] for m in input if m["role"] == "user"), "")
    else:
        user_msg = input
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    return response.choices[0].message.content

Eval(
    "Customer Support Chatbot",
    data=lambda: braintrust.init_dataset(project="Customer Support Chatbot", name="Account And Login Issues"),
    task=task,
    scores=[brand_alignment_scorer],
    max_concurrency=1,
    experiment_name="module_14_original",
)
