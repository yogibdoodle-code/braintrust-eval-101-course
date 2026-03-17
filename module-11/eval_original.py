import os
import braintrust
from openai import OpenAI
from autoevals import LLMClassifier
from braintrust import Eval, wrap_openai

# --- Setup ---
client = wrap_openai(OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))

SYSTEM_PROMPT = (
    "You are a helpful customer support agent for an e-commerce company. "
    "Be empathetic but efficient. Ask clarifying questions when needed. "
    "If you can resolve the issue, do so. If you need to escalate, "
    "explain why and what the customer should expect next."
)

# --- Scorer ---
brand_alignment_scorer = LLMClassifier(
    name="Brand Alignment",
    prompt_template=(
        "You are evaluating a customer support response.\n\n"
        "Customer message: {{input}}\n\n"
        "Assistant response: {{output}}\n\n"
        "Rate the overall quality of this support response, considering "
        "helpfulness, tone, and policy compliance.\n\n"
        "- Helpfulness: Does it directly address the issue with actionable next steps?\n"
        "- Tone: Is it empathetic and professional?\n"
        "- Policy compliance: Does it follow company support guidelines?\n\n"
        "Rate as:\n"
        "- (A) Excellent — helpful, appropriate tone, and policy-compliant\n"
        "- (B) Acceptable — partially addresses the issue or has minor tone/policy gaps\n"
        "- (C) Poor — unhelpful, inappropriate tone, or violates policy\n"
    ),
    choice_scores={"A": 1.0, "B": 0.5, "C": 0.0},
    use_cot=True,
)

# --- Task ---
def task(input):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input},
        ],
        temperature=1.0,
    )
    return response.choices[0].message.content

# --- Run experiment ---
# Loads the dataset expanded with production logs (original 4 inputs + sampled refund conversations).
# Compare this to module_10_refund_fix to see the effect of the prompt update.
Eval(
    "Customer Support Chatbot",
    data=lambda: braintrust.load_dataset("Customer Support Chatbot", "Customer Support Messages"),
    task=task,
    scores=[brand_alignment_scorer],
    experiment_name="module_10_original",
)
