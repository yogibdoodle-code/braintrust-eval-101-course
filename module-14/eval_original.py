import os
import braintrust
from braintrust import Eval, init_function, init_logger, wrap_anthropic
from anthropic import Anthropic

from dotenv import load_dotenv
load_dotenv()

BRAINTRUST_API_KEY = os.environ.get("BRAINTRUST_API_KEY")
PROJECT_NAME = "Customer Support Chat Bot"
model = os.environ.get("CLAUDE_MODEL")

braintrust.init(project=PROJECT_NAME)
client = wrap_anthropic(Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")))

SYSTEM_PROMPT = (
    "You are a helpful customer support agent for an e-commerce company. "
    "Be empathetic but efficient. Ask clarifying questions when needed. "
    "If you can resolve the issue, do so. If you need to escalate, "
    "explain why and what the customer should expect next."
)

# References the Brand Alignment scorer defined in the Braintrust UI.
brand_alignment_scorer = init_function(
    project_name=PROJECT_NAME,
    slug="brand-alignment-5133",
)

def task(input):
    # Input may be a conversation history list (from logs) or a plain string.
    if isinstance(input, list):
        user_msg = next((m["content"] for m in input if m["role"] == "user"), "")
    else:
        user_msg = input
    response = client.messages.create(
        model=model,
        messages=[
            {"role": "user", "content": user_msg},
        ],
        max_tokens=500,
        system = SYSTEM_PROMPT,
    )
    return response.content[0].text   

Eval(
    "Customer Support Chatbot",
    data=lambda: braintrust.init_dataset(project=PROJECT_NAME, name="Account And Login Issues"),
    task=task,
    scores=[brand_alignment_scorer],
    max_concurrency=1,
    experiment_name="module_14_original",
)
