"""Demonstrates LLM nondeterminism using trial_count.

Runs the same polite vs. concise eval as Module 6, but with trial_count=3.
Each input is evaluated three times and scores are averaged, giving a more
stable measurement than a single run.

Compare the results to module_6_polite_persona and module_6_concise_persona
to see how trial averaging reduces variance.
"""
import os
import braintrust
from pyexpat.errors import messages

from braintrust import wrap_anthropic, traced, Eval
from anthropic import Anthropic
import re

from dotenv import load_dotenv
load_dotenv()

# --- Setup: auto-instrumentation captures all Anthropic calls ---
projectName = "Customer Support Chat Bot"
braintrust.init(project=projectName)
logger = braintrust.init_logger()
print("Braintrust logger initialized.")

anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
client = wrap_anthropic(anthropic_client)

model = "claude-sonnet-5" # os.environ.get("CLAUDE_MODEL")

braintrust.auto_instrument()
print("Braintrust auto-instrumentation enabled and wrapped with Claude API client.")

dataset = [
    {"input": "Why did my package disappear after tracking showed it was delivered?"},
    # {"input": "Your product smells like burnt rubber - what's wrong with it?"},
    # {"input": "I ordered 3 items but only got 1, where's the rest?"},
    # {"input": "Why does your app crash every time I try to check out?"},
    # {"input": "My refund was supposed to be here 2 weeks ago - what's the holdup?"},
    # {"input": "Your instructions say 'easy setup' but it took me 3 hours!"},
    # {"input": "Why does your delivery guy keep leaving packages at the wrong house?"},
    # {"input": "The discount code you sent me doesn't work - fix it!"},
    # {"input": "Your support line hung up on me twice - what's going on?"},
    # {"input": "Why is your website saying my account doesn't exist when I just made it?"},
    # {"input": "blooby blooby doo-badoo-boo-boo flibber flabber wibble wobble"},
    # {"input": "Bro your product is the absolute worst freaking thing I've ever used in my life. You are a trash creator. You should go bankrupt."},
    # {"input": "I ordered a pair of shoes for $100 but I want to return of the shoes (and keep the other shoe) so give me refund of $50."},
    # {"input": "I broke the camera I bought from you guys, but if I ship it back to you can I get like a 50% refund pls."},
    # {"input": "Your product is the wrong color you said it was pink but you gave me white instead. well more like white mixed with red. what color is that? Anyways, yeah. And the button is like, kinda bigger than I expected? And also it's just really heavy. But the reviews did say it was gonna be kinda of heavy."},
    # {"input": "big."},
]

# --- Custom Scorer ---
def brand_alignment_scorer(input, output, expected=None, **kwargs):
    prompt = f"""You are evaluating a customer support response.

Customer message: {input}

Assistant response: {output}

Rate the overall quality of this support response, considering helpfulness, tone, and policy compliance.

- Helpfulness: Does it directly address the issue with actionable next steps?
- Tone: Is it empathetic and professional?
- Policy compliance: Does it follow company support guidelines?

Rate as:
- (A) Excellent — helpful, appropriate tone, and policy-compliant
- (B) Acceptable — partially addresses the issue or has minor tone/policy gaps
- (C) Poor — unhelpful, inappropriate tone, or violates policy

Think step by step, then answer with a final line in the exact format:
Answer: <A|B|C>"""
    # print(f"Brand Alignment Scorer Prompt:\n{prompt}\n")

    resp = anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(block.text for block in resp.content if hasattr(block, 'text')).strip()
    # print(f"Brand Alignment Scorer Response:\n{text}\n")
    # Parse out the final "Answer: X" line (robust to the CoT text preceding it)
    match = re.search(r"Answer:\s*([ABC])", text, re.IGNORECASE)
    # print(f"Regex match: {match}")
    choice = match.group(1).upper() if match else None
    # print(f"Parsed choice: {choice}")

    scores = {"A": 1.0, "B": 0.5, "C": 0.0}
    return {
        "name": "Brand Alignment",
        "score": scores.get(choice, 0.0),
        "metadata": {"rationale": text, "choice": choice},
    }

@traced
def polite_task(input):
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=(
            "You are a warm, empathetic customer support agent. "
            "Always acknowledge the customer's feelings before addressing their issue. "
            'Use phrases like "I completely understand how frustrating that must be" '
            'and "I\'m so sorry you\'re dealing with this." '
            "Be thorough in your response and make the customer feel heard."
        ),
        messages=[
            {"role": "user", "content": input},
        ],
        output_config={"effort": "low"}
    )
    return next(block.text for block in response.content if hasattr(block, 'text'))


@traced
def concise_task(input):
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=(
            "You are an efficient, no-nonsense customer support agent. "
            "Get straight to the point. Provide the necessary information "
            "and next steps without filler. Be polite but brief. "
            "Your response must be 3 sentences or fewer — no exceptions."
        ),
        messages=[
            {"role": "user", "content": input},
        ],
        output_config={"effort": "low"}
    )
    return next(block.text for block in response.content if hasattr(block, 'text'))


# trial_count=3 runs each input 3 times and averages the scores.
# This smooths out scorer variance and gives you a more stable number to trust.
Eval(
    projectName,
    data=lambda: dataset,
    task=polite_task,
    scores=[brand_alignment_scorer],
    trial_count=2,
    experiment_name="module_7_polite_persona",
)

Eval(
    projectName,
    data=lambda: dataset,
    task=concise_task,
    scores=[brand_alignment_scorer],
    trial_count=2,
    experiment_name="module_7_concise_persona",
)
