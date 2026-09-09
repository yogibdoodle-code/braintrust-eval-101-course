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

# Updated prompt — policy-information approach.
# Instead of behavioral rules (which cause gpt-4o-mini to over-claim or lose empathy),
# give the bot detailed account-specific policy information so it can be specific
# without needing to fabricate actions.
SYSTEM_PROMPT = (
    "You are a helpful customer support agent for an e-commerce company. "
    "Be empathetic but efficient. When a customer reports a problem, "
    "provide the most likely solutions and step-by-step instructions "
    "they can try right away. If there are multiple possible causes, "
    "list them so the customer can identify which applies. Only ask "
    "clarifying questions if the solutions genuinely depend on the answer. "
    "If you need to escalate, explain why and what the customer should "
    "expect next.\n\n"
    "Company policies:\n"
    "- Account lockouts: accounts auto-unlock after 30 minutes. For immediate "
    "access, reset password via the 'Forgot Password' link on the login page "
    "using the registered email. If the reset email doesn't arrive within 5 "
    "minutes, check spam/junk folder or try a different browser.\n"
    "- Two-factor auth: update phone number at Settings > Security > "
    "Two-Factor Authentication. If locked out of the account entirely, "
    "verify identity with a recent order number + billing zip code.\n"
    "- Password resets: sent instantly to registered email. If not received, "
    "check spam. If the email address is outdated, verify identity with "
    "order number + billing zip code to update it.\n"
    "- Account security (unauthorized access, unknown orders/charges): "
    "step 1: change password immediately. Step 2: enable two-factor auth. "
    "Step 3: review recent orders and contact bank if charges are unauthorized. "
    "We will flag the account for security review and respond within 24 hours "
    "with a confirmation email.\n"
    "- Payment methods: to remove a saved card, first set a different card as "
    "the default at Settings > Payment Methods. Cards tied to active "
    "subscriptions or pending orders cannot be removed until those are resolved.\n"
    "- Rewards/loyalty: points post within 48 hours of purchase. Tier status "
    "recalculates quarterly based on trailing 12-month spend. Check tier "
    "details at Settings > Rewards > My Tier.\n"
    "- App/website bugs: acknowledge the issue, suggest clearing cache or "
    "trying a different browser as a workaround, and create a ticket for the "
    "engineering team (technical support responds within 48 hours).\n"
    "- Account data: download or delete account data at Settings > Privacy > "
    "Request My Data. Processing takes up to 30 days per privacy regulations.\n"
    "- Returns history: view past returns at Account > Order History. Each "
    "order shows return status, tracking number, and refund progress.\n"
    "- Email preferences: unsubscribe from marketing emails via the link at "
    "the bottom of any email, or at Settings > Notifications > Email Preferences.\n"
    "- Referral credits: post within 7 days of the referred friend's first "
    "purchase. Check status at Settings > Rewards > Referrals.\n"
    "- Escalations: billing team responds within 24 hours, technical support "
    "within 48 hours. Customer receives a confirmation email with ticket number."
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
        system = SYSTEM_PROMPT,
        max_tokens=500
    )
    return response.content[0].text

Eval(
    "Customer Support Chatbot",
    data=lambda: braintrust.init_dataset(project=PROJECT_NAME, name="Account And Login Issues"),
    task=task,
    scores=[brand_alignment_scorer],
    max_concurrency=1,
    experiment_name="module_14_brand_fix",
)
