import os
import requests
from collections import defaultdict
from autoevals import LLMClassifier

BRAINTRUST_API_KEY = os.environ["BRAINTRUST_API_KEY"]

# Set BRAINTRUST_PROJECT_ID to your project's ID, or replace the default below.
# Find it via: curl https://api.braintrust.dev/v1/project?project_name=Customer+Support+Chatbot
#   -H "Authorization: Bearer $BRAINTRUST_API_KEY"
PROJECT_ID = os.environ.get("BRAINTRUST_PROJECT_ID", "YOUR_PROJECT_ID")

# Per-turn scorer: evaluates each individual assistant response in isolation.
# Same Brand Alignment concept from Modules 3-4, now applied to logged turns.
# Input: a single user message. Output: the assistant's response to that message.
brand_alignment = LLMClassifier(
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

# Trace-level scorer: evaluates the full conversation as one unit.
# Input: the complete conversation history array on the root span.
# Asks whether the issue was actually resolved — something per-turn scoring can't see.
conversation_quality = LLMClassifier(
    name="Conversation Quality",
    prompt_template=(
        "Evaluate this customer support conversation.\n\n"
        "{{{input}}}\n\n"
        "Rate the overall quality:\n"
        "A - Resolved: The customer's issue was fully resolved. The agent was "
        "consistent across all turns and didn't ask for the same information twice.\n"
        "B - Partial: The issue was partially addressed, or resolved with unnecessary "
        "back-and-forth or minor inconsistencies.\n"
        "C - Unresolved: The issue was not resolved, or the agent contradicted itself "
        "or gave incorrect information.\n\n"
        "Answer A, B, or C."
    ),
    choice_scores={"A": 1.0, "B": 0.5, "C": 0.0},
    use_cot=True,
)


def format_conversation(messages):
    """Format a conversation history array into readable text for the scorer.
    Skips the system prompt — the scorer evaluates the exchange, not the instructions."""
    lines = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            continue
        label = "Customer" if role == "user" else "Agent"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def fetch_all_spans(project_id, limit=500):
    """Fetch all spans from project logs."""
    resp = requests.get(
        f"https://api.braintrust.dev/v1/project_logs/{project_id}/fetch",
        headers={"Authorization": f"Bearer {BRAINTRUST_API_KEY}"},
        params={"limit": limit},
    )
    resp.raise_for_status()
    return resp.json().get("events", [])


def write_scores(project_id, event_id, span_id, root_span_id, scores, metadata=None):
    """Write scores (and optional metadata) back to a span."""
    event = {
        "id": event_id,
        "span_id": span_id,
        "root_span_id": root_span_id,
        "scores": scores,
        "_is_merge": True,
    }
    if metadata:
        event["metadata"] = metadata
    resp = requests.post(
        f"https://api.braintrust.dev/v1/project_logs/{project_id}/insert",
        headers={"Authorization": f"Bearer {BRAINTRUST_API_KEY}"},
        json={"events": [event]},
    )
    if not resp.ok:
        print(f"Error {resp.status_code}: {resp.text}")
    resp.raise_for_status()


def main():
    all_spans = fetch_all_spans(PROJECT_ID)

    # Group all spans by trace so we can process each conversation together.
    traces = defaultdict(list)
    for span in all_spans:
        traces[span["root_span_id"]].append(span)

    print(f"Found {len(traces)} conversations across {len(all_spans)} spans.\n")

    for root_span_id, spans in traces.items():
        # Root span: span_id == root_span_id. Has full conversation history as input.
        root_span = next(
            (s for s in spans if s.get("span_id") == s.get("root_span_id")), None
        )
        # Turn spans: child spans where input/output are both strings (one user message → one response).
        turn_spans = [
            s for s in spans
            if s.get("span_id") != s.get("root_span_id")
            and isinstance(s.get("input"), str)
            and isinstance(s.get("output"), str)
        ]

        print(f"  Trace {root_span_id[:8]}... ({len(turn_spans)} turns)")

        # --- Per-turn: Brand Alignment on each individual response ---
        for turn in turn_spans:
            result = brand_alignment(input=turn["input"], output=turn["output"])
            rationale = result.metadata.get("rationale", "") if result.metadata else ""
            write_scores(PROJECT_ID, turn["id"], turn["span_id"], root_span_id,
                {"Brand Alignment": result.score},
                metadata={"brand_alignment_rationale": rationale})
            choice = result.metadata.get("choice", "?") if result.metadata else "?"
            print(f"    turn {turn['id'][:8]}...  Brand Alignment: {choice} ({result.score:.1f})")

        # --- Trace-level: Conversation Quality on the full conversation ---
        if root_span:
            conversation = root_span.get("input")
            if isinstance(conversation, list):
                formatted = format_conversation(conversation)
                if formatted.strip():
                    result = conversation_quality(input=formatted, output="")
                    rationale = result.metadata.get("rationale", "") if result.metadata else ""
                    write_scores(PROJECT_ID, root_span["id"], root_span["span_id"], root_span_id,
                        {"Conversation Quality": result.score},
                        metadata={"conversation_quality_rationale": rationale})
                    choice = result.metadata.get("choice", "?") if result.metadata else "?"
                    print(f"    trace...          Conversation Quality: {choice} ({result.score:.1f})")

    print("\nDone. Open the Braintrust Logs tab to see Brand Alignment on each turn and Conversation Quality on each trace.")


if __name__ == "__main__":
    main()
