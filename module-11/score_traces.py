import os
from braintrust import wrap_anthropic, traced, init_logger
import requests
from collections import defaultdict

from anthropic import Anthropic

from dotenv import load_dotenv
load_dotenv()

BRAINTRUST_API_KEY = os.environ.get("BRAINTRUST_API_KEY")
PROJECT_NAME = "Customer Support Chat Bot"
# model = "claude-sonnet-5" # os.environ.get("CLAUDE_MODEL")
model = os.environ.get("CLAUDE_MODEL")

logger = init_logger(project=PROJECT_NAME)
client = wrap_anthropic(Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")))

def resolve_project_id(project_name):
    """Look up a project's UUID by name. The REST API requires an ID, not a name."""
    resp = requests.get(
        "https://api.braintrust.dev/v1/project",
        headers={"Authorization": f"Bearer {BRAINTRUST_API_KEY}"},
        params={"project_name": project_name},
    )
    resp.raise_for_status()
    objects = resp.json().get("objects", [])
    if not objects:
        raise ValueError(f"Project '{project_name}' not found.")
    return objects[0]["id"]

def brand_alignment(input_text, output_text):
    """Score a single support response for Brand Alignment quality."""
    prompt = f"""You are evaluating a customer support response.

Customer message: {input_text}

Assistant response: {output_text}

Rate the overall quality of this support response, considering helpfulness, tone, and policy compliance.

- Helpfulness: Does it directly address the issue with actionable next steps?
- Tone: Is it empathetic and professional?
- Policy compliance: Does it follow company support guidelines?

Rate as:
- (A) Excellent — helpful, appropriate tone, and policy-compliant
- (B) Acceptable — partially addresses the issue or has minor tone/policy gaps
- (C) Poor — unhelpful, inappropriate tone, or violates policy

First, provide your reasoning, then answer with ONLY the letter (A, B, or C) on the final line."""

    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    choice_scores = {"A": 1.0, "B": 0.5, "C": 0.0}

    # Extract the last line which should be the choice
    lines = text.strip().split("\n")
    choice = lines[-1].strip().upper()
    if choice in choice_scores:
        score = choice_scores[choice]
        rationale = "\n".join(lines[:-1])
    else:
        # Fallback: try to find A, B, or C in the text
        for c in ["A", "B", "C"]:
            if c in text:
                choice = c
                score = choice_scores[c]
                rationale = text
                break
        else:
            choice = "?"
            score = 0.5
            rationale = text

    class Result:
        def __init__(self, s, c, r):
            self.score = s
            self.metadata = {"choice": c, "rationale": r}

    return Result(score, choice, rationale)


def conversation_quality(input_text):
    """Score a full conversation for overall quality."""
    prompt = f"""Did this customer support conversation successfully resolve the customer's issue?

{input_text}

Answer Y if the issue was fully resolved — the agent addressed the problem, provided concrete next steps, and didn't ask for the same information twice.
Answer N if the issue was not resolved, or if the conversation had significant problems — the agent contradicted itself, asked for information already provided, or ended without a clear resolution.

First provide your reasoning, then answer with ONLY Y or N on the final line."""

    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    choice_scores = {"Y": 1.0, "N": 0.0}

    # Extract the last line which should be the choice
    lines = text.strip().split("\n")
    choice = lines[-1].strip().upper()
    if choice in choice_scores:
        score = choice_scores[choice]
        rationale = "\n".join(lines[:-1])
    else:
        # Fallback
        for c in ["Y", "N"]:
            if c in text:
                choice = c
                score = choice_scores[c]
                rationale = text
                break
        else:
            choice = "?"
            score = 0.5
            rationale = text

    class Result:
        def __init__(self, s, c, r):
            self.score = s
            self.metadata = {"choice": c, "rationale": r}

    return Result(score, choice, rationale)


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
    project_id = resolve_project_id(PROJECT_NAME)
    all_spans = fetch_all_spans(project_id)

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

        print(f"Trace {root_span_id[:8]}... ({len(spans)} spans)")

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
            result = brand_alignment(turn["input"], turn["output"])
            rationale = result.metadata.get("rationale", "") if result.metadata else ""
            write_scores(project_id, turn["id"], turn["span_id"], root_span_id,
                {"Brand Alignment": result.score},
                metadata={"brand_alignment_rationale": rationale})
            choice = result.metadata.get("choice", "?") if result.metadata else "?"
            print(f"    turn {turn['id'][:8]}...  Brand Alignment: {choice} ({result.score:.1f})")

        # --- Trace-level: Conversation Quality on the full conversation ---
        if root_span:
            conversation = root_span.get("input")
            print(conversation)
            if isinstance(conversation, list):
                formatted = format_conversation(conversation)
                print(f"    trace...          Conversation:\n{formatted}\n")
                if formatted.strip():
                    result = conversation_quality(formatted)
                    print(f"    trace...          Conversation Quality: {result.metadata.get('choice', '?')} ({result.score:.1f})")
                    rationale = result.metadata.get("rationale", "") if result.metadata else ""
                    write_scores(project_id, root_span["id"], root_span["span_id"], root_span_id,
                        {"Conversation Quality": result.score},
                        metadata={"conversation_quality_rationale": rationale})
                    choice = result.metadata.get("choice", "?") if result.metadata else "?"
                    print(f"    trace...          Conversation Quality: {choice} ({result.score:.1f})")

    print("\nDone. Open the Braintrust Logs tab to see Brand Alignment on each turn and Conversation Quality on each trace.")


if __name__ == "__main__":
    main()
