"""Replay the 5-turn customer support conversation from production logs.
Runs the same conversation through the chatbot non-interactively and logs to Braintrust.
"""
import os
from openai import OpenAI
from braintrust import init_logger, traced, wrap_openai

logger = init_logger(project="Customer Support Chatbot")
client = wrap_openai(OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))

SYSTEM_PROMPT = (
    "You are a helpful customer support agent for an e-commerce company. "
    "Be empathetic but efficient. Ask clarifying questions when needed. "
    "If you can resolve the issue, do so. If you need to escalate, "
    "explain why and what the customer should expect next."
)

# The 5 customer messages from the production conversation.
USER_MESSAGES = [
    "I ordered a tshirt and it come a size too big",
    "Yes, my order number is 6767 and the size I ordered was S and I got a M",
    "I want to exchange for correct size",
    "But you don't know what my email is",
    "can't you get it from my previous order?",
]

@traced
def chat(conversation_history):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation_history,
        temperature=1.0,
    )
    return response.choices[0].message.content

def main():
    print("Replaying 5-turn customer support conversation")
    print("-" * 50)

    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    turn_number = 0

    with logger.start_span(name="conversation") as conversation_span:
        for user_input in USER_MESSAGES:
            conversation_history.append({"role": "user", "content": user_input})
            turn_number += 1

            print(f"\nYou: {user_input}")

            with conversation_span.start_span(name=f"turn_{turn_number}") as turn_span:
                response = chat(conversation_history)
                conversation_history.append({"role": "assistant", "content": response})
                turn_span.log(
                    input=user_input,
                    output=response,
                    metadata={"turn_number": turn_number},
                )

            print(f"\nAgent: {response}")

        conversation_span.log(
            input=conversation_history,
            output=conversation_history[-1]["content"],
            metadata={"total_turns": turn_number},
        )

    print("\nDone. Check the Braintrust Logs tab for the new trace.")

if __name__ == "__main__":
    main()
