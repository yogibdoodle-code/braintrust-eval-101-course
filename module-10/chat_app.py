import os
from braintrust import wrap_anthropic, traced, init_logger
from anthropic import Anthropic

from dotenv import load_dotenv
load_dotenv()

projectName = "Customer Support Chat Bot"
# model = "claude-sonnet-5" # os.environ.get("CLAUDE_MODEL")
model = os.environ.get("CLAUDE_MODEL")

logger = init_logger(project=projectName)
client = wrap_anthropic(Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")))

SYSTEM_PROMPT = (
    "You are a helpful customer support agent for an e-commerce company. "
    "Be empathetic but efficient. Ask clarifying questions when needed. "
    "If you can resolve the issue, do so. If you need to escalate, "
    "explain why and what the customer should expect next."
)

# Toggle this to see the difference in how traces are grouped.
# True:  entire conversation is one log entry with nested turn spans.
# False: each turn is its own top-level log entry.
GROUP_AS_CONVERSATION = True

@traced
def chat(conversation_history):
    print(f"Conversation history: {conversation_history}")
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=conversation_history,
        #output_config={"effort": "high"}
    )
    return next(block.text for block in response.content if hasattr(block, 'text'))

def main():
    print("Customer Support Chat (type 'quit' to exit)")
    print("-" * 45)

    conversation_history = []
    turn_number = 0

    if GROUP_AS_CONVERSATION:
        with logger.start_span(name="conversation") as conversation_span:
            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == "quit":
                    break

                conversation_history.append({"role": "user", "content": user_input})
                turn_number += 1

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
                output=conversation_history[-1]["content"] if len(conversation_history) > 1 else None,
                metadata={"total_turns": turn_number},
            )
    else:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "quit":
                break

            conversation_history.append({"role": "user", "content": user_input})
            turn_number += 1

            with logger.start_span(name="conversation_turn") as span:
                response = chat(conversation_history)
                conversation_history.append({"role": "assistant", "content": response})
                span.log(
                    input=user_input,
                    output=response,
                    metadata={"turn_number": turn_number},
                )

            print(f"\nAgent: {response}")

if __name__ == "__main__":
    main()
