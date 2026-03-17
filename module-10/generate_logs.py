"""Generate 200 customer support conversations and log them to Braintrust.

Braintrust Topics requires at least 200 traces to generate topic clusters.
This script creates 200 unique single-turn conversations across 5 topic areas
(40 per topic) to meet that threshold.

Uses gpt-4o-mini to keep costs low.
"""
import os
import random
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

# 200 unique customer messages across 5 topic areas (40 per topic).
MESSAGES = {
    "shipping_delay": [
        "My order hasn't arrived yet. It's been 10 days.",
        "I ordered a jacket 2 weeks ago and it still hasn't shipped.",
        "Where is my package? I ordered it 8 days ago.",
        "My delivery is showing as delayed. What does that mean?",
        "I paid for 2-day shipping and it's been 5 days.",
        "My order was supposed to arrive Monday and it's now Thursday.",
        "The tracking hasn't updated in 3 days. Is my package lost?",
        "I need this by Friday for a birthday. Will it make it?",
        "My package has been sitting in the same city for a week.",
        "I ordered two items and only one has shipped. Where's the other?",
        "Why does it take so long to ship from your warehouse?",
        "My order shipped but the estimated delivery keeps changing.",
        "The carrier says they attempted delivery but no one came to my door.",
        "I'm going out of town next week. Can you hold my package?",
        "My order has been in 'processing' for 4 days. Is something wrong?",
        "I live in Alaska. Your site said 5-7 days but it's been 12.",
        "My package was supposed to arrive today but tracking says it's still 3 states away.",
        "I ordered something for my kid's school project and it's already late.",
        "Can I upgrade to overnight shipping on an order that's already placed?",
        "The delivery driver left my package in the rain. Who do I talk to?",
        "I got a shipping confirmation but no tracking number. What gives?",
        "My order says it shipped from two different warehouses. Is that normal?",
        "I ordered on Black Friday and it still hasn't arrived. It's been 3 weeks.",
        "Can you tell me which carrier is delivering my order?",
        "My neighbor got their order from you in 2 days. Mine has taken 9.",
        "I selected express shipping at checkout but it's showing standard on my confirmation.",
        "The package was out for delivery yesterday but was never delivered.",
        "I need to know the exact delivery date. I won't be home after Wednesday.",
        "My order has been stuck at 'label created' for 6 days.",
        "Do you ship to PO boxes? My order keeps getting returned.",
        "I placed two orders on the same day. One arrived, the other hasn't.",
        "Your website said in stock and ready to ship, but it's been a week.",
        "My package got sent to the wrong distribution center.",
        "I'm worried my package was stolen. It says delivered but I have nothing.",
        "Is there a phone number I can call to check on my delivery?",
        "I ordered perishable items and they're taking way too long to arrive.",
        "My tracking says 'exception' — what does that mean?",
        "I've been refreshing the tracking page all day and nothing has changed.",
        "Can you file a claim with the carrier? My package is clearly lost.",
        "I moved last month. Can you update the shipping address on my open order?",
    ],
    "refund_request": [
        "I want to return this sweater. It's not what I expected.",
        "I'd like a refund on my recent order.",
        "The shoes I ordered are the wrong color. I want my money back.",
        "I changed my mind about the backpack. Can I return it?",
        "The item I received was damaged. I want a full refund.",
        "I ordered two shirts and got the same one twice. Refund please.",
        "This product doesn't match the description on your website at all.",
        "I bought this as a gift but the recipient already has one. Can I return it?",
        "The quality is terrible compared to what the reviews said.",
        "I ordered the wrong size. What's your exchange or refund policy?",
        "My order arrived open and items were missing. I need a refund.",
        "I was charged twice for the same order. Please refund the duplicate.",
        "I want a refund but I already threw away the packaging.",
        "The product broke after 2 days of normal use. This is unacceptable.",
        "I returned my order 2 weeks ago but still no refund.",
        "Can I get store credit instead of a refund to my card?",
        "The color looks completely different in person than on the website.",
        "I received someone else's order instead of mine.",
        "The zipper on the jacket broke the first time I used it.",
        "I want to return all 3 items from my order, not just one.",
        "My refund was less than what I paid. Why was I charged a restocking fee?",
        "The product arrived after I already bought a replacement elsewhere.",
        "I never received a return shipping label after I requested one.",
        "This was supposed to be new but it looks like it's been used before.",
        "Can I exchange this for a different product entirely?",
        "I accidentally placed a duplicate order. Can you cancel and refund the second one?",
        "The electronics I ordered don't turn on. I want a refund, not a replacement.",
        "Your return policy says 30 days but I'm at day 32. Can you make an exception?",
        "I used a coupon on this order. Will I get the coupon back if I return it?",
        "The stitching on the bag is already coming undone.",
        "I got a partial refund but I returned everything. Where's the rest?",
        "The item smells like chemicals. I don't feel safe using it.",
        "I ordered a medium but the tag says large even though it fits like a small.",
        "I want a refund but the item was a final sale. What are my options?",
        "My package arrived damaged and the product inside is scratched.",
        "I've been waiting 10 business days for my refund. Your policy says 5-7.",
        "Can I return an item I bought in store through your website?",
        "The sunglasses I ordered have a scratch on the lens right out of the box.",
        "I ordered the wrong thing. Can I return it without the original receipt?",
        "I'd like to return this mattress topper. It's not firm enough.",
    ],
    "product_question": [
        "Does the blue hoodie come in a size XL?",
        "What's the difference between the Pro and Standard backpack?",
        "Is this jacket waterproof or just water resistant?",
        "What material are the running shorts made of?",
        "Is the wallet available in brown leather?",
        "Do you have this dress in a petite size?",
        "What's the battery life on the wireless earbuds?",
        "Is this shirt machine washable or dry clean only?",
        "Do the hiking boots run true to size?",
        "What's the weight limit on the carry-on luggage?",
        "Are your t-shirts pre-shrunk?",
        "Does this watch work with both iPhone and Android?",
        "What's the fill power on the down jacket?",
        "Do you sell replacement laces for the sneakers?",
        "Is the laptop sleeve padded enough for a MacBook Pro?",
        "What colors does the ceramic mug come in?",
        "How do I care for the leather boots? Do they need waterproofing?",
        "What's the thread count on your bed sheets?",
        "Is the kids' jacket machine washable?",
        "Do the wireless headphones have noise cancellation?",
        "What's the return policy on clearance items?",
        "Is the stainless steel water bottle dishwasher safe?",
        "How much does the duffel bag weigh when empty?",
        "Do you offer gift wrapping?",
        "What size should I get if I'm between a medium and large?",
        "Are the yoga pants see-through? I've had issues with other brands.",
        "Does the portable charger work with USB-C?",
        "Is the flannel shirt true to the color shown online?",
        "How long is the warranty on the Bluetooth speaker?",
        "Can I use the rain jacket for skiing or is it just for light rain?",
        "What's the difference between the regular and slim fit jeans?",
        "Do you have any vegan leather options?",
        "Is the camping tent rated for winter weather?",
        "What are the dimensions of the weekender bag?",
        "Does the smartwatch track sleep?",
        "Are the socks moisture-wicking?",
        "Can I monogram the leather journal?",
        "Is the insulated lunch bag big enough for adult meals?",
        "Do the sandals have arch support?",
        "What's the lumen output on the flashlight?",
    ],
    "account_issue": [
        "I can't log into my account. I keep getting an error.",
        "How do I change the email address on my account?",
        "I want to delete my account and remove my data.",
        "I think someone accessed my account without permission.",
        "My saved addresses disappeared after the app update.",
        "I forgot my password and the reset email isn't coming.",
        "My account shows orders that aren't mine.",
        "How do I turn off marketing emails?",
        "I can't update my payment method. The button doesn't work.",
        "My account was locked after too many login attempts.",
        "I need to change my phone number for two-factor auth.",
        "Why does your app keep logging me out?",
        "I created a duplicate account by accident. Can you merge them?",
        "My rewards points disappeared. I had over 2000.",
        "I can't see my order history anymore. It's completely blank.",
        "How do I add a family member to my account?",
        "I signed up with Google but now I want to use a regular password instead.",
        "My account says I have a gift card balance but I never added one.",
        "I can't remove an old credit card from my account.",
        "The app crashes every time I try to open my account settings.",
        "I changed my name and need it updated on my account.",
        "Someone used my email to create an account. I never signed up.",
        "My wishlist is gone after I updated the app.",
        "I can't unsubscribe from your texts. The link doesn't work.",
        "How do I see my past returns in my account?",
        "I'm getting login codes I didn't request. Is someone trying to hack me?",
        "My account was deactivated for no reason. What happened?",
        "I want to change my username but there's no option for it.",
        "My loyalty tier got downgraded even though I've been spending more.",
        "Can I transfer my rewards points to someone else?",
        "I can't link my PayPal account. It keeps erroring out.",
        "The birthday discount I was promised never showed up.",
        "I opted out of data sharing but I'm still getting targeted ads.",
        "My account shows a different default shipping address than what I set.",
        "I got an email saying my password was changed but I didn't change it.",
        "How do I download all my account data?",
        "I can't access the members-only sale even though I'm a member.",
        "My referral credits never posted after my friend made a purchase.",
        "The two-factor auth code is being sent to my old phone number.",
        "I signed up for the newsletter but never received the welcome discount.",
    ],
    "order_tracking": [
        "Can you tell me where my order is right now?",
        "The tracking link in my email isn't working.",
        "My order shows as delivered but I haven't received anything.",
        "Is there a way to get SMS updates on my delivery?",
        "My order was split into two shipments. When does the second arrive?",
        "The tracking number you gave me shows no results.",
        "My package was marked delivered 3 days ago but it never came.",
        "Can I redirect my package to a different address?",
        "I see my order is out for delivery. What time will it arrive?",
        "Your tracking says delivered to front door but nothing is there.",
        "I need proof of delivery for an insurance claim.",
        "My package was sent back to your warehouse. Why?",
        "The tracking shows it went to the wrong state. What happened?",
        "Can I pick up my package at the carrier facility instead?",
        "How do I track a replacement order?",
        "My order has two tracking numbers. Which one is correct?",
        "I got a delivery notification but the package wasn't at my door.",
        "Can I schedule a specific delivery time?",
        "My tracking shows it's been in customs for a week. Is that normal?",
        "The delivery was attempted but I wasn't home. How do I reschedule?",
        "I never got a shipping confirmation email. Did my order ship?",
        "My tracking shows the package is going back and forth between two cities.",
        "Can someone else sign for my package if I'm not home?",
        "The estimated delivery window is 5 days wide. Can you narrow it down?",
        "I have a Ring camera and no one came to my door even though it says delivered.",
        "My order shows 'pending' for tracking. When will it update?",
        "I need the tracking info for a gift I sent to someone else.",
        "The carrier left my package at the wrong apartment number.",
        "How do I opt into delivery photo confirmation?",
        "My order was marked as undeliverable. What does that mean?",
        "I placed an order for store pickup. How will I know when it's ready?",
        "Tracking says it was handed to a resident but I live alone.",
        "My order went through 3 shipping carriers. Is that why it's slow?",
        "Can you confirm the package dimensions? I want to make sure it fits in my mailbox.",
        "I got two emails with different tracking numbers for the same order.",
        "The tracking page shows a map but the pin is in the wrong location.",
        "My order was delivered to my old address even though I updated it.",
        "I want real-time tracking like Uber. Do you offer that?",
        "How do I track an international order?",
        "My package was marked as delivered to the mailroom but there is no mailroom here.",
    ],
}

# Build flat list — 200 unique messages, no repeats.
ALL_MESSAGES = []
for topic, messages in MESSAGES.items():
    for msg in messages:
        ALL_MESSAGES.append({"topic": topic, "message": msg})

@traced
def chat(conversation_history):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history,
        temperature=1.0,
    )
    return response.choices[0].message.content


def run_conversation(topic, user_message, index, total):
    """Run a single-turn conversation and log it to Braintrust."""
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    conversation_history.append({"role": "user", "content": user_message})

    with logger.start_span(name="conversation") as conversation_span:
        with conversation_span.start_span(name="turn_1") as turn_span:
            response = chat(conversation_history)
            conversation_history.append({"role": "assistant", "content": response})
            turn_span.log(
                input=user_message,
                output=response,
                metadata={"turn_number": 1},
            )

        conversation_span.log(
            input=conversation_history,
            output=response,
            metadata={"total_turns": 1, "topic": topic},
        )

    if index % 20 == 0 or index == total:
        print(f"  [{index}/{total}] {topic}: {user_message[:50]}...")


def main():
    total = len(ALL_MESSAGES)
    print(f"Generating {total} conversations across 5 topics...")
    print("(Using gpt-4o-mini to keep costs low)\n")

    # Shuffle so topics are interleaved.
    random.seed(42)
    shuffled = list(enumerate(ALL_MESSAGES))
    random.shuffle(shuffled)

    for new_index, (_, conv) in enumerate(shuffled, 1):
        run_conversation(conv["topic"], conv["message"], new_index, total)

    print(f"\nDone. Logged {total} conversations to Braintrust.")
    print("Go to Topics in your project to set up topic maps.")


if __name__ == "__main__":
    main()
