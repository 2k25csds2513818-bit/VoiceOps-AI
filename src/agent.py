"""ZentroMart support agent: RAG, guarded order tools, ticket handover, optional LLM."""
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from src.retrieval import KnowledgeBase

# Load environment settings from .env.
load_dotenv(override=True)

# Locate the dataset and make its tools package importable.
DATA_ROOT = Path(
    os.getenv("DATA_ROOT", "Capstone_Project_Dataset_FTTT_2026")
).resolve()
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from tools import lookup_order, create_ticket  # noqa: E402

KB = KnowledgeBase(DATA_ROOT / "kb")

SYSTEM_PROMPT = """You are ZentroMart's customer support AI.
Always identify yourself as an AI assistant.
Use supplied company documents for company policies and verified
 tool results for private order information.
For general policy questions, answer directly from supplied evidence.
Do not ask for order verification for general questions.
Ask for order verification before revealing private order details
or performing an order-specific action that requires verification.
If evidence does not answer a question, clearly say that you cannot
confirm the answer from the available company documents.
Never invent company policies, dates, prices, or other facts.
Be concise, polite, and conversational.
Do not claim an order lookup or ticket succeeded unless the tool
confirms it.
Never disclose private order details before verification.
Never reveal or store verification values in conversation history.
Allow at most two verification attempts per order.
Escalate when policy requires a human or an exception needs review.
For product safety issues, such as a swelling battery, burning smell,
sparks, or overheating, tell the customer to stop using the product
immediately and arrange an immediate human handoff.
Do not wait for order verification to give safety guidance.
"""


class SupportAgent:
    def __init__(self):
        self.history = []
        self.pending_order_id = None
        self.verification_attempts = 0
        self.verified_order = None

    def _llm(self, messages):
        """Call the local Ollama model."""
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
            result = client.chat.completions.create(
                model="llama3.2:1b",
                messages=messages,
                temperature=0.2,
            )
            content = result.choices[0].message.content
            return content.strip() if content else None
        except Exception as exc:
            print(f"LLM error: {exc}")
            return None

    def _save_exchange(self, user_message, assistant_message):
        """Save conversation history without verification secrets."""
        self.history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        self.history = self.history[-12:]

    def _is_safety_issue(self, message):
        """Detect product safety issues using common customer wording."""
        low = message.lower()
        return bool(
            re.search(
                r"\b("
                r"swelling battery|"
                r"swollen battery|"
                r"battery swelling|"
                r"battery is swelling|"
                r"battery has swollen|"
                r"battery is swollen|"
                r"burning smell|"
                r"smell of burning|"
                r"sparks|"
                r"sparking|"
                r"overheating|"
                r"overheated|"
                r"overheating battery|"
                r"safety risk"
                r")\b",
                low,
            )
        )

    def _retrieve(self, message):
        """Retrieve company evidence safely."""
        try:
            return KB.search(message, k=4)
        except Exception as exc:
            print(f"Knowledge-base retrieval error: {exc}")
            return []

    def _handle_safety_issue(self, message, retrieved):
        """Give immediate stop-use guidance and request human handoff."""
        order_id = (self.verified_order or {}).get("order_id")
        summary = (
            f"Product safety issue reported by customer: {message}. "
            "Customer was instructed to stop using the product. "
            "Immediate human review and handoff required."
        )
        try:
            ticket = create_ticket(
                summary=summary,
                order_id=order_id,
                reason="customer_requested_human_or_ticket",
                intent="escalation",
            )
        except Exception as exc:
            print(f"Safety ticket creation error: {exc}")
            ticket = {"created": False}

        if ticket.get("created") and ticket.get("ticket_id"):
            answer = (
                "I'm ZentroMart's customer support AI. This may be a "
                "product safety issue. Stop using the product immediately. "
                "Please contact a human support agent now for urgent help. "
                "I've created a support ticket for human review. "
                f"Ticket ID: {ticket['ticket_id']}."
            )
        else:
            answer = (
                "I'm ZentroMart's customer support AI. This may be a "
                "product safety issue. Stop using the product immediately "
                "and contact a human support agent now. I couldn't confirm "
                "that a support ticket was created, so please contact "
                "ZentroMart's official support channel for urgent assistance."
            )

        self._save_exchange(message, answer)
        return answer, retrieved

    def respond(self, message):
        message = message.strip()
        if not message:
            return "Sorry, I didn't catch that. Could you say it again?", []

        low = message.lower()

        # 1. Safety issues come first; never delay safety guidance for verification.
        if self._is_safety_issue(message):
            retrieved = self._retrieve(message)
            return self._handle_safety_issue(message, retrieved)

        # 2. Handle return-window questions separately from damaged-item policy.
        return_keywords = ("return", "return window", "return period")
        return_time_keywords = (
            "after 10 days",
            "after 10 day",
            "after the 10-day",
            "after the 10 day",
            "expired",
            "passed",
            "exceeded",
            "outside the return window",
            "return window has expired",
            "return period has expired",
        )
        if (
            any(keyword in low for keyword in return_keywords)
            and any(keyword in low for keyword in return_time_keywords)
        ):
            retrieved = self._retrieve(message)
            answer = (
                "ZentroMart's standard return window is 10 days. If that "
                "window has expired, I can't confirm eligibility for a "
                "standard return from the available policy. Please contact "
                "a human support agent to review your case."
            )
            self._save_exchange(message, answer)
            return answer, retrieved

        # 3. Handle damaged/wrong/missing-item policy questions deterministically.
        # Keep this after safety and return-window checks.
        damaged_keywords = (
            "damaged",
            "defective",
            "expired",
            "wrong item",
            "missing item",
        )
        if any(keyword in low for keyword in damaged_keywords):
            retrieved = self._retrieve(message)
            answer = (
                "Please report a damaged, defective, expired, wrong, or missing "
                "item within 48 hours of delivery. Please provide photos of the "
                "item and its packaging. After verification, you may receive a "
                "free replacement or a full refund, including delivery fees. "
                "No reverse pickup fee applies."
            )
            self._save_exchange(message, answer)
            return answer, retrieved

        # 4. Handle pending order verification.
        if self.pending_order_id:
            digits_match = re.fullmatch(r"\s*(\d{4}|\d{6})\s*", message)
            if digits_match:
                digits = digits_match.group(1)
                self.verification_attempts += 1
                try:
                    result = lookup_order(
                        self.pending_order_id,
                        verification_value=digits,
                    )
                except Exception as exc:
                    print(f"Order lookup error: {exc}")
                    result = {"verified": False}

                if result.get("verified"):
                    self.verified_order = result
                    self.pending_order_id = None
                    self.verification_attempts = 0
                    summary = (
                        f"Order {result.get('order_id')} is currently "
                        f"{result.get('status')}. "
                    )
                    if result.get("promised_date"):
                        summary += (
                            "The promised delivery date is "
                            f"{result['promised_date']}. "
                        )
                    summary += "What would you like help with?"
                    self._save_exchange(
                        "[verification value redacted]",
                        summary,
                    )
                    return summary, []

                if self.verification_attempts >= 2:
                    self.pending_order_id = None
                    self.verification_attempts = 0
                    answer = (
                        "I couldn't verify those details, so I can't share "
                        "order information. Please use the official support "
                        "channel for further help."
                    )
                    self._save_exchange(
                        "[verification attempt redacted]",
                        answer,
                    )
                    return answer, []

                answer = (
                    "That didn't match. Please try once more with the "
                    "registered phone's last four digits or delivery pincode."
                )
                self._save_exchange(
                    "[verification attempt redacted]",
                    answer,
                )
                return answer, []

            # Allow general policy questions while verification is pending.
            general_question = any(
                phrase in low
                for phrase in (
                    "damaged product",
                    "damaged item",
                    "wrong item",
                    "missing item",
                    "return policy",
                    "refund policy",
                    "warranty",
                    "how do returns work",
                    "what is the policy",
                    "defective product",
                    "expired item",
                )
            )
            if not general_question:
                answer = (
                    "I'm still waiting for verification for your order. "
                    "Please enter the registered phone's last four digits "
                    "or the delivery pincode. You can also ask a general "
                    "company-policy question separately."
                )
                self._save_exchange(message, answer)
                return answer, []

        # 5. Detect a new eight-digit order ID.
        order_match = re.search(
            r"\b(?:order\s*(?:id|number|no\.?)\s*[:#]?\s*)?(\d{8})\b",
            message,
            re.IGNORECASE,
        )
        if order_match and any(
            word in low
            for word in (
                "order",
                "delivery",
                "track",
                "cancel",
                "refund",
                "return",
                "where",
                "delayed",
                "shipped",
            )
        ):
            order_id = order_match.group(1)
            self.pending_order_id = order_id
            self.verification_attempts = 0
            self.verified_order = None
            answer = (
                "I can help with that order. For privacy, please confirm "
                "either the last four digits of the registered phone number "
                "or the delivery pincode."
            )
            self._save_exchange(message, answer)
            return answer, []

        # 6. Detect ticket requests, delivery issues, and escalation.
        ticket_requested = any(
            phrase in low
            for phrase in (
                "create a support ticket",
                "create ticket",
                "raise a ticket",
                "open a ticket",
                "support ticket",
                "create a ticket",
            )
        )
        delivery_issue = any(
            phrase in low
            for phrase in (
                "delayed",
                "delay",
                "late delivery",
                "not delivered",
                "hasn't arrived",
                "has not arrived",
            )
        )
        needs_human = any(
            phrase in low
            for phrase in (
                "human agent",
                "talk to a person",
                "talk to a human",
                "speak to a person",
                "compensation",
                "supervisor",
                "exception",
            )
        )
        order_related = delivery_issue or any(
            word in low
            for word in (
                "order",
                "delivery",
                "refund",
                "return",
                "shipment",
            )
        )

        # 7. Require verification for order-specific delivery tickets.
        if (
            (ticket_requested or delivery_issue or needs_human)
            and order_related
            and not self.verified_order
        ):
            answer = (
                "For privacy, please provide your order ID and verify it "
                "first. Then I can help raise a support ticket."
            )
            self._save_exchange(message, answer)
            return answer, []

        # 8. Retrieve relevant company evidence.
        retrieved = self._retrieve(message)

        # 9. Create a ticket only when requested or escalation is needed.
        if (
            ticket_requested
            or needs_human
            or (delivery_issue and self.verified_order)
        ):
            order_id = (self.verified_order or {}).get("order_id")
            summary = (
                f"Customer request: {message}. "
                f"Order ID: {order_id or 'Not provided'}. "
                "Human review requested; verify the facts and applicable "
                "policy before deciding on a resolution."
            )
            try:
                ticket = create_ticket(
                    summary=summary,
                    order_id=order_id,
                    reason=(
                        "delivery_delay"
                        if delivery_issue
                        else "customer_requested_human_or_ticket"
                    ),
                    intent="delivery" if delivery_issue else "escalation",
                )
            except Exception as exc:
                print(f"Ticket creation error: {exc}")
                ticket = {"created": False}

            if ticket.get("created") and ticket.get("ticket_id"):
                answer = (
                    "Your support ticket has been created successfully. "
                    f"Ticket ID: {ticket['ticket_id']}. "
                    "A human support team can review your request."
                )
            else:
                answer = (
                    "I couldn't create the support ticket right now. "
                    "Please try again or contact support."
                )
            self._save_exchange(message, answer)
            return answer, retrieved

        # 10. Do not ask the LLM to answer without relevant evidence.
        if not retrieved:
            answer = (
                "I couldn't find enough relevant information in the "
                "available company documents to answer that reliably. "
                "I don't want to guess. Can I help with another question?"
            )
            self._save_exchange(message, answer)
            return answer, []

        # 11. Prepare retrieved evidence for the LLM.
        evidence = "\n\n".join(
            f"[{item['id']} | {item['source']}]\n{item['text'][:1800]}"
            for item in retrieved
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += self.history[-8:]
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Customer message: {message}\n\n"
                    f"Retrieved company evidence:\n{evidence}\n\n"
                    "Answer using only evidence that directly supports "
                    "the answer. Do not treat unrelated documents as evidence. "
                    "If the evidence does not answer the question, explicitly "
                    "say that you cannot confirm the answer from the available "
                    "company documents. Do not guess. Keep the answer concise. "
                    "Do not claim that a ticket was created unless a tool "
                    "confirmed it."
                ),
            }
        )
        answer = self._llm(messages)
        if answer is None:
            answer = (
                "I'm unable to generate an answer right now. "
                "Please try again shortly."
            )
        self._save_exchange(message, answer)
        return answer, retrieved
