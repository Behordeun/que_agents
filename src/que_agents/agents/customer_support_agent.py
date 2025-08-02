# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-01 23:53:31
# @Description: This module implements a customer support agent using LangChain and SQLAlchemy


from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from src.que_agents.core.database import (
    Customer,
    CustomerInteraction,
    SupportTicket,
    get_session,
)
from src.que_agents.core.llm_factory import LLMFactory
from src.que_agents.knowledge_base.kb_manager import search_knowledge_base


@dataclass
class CustomerContext:
    """Customer context information"""

    customer_id: int
    name: str
    email: str
    tier: str
    company: str
    recent_interactions: List[Dict]
    open_tickets: List[Dict]


@dataclass
class AgentResponse:
    """Agent response structure"""

    message: str
    confidence: float
    escalate: bool
    suggested_actions: List[str]
    knowledge_sources: List[str]
    sentiment: str


# Load agent configuration
with open("/configs/agent_config.yaml", "r") as f:
    agent_config = yaml.safe_load(f)


class CustomerSupportAgent:
    """Customer Support Agent using LangChain"""

    def __init__(self):
        config = agent_config["customer_support_agent"]
        self.llm = LLMFactory.get_llm(
            agent_type="customer_support",
            model_name=config["model_name"],
            temperature=config["temperature"],
            max_tokens=500,
        )
        # Memory for conversation history
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True,
            memory_key="chat_history",
        )

        # Escalation keywords
        self.escalation_keywords = [
            "angry",
            "furious",
            "lawsuit",
            "legal",
            "cancel",
            "refund",
            "manager",
            "supervisor",
            "complaint",
            "terrible",
            "awful",
            "unacceptable",
            "frustrated",
            "disappointed",
        ]

        # Initialize prompt template
        self.prompt = self._create_prompt_template()

        # Create the chain
        self.chain = self._create_chain()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for the customer support agent"""
        system_message = """You are a helpful and empathetic customer support agent for an AI Analytics Platform. Your role is to:

1. Provide excellent customer service with a professional and friendly tone
2. Help customers resolve their issues quickly and effectively
3. Use the provided knowledge base and customer context to give accurate information
4. Escalate issues when necessary based on the escalation criteria
5. Document interactions properly for follow-up

ESCALATION CRITERIA:
- Customer mentions legal action, lawsuits, or regulatory complaints
- Customer requests to speak with a manager or supervisor
- Issue involves billing disputes over $500
- Technical issues affecting multiple customers
- Customer satisfaction appears very low
- Issue remains unresolved after multiple attempts

RESPONSE GUIDELINES:
- Always acknowledge the customer's concern first
- Be empathetic and understanding
- Provide clear, step-by-step solutions when possible
- Reference relevant knowledge base articles
- Offer follow-up if needed
- Keep responses concise but complete

Customer Context: {customer_context}
Knowledge Base Results: {knowledge_base_results}

Current customer message: {customer_message}

Provide a helpful response that addresses the customer's concern. If escalation is needed, clearly indicate this in your response."""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{customer_message}"),
            ]
        )

    def _create_chain(self):
        """Create the LangChain processing chain"""
        return (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.chat_memory.messages
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def get_customer_context(self, customer_id: int) -> Optional[CustomerContext]:
        """Retrieve customer context from database"""
        session = get_session()
        try:
            customer = (
                session.query(Customer).filter(Customer.id == customer_id).first()
            )
            if not customer:
                return None

            # Get recent interactions
            recent_interactions = (
                session.query(CustomerInteraction)
                .filter(CustomerInteraction.customer_id == customer_id)
                .order_by(CustomerInteraction.created_at.desc())
                .limit(5)
                .all()
            )

            # Get open tickets
            open_tickets = (
                session.query(SupportTicket)
                .filter(SupportTicket.customer_id == customer_id)
                .filter(SupportTicket.status.in_(["open", "in_progress"]))
                .all()
            )

            return CustomerContext(
                customer_id=customer.id,
                name=customer.name,
                email=customer.email,
                tier=customer.tier,
                company=customer.company or "N/A",
                recent_interactions=[
                    {
                        "type": i.interaction_type,
                        "message": i.message,
                        "response": i.response,
                        "sentiment": i.sentiment,
                        "date": i.created_at.isoformat() if i.created_at else None,
                    }
                    for i in recent_interactions
                ],
                open_tickets=[
                    {
                        "id": t.id,
                        "title": t.title,
                        "category": t.category,
                        "priority": t.priority,
                        "status": t.status,
                        "created_at": (
                            t.created_at.isoformat() if t.created_at else None
                        ),
                    }
                    for t in open_tickets
                ],
            )
        finally:
            session.close()

    def analyze_sentiment(self, message: str) -> str:
        """Simple sentiment analysis based on keywords"""
        message_lower = message.lower()

        negative_words = [
            "angry",
            "frustrated",
            "terrible",
            "awful",
            "hate",
            "worst",
            "horrible",
            "disappointed",
        ]
        positive_words = [
            "great",
            "excellent",
            "love",
            "amazing",
            "wonderful",
            "fantastic",
            "perfect",
        ]

        negative_count = sum(1 for word in negative_words if word in message_lower)
        positive_count = sum(1 for word in positive_words if word in message_lower)

        if negative_count > positive_count:
            return "negative"
        elif positive_count > negative_count:
            return "positive"
        else:
            return "neutral"

    def should_escalate(self, message: str, customer_context: CustomerContext) -> bool:
        """Determine if the issue should be escalated"""
        message_lower = message.lower()

        # Check for escalation keywords
        for keyword in self.escalation_keywords:
            if keyword in message_lower:
                return True

        # Check for high-priority customer tier
        if customer_context.tier == "enterprise" and any(
            ticket["priority"] == "urgent" for ticket in customer_context.open_tickets
        ):
            return True

        # Check for multiple open tickets
        if len(customer_context.open_tickets) > 2:
            return True

        return False

    def search_knowledge_base_for_query(self, query: str) -> List[Dict]:
        """Search knowledge base for relevant information"""
        try:
            results = search_knowledge_base(query, limit=3)
            return results
        except Exception as e:
            print(f"Error searching knowledge base: {e}")
            return []

    def process_customer_message(self, customer_id: int, message: str) -> AgentResponse:
        """Process a customer message and generate a response"""
        # Get customer context
        customer_context = self.get_customer_context(customer_id)
        if not customer_context:
            return AgentResponse(
                message="I'm sorry, I couldn't find your customer information. Please verify your customer ID.",
                confidence=0.0,
                escalate=True,
                suggested_actions=["Verify customer identity"],
                knowledge_sources=[],
                sentiment="neutral",
            )

        # Search knowledge base
        kb_results = self.search_knowledge_base_for_query(message)

        # Analyze sentiment
        sentiment = self.analyze_sentiment(message)

        # Check for escalation
        should_escalate = self.should_escalate(message, customer_context)

        # Prepare context for the LLM
        context_str = f"""
Customer: {customer_context.name} ({customer_context.email})
Tier: {customer_context.tier}
Company: {customer_context.company}
Open Tickets: {len(customer_context.open_tickets)}
Recent Interactions: {len(customer_context.recent_interactions)}
"""

        kb_str = "\n".join(
            [
                f"- {result['title']}: {result['content'][:200]}..."
                for result in kb_results
            ]
        )

        # Generate response using the chain
        try:
            response = self.chain.invoke(
                {
                    "customer_context": context_str,
                    "knowledge_base_results": kb_str,
                    "customer_message": message,
                }
            )

            # Update memory
            self.memory.chat_memory.add_user_message(message)
            self.memory.chat_memory.add_ai_message(response)

            # Determine confidence based on knowledge base matches
            confidence = min(0.9, 0.5 + (len(kb_results) * 0.1))

            # Generate suggested actions
            suggested_actions = []
            if should_escalate:
                suggested_actions.append("Escalate to supervisor")
            if kb_results:
                suggested_actions.append("Follow knowledge base procedures")
            if sentiment == "negative":
                suggested_actions.append("Follow up within 24 hours")

            return AgentResponse(
                message=response,
                confidence=confidence,
                escalate=should_escalate,
                suggested_actions=suggested_actions,
                knowledge_sources=[result["title"] for result in kb_results],
                sentiment=sentiment,
            )

        except Exception as e:
            print(f"Error generating response: {e}")
            return AgentResponse(
                message="I apologize, but I'm experiencing technical difficulties. Let me escalate this to a human agent who can assist you immediately.",
                confidence=0.0,
                escalate=True,
                suggested_actions=["Escalate to human agent"],
                knowledge_sources=[],
                sentiment=sentiment,
            )

    def log_interaction(self, customer_id: int, message: str, response: AgentResponse):
        """Log the interaction to the database"""
        session = get_session()
        try:
            interaction = CustomerInteraction(
                customer_id=customer_id,
                interaction_type="chat",
                message=message,
                response=response.message,
                sentiment=response.sentiment,
                satisfaction_score=4.0 if response.confidence > 0.7 else 3.0,
                agent_id="customer_support_agent_ai",
            )
            session.add(interaction)
            session.commit()
        except Exception as e:
            print(f"Error logging interaction: {e}")
            session.rollback()
        finally:
            session.close()

    def handle_customer_request(self, customer_id: int, message: str) -> Dict[str, Any]:
        """Main method to handle a customer request"""
        # Process the message
        response = self.process_customer_message(customer_id, message)

        # Log the interaction
        self.log_interaction(customer_id, message, response)

        # Return structured response
        return {
            "response": response.message,
            "confidence": response.confidence,
            "escalate": response.escalate,
            "suggested_actions": response.suggested_actions,
            "knowledge_sources": response.knowledge_sources,
            "sentiment": response.sentiment,
            "timestamp": datetime.now().isoformat(),
        }


def test_customer_support_agent():
    """Test the customer support agent with sample interactions"""
    agent = CustomerSupportAgent()

    # Test scenarios
    test_cases = [
        {
            "customer_id": 1,
            "message": "I can't log into my account. I keep getting an error message.",
        },
        {
            "customer_id": 2,
            "message": "I was charged twice this month and I want a refund immediately!",
        },
        {
            "customer_id": 3,
            "message": "The API is returning 500 errors and it's affecting our production system.",
        },
        {
            "customer_id": 1,
            "message": "Thank you for helping me reset my password. Everything is working now.",
        },
    ]

    print("=== Customer Support Agent Test ===\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Customer ID: {test_case['customer_id']}")
        print(f"Message: {test_case['message']}")

        result = agent.handle_customer_request(
            test_case["customer_id"], test_case["message"]
        )

        print(f"Response: {result['response']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Escalate: {result['escalate']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Suggested Actions: {', '.join(result['suggested_actions'])}")
        print(f"Knowledge Sources: {', '.join(result['knowledge_sources'])}")
        print("-" * 80)


if __name__ == "__main__":
    test_customer_support_agent()
