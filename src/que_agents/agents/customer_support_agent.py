# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-05 15:00:00
# @Description: This module implements a customer support agent using LangChain and SQLAlchemy

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
from src.que_agents.core.schemas import AgentResponse, CustomerContext

try:
    from src.que_agents.knowledge_base.kb_manager import (
        search_agent_knowledge_base,
        search_knowledge_base,
    )
except ImportError:
    def search_agent_knowledge_base(agent_type: str, query: str, limit: int = 5) -> List[Dict]:
        """Fallback when knowledge base is not available"""
        return []
    
    def search_knowledge_base(query: str, limit: int = 5) -> List[Dict]:
        """Fallback when knowledge base is not available"""
        return []

# Load agent configuration
with open("configs/agent_config.yaml", "r") as f:
    agent_config = yaml.safe_load(f)


class CustomerSupportAgent:
    """Enhanced Customer Support Agent using LangChain with Knowledge Base Integration"""

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

        # Enhanced escalation keywords categorized by type
        self.escalation_keywords = {
            "anger": ["angry", "furious", "rage", "mad", "pissed", "outraged"],
            "legal": ["lawsuit", "legal", "attorney", "lawyer", "sue", "court"],
            "cancellation": ["cancel", "unsubscribe", "terminate", "quit"],
            "refund": ["refund", "money back", "chargeback", "dispute"],
            "management": ["manager", "supervisor", "boss", "escalate", "senior"],
            "complaint": ["complaint", "complain", "terrible", "awful", "horrible", "worst"],
            "dissatisfaction": ["unacceptable", "frustrated", "disappointed", "disgusted"],
            "urgency": ["urgent", "immediately", "asap", "emergency", "critical"],
        }

        # Support categories for better knowledge base searching
        self.support_categories = [
            "account_access",
            "billing",
            "technical_issues",
            "api_problems",
            "password_reset",
            "subscription",
            "integration",
            "data_export",
            "security",
            "troubleshooting",
        ]

        # Initialize prompt templates
        self.main_prompt = self._create_main_prompt_template()
        self.sentiment_prompt = self._create_sentiment_prompt()
        self.escalation_prompt = self._create_escalation_prompt()
        self.category_prompt = self._create_category_prompt()

        # Create chains
        self.main_chain = self._create_main_chain()
        self.sentiment_chain = self._create_sentiment_chain()
        self.escalation_chain = self._create_escalation_chain()
        self.category_chain = self._create_category_chain()

    def get_support_knowledge(self, query: str) -> List[Dict]:
        """Get customer support knowledge from knowledge base"""
        try:
            return search_agent_knowledge_base("customer_support", query, limit=3)
        except Exception as e:
            print(f"Error searching support knowledge: {e}")
            return []

    def get_enhanced_context(self, customer_message: str, customer_context: CustomerContext) -> str:
        """Get enhanced context from knowledge base"""
        try:
            # Categorize the issue first
            category = self.categorize_issue(customer_message)
            
            # Search for category-specific knowledge
            category_knowledge = self.get_support_knowledge(f"{category} {customer_message}")
            
            # Search for tier-specific knowledge
            tier_knowledge = self.get_support_knowledge(f"{customer_context.tier} customer support")
            
            enhanced_context = ""
            if category_knowledge:
                enhanced_context += "Relevant Support Knowledge:\n"
                for kb_item in category_knowledge:
                    enhanced_context += f"- {kb_item['title']}: {kb_item['content'][:200]}...\n"
            
            if tier_knowledge:
                enhanced_context += f"\n{customer_context.tier.title()} Tier Guidelines:\n"
                for kb_item in tier_knowledge:
                    enhanced_context += f"- {kb_item['title']}: {kb_item['content'][:150]}...\n"
            
            return enhanced_context
        except Exception as e:
            print(f"Error getting enhanced context: {e}")
            return ""

    def _create_main_prompt_template(self) -> ChatPromptTemplate:
        """Create the enhanced main prompt template for the customer support agent"""
        system_message = """You are an expert customer support agent for an AI Analytics Platform. Your role is to provide exceptional customer service with:

CORE RESPONSIBILITIES:
1. Provide empathetic, professional, and solution-focused responses
2. Use knowledge base information to give accurate, helpful solutions
3. Acknowledge customer emotions and validate their concerns
4. Escalate appropriately based on defined criteria
5. Document interactions for seamless follow-up
6. Maintain brand voice and customer satisfaction

CUSTOMER TIER CONSIDERATIONS:
- Enterprise: Priority support, dedicated resources, immediate escalation for critical issues
- Business: Standard support, prompt responses, escalate for billing issues >$200
- Free: Community support focus, self-service resources, escalate for account security

ESCALATION CRITERIA:
- Legal threats or regulatory compliance issues
- Requests for management/supervisor
- Billing disputes: >$500 (Free/Business), >$1000 (Enterprise)
- Critical technical issues affecting production systems
- Security breaches or data concerns
- Multiple unresolved tickets or repeated complaints
- Customer satisfaction clearly deteriorating

RESPONSE STRUCTURE:
1. Acknowledge the issue and show empathy
2. Reference relevant knowledge base solutions
3. Provide clear, actionable steps
4. Offer additional assistance
5. Set expectations for follow-up

Customer Context: {customer_context}
Enhanced Knowledge: {enhanced_context}
Knowledge Base Results: {knowledge_base_results}
Issue Category: {issue_category}
Escalation Analysis: {escalation_analysis}

Current customer message: {customer_message}

Provide a comprehensive, empathetic response that addresses the customer's specific concern using available knowledge and context."""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{customer_message}"),
            ]
        )

    def _create_sentiment_prompt(self) -> ChatPromptTemplate:
        """Create prompt for enhanced sentiment analysis"""
        system_message = """You are an expert in customer sentiment analysis. Analyze the emotional tone and satisfaction level in customer messages.

SENTIMENT CATEGORIES:
- very_positive: Extremely happy, praising, grateful
- positive: Satisfied, pleased, thankful
- neutral: Factual, informational, no strong emotion
- negative: Frustrated, disappointed, concerned
- very_negative: Angry, furious, threatening, extremely dissatisfied

Consider context like:
- Specific emotional words and phrases
- Urgency indicators
- Escalation language
- Satisfaction expressions
- Problem severity indicators

Customer message: {message}

Respond with ONLY the sentiment category from the list above."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{message}")
        ])

    def _create_escalation_prompt(self) -> ChatPromptTemplate:
        """Create prompt for escalation analysis"""
        system_message = """You are an escalation specialist. Analyze whether this customer interaction requires escalation to human agents or management.

ESCALATION INDICATORS:
- Legal threats or compliance issues
- Requests for managers/supervisors
- Security or data breach concerns
- High-value billing disputes
- Production system outages
- Repeated unresolved issues
- Extreme customer dissatisfaction
- Complex technical issues beyond standard support

Customer Tier: {customer_tier}
Open Tickets: {open_tickets}
Recent Interactions: {recent_interactions}
Customer Message: {customer_message}

Respond with:
1. "YES" or "NO" for escalation needed
2. Brief reason (if YES)

Format: ESCALATE: YES/NO - [reason if applicable]"""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "Analyze escalation need for: {customer_message}")
        ])

    def _create_category_prompt(self) -> ChatPromptTemplate:
        """Create prompt for issue categorization"""
        system_message = f"""You are an expert at categorizing customer support issues. Classify the customer's issue into one of these categories:

CATEGORIES:
{', '.join(self.support_categories)}

CATEGORY DEFINITIONS:
- account_access: Login issues, password problems, account lockouts
- billing: Payment issues, charges, invoices, subscription problems
- technical_issues: Platform bugs, performance problems, feature malfunctions
- api_problems: API errors, integration issues, rate limiting
- password_reset: Specific password reset requests
- subscription: Plan changes, upgrades, downgrades
- integration: Third-party integrations, setup help
- data_export: Data export, import, migration issues
- security: Security concerns, breach reports, permissions
- troubleshooting: General problem-solving, how-to questions

Customer message: {customer_message}

Respond with ONLY the category name from the list above."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{customer_message}")
        ])

    def _create_main_chain(self):
        """Create the main LangChain processing chain"""
        return (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.chat_memory.messages
            )
            | self.main_prompt
            | self.llm
            | StrOutputParser()
        )

    def _create_sentiment_chain(self):
        """Create sentiment analysis chain"""
        return self.sentiment_prompt | self.llm | StrOutputParser()

    def _create_escalation_chain(self):
        """Create escalation analysis chain"""
        return self.escalation_prompt | self.llm | StrOutputParser()

    def _create_category_chain(self):
        """Create category analysis chain"""
        return self.category_prompt | self.llm | StrOutputParser()

    def get_customer_context(self, customer_id: int) -> Optional[CustomerContext]:
        """Retrieve enhanced customer context from database"""
        session = get_session()
        try:
            customer = (
                session.query(Customer).filter(Customer.id == customer_id).first()
            )
            if not customer:
                # Create a default customer for testing
                customer = Customer(
                    id=customer_id,
                    name=f"Customer {customer_id}",
                    email=f"customer{customer_id}@example.com",
                    tier="business",
                    company=f"Company {customer_id}",
                    created_at=datetime.now()
                )
                session.add(customer)
                session.commit()

            # Get recent interactions (last 10)
            recent_interactions = (
                session.query(CustomerInteraction)
                .filter(CustomerInteraction.customer_id == customer_id)
                .order_by(CustomerInteraction.created_at.desc())
                .limit(10)
                .all()
            )

            # Get open tickets
            open_tickets = (
                session.query(SupportTicket)
                .filter(SupportTicket.customer_id == customer_id)
                .filter(SupportTicket.status.in_(["open", "in_progress", "pending"]))
                .all()
            )

            # Calculate satisfaction trend
            satisfaction_scores = [i.satisfaction_score for i in recent_interactions if i.satisfaction_score]
            avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 3.5

            return CustomerContext(
                customer_id=customer.id,
                name=customer.name,
                email=customer.email,
                tier=customer.tier,
                company=customer.company or "N/A",
                average_satisfaction=avg_satisfaction,
                recent_interactions=[
                    {
                        "type": i.interaction_type,
                        "message": i.message[:100] + "..." if len(i.message) > 100 else i.message,
                        "response": i.response[:100] + "..." if len(i.response) > 100 else i.response,
                        "sentiment": i.sentiment,
                        "satisfaction": i.satisfaction_score,
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
                        "updated_at": (
                            t.updated_at.isoformat() if t.updated_at else None
                        ),
                    }
                    for t in open_tickets
                ],
            )
        finally:
            session.close()

    def analyze_sentiment_enhanced(self, message: str) -> str:
        """Enhanced sentiment analysis using the LLM"""
        try:
            sentiment = self.sentiment_chain.invoke({"message": message}).strip().lower()
            
            valid_sentiments = ["very_positive", "positive", "neutral", "negative", "very_negative"]
            if sentiment in valid_sentiments:
                return sentiment
            else:
                # Fallback to keyword-based analysis
                return self._fallback_sentiment_analysis(message)
        except Exception as e:
            print(f"Error in enhanced sentiment analysis: {e}")
            return self._fallback_sentiment_analysis(message)

    def _fallback_sentiment_analysis(self, message: str) -> str:
        """Fallback sentiment analysis using keywords"""
        message_lower = message.lower()
        
        very_positive_words = ["excellent", "amazing", "fantastic", "perfect", "love", "thrilled"]
        positive_words = ["good", "great", "thanks", "thank you", "helpful", "resolved"]
        negative_words = ["bad", "poor", "slow", "problem", "issue", "error", "broken"]
        very_negative_words = ["terrible", "awful", "horrible", "hate", "angry", "furious"]
        
        # Count sentiment indicators
        very_negative_count = sum(1 for word in very_negative_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        positive_count = sum(1 for word in positive_words if word in message_lower)
        very_positive_count = sum(1 for word in very_positive_words if word in message_lower)
        
        # Check for escalation keywords
        escalation_count = sum(
            1 for category in self.escalation_keywords.values()
            for word in category if word in message_lower
        )
        
        if very_negative_count > 0 or escalation_count > 2:
            return "very_negative"
        elif negative_count > positive_count:
            return "negative"
        elif very_positive_count > 0:
            return "very_positive"
        elif positive_count > 0:
            return "positive"
        else:
            return "neutral"

    def categorize_issue(self, message: str) -> str:
        """Categorize the customer issue"""
        try:
            category = self.category_chain.invoke({"customer_message": message}).strip().lower()
            if category in self.support_categories:
                return category
            else:
                return self._fallback_categorization(message)
        except Exception as e:
            print(f"Error categorizing issue: {e}")
            return self._fallback_categorization(message)

    def _fallback_categorization(self, message: str) -> str:
        """Fallback issue categorization using keywords"""
        message_lower = message.lower()
        
        category_keywords = {
            "account_access": ["login", "log in", "sign in", "access", "locked", "password"],
            "billing": ["bill", "charge", "payment", "invoice", "refund", "cost", "price"],
            "technical_issues": ["bug", "error", "broken", "not working", "crash", "slow"],
            "api_problems": ["api", "endpoint", "integration", "webhook", "rate limit"],
            "password_reset": ["password", "reset", "forgot", "change password"],
            "subscription": ["plan", "upgrade", "downgrade", "subscription", "tier"],
            "data_export": ["export", "download", "backup", "migration", "import"],
            "security": ["security", "breach", "hack", "unauthorized", "permission"],
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return category
        
        return "troubleshooting"  # Default category

    def should_escalate_enhanced(self, message: str, customer_context: CustomerContext) -> tuple[bool, str]:
        """Enhanced escalation analysis"""
        try:
            escalation_result = self.escalation_chain.invoke({
                "customer_tier": customer_context.tier,
                "open_tickets": len(customer_context.open_tickets),
                "recent_interactions": len(customer_context.recent_interactions),
                "customer_message": message
            }).strip()
            
            if escalation_result.startswith("ESCALATE: YES"):
                reason = escalation_result.split(" - ", 1)[1] if " - " in escalation_result else "Multiple escalation indicators"
                return True, reason
            else:
                return False, ""
                
        except Exception as e:
            print(f"Error in enhanced escalation analysis: {e}")
            return self._fallback_escalation_analysis(message, customer_context)

    def _fallback_escalation_analysis(self, message: str, customer_context: CustomerContext) -> tuple[bool, str]:
        """Fallback escalation analysis"""
        message_lower = message.lower()
        reasons = []

        # Check escalation keywords by category
        for category, keywords in self.escalation_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                reasons.append(f"{category} indicators detected")

        # Check customer tier and ticket count
        if customer_context.tier == "enterprise" and len(customer_context.open_tickets) > 1:
            reasons.append("Enterprise customer with multiple open tickets")

        # Check satisfaction trend
        if hasattr(customer_context, 'average_satisfaction') and customer_context.average_satisfaction < 2.5:
            reasons.append("Low customer satisfaction trend")

        # Check for urgent issues
        urgent_tickets = [t for t in customer_context.open_tickets if t.get("priority") == "urgent"]
        if urgent_tickets:
            reasons.append("Urgent tickets present")

        return len(reasons) > 0, "; ".join(reasons) if reasons else ""

    def search_knowledge_base_enhanced(self, query: str, category: str = None) -> List[Dict]:
        """Enhanced knowledge base search with category filtering"""
        try:
            # Search agent-specific knowledge first
            agent_results = self.get_support_knowledge(query)
            
            # Search general knowledge base
            general_results = search_knowledge_base(query, limit=3)
            
            # Combine and deduplicate results
            all_results = agent_results + general_results
            seen_titles = set()
            unique_results = []
            
            for result in all_results:
                if result['title'] not in seen_titles:
                    seen_titles.add(result['title'])
                    unique_results.append(result)
            
            return unique_results[:5]  # Return top 5 unique results
            
        except Exception as e:
            print(f"Error in enhanced knowledge base search: {e}")
            return []

    def process_customer_message(self, customer_id: int, message: str) -> AgentResponse:
        """Enhanced customer message processing with knowledge base integration"""
        # Get customer context
        customer_context = self.get_customer_context(customer_id)
        if not customer_context:
            return AgentResponse(
                message="I apologize, but I'm having trouble accessing your customer information. Let me escalate this to ensure you receive immediate assistance.",
                confidence=0.0,
                escalate=True,
                suggested_actions=["Verify customer identity", "Manual account lookup"],
                knowledge_sources=[],
                sentiment="neutral",
            )

        # Categorize the issue
        issue_category = self.categorize_issue(message)

        # Enhanced knowledge base search
        kb_results = self.search_knowledge_base_enhanced(message, issue_category)

        # Get enhanced context
        enhanced_context = self.get_enhanced_context(message, customer_context)

        # Analyze sentiment
        sentiment = self.analyze_sentiment_enhanced(message)

        # Check for escalation
        should_escalate, escalation_reason = self.should_escalate_enhanced(message, customer_context)

        # Prepare comprehensive context for the LLM
        context_str = f"""
Customer: {customer_context.name} ({customer_context.email})
Tier: {customer_context.tier} | Company: {customer_context.company}
Satisfaction: {customer_context.average_satisfaction:.1f}/5.0
Open Tickets: {len(customer_context.open_tickets)} | Recent Interactions: {len(customer_context.recent_interactions)}

Recent Ticket Details:
{self._format_tickets(customer_context.open_tickets[:3])}

Recent Interaction Summary:
{self._format_interactions(customer_context.recent_interactions[:3])}
"""

        kb_str = self._format_knowledge_results(kb_results)
        escalation_str = f"Escalation: {'Required' if should_escalate else 'Not Required'}"
        if should_escalate:
            escalation_str += f" - Reason: {escalation_reason}"

        # Generate response using the enhanced chain
        try:
            response = self.main_chain.invoke({
                "customer_context": context_str,
                "enhanced_context": enhanced_context,
                "knowledge_base_results": kb_str,
                "issue_category": issue_category,
                "escalation_analysis": escalation_str,
                "customer_message": message,
            })

            # Update memory
            self.memory.chat_memory.add_user_message(message)
            self.memory.chat_memory.add_ai_message(response)

            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(kb_results, sentiment, customer_context)

            # Generate comprehensive suggested actions
            suggested_actions = self._generate_suggested_actions(
                should_escalate, escalation_reason, sentiment, issue_category, customer_context
            )

            return AgentResponse(
                message=response,
                confidence=confidence,
                escalate=should_escalate,
                suggested_actions=suggested_actions,
                knowledge_sources=[result["title"] for result in kb_results],
                sentiment=sentiment,
            )

        except Exception as e:
            print(f"Error generating enhanced response: {e}")
            return AgentResponse(
                message="I sincerely apologize for the technical difficulty. To ensure you receive the best possible service, I'm immediately connecting you with one of our specialist agents who can provide hands-on assistance.",
                confidence=0.0,
                escalate=True,
                suggested_actions=["Immediate escalation to human agent", "Technical support review"],
                knowledge_sources=[],
                sentiment=sentiment,
            )

    def _format_tickets(self, tickets: List[Dict]) -> str:
        """Format ticket information for context"""
        if not tickets:
            return "No recent tickets"
        
        formatted = []
        for ticket in tickets:
            formatted.append(f"#{ticket['id']}: {ticket['title']} ({ticket['status']}, {ticket['priority']})")
        return "\n".join(formatted)

    def _format_interactions(self, interactions: List[Dict]) -> str:
        """Format interaction information for context"""
        if not interactions:
            return "No recent interactions"
        
        formatted = []
        for interaction in interactions:
            formatted.append(f"- {interaction['type']}: {interaction['sentiment']} sentiment")
        return "\n".join(formatted)

    def _format_knowledge_results(self, kb_results: List[Dict]) -> str:
        """Format knowledge base results for context"""
        if not kb_results:
            return "No specific knowledge base matches found"
        
        formatted = []
        for result in kb_results:
            formatted.append(f"- {result['title']}: {result['content'][:200]}...")
        return "\n".join(formatted)

    def _calculate_confidence(self, kb_results: List[Dict], sentiment: str, customer_context: CustomerContext) -> float:
        """Calculate confidence score based on multiple factors"""
        base_confidence = 0.5
        
        # Knowledge base match boost
        kb_boost = min(0.3, len(kb_results) * 0.1)
        
        # Sentiment factor
        sentiment_factors = {
            "very_positive": 0.1,
            "positive": 0.05,
            "neutral": 0.0,
            "negative": -0.05,
            "very_negative": -0.1
        }
        sentiment_factor = sentiment_factors.get(sentiment, 0.0)
        
        # Customer tier factor
        tier_factors = {"enterprise": 0.1, "business": 0.05, "free": 0.0}
        tier_factor = tier_factors.get(customer_context.tier, 0.0)
        
        # Customer satisfaction factor
        if hasattr(customer_context, 'average_satisfaction'):
            satisfaction_factor = (customer_context.average_satisfaction - 3.0) * 0.1
        else:
            satisfaction_factor = 0.0
        
        confidence = base_confidence + kb_boost + sentiment_factor + tier_factor + satisfaction_factor
        return min(0.95, max(0.1, confidence))

    def _generate_suggested_actions(self, should_escalate: bool, escalation_reason: str, 
                                   sentiment: str, category: str, customer_context: CustomerContext) -> List[str]:
        """Generate comprehensive suggested actions"""
        actions = []
        
        if should_escalate:
            actions.append(f"Escalate to supervisor: {escalation_reason}")
        
        # Category-specific actions
        category_actions = {
            "account_access": ["Verify identity", "Check account status", "Password reset assistance"],
            "billing": ["Review billing history", "Process refund if applicable", "Update payment method"],
            "technical_issues": ["Technical diagnostics", "Check system status", "Submit bug report"],
            "api_problems": ["Check API status", "Review integration logs", "Rate limit analysis"],
            "security": ["Security review", "Account audit", "Enable 2FA"],
        }
        
        if category in category_actions:
            actions.extend(category_actions[category][:2])
        
        # Sentiment-based actions
        if sentiment in ["negative", "very_negative"]:
            actions.append("Follow up within 4 hours")
            if customer_context.tier == "enterprise":
                actions.append("Priority handling required")
        elif sentiment in ["positive", "very_positive"]:
            actions.append("Document success for best practices")
        
        # Tier-specific actions
        if customer_context.tier == "enterprise":
            actions.append("Dedicated support specialist assignment")
        elif customer_context.tier == "free":
            actions.append("Direct to self-service resources")
        
        return actions[:5]  # Limit to 5 actions

    def create_support_ticket(self, customer_id: int, message: str, category: str, priority: str = "medium") -> int:
        """Create a support ticket for tracking"""
        session = get_session()
        try:
            ticket = SupportTicket(
                customer_id=customer_id,
                title=message[:100] + "..." if len(message) > 100 else message,
                description=message,
                category=category,
                priority=priority,
                status="open",
                created_at=datetime.now()
            )
            session.add(ticket)
            session.commit()
            return ticket.id
        except Exception as e:
            print(f"Error creating support ticket: {e}")
            session.rollback()
            return None
        finally:
            session.close()

    def log_interaction_enhanced(self, customer_id: int, message: str, response: AgentResponse, 
                               category: str, ticket_id: int = None):
        """Enhanced interaction logging with more metadata"""
        session = get_session()
        try:
            # Calculate satisfaction score based on sentiment and confidence
            satisfaction_mapping = {
                "very_positive": 5.0,
                "positive": 4.0,
                "neutral": 3.0,
                "negative": 2.0,
                "very_negative": 1.0
            }
            satisfaction = satisfaction_mapping.get(response.sentiment, 3.0)
            
            # Adjust based on confidence
            if response.confidence > 0.8:
                satisfaction = min(5.0, satisfaction + 0.5)
            elif response.confidence < 0.5:
                satisfaction = max(1.0, satisfaction - 0.5)

            interaction = CustomerInteraction(
                customer_id=customer_id,
                interaction_type="enhanced_chat",
                message=message,
                response=response.message,
                sentiment=response.sentiment,
                satisfaction_score=satisfaction,
                agent_id="customer_support_agent_ai_enhanced",
                metadata={
                    "category": category,
                    "confidence": response.confidence,
                    "escalated": response.escalate,
                    "knowledge_sources": response.knowledge_sources,
                    "suggested_actions": response.suggested_actions,
                    "ticket_id": ticket_id
                },
                created_at=datetime.now()
            )
            session.add(interaction)
            session.commit()
        except Exception as e:
            print(f"Error logging enhanced interaction: {e}")
            session.rollback()
        finally:
            session.close()

    def handle_customer_request_enhanced(self, customer_id: int, message: str, 
                                       create_ticket: bool = False) -> Dict[str, Any]:
        """Enhanced main method to handle a customer request"""
        # Categorize the issue first
        category = self.categorize_issue(message)
        
        # Create ticket if requested or if it's a complex issue
        ticket_id = None
        if create_ticket or category in ["technical_issues", "billing", "security"]:
            priority = "urgent" if any(word in message.lower() for word in ["urgent", "critical", "emergency"]) else "medium"
            ticket_id = self.create_support_ticket(customer_id, message, category, priority)

        # Process the message
        response = self.process_customer_message(customer_id, message)

        # Log the enhanced interaction
        self.log_interaction_enhanced(customer_id, message, response, category, ticket_id)

        # Return comprehensive structured response
        return {
            "response": response.message,
            "confidence": response.confidence,
            "escalate": response.escalate,
            "suggested_actions": response.suggested_actions,
            "knowledge_sources": response.knowledge_sources,
            "sentiment": response.sentiment,
            "category": category,
            "ticket_id": ticket_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "agent_version": "enhanced_v2.0",
                "processing_time": "< 3 seconds",
                "knowledge_base_used": len(response.knowledge_sources) > 0
            }
        }

    def get_customer_insights(self, customer_id: int) -> Dict[str, Any]:
        """Get comprehensive customer insights"""
        customer_context = self.get_customer_context(customer_id)
        if not customer_context:
            return {"error": "Customer not found"}

        session = get_session()
        try:
            # Get interaction statistics
            interactions = session.query(CustomerInteraction).filter(
                CustomerInteraction.customer_id == customer_id
            ).all()

            sentiment_distribution = {}
            category_distribution = {}
            satisfaction_scores = []

            for interaction in interactions:
                # Sentiment distribution
                sentiment = interaction.sentiment
                sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1

                # Category distribution (from metadata)
                if interaction.metadata and 'category' in interaction.metadata:
                    category = interaction.metadata['category']
                    category_distribution[category] = category_distribution.get(category, 0) + 1

                # Satisfaction scores
                if interaction.satisfaction_score:
                    satisfaction_scores.append(interaction.satisfaction_score)

            return {
                "customer_info": {
                    "name": customer_context.name,
                    "email": customer_context.email,
                    "tier": customer_context.tier,
                    "company": customer_context.company
                },
                "interaction_stats": {
                    "total_interactions": len(interactions),
                    "sentiment_distribution": sentiment_distribution,
                    "category_distribution": category_distribution,
                    "average_satisfaction": sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0,
                    "satisfaction_trend": satisfaction_scores[-5:] if len(satisfaction_scores) >= 5 else satisfaction_scores
                },
                "current_status": {
                    "open_tickets": len(customer_context.open_tickets),
                    "recent_sentiment": customer_context.recent_interactions[0]['sentiment'] if customer_context.recent_interactions else 'neutral'
                }
            }
        finally:
            session.close()


def test_customer_support_agent_enhanced():
    """Comprehensive test of the enhanced customer support agent"""
    agent = CustomerSupportAgent()

    # Enhanced test scenarios
    test_cases = [
        {
            "customer_id": 1,
            "message": "I can't log into my account. I keep getting an error message saying 'invalid credentials' but I'm sure my password is correct.",
            "description": "Account access issue",
            "create_ticket": False
        },
        {
            "customer_id": 2,
            "message": "This is absolutely ridiculous! I was charged twice this month for $299 each and I want my money back immediately or I'm calling my lawyer!",
            "description": "Angry billing dispute",
            "create_ticket": True
        },
        {
            "customer_id": 3,
            "message": "Our production API is returning 500 errors since this morning. This is affecting thousands of our customers. We need immediate assistance.",
            "description": "Critical technical issue",
            "create_ticket": True
        },
        {
            "customer_id": 1,
            "message": "Thank you so much for helping me reset my password yesterday. Everything is working perfectly now and your support was excellent!",
            "description": "Positive feedback",
            "create_ticket": False
        },
        {
            "customer_id": 4,
            "message": "I need to upgrade my subscription plan but I'm not sure which one would be best for our company's needs.",
            "description": "Subscription inquiry",
            "create_ticket": False
        },
        {
            "customer_id": 5,
            "message": "We've had a security breach and unauthorized access to our API keys. I need to speak with your security team immediately.",
            "description": "Security incident",
            "create_ticket": True
        },
    ]

    print("=== Enhanced Customer Support Agent Test ===\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['description']}")
        print(f"Customer ID: {test_case['customer_id']}")
        print(f"Message: {test_case['message']}")
        print(f"Create Ticket: {test_case['create_ticket']}")

        result = agent.handle_customer_request_enhanced(
            test_case["customer_id"], 
            test_case["message"],
            test_case["create_ticket"]
        )

        print(f"Response: {result['response'][:200]}...")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Escalate: {result['escalate']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Category: {result['category']}")
        print(f"Ticket ID: {result['ticket_id']}")
        print(f"Suggested Actions: {', '.join(result['suggested_actions'][:3])}")
        print(f"Knowledge Sources: {', '.join(result['knowledge_sources'][:2])}")
        print("-" * 80)

    # Test knowledge base integration
    print("\n=== Knowledge Base Integration Test ===")
    knowledge = agent.get_support_knowledge("billing issues enterprise customers")
    print(f"Knowledge base results: {len(knowledge)} items found")
    if knowledge:
        print(f"First result: {knowledge[0]['title']}")

    # Test customer insights
    print("\n=== Customer Insights Test ===")
    insights = agent.get_customer_insights(1)
    print(f"Customer insights: {insights}")


if __name__ == "__main__":
    test_customer_support_agent_enhanced()
