# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-05 15:00:00
# @Description: This module implements a customer support agent using LangChain and SQLAlchemy with CSV feedback integration

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from langchain.schema.output_parser import StrOutputParser
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.que_agents.core.database import (
    Customer,
    CustomerInteraction,
    SupportTicket,
    get_session,
)
from src.que_agents.core.llm_factory import LLMFactory
from src.que_agents.core.schemas import AgentResponse, CustomerContext
from src.que_agents.error_trace.errorlogger import system_logger
from src.que_agents.knowledge_base.kb_manager import (
    search_agent_knowledge_base,
    search_knowledge_base,
)

system_logger.info(
    "Customer Support Agent initialized", {"timestamp": datetime.now().isoformat()}
)

FEEDBACK_TEXT = "Feedback Text"
FEEDBACK_DATE = "Feedback Date"
CUSTOMER_ID = "Customer ID"

# Load agent configuration
try:
    with open("./configs/agent_config.yaml", "r") as f:
        agent_config = yaml.safe_load(f)
except FileNotFoundError:
    system_logger.error("agent_config.yaml not found. Please check the file path.")
    agent_config = {}


class CustomerFeedbackManager:
    """Manager for handling customer feedback CSV data"""

    FEEDBACK_DATE_COL = FEEDBACK_DATE
    RESOLUTION_DATE_COL = "Resolution Date"
    CUSTOMER_ID_COL = CUSTOMER_ID
    RATING_COL = "Rating"
    CATEGORY_COL = "Category"
    RESOLUTION_STATUS_COL = "Resolution Status"
    ESCALATED_COL = "Escalated"
    RESPONSE_TIME_COL = "Response Time (hours)"
    SATISFACTION_SCORE_COL = "Satisfaction Score"

    def __init__(self, csv_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "semi_structured", "customer_feedback.csv")):
        self.csv_path = csv_path
        self.feedback_data: Optional[pd.DataFrame] = None
        self.load_feedback_data()

    def load_feedback_data(self):
        """Load customer feedback data from CSV with improved date handling"""
        try:
            if os.path.exists(self.csv_path):
                self.feedback_data = pd.read_csv(self.csv_path)

                # Improved date parsing with multiple format support
                for date_col in [self.FEEDBACK_DATE_COL, self.RESOLUTION_DATE_COL]:
                    if date_col in self.feedback_data.columns:
                        # Handle multiple date formats
                        self.feedback_data[date_col] = pd.to_datetime(
                            self.feedback_data[date_col],
                            format="mixed",  # This handles multiple formats automatically
                            errors="coerce",
                        )

                        # Fill any NaT values with a default date
                        self.feedback_data[date_col] = self.feedback_data[
                            date_col
                        ].fillna(pd.Timestamp("2023-01-01"))

                system_logger.info(
                    f"Loaded customer feedback data from {self.csv_path}",
                    {"row_count": len(self.feedback_data)},
                )
            else:
                system_logger.warning(f"Feedback CSV file not found at {self.csv_path}")
                self.feedback_data = self._create_empty_feedback_dataframe()
        except Exception as e:
            system_logger.error(
                f"Error loading feedback data from {self.csv_path}: {e}", exc_info=True
            )
            self.feedback_data = self._create_empty_feedback_dataframe()

    def get_customer_feedback_history(self, customer_id: int) -> List[Dict]:
        """Get feedback history for a specific customer"""
        if self.feedback_data is None or self.feedback_data.empty:
            return []

        customer_feedback = self.feedback_data[
            self.feedback_data[self.CUSTOMER_ID_COL] == customer_id
        ].sort_values(self.FEEDBACK_DATE_COL, ascending=False)

        return customer_feedback.to_dict("records")

    def _create_empty_feedback_dataframe(self) -> pd.DataFrame:
        """Create an empty DataFrame with proper column types"""
        return pd.DataFrame(
            {
                self.CUSTOMER_ID_COL: pd.Series(dtype="int64"),
                self.FEEDBACK_DATE_COL: pd.Series(dtype="datetime64[ns]"),
                self.RATING_COL: pd.Series(dtype="float64"),
                self.CATEGORY_COL: pd.Series(dtype="str"),
                self.RESOLUTION_STATUS_COL: pd.Series(dtype="str"),
                self.ESCALATED_COL: pd.Series(dtype="str"),
                self.RESPONSE_TIME_COL: pd.Series(dtype="float64"),
                self.SATISFACTION_SCORE_COL: pd.Series(dtype="float64"),
                self.RESOLUTION_DATE_COL: pd.Series(dtype="datetime64[ns]"),
                "Sentiment": pd.Series(dtype="str"),
                "Subcategory": pd.Series(dtype="str"),
                FEEDBACK_TEXT: pd.Series(dtype="str"),
            }
        )

    def get_feedback_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback trends for the last N days with improved date handling"""
        if self.feedback_data is None or self.feedback_data.empty:
            return {}

        try:
            # Ensure cutoff_date is a pandas Timestamp
            cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=days))

            # Ensure the date column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(
                self.feedback_data[self.FEEDBACK_DATE_COL]
            ):
                self.feedback_data[self.FEEDBACK_DATE_COL] = pd.to_datetime(
                    self.feedback_data[self.FEEDBACK_DATE_COL],
                    format="mixed",
                    errors="coerce",
                )

            # Filter recent feedback with proper datetime comparison
            recent_feedback = self.feedback_data[
                self.feedback_data[self.FEEDBACK_DATE_COL] >= cutoff_date
            ].copy()

            if recent_feedback.empty:
                return {
                    "total_feedback": 0,
                    "average_rating": 0.0,
                    "category_distribution": {},
                    "sentiment_distribution": {},
                    "resolution_rate": 0.0,
                    "escalation_rate": 0.0,
                    "average_response_time": 0.0,
                }

            # Calculate metrics safely
            resolution_rate = self._calculate_resolution_rate(recent_feedback)
            escalation_rate = self._calculate_escalation_rate(recent_feedback)

            return {
                "total_feedback": len(recent_feedback),
                "average_rating": (
                    recent_feedback[self.RATING_COL].mean()
                    if not recent_feedback[self.RATING_COL].isna().all()
                    else 0.0
                ),
                "category_distribution": recent_feedback[self.CATEGORY_COL]
                .value_counts()
                .to_dict(),
                "sentiment_distribution": (
                    recent_feedback["Sentiment"].value_counts().to_dict()
                    if "Sentiment" in recent_feedback.columns
                    else {}
                ),
                "resolution_rate": resolution_rate,
                "escalation_rate": escalation_rate,
                "average_response_time": (
                    recent_feedback[self.RESPONSE_TIME_COL].mean()
                    if not recent_feedback[self.RESPONSE_TIME_COL].isna().all()
                    else 0.0
                ),
            }

        except Exception as e:
            system_logger.error(
                f"Error calculating feedback trends: {e}", exc_info=True
            )
            return {}

    def _calculate_resolution_rate(self, data: pd.DataFrame) -> float:
        """Calculate resolution rate safely"""
        try:
            if data.empty:
                return 0.0
            resolved_count = (
                data[self.RESOLUTION_STATUS_COL].isin(["Resolved", "Closed"]).sum()
            )
            return (resolved_count / len(data) * 100) if len(data) > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_escalation_rate(self, data: pd.DataFrame) -> float:
        """Calculate escalation rate safely"""
        try:
            if data.empty:
                return 0.0
            escalated_count = (data[self.ESCALATED_COL] == "Yes").sum()
            return (escalated_count / len(data) * 100) if len(data) > 0 else 0.0
        except Exception:
            return 0.0

    def get_similar_issues(
        self, category: str, subcategory: Optional[str] = None, limit: int = 5
    ) -> List[Dict]:
        """Get similar resolved issues for reference with improved date handling"""
        if self.feedback_data is None or self.feedback_data.empty:
            return []

        try:
            similar_issues = self.feedback_data[
                (self.feedback_data[self.CATEGORY_COL] == category)
                & (
                    self.feedback_data[self.RESOLUTION_STATUS_COL].isin(
                        ["Resolved", "Closed"]
                    )
                )
            ].copy()

            if subcategory and "Subcategory" in similar_issues.columns:
                similar_issues = similar_issues[
                    similar_issues["Subcategory"] == subcategory
                ]

            # Sort by rating and date - ensure date column is datetime
            if not similar_issues.empty:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(
                    similar_issues[self.FEEDBACK_DATE_COL]
                ):
                    similar_issues[self.FEEDBACK_DATE_COL] = pd.to_datetime(
                        similar_issues[self.FEEDBACK_DATE_COL],
                        format="mixed",
                        errors="coerce",
                    )

                similar_issues = similar_issues.sort_values(
                    [self.RATING_COL, self.FEEDBACK_DATE_COL],
                    ascending=[False, False],
                    na_position="last",
                ).head(limit)

            return similar_issues.to_dict("records")

        except Exception as e:
            system_logger.error(f"Error getting similar issues: {e}", exc_info=True)
            return []

    def add_feedback_entry(self, feedback_data: Dict[str, Any]):
        """Add new feedback entry to CSV"""
        try:
            new_row = pd.DataFrame([feedback_data])
            if self.feedback_data is not None and not self.feedback_data.empty:
                self.feedback_data = pd.concat(
                    [self.feedback_data, new_row], ignore_index=True
                )
            else:
                self.feedback_data = new_row

            # Ensure the feedback directory exists
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            self.feedback_data.to_csv(self.csv_path, index=False)

            system_logger.info(
                "Added new feedback entry",
                {"customer_id": feedback_data.get(self.CUSTOMER_ID_COL)},
            )
        except Exception as e:
            system_logger.error(f"Error adding feedback entry: {e}", exc_info=True)

    def get_customer_satisfaction_trend(self, customer_id: int) -> Dict[str, Any]:
        """Get satisfaction trend for a specific customer with improved error handling"""
        if self.feedback_data is None or self.feedback_data.empty:
            return {}

        try:
            customer_feedback = self.feedback_data[
                self.feedback_data[self.CUSTOMER_ID_COL] == customer_id
            ].copy()

            if customer_feedback.empty:
                return {}

            # Ensure date column is datetime and sort
            if not pd.api.types.is_datetime64_any_dtype(
                customer_feedback[self.FEEDBACK_DATE_COL]
            ):
                customer_feedback[self.FEEDBACK_DATE_COL] = pd.to_datetime(
                    customer_feedback[self.FEEDBACK_DATE_COL],
                    format="mixed",
                    errors="coerce",
                )

            customer_feedback = customer_feedback.sort_values(
                self.FEEDBACK_DATE_COL, na_position="first"
            )

            # Get ratings and satisfaction scores safely
            ratings = customer_feedback[self.RATING_COL].dropna().tolist()
            satisfaction_scores = (
                customer_feedback[self.SATISFACTION_SCORE_COL].dropna().tolist()
            )

            result = {
                "rating_trend": ratings,
                "satisfaction_trend": satisfaction_scores,
                "latest_rating": ratings[-1] if ratings else None,
                "latest_satisfaction": (
                    satisfaction_scores[-1] if satisfaction_scores else None
                ),
                "average_rating": sum(ratings) / len(ratings) if ratings else None,
                "trend_direction": self._calculate_trend_direction(ratings),
                "feedback_count": len(customer_feedback),
            }

            return result

        except Exception as e:
            system_logger.error(
                f"Error getting customer satisfaction trend: {e}", exc_info=True
            )
            return {}

    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate if trend is improving, declining, or stable"""
        if len(values) < 2:
            return "insufficient_data"

        # Calculate average of the last 3 values and compare to the average of previous values
        if len(values) > 3:
            recent_avg = sum(values[-3:]) / 3
            older_avg = sum(values[:-3]) / len(values[:-3])
        else:
            recent_avg = values[-1]
            older_avg = values[0]

        if recent_avg > older_avg + 0.5:
            return "improving"
        elif recent_avg < older_avg - 0.5:
            return "declining"
        else:
            return "stable"


class CustomerSupportAgent:
    """Enhanced Customer Support Agent using LangChain with Knowledge Base and Feedback Integration"""

    def __init__(self):
        config = agent_config.get("customer_support_agent", {})
        self.llm = LLMFactory.get_llm(
            agent_type="customer_support",
            model_name=config.get("model_name", "gpt-4"),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 500),
        )

        # Initialize feedback manager
        self.feedback_manager = CustomerFeedbackManager()

        # New memory system using InMemoryChatMessageHistory
        self.store = {}  # Session storage for conversation history

        # Enhanced escalation keywords categorized by type
        self.escalation_keywords = {
            "anger": ["angry", "furious", "rage", "mad", "pissed", "outraged"],
            "legal": ["lawsuit", "legal", "attorney", "lawyer", "sue", "court"],
            "cancellation": ["cancel", "unsubscribe", "terminate", "quit"],
            "refund": ["refund", "money back", "chargeback", "dispute"],
            "management": ["manager", "supervisor", "boss", "escalate", "senior"],
            "complaint": [
                "complaint",
                "complain",
                "terrible",
                "awful",
                "horrible",
                "worst",
            ],
            "dissatisfaction": [
                "unacceptable",
                "frustrated",
                "disappointed",
                "disgusted",
            ],
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
            "performance",
            "features",
        ]

        # Initialize prompt templates
        self.main_prompt = self._create_main_prompt_template()
        self.sentiment_prompt = self._create_sentiment_prompt()
        self.escalation_prompt = self._create_escalation_prompt()
        self.category_prompt = self._create_category_prompt()

        # Create chains with new memory system
        self.main_chain = self._create_main_chain()
        self.sentiment_chain = self._create_sentiment_chain()
        self.escalation_chain = self._create_escalation_chain()
        self.category_chain = self._create_category_chain()

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create session history for conversation memory"""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def get_support_knowledge(self, query: str) -> List[Dict]:
        """Get customer support knowledge from knowledge base"""
        try:
            kb_results = search_agent_knowledge_base("customer_support", query, limit=3)
            system_logger.info(
                "Knowledge base queried", {"query": query, "results": len(kb_results)}
            )
            return kb_results
        except Exception as e:
            system_logger.error(
                f"Error searching support knowledge: {e}", exc_info=True
            )
            return []

    def get_enhanced_context(
        self, customer_message: str, customer_context: CustomerContext
    ) -> str:
        """Get enhanced context from knowledge base and feedback data"""
        try:
            # Categorize the issue first
            category = self.categorize_issue(customer_message)

            # Search for category-specific knowledge
            category_knowledge = self.get_support_knowledge(
                f"{category} {customer_message}"
            )

            # Search for tier-specific knowledge
            tier_knowledge = self.get_support_knowledge(
                f"{customer_context.tier} customer support"
            )

            # Get customer feedback history
            feedback_history = self.feedback_manager.get_customer_feedback_history(
                customer_context.customer_id
            )

            # Get similar resolved issues
            similar_issues = self.feedback_manager.get_similar_issues(category)

            # Get customer satisfaction trend
            satisfaction_trend = self.feedback_manager.get_customer_satisfaction_trend(
                customer_context.customer_id
            )

            enhanced_context = ""

            if category_knowledge:
                enhanced_context += "Relevant Support Knowledge:\n"
                for kb_item in category_knowledge:
                    enhanced_context += (
                        f"- {kb_item['title']}: {kb_item['content'][:200]}...\n"
                    )

            if tier_knowledge:
                enhanced_context += (
                    f"\n{customer_context.tier.title()} Tier Guidelines:\n"
                )
                for kb_item in tier_knowledge:
                    enhanced_context += (
                        f"- {kb_item['title']}: {kb_item['content'][:150]}...\n"
                    )

            if feedback_history:
                enhanced_context += f"\nCustomer Feedback History (Last {len(feedback_history)} entries):\n"
                for feedback in feedback_history[:3]:  # Show last 3 feedback entries
                    enhanced_context += (
                        f"- {feedback.get('Feedback Date', 'N/A')}: Rating {feedback.get('Rating', 'N/A')}/5 "
                        f"({feedback.get('Category', 'N/A')}) - {feedback.get('Resolution Status', 'N/A')}\n"
                    )

            if (
                satisfaction_trend
                and satisfaction_trend.get("latest_rating") is not None
            ):
                enhanced_context += "\nCustomer Satisfaction Trend:\n"
                enhanced_context += f"- Latest Rating: {satisfaction_trend.get('latest_rating', 'N/A')}/5\n"
                enhanced_context += f"- Trend Direction: {satisfaction_trend.get('trend_direction', 'N/A')}\n"
                enhanced_context += f"- Average Rating: {satisfaction_trend.get('average_rating', 0):.1f}/5\n"

            if similar_issues:
                enhanced_context += "\nSimilar Resolved Issues:\n"
                for issue in similar_issues[:2]:  # Show top 2 similar issues
                    enhanced_context += (
                        f"- {issue.get('Feedback Text', '')[:100]}... "
                        f"(Rating: {issue.get('Rating', 'N/A')}/5, Status: {issue.get('Resolution Status', 'N/A')})\n"
                    )

            return enhanced_context
        except Exception as e:
            system_logger.error(f"Error getting enhanced context: {e}", exc_info=True)
            return ""

    def _create_main_prompt_template(self) -> ChatPromptTemplate:
        """Create the enhanced main prompt template for the customer support agent"""
        system_message = """You are an expert customer support agent for an AI Analytics Platform. Your role is to provide exceptional customer service with:

CORE RESPONSIBILITIES:
1. Provide empathetic, professional, and solution-focused responses
2. Use knowledge base information and historical feedback to give accurate, helpful solutions
3. Acknowledge customer emotions and validate their concerns
4. Consider customer's feedback history and satisfaction trends
5. Learn from similar resolved issues to provide better solutions
6. Escalate appropriately based on defined criteria and customer history
7. Document interactions for seamless follow-up
8. Maintain brand voice and customer satisfaction

CUSTOMER TIER CONSIDERATIONS:
- Enterprise: Priority support, dedicated resources, immediate escalation for critical issues
- Business: Standard support, prompt responses, escalate for billing issues >$200
- Free: Community support focus, self-service resources, escalate for account security

FEEDBACK-DRIVEN INSIGHTS:
- Consider customer's historical satisfaction trends
- Reference similar resolved issues for solution guidance
- Adapt response style based on customer's feedback patterns
- Prioritize customers with declining satisfaction trends

ESCALATION CRITERIA:
- Legal threats or regulatory compliance issues
- Requests for management/supervisor
- Billing disputes: >$500 (Free/Business), >$1000 (Enterprise)
- Critical technical issues affecting production systems
- Security breaches or data concerns
- Multiple unresolved tickets or repeated complaints
- Customer satisfaction clearly deteriorating (based on feedback trends)
- Customers with history of escalations

RESPONSE STRUCTURE:
1. Acknowledge the issue and show empathy (consider feedback history)
2. Reference relevant knowledge base solutions and similar resolved cases
3. Provide clear, actionable steps based on successful resolutions
4. Offer additional assistance
5. Set expectations for follow-up

Customer Context: {customer_context}
Enhanced Knowledge: {enhanced_context}
Knowledge Base Results: {knowledge_base_results}
Issue Category: {issue_category}
Escalation Analysis: {escalation_analysis}
Feedback Insights: {feedback_insights}

Provide a comprehensive, empathetic response that addresses the customer's specific concern using available knowledge, context, and feedback insights."""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="history"),
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

        return ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", "{message}")]
        )

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
- Declining customer satisfaction trends
- History of escalations

Customer Tier: {customer_tier}
Open Tickets: {open_tickets}
Recent Interactions: {recent_interactions}
Feedback History: {feedback_history}
Customer Message: {customer_message}

Respond with:
1. "YES" or "NO" for escalation needed
2. Brief reason (if YES)

Format: ESCALATE: YES/NO - [reason if applicable]"""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "Analyze escalation need for: {customer_message}"),
            ]
        )

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
- performance: Speed, latency, system performance issues
- features: Feature requests, missing functionality

Respond with ONLY the category name from the list above."""

        return ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", "{customer_message}")]
        )

    def _create_main_chain(self):
        """Create the main LangChain processing chain with new memory system"""
        chain = self.main_prompt | self.llm | StrOutputParser()

        # Wrap with message history
        return RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="customer_message",
            history_messages_key="history",
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
                system_logger.warning(
                    f"Customer with ID {customer_id} not found. Creating a default entry."
                )
                customer = Customer(
                    id=customer_id,
                    name=f"Customer {customer_id}",
                    email=f"customer{customer_id}@example.com",
                    tier="business",
                    company=f"Company {customer_id}",
                    created_at=datetime.now(),
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

            # Calculate satisfaction trend from feedback data
            satisfaction_trend = self.feedback_manager.get_customer_satisfaction_trend(
                customer_id
            )
            avg_satisfaction = satisfaction_trend.get("average_rating", 3.5)

            return CustomerContext(
                customer_id=customer.id,
                name=customer.name,
                email=customer.email,
                tier=customer.tier,
                company=customer.company or "N/A",
                recent_interactions=[
                    {
                        "type": i.interaction_type,
                        "message": (
                            i.message[:100] + "..."
                            if len(i.message) > 100
                            else i.message
                        ),
                        "response": (
                            i.response[:100] + "..."
                            if len(i.response) > 100
                            else i.response
                        ),
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
                purchase_history=[],
                preferences={},
                satisfaction_score=avg_satisfaction,
                lifetime_value=1000.0,  # Default value, should be calculated
                risk_score=0.1,  # Default low risk, should be calculated
            )
        finally:
            if session:
                session.close()

    def analyze_sentiment_enhanced(self, message: str) -> str:
        """Enhanced sentiment analysis using the LLM"""
        try:
            sentiment = (
                self.sentiment_chain.invoke({"message": message}).strip().lower()
            )
            if sentiment in [
                "very_positive",
                "positive",
                "neutral",
                "negative",
                "very_negative",
            ]:
                system_logger.info(
                    "Sentiment analyzed using LLM", {"sentiment": sentiment}
                )
                return sentiment
            else:
                system_logger.warning(
                    "LLM sentiment out of range, using fallback",
                    {"sentiment": sentiment},
                )
                return self._fallback_sentiment_analysis(message)
        except Exception as e:
            system_logger.error(
                f"Error in enhanced sentiment analysis: {e}", exc_info=True
            )
            return self._fallback_sentiment_analysis(message)

    def _fallback_sentiment_analysis(self, message: str) -> str:
        """Fallback sentiment analysis using keywords"""
        message_lower = message.lower()

        very_positive_words = [
            "excellent",
            "amazing",
            "fantastic",
            "perfect",
            "love",
            "thrilled",
        ]
        positive_words = ["good", "great", "thanks", "thank you", "helpful", "resolved"]
        negative_words = ["bad", "poor", "slow", "problem", "issue", "error", "broken"]
        very_negative_words = [
            "terrible",
            "awful",
            "horrible",
            "hate",
            "angry",
            "furious",
        ]

        # Count sentiment indicators
        very_negative_count = sum(
            1 for word in very_negative_words if word in message_lower
        )
        negative_count = sum(1 for word in negative_words if word in message_lower)
        positive_count = sum(1 for word in positive_words if word in message_lower)
        very_positive_count = sum(
            1 for word in very_positive_words if word in message_lower
        )

        # Check for escalation keywords
        escalation_count = sum(
            1
            for category in self.escalation_keywords.values()
            for word in category
            if word in message_lower
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
            category = (
                self.category_chain.invoke({"customer_message": message})
                .strip()
                .lower()
            )
            if category in self.support_categories:
                return category
            else:
                return self._fallback_categorization(message)
        except Exception as e:
            system_logger.error(f"Error categorizing issue: {e}", exc_info=True)
            return self._fallback_categorization(message)

    def _fallback_categorization(self, message: str) -> str:
        """Fallback issue categorization using keywords"""
        message_lower = message.lower()

        category_keywords = {
            "account_access": [
                "login",
                "log in",
                "sign in",
                "access",
                "locked",
                "password",
            ],
            "billing": [
                "bill",
                "charge",
                "payment",
                "invoice",
                "refund",
                "cost",
                "price",
            ],
            "technical_issues": [
                "bug",
                "error",
                "broken",
                "not working",
                "crash",
                "slow",
            ],
            "api_problems": ["api", "endpoint", "integration", "webhook", "rate limit"],
            "password_reset": ["password", "reset", "forgot", "change password"],
            "subscription": ["plan", "upgrade", "downgrade", "subscription", "tier"],
            "data_export": ["export", "download", "backup", "migration", "import"],
            "security": ["security", "breach", "hack", "unauthorized", "permission"],
            "performance": ["slow", "speed", "performance", "latency", "timeout"],
            "features": ["feature", "functionality", "missing", "request"],
        }

        for category, keywords in category_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return category

        return "troubleshooting"  # Default category

    def should_escalate_enhanced(
        self, message: str, customer_context: CustomerContext
    ) -> tuple[bool, str]:
        """Enhanced escalation analysis with feedback history"""
        try:
            # Get customer feedback history for escalation context
            feedback_history = self.feedback_manager.get_customer_feedback_history(
                customer_context.customer_id
            )

            feedback_summary = ""
            if feedback_history:
                recent_ratings = [f.get("Rating", 3) for f in feedback_history[:3]]
                escalated_count = sum(
                    1 for f in feedback_history if f.get("Escalated") == "Yes"
                )
                feedback_summary = f"Recent ratings: {recent_ratings}, Previous escalations: {escalated_count}"

            escalation_result = self.escalation_chain.invoke(
                {
                    "customer_tier": customer_context.tier,
                    "open_tickets": len(customer_context.open_tickets),
                    "recent_interactions": len(customer_context.recent_interactions),
                    "feedback_history": feedback_summary,
                    "customer_message": message,
                }
            ).strip()

            if escalation_result.startswith("ESCALATE: YES"):
                reason = (
                    escalation_result.split(" - ", 1)[1]
                    if " - " in escalation_result
                    else "Multiple escalation indicators"
                )
                return True, reason
            else:
                return False, ""

        except Exception as e:
            system_logger.error(
                f"Error in enhanced escalation analysis: {e}", exc_info=True
            )
            return self._fallback_escalation_analysis(message, customer_context)

    def _fallback_escalation_analysis(
        self, message: str, customer_context: CustomerContext
    ) -> tuple[bool, str]:
        """Fallback escalation analysis"""
        message_lower = message.lower()
        reasons = []

        # Check escalation keywords by category
        for category, keywords in self.escalation_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                reasons.append(f"{category} indicators detected")

        # Check customer tier and ticket count
        if (
            customer_context.tier == "enterprise"
            and len(customer_context.open_tickets) > 1
        ):
            reasons.append("Enterprise customer with multiple open tickets")

        # Check satisfaction trend from feedback
        satisfaction_trend = self.feedback_manager.get_customer_satisfaction_trend(
            customer_context.customer_id
        )
        if satisfaction_trend.get("trend_direction") == "declining":
            reasons.append("Declining customer satisfaction trend")

        # Check for urgent issues
        urgent_tickets = [
            t for t in customer_context.open_tickets if t.get("priority") == "urgent"
        ]
        if urgent_tickets:
            reasons.append("Urgent tickets present")

        return len(reasons) > 0, "; ".join(reasons) if reasons else ""

    def search_knowledge_base_enhanced(
        self, query: str, category: Optional[str] = None
    ) -> List[Dict]:
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
                if result["title"] not in seen_titles:
                    seen_titles.add(result["title"])
                    unique_results.append(result)

            return unique_results[:5]  # Return top 5 unique results

        except Exception as e:
            system_logger.error(f"Error searching knowledge base: {e}", exc_info=True)
            return []

    def get_feedback_insights(self, customer_id: int, category: str) -> str:
        """Get insights from feedback data for better responses with improved error handling"""
        try:
            # Get customer satisfaction trend
            satisfaction_trend = self.feedback_manager.get_customer_satisfaction_trend(
                customer_id
            )

            # Get similar resolved issues
            similar_issues = self.feedback_manager.get_similar_issues(category, limit=3)

            # Get overall feedback trends
            feedback_trends = self.feedback_manager.get_feedback_trends(days=30)

            insights = ""

            if satisfaction_trend and satisfaction_trend.get("trend_direction"):
                insights += f"Customer Satisfaction: {satisfaction_trend.get('trend_direction', 'stable')} trend, "
                latest_rating = satisfaction_trend.get("latest_rating")
                if latest_rating is not None:
                    insights += f"latest rating {latest_rating}/5\n"
                else:
                    insights += "no recent ratings\n"

            if similar_issues:
                insights += f"Similar Issues Resolved: {len(similar_issues)} cases with avg rating "
                ratings = [
                    issue.get("Rating")
                    for issue in similar_issues
                    if issue.get("Rating") is not None
                ]
                if ratings:
                    avg_rating = sum(ratings) / len(ratings)
                    insights += f"{avg_rating:.1f}/5\n"
                else:
                    insights += "no ratings available\n"

            if feedback_trends and feedback_trends.get("category_distribution"):
                category_issues_count = feedback_trends["category_distribution"].get(
                    category, 0
                )
                resolution_rate = feedback_trends.get("resolution_rate", 0)
                insights += f"Category Trends: {category_issues_count} "
                insights += f"recent {category} issues, {resolution_rate:.1f}% resolution rate\n"

            return insights

        except Exception as e:
            system_logger.error(f"Error getting feedback insights: {e}", exc_info=True)
            return "Unable to retrieve feedback insights at this time."

    def process_customer_message(
        self, customer_id: int, message: str, session_id: Optional[str] = None
    ) -> AgentResponse:
        """Enhanced customer message processing with feedback integration"""
        if not session_id:
            session_id = f"customer_{customer_id}"

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

        # Get enhanced context with feedback data
        enhanced_context = self.get_enhanced_context(message, customer_context)

        # Get feedback insights
        feedback_insights = self.get_feedback_insights(customer_id, issue_category)

        # Analyze sentiment
        sentiment = self.analyze_sentiment_enhanced(message)

        # Check for escalation with feedback history
        should_escalate, escalation_reason = self.should_escalate_enhanced(
            message, customer_context
        )

        # Prepare comprehensive context for the LLM
        context_str = f"""
Customer: {customer_context.name} ({customer_context.email})
Tier: {customer_context.tier} | Company: {customer_context.company}
Satisfaction: {customer_context.satisfaction_score:.1f}/5.0
Open Tickets: {len(customer_context.open_tickets)} | Recent Interactions: {len(customer_context.recent_interactions)}

Recent Ticket Details:
{self._format_tickets(customer_context.open_tickets[:3])}

Recent Interaction Summary:
{self._format_interactions(customer_context.recent_interactions[:3])}
"""

        kb_str = self._format_knowledge_results(kb_results)
        escalation_str = (
            f"Escalation: {'Required' if should_escalate else 'Not Required'}"
        )
        if should_escalate:
            escalation_str += f" - Reason: {escalation_reason}"

        # Generate response using the enhanced chain with feedback insights
        try:
            response = self.main_chain.invoke(
                {
                    "customer_context": context_str,
                    "enhanced_context": enhanced_context,
                    "knowledge_base_results": kb_str,
                    "issue_category": issue_category,
                    "escalation_analysis": escalation_str,
                    "feedback_insights": feedback_insights,
                    "customer_message": message,
                },
                config={"configurable": {"session_id": session_id}},
            )

            # Calculate confidence with feedback data
            confidence = self._calculate_confidence_with_feedback(
                kb_results, sentiment, customer_context, customer_id
            )

            # Generate comprehensive suggested actions
            suggested_actions = self._generate_suggested_actions(
                should_escalate,
                escalation_reason,
                sentiment,
                issue_category,
                customer_context,
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
            system_logger.error(
                f"Error generating enhanced response: {e}", exc_info=True
            )
            return AgentResponse(
                message="I sincerely apologize for the technical difficulty. To ensure you receive the best possible service, I'm immediately connecting you with one of our specialist agents who can provide hands-on assistance.",
                confidence=0.0,
                escalate=True,
                suggested_actions=[
                    "Immediate escalation to human agent",
                    "Technical support review",
                ],
                knowledge_sources=[],
                sentiment=sentiment,
            )

    def _calculate_confidence_with_feedback(
        self,
        kb_results: List[Dict],
        sentiment: str,
        customer_context: CustomerContext,
        customer_id: int,
    ) -> float:
        """Calculate confidence score with feedback data"""
        base_confidence = 0.5

        # Knowledge base match boost
        kb_boost = min(0.3, len(kb_results) * 0.1)

        # Sentiment factor
        sentiment_factors = {
            "very_positive": 0.1,
            "positive": 0.05,
            "neutral": 0.0,
            "negative": -0.05,
            "very_negative": -0.1,
        }
        sentiment_factor = sentiment_factors.get(sentiment, 0.0)

        # Customer tier factor
        tier_factors = {"enterprise": 0.1, "business": 0.05, "free": 0.0}
        tier_factor = tier_factors.get(customer_context.tier, 0.0)

        # Feedback-based satisfaction factor
        satisfaction_trend = self.feedback_manager.get_customer_satisfaction_trend(
            customer_id
        )
        satisfaction_factor = 0.0
        if satisfaction_trend and satisfaction_trend.get("latest_rating"):
            latest_rating = satisfaction_trend["latest_rating"]
            satisfaction_factor = (latest_rating - 3.0) * 0.1

            # Boost confidence if trend is improving
            if satisfaction_trend.get("trend_direction") == "improving":
                satisfaction_factor += 0.05
            elif satisfaction_trend.get("trend_direction") == "declining":
                satisfaction_factor -= 0.05

        confidence = (
            base_confidence
            + kb_boost
            + sentiment_factor
            + tier_factor
            + satisfaction_factor
        )
        return min(0.95, max(0.1, confidence))

    def _format_tickets(self, tickets: List[Dict]) -> str:
        """Format ticket information for context"""
        if not tickets:
            return "No recent tickets"

        formatted = []
        for ticket in tickets:
            formatted.append(
                f"#{ticket['id']}: {ticket['title']} ({ticket['status']}, {ticket['priority']})"
            )
        return "\n".join(formatted)

    def _format_interactions(self, interactions: List[Dict]) -> str:
        """Format interaction information for context"""
        if not interactions:
            return "No recent interactions"

        formatted = []
        for interaction in interactions:
            formatted.append(
                f"- {interaction['type']}: {interaction['sentiment']} sentiment"
            )
        return "\n".join(formatted)

    def _format_knowledge_results(self, kb_results: List[Dict]) -> str:
        """Format knowledge base results for context"""
        if not kb_results:
            return "No specific knowledge base matches found"

        formatted = []
        for result in kb_results:
            formatted.append(f"- {result['title']}: {result['content'][:200]}...")
        return "\n".join(formatted)

    def _generate_suggested_actions(
        self,
        should_escalate: bool,
        escalation_reason: str,
        sentiment: str,
        category: str,
        customer_context: CustomerContext,
    ) -> List[str]:
        """Generate comprehensive suggested actions with feedback insights"""
        actions = []

        if should_escalate:
            actions.append(f"Escalate to supervisor: {escalation_reason}")

        # Category-specific actions
        category_actions = {
            "account_access": [
                "Verify identity",
                "Check account status",
                "Password reset assistance",
            ],
            "billing": [
                "Review billing history",
                "Process refund if applicable",
                "Update payment method",
            ],
            "technical_issues": [
                "Technical diagnostics",
                "Check system status",
                "Submit bug report",
            ],
            "api_problems": [
                "Check API status",
                "Review integration logs",
                "Rate limit analysis",
            ],
            "security": ["Security review", "Account audit", "Enable 2FA"],
            "performance": [
                "Performance analysis",
                "System optimization",
                "Load testing",
            ],
            "features": [
                "Feature request review",
                "Product roadmap check",
                "Alternative solutions",
            ],
        }

        if category in category_actions:
            actions.extend(category_actions[category][:2])

        # Feedback-based actions
        satisfaction_trend = self.feedback_manager.get_customer_satisfaction_trend(
            customer_context.customer_id
        )
        if satisfaction_trend:
            if satisfaction_trend.get("trend_direction") == "declining":
                actions.append("Priority follow-up due to satisfaction decline")
            elif satisfaction_trend.get("trend_direction") == "improving":
                actions.append("Continue positive engagement")

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

    def create_support_ticket(
        self, customer_id: int, message: str, category: str, priority: str = "medium"
    ) -> Optional[int]:
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
                created_at=datetime.now(),
            )
            session.add(ticket)
            session.commit()
            system_logger.info(
                "Support ticket created",
                {
                    "customer_id": customer_id,
                    "ticket_id": ticket.id,
                    "category": category,
                    "priority": priority,
                },
            )
            return int(ticket.id)
        except Exception as e:
            system_logger.error(f"Error creating support ticket: {e}", exc_info=True)
            session.rollback()
            return None
        finally:
            if session:
                session.close()

    def log_interaction_enhanced(
        self,
        customer_id: int,
        message: str,
        response: AgentResponse,
        category: str,
        ticket_id: Optional[int] = None,
    ):
        """Enhanced interaction logging with feedback data update"""
        session = get_session()
        try:
            # Calculate satisfaction score based on sentiment and confidence
            satisfaction_mapping = {
                "very_positive": 5.0,
                "positive": 4.0,
                "neutral": 3.0,
                "negative": 2.0,
                "very_negative": 1.0,
            }
            satisfaction = satisfaction_mapping.get(response.sentiment, 3.0)

            # Adjust based on confidence
            if response.confidence > 0.8:
                satisfaction = min(5.0, satisfaction + 0.5)
            elif response.confidence < 0.5:
                satisfaction = max(1.0, satisfaction - 0.5)

            interaction = CustomerInteraction(
                customer_id=customer_id,
                interaction_type="enhanced_chat_with_feedback",
                message=message,
                response=response.message,
                sentiment=response.sentiment,
                satisfaction_score=satisfaction,
                agent_id="customer_support_agent_ai_enhanced_v2",
                metadata={
                    "category": category,
                    "confidence": response.confidence,
                    "escalated": response.escalate,
                    "knowledge_sources": response.knowledge_sources,
                    "suggested_actions": response.suggested_actions,
                    "ticket_id": ticket_id,
                },
                created_at=datetime.now(),
            )
            session.add(interaction)
            session.commit()

            # Also add to feedback CSV for future reference
            customer_context = self.get_customer_context(customer_id)
            customer_tier = customer_context.tier if customer_context else "unknown"

            feedback_entry = {
                CUSTOMER_ID: customer_id,
                FEEDBACK_DATE: datetime.now().strftime("%Y-%m-%d"),
                "Rating": int(round(satisfaction)),
                "Category": category.title().replace("_", " "),
                "Subcategory": "General",
                FEEDBACK_TEXT: (
                    message[:200] + "..." if len(message) > 200 else message
                ),
                "Resolution Status": (
                    "Resolved" if response.confidence > 0.7 else "In Progress"
                ),
                "Agent ID": "ai_agent_v2",
                "Response Time (hours)": 0.1,  # AI response time
                "Follow Up Required": "Yes" if response.escalate else "No",
                "Sentiment": response.sentiment.title().replace("_", " "),
                "Priority": "High" if response.escalate else "Medium",
                "Product Version": "v2.1",
                "Channel": "Chat",
                "Customer Tier": customer_tier,
                "Escalated": "Yes" if response.escalate else "No",
                "Resolution Date": (
                    datetime.now().strftime("%Y-%m-%d")
                    if response.confidence > 0.7
                    else ""
                ),
                "Satisfaction Score": satisfaction,
                "Tags": f"{category},ai_response",
                "Internal Notes": f"AI confidence: {response.confidence:.2f}",
            }
            self.feedback_manager.add_feedback_entry(feedback_entry)
            system_logger.info(
                "Interaction logged",
                {
                    "customer_id": customer_id,
                    "ticket_id": ticket_id,
                    "satisfaction": satisfaction,
                },
            )
        except Exception as e:
            system_logger.error(
                f"Error logging enhanced interaction: {e}", exc_info=True
            )
            session.rollback()
        finally:
            if session:
                session.close()

    def handle_customer_request_enhanced(
        self,
        customer_id: int,
        message: str,
        create_ticket: bool = False,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Enhanced main method to handle a customer request with feedback integration"""
        # Categorize the issue first
        category = self.categorize_issue(message)

        # Create ticket if requested or if it's a complex issue
        ticket_id = None
        if create_ticket or category in ["technical_issues", "billing", "security"]:
            priority = (
                "urgent"
                if any(
                    word in message.lower()
                    for word in ["urgent", "critical", "emergency"]
                )
                else "medium"
            )
            ticket_id = self.create_support_ticket(
                customer_id, message, category, priority
            )

        # Process the message with feedback integration
        response = self.process_customer_message(customer_id, message, session_id)

        # Log the enhanced interaction with feedback update
        self.log_interaction_enhanced(
            customer_id, message, response, category, ticket_id
        )

        # Get feedback insights for response metadata
        feedback_insights = self.get_feedback_insights(customer_id, category)

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
            "feedback_insights": feedback_insights,
            "metadata": {
                "agent_version": "enhanced_v2.1_with_feedback",
                "processing_time": "< 3 seconds",
                "knowledge_base_used": len(response.knowledge_sources) > 0,
                "feedback_integration": True,
            },
        }

    def get_customer_insights(self, customer_id: int) -> Dict[str, Any]:
        """Get comprehensive customer insights with feedback data"""
        customer_context = self.get_customer_context(customer_id)
        if not customer_context:
            system_logger.error(f"Customer {customer_id} not found", exc_info=True)
            return {"error": "Customer not found"}

        session = get_session()
        try:
            # Get interaction statistics
            interactions = (
                session.query(CustomerInteraction)
                .filter(CustomerInteraction.customer_id == customer_id)
                .all()
            )

            sentiment_distribution = {}
            category_distribution = {}
            satisfaction_scores = []

            for interaction in interactions:
                # Sentiment distribution
                sentiment = interaction.sentiment
                sentiment_distribution[sentiment] = (
                    sentiment_distribution.get(sentiment, 0) + 1
                )

                # Category distribution (from metadata)
                if interaction.metadata and "category" in interaction.metadata:
                    category = interaction.metadata["category"]
                    category_distribution[category] = (
                        category_distribution.get(category, 0) + 1
                    )

                # Satisfaction scores
                if interaction.satisfaction_score:
                    satisfaction_scores.append(interaction.satisfaction_score)

            # Get feedback data insights
            feedback_history = self.feedback_manager.get_customer_feedback_history(
                customer_id
            )
            satisfaction_trend = self.feedback_manager.get_customer_satisfaction_trend(
                customer_id
            )

            # Calculate metrics
            avg_satisfaction = (
                sum(satisfaction_scores) / len(satisfaction_scores)
                if satisfaction_scores
                else 0.0
            )

            # Get recent tickets
            recent_tickets = (
                session.query(SupportTicket)
                .filter(SupportTicket.customer_id == customer_id)
                .order_by(SupportTicket.created_at.desc())
                .limit(10)
                .all()
            )

            risk_indicators = self._assess_customer_risk(
                customer_context, sentiment_distribution, satisfaction_trend
            )

            # FIX: Use proper attribute access for CustomerContext object
            return {
                "customer_context": {
                    "customer_id": customer_context.customer_id,  # Fix: Use attribute access
                    "name": customer_context.name,
                    "email": customer_context.email,
                    "tier": customer_context.tier,
                    "company": customer_context.company,
                    "satisfaction_score": customer_context.satisfaction_score,
                },
                "interaction_stats": {
                    "total_interactions": len(interactions),
                    "sentiment_distribution": sentiment_distribution,
                    "category_distribution": category_distribution,
                    "average_satisfaction": avg_satisfaction,
                    "satisfaction_scores": satisfaction_scores[-10:],  # Last 10 scores
                    "interaction_stats": {  # Add nested structure for API compatibility
                        "recent_interactions": [
                            {
                                "type": interaction.get("type", "chat"),
                                "message": interaction.get("message", "No message"),
                                "sentiment": interaction.get("sentiment", "neutral"),
                                "satisfaction": interaction.get("satisfaction", 0),
                                "date": interaction.get(
                                    "date", datetime.now().isoformat()
                                ),
                            }
                            for interaction in customer_context.recent_interactions
                        ][:5]
                    },
                },
                "feedback_insights": {
                    "feedback_count": len(feedback_history),
                    "satisfaction_trend": satisfaction_trend,
                    "recent_feedback": feedback_history[:5],  # Last 5 feedback entries
                },
                "support_tickets": {
                    "total_tickets": len(recent_tickets),
                    "open_tickets": len(customer_context.open_tickets),
                    "recent_tickets": [
                        {
                            "id": t.id,
                            "title": t.title,
                            "category": t.category,
                            "status": t.status,
                            "priority": t.priority,
                            "created_at": (
                                t.created_at.isoformat() if t.created_at else None
                            ),
                        }
                        for t in recent_tickets
                    ],
                },
                "recommendations": self._generate_customer_recommendations(
                    customer_context, sentiment_distribution, satisfaction_trend
                ),
                "risk_indicators": risk_indicators,
            }

        except Exception as e:
            system_logger.error(f"Error getting customer insights: {e}", exc_info=True)
            return {"error": f"Failed to retrieve customer insights: {str(e)}"}
        finally:
            if session:
                session.close()

    def _generate_customer_recommendations(
        self,
        customer_context: CustomerContext,
        sentiment_dist: Dict,
        satisfaction_trend: Dict,
    ) -> List[str]:
        """Generate recommendations for customer management"""
        recommendations = []
        recommendations.extend(self._satisfaction_recommendations(satisfaction_trend))
        recommendations.extend(self._sentiment_recommendations(sentiment_dist))
        recommendations.extend(self._tier_recommendations(customer_context))
        recommendations.extend(self._ticket_recommendations(customer_context))
        return recommendations[:5]  # Limit to top 5 recommendations

    def _satisfaction_recommendations(self, satisfaction_trend: Dict) -> List[str]:
        recs = []
        if satisfaction_trend:
            trend_direction = satisfaction_trend.get("trend_direction", "stable")
            latest_rating = satisfaction_trend.get("latest_rating", 3.0)
            if trend_direction == "declining":
                recs.append("Priority attention needed - satisfaction declining")
                recs.append("Schedule proactive check-in call")
            elif trend_direction == "improving":
                recs.append("Positive momentum - continue current approach")
            if latest_rating and latest_rating < 3.0:
                recs.append("Low satisfaction score - immediate intervention required")
        return recs

    def _sentiment_recommendations(self, sentiment_dist: Dict) -> List[str]:
        recs = []
        total_interactions = sum(sentiment_dist.values()) if sentiment_dist else 1
        negative_sentiment = sentiment_dist.get("negative", 0) + sentiment_dist.get(
            "very_negative", 0
        )
        if total_interactions > 0 and (negative_sentiment / total_interactions) > 0.3:
            recs.append("High negative sentiment ratio - review service approach")
        return recs

    def _tier_recommendations(self, customer_context: CustomerContext) -> List[str]:
        recs = []
        if customer_context.tier == "enterprise":
            recs.append("Enterprise customer - ensure dedicated support")
            if len(customer_context.open_tickets) > 2:
                recs.append("Multiple open tickets - escalate to account manager")
        elif customer_context.tier == "free":
            if len(customer_context.open_tickets) > 0:
                recs.append("Consider upgrade conversation for better support")
        return recs

    def _ticket_recommendations(self, customer_context: CustomerContext) -> List[str]:
        recs = []
        if len(customer_context.open_tickets) > 3:
            recs.append("Multiple open issues - consolidate and prioritize")
        return recs

    def _assess_customer_risk(
        self,
        customer_context: CustomerContext,
        sentiment_dist: Dict,
        satisfaction_trend: Dict,
    ) -> Dict[str, Any]:
        """Assess customer churn/escalation risk"""
        risk_score = 0.0
        risk_factors = []

        # Satisfaction trend risk
        score, factors = self._satisfaction_trend_risk(satisfaction_trend)
        risk_score += score
        risk_factors.extend(factors)

        # Sentiment risk
        score, factors = self._sentiment_risk(sentiment_dist)
        risk_score += score
        risk_factors.extend(factors)

        # Ticket volume risk
        score, factors = self._ticket_volume_risk(customer_context)
        risk_score += score
        risk_factors.extend(factors)

        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "risk_score": min(1.0, risk_score),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommended_actions": self._get_risk_mitigation_actions(
                risk_level, risk_factors
            ),
        }

    def _satisfaction_trend_risk(self, satisfaction_trend: Dict) -> tuple[float, list]:
        score = 0.0
        factors = []
        if satisfaction_trend:
            trend_direction = satisfaction_trend.get("trend_direction", "stable")
            latest_rating = satisfaction_trend.get("latest_rating", 3.0)
            if trend_direction == "declining":
                score += 0.3
                factors.append("Declining satisfaction trend")
            if latest_rating and latest_rating < 2.5:
                score += 0.4
                factors.append("Very low satisfaction rating")
        return score, factors

    def _sentiment_risk(self, sentiment_dist: Dict) -> tuple[float, list]:
        score = 0.0
        factors = []
        if sentiment_dist:
            total_interactions = sum(sentiment_dist.values())
            if total_interactions > 0:
                negative_ratio = (
                    sentiment_dist.get("negative", 0)
                    + sentiment_dist.get("very_negative", 0)
                ) / total_interactions
                if negative_ratio > 0.4:
                    score += 0.2
                    factors.append("High negative sentiment ratio")
        return score, factors

    def _ticket_volume_risk(
        self, customer_context: CustomerContext
    ) -> tuple[float, list]:
        score = 0.0
        factors = []
        if len(customer_context.open_tickets) > 2:
            score += 0.1
            factors.append("Multiple open tickets")
        return score, factors

    def _get_risk_mitigation_actions(
        self, risk_level: str, risk_factors: List[str]
    ) -> List[str]:
        """Get recommended actions based on risk assessment"""
        actions = []

        if risk_level == "high":
            actions.extend(
                [
                    "Immediate escalation to account manager",
                    "Schedule urgent customer call",
                    "Review all open tickets and prioritize resolution",
                ]
            )
        elif risk_level == "medium":
            actions.extend(
                [
                    "Proactive outreach within 24 hours",
                    "Review recent interactions for improvement opportunities",
                    "Consider offering additional support resources",
                ]
            )
        else:
            actions.extend(
                [
                    "Continue standard support approach",
                    "Monitor for any changes in satisfaction",
                ]
            )

        # Factor-specific actions
        if "Declining satisfaction trend" in risk_factors:
            actions.append("Conduct satisfaction survey to identify specific issues")

        if "Multiple open tickets" in risk_factors:
            actions.append("Consolidate tickets and assign dedicated agent")

        if "High negative sentiment ratio" in risk_factors:
            actions.append("Review communication approach and agent training")

        return actions[:5]  # Limit to top 5 actions

    def get_agent_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        session = get_session()
        try:
            feedback_trends = self.feedback_manager.get_feedback_trends(days)
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_interactions = (
                session.query(CustomerInteraction)
                .filter(CustomerInteraction.created_at >= cutoff_date)
                .filter(CustomerInteraction.agent_id.like("%customer_support_agent%"))
                .all()
            )

            if not recent_interactions:
                return {"message": "No recent interactions found"}

            metrics = self._calculate_agent_metrics(recent_interactions)
            avg_satisfaction = metrics["avg_satisfaction"]
            escalation_rate = metrics["escalation_rate"]
            sentiment_distribution = metrics["sentiment_distribution"]
            satisfaction_scores = metrics["satisfaction_scores"]
            metrics["escalation_count"]
            total_interactions = metrics["total_interactions"]

            response_quality = self._get_response_quality(avg_satisfaction)
            escalation_management = (
                "good" if escalation_rate < 10 else "needs_improvement"
            )

            return {
                "period_days": days,
                "total_interactions": total_interactions,
                "average_satisfaction": round(avg_satisfaction, 2),
                "satisfaction_distribution": self._get_satisfaction_distribution(
                    satisfaction_scores
                ),
                "sentiment_distribution": sentiment_distribution,
                "escalation_rate": round(escalation_rate, 2),
                "feedback_integration_metrics": feedback_trends,
                "performance_indicators": {
                    "customer_satisfaction_trend": (
                        "improving" if avg_satisfaction > 3.5 else "needs_attention"
                    ),
                    "response_quality": response_quality,
                    "escalation_management": escalation_management,
                },
                "recommendations": self._get_agent_improvement_recommendations(
                    avg_satisfaction, escalation_rate, sentiment_distribution
                ),
            }
        except Exception as e:
            system_logger.error(
                f"Error getting agent performance metrics: {e}", exc_info=True
            )
            return {"error": f"Failed to retrieve performance metrics: {str(e)}"}
        finally:
            if session:
                session.close()

    def _calculate_agent_metrics(self, interactions: List[Any]) -> Dict[str, Any]:
        satisfaction_scores = [
            i.satisfaction_score
            for i in interactions
            if i.satisfaction_score is not None
        ]
        sentiment_distribution = {}
        escalation_count = 0

        for interaction in interactions:
            sentiment = interaction.sentiment or "neutral"
            sentiment_distribution[sentiment] = (
                sentiment_distribution.get(sentiment, 0) + 1
            )
            if interaction.metadata and interaction.metadata.get("escalated"):
                escalation_count += 1

        total_interactions = len(interactions)
        avg_satisfaction = (
            sum(satisfaction_scores) / len(satisfaction_scores)
            if satisfaction_scores
            else 0.0
        )
        escalation_rate = (
            (escalation_count / total_interactions) * 100
            if total_interactions > 0
            else 0.0
        )

        return {
            "satisfaction_scores": satisfaction_scores,
            "sentiment_distribution": sentiment_distribution,
            "escalation_count": escalation_count,
            "total_interactions": total_interactions,
            "avg_satisfaction": avg_satisfaction,
            "escalation_rate": escalation_rate,
        }

    def _get_satisfaction_distribution(self, scores: List[float]) -> Dict[str, int]:
        return {
            "excellent": len([s for s in scores if s >= 4.5]),
            "good": len([s for s in scores if 3.5 <= s < 4.5]),
            "average": len([s for s in scores if 2.5 <= s < 3.5]),
            "poor": len([s for s in scores if s < 2.5]),
        }

    def _get_response_quality(self, avg_satisfaction: float) -> str:
        try:
            avg_satisfaction_float = float(avg_satisfaction)
        except Exception:
            avg_satisfaction_float = 0.0

        if avg_satisfaction_float > 4.0:
            return "high"
        elif avg_satisfaction_float > 3.0:
            return "medium"
        else:
            return "low"

    def _get_agent_improvement_recommendations(
        self, avg_satisfaction: float, escalation_rate: float, sentiment_dist: Dict
    ) -> List[str]:
        """Get recommendations for improving agent performance"""
        recommendations = []

        if avg_satisfaction < 3.5:
            recommendations.append("Focus on improving response quality and accuracy")
            recommendations.append("Review knowledge base coverage for common issues")

        if escalation_rate > 15:
            recommendations.append(
                "Analyze escalation patterns to improve first-contact resolution"
            )
            recommendations.append("Enhance escalation criteria and thresholds")

        total_sentiment = sum(sentiment_dist.values()) if sentiment_dist else 1
        negative_sentiment_ratio = (
            (sentiment_dist.get("negative", 0) + sentiment_dist.get("very_negative", 0))
            / total_sentiment
            if total_sentiment > 0
            else 0
        )

        if negative_sentiment_ratio > 0.3:
            recommendations.append(
                "Improve empathy and emotional intelligence in responses"
            )
            recommendations.append("Review communication tone and approach")

        if avg_satisfaction > 4.0:
            recommendations.append("Excellent performance - maintain current approach")
            recommendations.append(
                "Consider expanding successful strategies to other agents"
            )

        return recommendations

    def export_customer_insights_report(
        self, customer_id: int, file_path: Optional[str] = None
    ) -> str:
        """Export comprehensive customer insights to a file"""
        insights = self.get_customer_insights(customer_id)

        if "error" in insights:
            system_logger.error(
                f"Error generating report for customer {customer_id}: {insights['error']}",
                exc_info=True,
            )
            return f"Error generating report: {insights['error']}"

        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"reports/customer_{customer_id}_insights_{timestamp}.txt"

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w") as f:
                f.write("CUSTOMER INSIGHTS REPORT\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")

                # Customer Information
                f.write("CUSTOMER INFORMATION\n")
                f.write("-" * 20 + "\n")
                customer_info = insights["customer_context"]
                f.write(f"Name: {customer_info['name']}\n")
                f.write(f"Email: {customer_info['email']}\n")
                f.write(f"Tier: {customer_info['tier']}\n")
                f.write(f"Company: {customer_info['company']}\n")
                f.write(
                    f"Current Satisfaction: {customer_info['satisfaction_score']:.1f}/5.0\n\n"
                )

                # Interaction Statistics
                f.write("INTERACTION STATISTICS\n")
                f.write("-" * 20 + "\n")
                stats = insights["interaction_stats"]
                f.write(f"Total Interactions: {stats['total_interactions']}\n")
                f.write(
                    f"Average Satisfaction: {stats['average_satisfaction']:.2f}/5.0\n"
                )
                f.write(f"Sentiment Distribution: {stats['sentiment_distribution']}\n")
                f.write(f"Category Distribution: {stats['category_distribution']}\n\n")

                # Feedback Insights
                f.write("FEEDBACK INSIGHTS\n")
                f.write("-" * 15 + "\n")
                feedback = insights["feedback_insights"]
                f.write(f"Total Feedback Entries: {feedback['feedback_count']}\n")
                if feedback["satisfaction_trend"]:
                    trend = feedback["satisfaction_trend"]
                    f.write(
                        f"Satisfaction Trend: {trend.get('trend_direction', 'N/A')}\n"
                    )
                    f.write(f"Latest Rating: {trend.get('latest_rating', 'N/A')}/5\n")
                    if trend.get("average_rating") is not None:
                        f.write(f"Average Rating: {trend['average_rating']:.1f}/5\n")
                f.write("\n")

                # Support Tickets
                f.write("SUPPORT TICKETS\n")
                f.write("-" * 15 + "\n")
                tickets = insights["support_tickets"]
                f.write(f"Total Tickets: {tickets['total_tickets']}\n")
                f.write(f"Open Tickets: {tickets['open_tickets']}\n\n")

                # Risk Assessment
                f.write("RISK ASSESSMENT\n")
                f.write("-" * 15 + "\n")
                risk = insights["risk_indicators"]
                f.write(f"Risk Level: {risk['risk_level'].upper()}\n")
                f.write(f"Risk Score: {risk['risk_score']:.2f}/1.0\n")
                f.write(f"Risk Factors: {', '.join(risk['risk_factors'])}\n\n")

                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 15 + "\n")
                for i, rec in enumerate(insights["recommendations"], 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")

                # Risk Mitigation Actions
                f.write("RISK MITIGATION ACTIONS\n")
                f.write("-" * 25 + "\n")
                for i, action in enumerate(risk["recommended_actions"], 1):
                    f.write(f"{i}. {action}\n")

            return f"Customer insights report exported to: {file_path}"

        except Exception as e:
            system_logger.error(f"Error exporting report: {e}", exc_info=True)
            return f"Error exporting report: {str(e)}"

    def clear_session_history(self, session_id: str = None):
        """Clear conversation history for a session or all sessions"""
        if session_id:
            if session_id in self.store:
                del self.store[session_id]
                return f"Cleared history for session: {session_id}"
            else:
                return f"No history found for session: {session_id}"
        else:
            self.store.clear()
            return "Cleared all session histories"

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a conversation session"""
        if session_id not in self.store:
            system_logger.error(f"Session {session_id} not found", exc_info=True)
            return {"error": "Session not found"}

        history = self.store[session_id]
        messages = history.messages

        if not messages:
            return {"message": "No messages in session"}

        # Analyze the conversation
        human_messages = [msg.content for msg in messages if msg.type == "human"]
        ai_messages = [msg.content for msg in messages if msg.type == "ai"]

        conversation_start_time = messages[0].additional_kwargs.get(
            "timestamp"
        ) or messages[0].additional_kwargs.get("created_at")
        last_interaction_time = messages[-1].additional_kwargs.get(
            "timestamp"
        ) or messages[-1].additional_kwargs.get("created_at")

        return {
            "session_id": session_id,
            "message_count": len(messages),
            "human_messages": len(human_messages),
            "ai_responses": len(ai_messages),
            "conversation_start": conversation_start_time,
            "last_interaction": last_interaction_time,
            "topics_discussed": self._extract_topics_from_messages(human_messages),
            "session_active": True,
        }

    def _extract_topics_from_messages(self, messages: List[str]) -> List[str]:
        """Extract main topics from conversation messages"""
        topics = set()
        for message in messages:
            category = self.categorize_issue(message)
            topics.add(category.replace("_", " ").title())
        return list(topics)

    def get_feedback_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive feedback summary for management reporting"""
        try:
            feedback_trends = self.feedback_manager.get_feedback_trends(days)

            if not feedback_trends:
                return {
                    "message": "No feedback data available for the specified period"
                }

            # Get top categories and their resolution rates
            category_dist = feedback_trends.get("category_distribution", {})
            top_categories = sorted(
                category_dist.items(), key=lambda x: x[1], reverse=True
            )[:5]

            # Get sentiment analysis
            sentiment_dist = feedback_trends.get("sentiment_distribution", {})

            # Calculate satisfaction metrics
            avg_rating = feedback_trends.get("average_rating", 0)
            resolution_rate = feedback_trends.get("resolution_rate", 0)
            escalation_rate = feedback_trends.get("escalation_rate", 0)

            # Calculate escalation_control, satisfaction_level, and resolution_efficiency
            if escalation_rate <= 5:
                escalation_control = "excellent"
            elif escalation_rate <= 15:
                escalation_control = "good"
            else:
                escalation_control = "needs_improvement"
            if avg_rating >= 4.5:
                satisfaction_level = "excellent"
            elif avg_rating >= 4.0:
                satisfaction_level = "good"
            else:
                satisfaction_level = "needs_improvement"

            if resolution_rate >= 90:
                resolution_efficiency = "excellent"
            elif resolution_rate >= 75:
                resolution_efficiency = "good"
            else:
                resolution_efficiency = "needs_improvement"

            return {
                "period_days": days,
                "summary_metrics": {
                    "total_feedback": feedback_trends.get("total_feedback", 0),
                    "average_rating": round(avg_rating, 2),
                    "resolution_rate": round(resolution_rate, 1),
                    "escalation_rate": round(escalation_rate, 1),
                    "average_response_time": round(
                        feedback_trends.get("average_response_time", 0), 1
                    ),
                },
                "top_categories": [
                    {"category": cat, "count": count} for cat, count in top_categories
                ],
                "sentiment_breakdown": sentiment_dist,
                "performance_indicators": {
                    "satisfaction_level": satisfaction_level,
                    "resolution_efficiency": resolution_efficiency,
                    "escalation_control": escalation_control,
                },
                "recommendations": self._generate_feedback_recommendations(
                    avg_rating, resolution_rate, escalation_rate, sentiment_dist
                ),
            }

        except Exception as e:
            system_logger.error(f"Error getting feedback summary: {e}", exc_info=True)
            return {"error": f"Failed to retrieve feedback summary: {str(e)}"}

    def _generate_feedback_recommendations(
        self,
        avg_rating: float,
        resolution_rate: float,
        escalation_rate: float,
        sentiment_dist: Dict,
    ) -> List[str]:
        """Generate recommendations based on feedback analysis"""
        recommendations = []

        if avg_rating < 3.5:
            recommendations.append("Focus on improving overall customer satisfaction")
            recommendations.append(
                "Conduct detailed analysis of low-rated interactions"
            )

        if resolution_rate < 75:
            recommendations.append("Improve first-contact resolution processes")
            recommendations.append("Enhance agent training and knowledge base")

        if escalation_rate > 15:
            recommendations.append("Review escalation triggers and thresholds")
            recommendations.append(
                "Implement proactive customer outreach for at-risk accounts"
            )

        total_sentiment = sum(sentiment_dist.values()) if sentiment_dist else 1
        negative_sentiment = sentiment_dist.get("Negative", 0) + sentiment_dist.get(
            "Very Negative", 0
        )
        if total_sentiment > 0 and (negative_sentiment / total_sentiment) > 0.3:
            recommendations.append(
                "Address communication tone and empathy in responses"
            )
            recommendations.append("Implement sentiment monitoring and alerts")

        if avg_rating > 4.0 and resolution_rate > 85:
            recommendations.append("Excellent performance - document best practices")
            recommendations.append("Consider expanding successful strategies")

        return recommendations[:5]

    def bulk_process_feedback(self, feedback_file_path: str) -> Dict[str, Any]:
        """Process feedback data in bulk from a CSV file"""
        try:
            if not os.path.exists(feedback_file_path):
                system_logger.error(
                    f"Feedback file not found: {feedback_file_path}", exc_info=True
                )
                return {"error": f"File not found: {feedback_file_path}"}

            feedback_df = pd.read_csv(feedback_file_path)
            processed_count = 0
            errors = []

            for index, row in feedback_df.iterrows():
                try:
                    # Validate required fields
                    required_fields = [
                        CUSTOMER_ID,
                        FEEDBACK_DATE,
                        "Rating",
                        "Category",
                        FEEDBACK_TEXT,
                    ]
                    if not all(
                        field in row and pd.notna(row[field])
                        for field in required_fields
                    ):
                        errors.append(f"Row {index + 1}: Missing required fields")
                        continue

                    # Add to feedback manager
                    feedback_entry = row.to_dict()
                    self.feedback_manager.add_feedback_entry(feedback_entry)
                    processed_count += 1

                except Exception as e:
                    system_logger.error(
                        f"Error processing row {index + 1}: {e}", exc_info=True
                    )
                    errors.append(f"Row {index + 1}: {str(e)}")
            system_logger.info(
                "Bulk feedback processing complete",
                {"file": feedback_file_path, "processed": processed_count},
            )
            return {
                "processed_count": processed_count,
                "total_rows": len(feedback_df),
                "success_rate": (
                    (processed_count / len(feedback_df)) * 100
                    if len(feedback_df) > 0
                    else 0
                ),
                "errors": errors[:10],  # Limit to first 10 errors
                "error_count": len(errors),
            }

        except Exception as e:
            system_logger.error(f"Error processing bulk feedback: {e}", exc_info=True)
            return {"error": f"Failed to process bulk feedback: {str(e)}"}

    def _get_satisfaction_level(self, avg_satisfaction):
        """Helper to determine satisfaction level from average satisfaction score."""
        try:
            avg = float(avg_satisfaction)
        except Exception:
            avg = 0.0
        if avg >= 4.5:
            return "excellent"
        elif avg >= 3.5:
            return "good"
        else:
            return "needs_attention"

    def generate_daily_report(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Generate daily performance report"""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        session = get_session()
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
            next_date = target_date + timedelta(days=1)

            # Get interactions for the day
            daily_interactions = (
                session.query(CustomerInteraction)
                .filter(CustomerInteraction.created_at >= target_date)
                .filter(CustomerInteraction.created_at < next_date)
                .filter(CustomerInteraction.agent_id.like("%customer_support_agent%"))
                .all()
            )

            # Get tickets created on the day
            daily_tickets = (
                session.query(SupportTicket)
                .filter(SupportTicket.created_at >= target_date)
                .filter(SupportTicket.created_at < next_date)
                .all()
            )

            interaction_metrics, escalation_count, avg_satisfaction = (
                self._calculate_daily_interaction_metrics(daily_interactions)
            )
            ticket_metrics = self._calculate_daily_ticket_metrics(daily_tickets)

            total_interactions = interaction_metrics["total_interactions"]

            if avg_satisfaction >= 4.5:
                satisfaction_level = "excellent"
            elif avg_satisfaction >= 3.5:
                satisfaction_level = "good"
            else:
                satisfaction_level = "needs_attention"

            return {
                "date": date,
                "interaction_metrics": interaction_metrics,
                "ticket_metrics": ticket_metrics,
                "performance_summary": {
                    "satisfaction_level": satisfaction_level,
                    "volume_status": (
                        "high" if total_interactions > 100 else "normal"
                    ),  # Example logic
                    "escalation_status": (
                        "concerning" if escalation_count > 5 else "normal"
                    ),
                },
            }

        except Exception as e:
            system_logger.error(
                f"Error generating daily report for {date}: {e}", exc_info=True
            )
            return {"error": f"Failed to generate daily report: {str(e)}"}
        finally:
            if session:
                session.close()

    def _calculate_daily_interaction_metrics(
        self, daily_interactions: List[Any]
    ) -> tuple:
        """Helper to calculate daily interaction metrics and reduce complexity."""
        total_interactions = len(daily_interactions)
        satisfaction_scores = [
            i.satisfaction_score for i in daily_interactions if i.satisfaction_score
        ]
        avg_satisfaction = (
            sum(satisfaction_scores) / len(satisfaction_scores)
            if satisfaction_scores
            else 0
        )

        sentiment_dist = {}
        category_dist = {}
        escalation_count = 0

        for interaction in daily_interactions:
            # Sentiment distribution
            sentiment = interaction.sentiment or "neutral"
            sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1

            # Category distribution
            if interaction.metadata and "category" in interaction.metadata:
                category = interaction.metadata["category"]
                category_dist[category] = category_dist.get(category, 0) + 1

            # Escalation count
            if interaction.metadata and interaction.metadata.get("escalated"):
                escalation_count += 1

        return (
            {
                "total_interactions": total_interactions,
                "average_satisfaction": round(avg_satisfaction, 2),
                "sentiment_distribution": sentiment_dist,
                "category_distribution": category_dist,
                "satisfaction_level": self._get_satisfaction_level(avg_satisfaction),
            },
            escalation_count,
            avg_satisfaction,
        )

    def _calculate_daily_ticket_metrics(
        self, daily_tickets: List[Any]
    ) -> Dict[str, Any]:
        """Helper to calculate daily ticket metrics and reduce complexity."""
        ticket_priorities = {}
        ticket_categories = {}

        for ticket in daily_tickets:
            ticket_priorities[ticket.priority] = (
                ticket_priorities.get(ticket.priority, 0) + 1
            )
            ticket_categories[ticket.category] = (
                ticket_categories.get(ticket.category, 0) + 1
            )

        return {
            "total_tickets": len(daily_tickets),
            "priority_distribution": ticket_priorities,
            "category_distribution": ticket_categories,
        }


# Example usage and testing functions
def test_enhanced_agent():
    """Test the enhanced customer support agent"""
    agent = CustomerSupportAgent()

    # Test customer requests
    test_cases = [
        {
            "customer_id": 1,
            "message": "I'm having trouble logging into my account. The password reset isn't working.",
            "expected_category": "account_access",
        },
        {
            "customer_id": 2,
            "message": "Your API is timing out constantly. This is affecting our production system!",
            "expected_category": "api_problems",
        },
        {
            "customer_id": 3,
            "message": "I was charged twice this month. I need a refund immediately.",
            "expected_category": "billing",
        },
    ]

    print("Testing Enhanced Customer Support Agent with Feedback Integration")
    print("=" * 70)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Customer ID: {test_case['customer_id']}")
        print(f"Message: {test_case['message']}")

        # Process the request
        result = agent.handle_customer_request_enhanced(
            customer_id=test_case["customer_id"],
            message=test_case["message"],
            create_ticket=True,
        )

        print(f"Category: {result['category']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Escalate: {result['escalate']}")
        print(f"Response: {result['response'][:200]}...")
        print(f"Ticket ID: {result['ticket_id']}")
        print("-" * 50)

    # Test performance metrics
    print("\nAgent Performance Metrics:")
    metrics = agent.get_agent_performance_metrics(days=30)
    print(f"Total Interactions: {metrics.get('total_interactions', 0)}")
    print(f"Average Satisfaction: {metrics.get('average_satisfaction', 0):.2f}")
    print(f"Escalation Rate: {metrics.get('escalation_rate', 0):.2f}%")

    # Test feedback summary
    print("\nFeedback Summary:")
    feedback_summary = agent.get_feedback_summary(days=30)
    if "summary_metrics" in feedback_summary:
        summary = feedback_summary["summary_metrics"]
        print(f"Total Feedback: {summary.get('total_feedback', 0)}")
        print(f"Average Rating: {summary.get('average_rating', 0):.2f}/5")
        print(f"Resolution Rate: {summary.get('resolution_rate', 0):.1f}%")


if __name__ == "__main__":
    test_enhanced_agent()
