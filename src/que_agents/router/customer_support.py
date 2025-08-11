# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-10
# @Description: Customer Support API routes and handlers

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from src.que_agents.core.schemas import CustomerSupportRequest, CustomerSupportResponse
from src.que_agents.error_trace.errorlogger import system_logger
from src.que_agents.utils.agent_manager import AgentManager
from src.que_agents.utils.auth import get_verified_token


class CustomerSupportService:
    """Service class for customer support operations"""

    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        self.CUSTOMER_SUPPORT_UNAVAILABLE = "Customer support agent not available"

    def get_agent(self):
        """Get customer support agent"""
        agent = self.agent_manager.get_agent("customer_support")
        if not agent:
            system_logger.error(
                "Customer support agent is not available in AgentManager.",
                additional_info={"context": "Customer Support Chat"},
            )
        return agent

    def handle_chat_request(
        self, request: CustomerSupportRequest
    ) -> CustomerSupportResponse:
        """Handle customer support chat request with improved error handling"""
        try:
            agent = self.get_agent()
            if not agent:
                raise HTTPException(
                    status_code=503, detail=self.CUSTOMER_SUPPORT_UNAVAILABLE
                )

            # Ensure customer_id is an integer
            customer_id = (
                int(request.customer_id)
                if isinstance(request.customer_id, str)
                else request.customer_id
            )

            result = agent.handle_customer_request_enhanced(
                customer_id=customer_id, message=request.message
            )

            return CustomerSupportResponse(
                response=result["response"],
                confidence=result["confidence"],
                escalate=result["escalate"],
                suggested_actions=result["suggested_actions"],
                knowledge_sources=result["knowledge_sources"],
                sentiment=result["sentiment"],
                timestamp=datetime.now().isoformat(),
            )
        except ValueError as ve:
            system_logger.error(
                f"Invalid customer ID format: {ve}",
                additional_info={
                    "context": "Customer Support Chat",
                    "customer_id": request.customer_id,
                },
                exc_info=True,
            )
            return CustomerSupportResponse(
                response="I apologize, but there seems to be an issue with your customer information. Please contact our support team directly.",
                confidence=0.0,
                escalate=True,
                suggested_actions=["contact_human_support"],
                knowledge_sources=["error_handling"],
                sentiment="neutral",
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            system_logger.error(
                f"Error handling customer support chat: {e}",
                additional_info={
                    "context": "Customer Support Chat",
                    "customer_id": getattr(request, "customer_id", "unknown"),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            # Return fallback response
            return CustomerSupportResponse(
                response="I apologize, but I'm experiencing technical difficulties. Please contact our support team directly for immediate assistance.",
                confidence=0.5,
                escalate=True,
                suggested_actions=["contact_human_support"],
                knowledge_sources=["error_handling"],
                sentiment="neutral",
                timestamp=datetime.now().isoformat(),
            )

    def get_customer_context_data(self, customer_id: int) -> Dict[str, Any]:
        """Get enhanced customer context and information"""
        try:
            agent = self.get_agent()
            if not agent:
                raise HTTPException(
                    status_code=503, detail=self.CUSTOMER_SUPPORT_UNAVAILABLE
                )

            # Get comprehensive customer insights instead of basic context
            customer_insights = agent.get_customer_insights(customer_id)

            if "error" in customer_insights:
                raise HTTPException(status_code=404, detail=customer_insights["error"])

            # Safely extract data with fallbacks
            customer_context = customer_insights.get("customer_context", {})
            interaction_stats = customer_insights.get("interaction_stats", {})
            support_tickets = customer_insights.get("support_tickets", {})
            feedback_insights = customer_insights.get("feedback_insights", {})
            risk_indicators = customer_insights.get("risk_indicators", {})

            # Safe extraction with defaults
            recent_interactions_data = interaction_stats.get(
                "interaction_stats", {}
            ).get("recent_interactions", [])
            if (
                not recent_interactions_data
                and "recent_interactions" in customer_context
            ):
                # Fallback to customer_context if available
                recent_interactions_data = customer_context.get(
                    "recent_interactions", []
                )

            return {
                "customer_id": customer_context.get("customer_id", customer_id),
                "customer_name": customer_context.get(
                    "name", f"Customer {customer_id}"
                ),
                "email": customer_context.get(
                    "email", f"customer{customer_id}@example.com"
                ),
                "support_tier": customer_context.get("tier", "standard").title(),
                "company": customer_context.get("company", "Unknown Company"),
                "satisfaction_score": customer_context.get("satisfaction_score", 3.5),
                # Enhanced interaction data with fallbacks
                "recent_interactions": [
                    {
                        "timestamp": interaction.get(
                            "date", datetime.now().isoformat()
                        ),
                        "message": interaction.get("message", "No message available")[
                            :100
                        ]
                        + ("..." if len(interaction.get("message", "")) > 100 else ""),
                        "sentiment": interaction.get("sentiment", "neutral").title(),
                        "satisfaction": interaction.get("satisfaction", 0),
                        "type": interaction.get("type", "chat"),
                    }
                    for interaction in recent_interactions_data
                    if interaction.get("message")
                    and interaction.get("message") != "No message"
                ][
                    :5
                ],  # Limit to 5 most recent
                # Support metrics with safe defaults
                "support_metrics": {
                    "total_interactions": interaction_stats.get(
                        "total_interactions", 0
                    ),
                    "average_satisfaction": round(
                        interaction_stats.get("average_satisfaction", 3.5), 2
                    ),
                    "open_tickets": support_tickets.get("open_tickets", 0),
                    "total_tickets": support_tickets.get("total_tickets", 0),
                },
                # Recent tickets with safe extraction
                "open_tickets": [
                    {
                        "ticket_id": ticket.get("id", "N/A"),
                        "title": ticket.get("title", "No title"),
                        "category": ticket.get("category", "general")
                        .replace("_", " ")
                        .title(),
                        "priority": ticket.get("priority", "medium").title(),
                        "status": ticket.get("status", "open").title(),
                        "created_at": ticket.get(
                            "created_at", datetime.now().isoformat()
                        ),
                    }
                    for ticket in support_tickets.get("recent_tickets", [])
                ][:3],
                # Risk assessment with safe defaults
                "risk_assessment": {
                    "risk_level": risk_indicators.get("risk_level", "low").title(),
                    "risk_score": round(risk_indicators.get("risk_score", 0.1), 2),
                    "risk_factors": risk_indicators.get("risk_factors", []),
                },
                # Feedback insights with safe defaults
                "feedback_summary": {
                    "feedback_count": feedback_insights.get("feedback_count", 0),
                    "satisfaction_trend": feedback_insights.get(
                        "satisfaction_trend", {}
                    ).get("trend_direction", "stable"),
                    "latest_rating": feedback_insights.get(
                        "satisfaction_trend", {}
                    ).get("latest_rating"),
                },
                # Recommendations with safe defaults
                "recommendations": customer_insights.get(
                    "recommendations", ["Continue providing excellent service"]
                )[:3],
                # Metadata
                "last_updated": datetime.now().isoformat(),
                "data_sources": ["database", "feedback_csv", "interaction_history"],
            }

        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except KeyError as ke:
            system_logger.error(
                f"KeyError in get_customer_context: {str(ke)}",
                additional_info={
                    "context": "Get Customer Context",
                    "customer_id": customer_id,
                    "error_type": "KeyError",
                },
                exc_info=True,
            )
            # Return a fallback response with basic customer information
            return self._get_fallback_customer_data(customer_id)
        except Exception as e:
            system_logger.error(
                f"Error getting customer context: {str(e)}",
                additional_info={
                    "context": "Get Customer Context",
                    "customer_id": customer_id,
                },
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Error retrieving customer context: Please try again or contact support",
            )

    def _get_fallback_customer_data(self, customer_id: int) -> Dict[str, Any]:
        """Get fallback customer data when main data retrieval fails"""
        return {
            "customer_id": customer_id,
            "customer_name": f"Customer {customer_id}",
            "email": f"customer{customer_id}@example.com",
            "support_tier": "Standard",
            "company": "Unknown Company",
            "satisfaction_score": 3.5,
            "recent_interactions": [],
            "support_metrics": {
                "total_interactions": 0,
                "average_satisfaction": 3.5,
                "open_tickets": 0,
                "total_tickets": 0,
            },
            "open_tickets": [],
            "risk_assessment": {
                "risk_level": "Low",
                "risk_score": 0.1,
                "risk_factors": [],
            },
            "feedback_summary": {
                "feedback_count": 0,
                "satisfaction_trend": "stable",
                "latest_rating": None,
            },
            "recommendations": ["No specific recommendations at this time"],
            "last_updated": datetime.now().isoformat(),
            "data_sources": ["fallback_data"],
            "note": "Using fallback data due to data retrieval issues",
        }

    def get_debug_info(self, customer_id: int) -> Dict[str, Any]:
        """Debug customer context issues"""
        try:
            agent = self.get_agent()
            if not agent:
                return {"error": self.CUSTOMER_SUPPORT_UNAVAILABLE}

            # Get raw customer context
            customer_context = agent.get_customer_context(customer_id)

            # Get customer insights step by step
            debug_info = {
                "customer_id": customer_id,
                "customer_context_exists": customer_context is not None,
                "customer_context_type": str(type(customer_context)),
                "customer_context_attributes": (
                    dir(customer_context) if customer_context else []
                ),
            }

            if customer_context:
                debug_info["customer_context_data"] = {
                    "customer_id": getattr(customer_context, "customer_id", "MISSING"),
                    "name": getattr(customer_context, "name", "MISSING"),
                    "email": getattr(customer_context, "email", "MISSING"),
                    "tier": getattr(customer_context, "tier", "MISSING"),
                    "company": getattr(customer_context, "company", "MISSING"),
                    "satisfaction_score": getattr(
                        customer_context, "satisfaction_score", "MISSING"
                    ),
                    "recent_interactions_count": len(
                        getattr(customer_context, "recent_interactions", [])
                    ),
                    "open_tickets_count": len(
                        getattr(customer_context, "open_tickets", [])
                    ),
                }

            # Test customer insights
            try:
                customer_insights = agent.get_customer_insights(customer_id)
                debug_info["customer_insights_success"] = True
                debug_info["customer_insights_keys"] = (
                    list(customer_insights.keys())
                    if isinstance(customer_insights, dict)
                    else "NOT_DICT"
                )
                debug_info["customer_insights_has_error"] = "error" in customer_insights
            except Exception as insights_error:
                debug_info["customer_insights_success"] = False
                debug_info["customer_insights_error"] = str(insights_error)

            return debug_info

        except Exception as e:
            return {"debug_error": str(e), "error_type": type(e).__name__}

    def get_customer_insights_data(self, customer_id: int) -> Dict[str, Any]:
        """Get comprehensive customer insights"""
        try:
            agent = self.get_agent()
            if not agent:
                raise HTTPException(
                    status_code=503, detail=self.CUSTOMER_SUPPORT_UNAVAILABLE
                )

            insights = agent.get_customer_insights(customer_id)

            if "error" in insights:
                raise HTTPException(status_code=404, detail=insights["error"])

            return insights

        except HTTPException:
            raise
        except Exception as e:
            system_logger.error(
                f"Error getting customer insights: {str(e)}",
                additional_info={
                    "context": "Get Customer Insights",
                    "customer_id": customer_id,
                },
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail=f"Error retrieving customer insights: {str(e)}"
            )


# Create router
router = APIRouter(tags=["Customer Support"])


# Dependency to get customer support service
def get_customer_support_service(
    agent_manager: AgentManager = Depends(),
) -> CustomerSupportService:
    """Get customer support service instance"""
    return CustomerSupportService(agent_manager)


# Customer Support endpoints
@router.post("/chat", response_model=CustomerSupportResponse)
async def customer_support_chat(
    request: CustomerSupportRequest,
    service: CustomerSupportService = Depends(get_customer_support_service),
    token: str = Depends(get_verified_token),
):
    """Handle customer support chat request with improved error handling"""
    return service.handle_chat_request(request)


@router.get("/customer/{customer_id}")
async def get_customer_context(
    customer_id: int,
    service: CustomerSupportService = Depends(get_customer_support_service),
    token: str = Depends(get_verified_token),
):
    """Get enhanced customer context and information"""
    return service.get_customer_context_data(customer_id)


@router.get("/debug/{customer_id}")
async def debug_customer_context(
    customer_id: int,
    service: CustomerSupportService = Depends(get_customer_support_service),
    token: str = Depends(get_verified_token),
):
    """Debug customer context issues"""
    return service.get_debug_info(customer_id)


@router.get("/customer/{customer_id}/insights")
async def get_customer_insights(
    customer_id: int,
    service: CustomerSupportService = Depends(get_customer_support_service),
    token: str = Depends(get_verified_token),
):
    """Get comprehensive customer insights"""
    return service.get_customer_insights_data(customer_id)
