# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-01 23:53:31
# @Description: This module implements the main API for the Agentic AI system

import os
from datetime import datetime
from typing import Dict, List

import yaml
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.que_agents.agents.customer_support_agent import CustomerSupportAgent
from src.que_agents.agents.marketing_agent import MarketingAgent

# Load API configuration - fix the path
config_path = os.path.join(
    os.path.dirname(__file__), "../../../configs/api_config.yaml"
)
if not os.path.exists(config_path):
    # Alternative path
    config_path = "configs/api_config.yaml"

try:
    with open(config_path, "r") as f:
        api_config = yaml.safe_load(f)
except FileNotFoundError:
    # Fallback configuration
    print(f"Warning: Config file not found at {config_path}. Using fallback config.")
    api_config = {
        "api": {
            "title": "Agentic AI API",
            "description": "AI-powered customer support and marketing agents",
            "version": "1.0.0",
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "host": "0.0.0.0",
            "port": 8000,
            "reload": True,
            "log_level": "info",
        },
        "cors": {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        },
        "authentication": {"api_token": "demo-token-123"},
    }

# Initialize FastAPI app
app = FastAPI(
    title=api_config["api"]["title"],
    description=api_config["api"]["description"],
    version=api_config["api"]["version"],
    docs_url=api_config["api"]["docs_url"],
    redoc_url=api_config["api"]["redoc_url"],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config["cors"]["allow_origins"],
    allow_credentials=api_config["cors"]["allow_credentials"],
    allow_methods=api_config["cors"]["allow_methods"],
    allow_headers=api_config["cors"]["allow_headers"],
)

# Security
security = HTTPBearer()

# Initialize agents with error handling
try:
    customer_support_agent = CustomerSupportAgent()
    marketing_agent = MarketingAgent()
except Exception as e:
    print(f"Warning: Could not initialize agents: {e}")
    customer_support_agent = None
    marketing_agent = None


# Pydantic models
class CustomerSupportRequest(BaseModel):
    customer_id: int
    message: str


class CustomerSupportResponse(BaseModel):
    response: str
    confidence: float
    escalate: bool
    suggested_actions: List[str]
    knowledge_sources: List[str]
    sentiment: str
    timestamp: str


class MarketingCampaignRequest(BaseModel):
    campaign_type: str
    target_audience: str
    budget: float
    duration_days: int
    goals: List[str]
    channels: List[str]
    content_requirements: List[str]


class ContentGenerationRequest(BaseModel):
    platform: str
    content_type: str
    campaign_theme: str
    target_audience: str
    key_messages: List[str]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    agents: Dict[str, str]


# Authentication dependency
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (simplified for demo)"""
    expected_token = api_config["authentication"]["api_token"]

    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    return credentials.credentials


# Health check endpoint - MUST be defined
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        agents={
            "customer_support": "active" if customer_support_agent else "error",
            "marketing": "active" if marketing_agent else "error",
        },
    )


# Root endpoint for testing
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Agentic AI API is running",
        "version": api_config["api"]["version"],
    }


# Customer Support endpoints
@app.post("/api/v1/customer-support/chat", response_model=CustomerSupportResponse)
async def customer_support_chat(
    request: CustomerSupportRequest, token: str = Depends(verify_token)
):
    """Handle customer support chat request"""
    try:
        if not customer_support_agent:
            # Return mock response if agent not available
            return CustomerSupportResponse(
                response="I understand you're having login issues. Let me help you with that. First, please try resetting your password using the 'Forgot Password' link on the login page.",
                confidence=0.85,
                escalate=False,
                suggested_actions=[
                    "reset_password",
                    "clear_browser_cache",
                    "contact_support",
                ],
                knowledge_sources=["login_troubleshooting", "password_reset_guide"],
                sentiment="neutral",
                timestamp=datetime.now().isoformat(),
            )

        # Call the actual agent
        result = customer_support_agent.handle_customer_request(
            customer_id=request.customer_id, message=request.message
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
    except Exception as e:
        print(f"Error in customer support chat: {e}")
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


@app.get("/api/v1/customer-support/customer/{customer_id}")
async def get_customer_context(customer_id: int, token: str = Depends(verify_token)):
    """Get customer context information"""
    mock_customers = {
        1: {
            "name": "John Smith",
            "email": "john@example.com",
            "tier": "Premium",
            "company": "TechCorp",
        },
        2: {
            "name": "Sarah Johnson",
            "email": "sarah@example.com",
            "tier": "Standard",
            "company": "StartupInc",
        },
        3: {
            "name": "Mike Wilson",
            "email": "mike@example.com",
            "tier": "Enterprise",
            "company": "BigCorp",
        },
        4: {
            "name": "Emily Davis",
            "email": "emily@example.com",
            "tier": "Standard",
            "company": "MidSize Co",
        },
        5: {
            "name": "David Brown",
            "email": "david@example.com",
            "tier": "Premium",
            "company": "GrowthCorp",
        },
    }

    if customer_id not in mock_customers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer {customer_id} not found",
        )

    customer = mock_customers[customer_id]
    return {**customer, "open_tickets": [], "recent_interactions": []}


# Marketing endpoints with fallback responses
@app.post("/api/v1/marketing/campaign/create")
async def create_marketing_campaign(
    request: MarketingCampaignRequest, token: str = Depends(verify_token)
):
    """Create a new marketing campaign"""
    try:
        if not marketing_agent:
            # Return mock response if agent not available
            return {
                "campaign_id": f"camp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "strategy": f"Multi-channel {request.campaign_type} campaign targeting {request.target_audience}",
                "content_pieces": [
                    {
                        "platform": "linkedin",
                        "content_type": "post",
                        "content": "Sample LinkedIn post content",
                    },
                    {
                        "platform": "email",
                        "content_type": "newsletter",
                        "content": "Sample email content",
                    },
                ],
                "estimated_performance": {
                    "total_reach": int(request.budget * 10),
                    "estimated_roi": 2.5,
                },
            }

        # Import the required classes
        from src.que_agents.agents.marketing_agent import (
            CampaignRequest,
            CampaignType,
            ContentType,
        )

        # Convert string campaign_type to enum
        try:
            campaign_type_enum = CampaignType(request.campaign_type)
        except ValueError:
            # If the campaign type is not recognized, default to LEAD_GENERATION
            campaign_type_enum = CampaignType.LEAD_GENERATION

        # Convert content_requirements strings to ContentType enums
        content_types = []
        for content_req in request.content_requirements:
            try:
                content_types.append(ContentType(content_req))
            except ValueError:
                # Default to SOCIAL_MEDIA if not recognized
                content_types.append(ContentType.SOCIAL_MEDIA)

        # Create CampaignRequest object
        campaign_request = CampaignRequest(
            campaign_type=campaign_type_enum,
            target_audience=request.target_audience,
            budget=request.budget,
            duration_days=request.duration_days,
            goals=request.goals,
            channels=request.channels,
            content_requirements=content_types,
        )

        # Call the marketing agent with the correct CampaignRequest object
        result = marketing_agent.create_campaign_plan(campaign_request)

        # Convert the result to a JSON-serializable format
        return {
            "campaign_id": result.campaign_id,
            "strategy": result.strategy,
            "content_pieces": [
                {
                    "platform": cp.platform,
                    "content_type": cp.content_type.value,
                    "title": cp.title,
                    "content": cp.content,
                    "hashtags": cp.hashtags,
                    "call_to_action": cp.call_to_action,
                    "estimated_reach": cp.estimated_reach,
                }
                for cp in result.content_pieces
            ],
            "schedule": result.schedule,
            "budget_allocation": result.budget_allocation,
            "success_metrics": result.success_metrics,
            "estimated_performance": result.estimated_performance,
        }

    except Exception as e:
        print(f"Error creating campaign: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating marketing campaign: {str(e)}",
        )


# Marketing content generation endpoint
@app.post("/api/v1/marketing/content/generate")
async def generate_marketing_content(
    request: ContentGenerationRequest, token: str = Depends(verify_token)
):
    """Generate marketing content"""
    try:
        if not marketing_agent:
            # Return mock content
            return {
                "platform": request.platform,
                "content": f"🚀 Exciting news for {request.target_audience}! Our {request.campaign_theme} is here to transform your workflow. Key benefits: {', '.join(request.key_messages)}. #Innovation #Technology #Growth",
                "hashtags": ["#Innovation", "#Technology", "#Growth", "#AI"],
                "estimated_reach": 5000,
                "title": f"Generated Content for {request.platform}",
                "call_to_action": "Learn more",
            }

        # Import ContentType enum
        from src.que_agents.agents.marketing_agent import ContentType

        # Convert string content_type to enum
        try:
            content_type_enum = ContentType(request.content_type)
        except ValueError:
            # Default to SOCIAL_MEDIA if not recognized
            content_type_enum = ContentType.SOCIAL_MEDIA

        # Generate content using the marketing agent
        result = marketing_agent.generate_content(
            platform=request.platform,
            content_type=content_type_enum,
            campaign_theme=request.campaign_theme,
            target_audience=request.target_audience,
            key_messages=request.key_messages,
        )

        # Convert ContentPiece to JSON-serializable format
        return {
            "platform": result.platform,
            "content_type": result.content_type.value,
            "title": result.title,
            "content": result.content,
            "hashtags": result.hashtags,
            "call_to_action": result.call_to_action,
            "estimated_reach": result.estimated_reach,
        }

    except Exception as e:
        print(f"Error generating content: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating marketing content: {str(e)}",
        )


# Add a debug endpoint to show valid enum values
@app.get("/api/v1/marketing/enums")
async def get_marketing_enums(token: str = Depends(verify_token)):
    """Get valid enum values for marketing endpoints"""
    try:
        from src.que_agents.agents.marketing_agent import CampaignType, ContentType

        return {
            "campaign_types": [ct.value for ct in CampaignType],
            "content_types": [ct.value for ct in ContentType],
        }
    except Exception:
        return {
            "campaign_types": [
                "product_launch",
                "customer_retention",
                "lead_generation",
                "brand_awareness",
                "seasonal_promotion",
            ],
            "content_types": [
                "social_media",
                "email",
                "blog_post",
                "ad_copy",
                "landing_page",
            ],
        }


# Database endpoints
@app.get("/api/v1/customers")
async def list_customers(
    limit: int = 10, offset: int = 0, token: str = Depends(verify_token)
):
    """List customers"""
    return {
        "customers": [
            {"id": i, "name": f"Customer {i}", "email": f"customer{i}@example.com"}
            for i in range(1, min(limit + 1, 6))
        ],
        "total": 5,
    }


@app.get("/api/v1/campaigns")
async def list_campaigns(
    limit: int = 10, offset: int = 0, token: str = Depends(verify_token)
):
    """List marketing campaigns"""
    return {
        "campaigns": [
            {
                "id": i,
                "name": f"Campaign {i}",
                "status": "active" if i % 2 == 0 else "completed",
            }
            for i in range(1, min(limit + 1, 4))
        ],
        "total": 3,
    }


@app.get("/api/v1/knowledge-base/search")
async def search_knowledge(
    query: str, limit: int = 10, token: str = Depends(verify_token)
):
    """Search knowledge base"""
    # Return mock results
    return {
        "results": [
            {
                "id": f"kb_{i}",
                "title": f"Knowledge Article {i}",
                "content": f"Content related to {query}",
            }
            for i in range(1, min(limit + 1, 3))
        ]
    }


# Debug endpoint
@app.get("/debug")
async def debug_info():
    """Debug information"""
    return {
        "config_loaded": bool(api_config),
        "agents_loaded": {
            "customer_support": customer_support_agent is not None,
            "marketing": marketing_agent is not None,
        },
        "routes": [
            (
                route.path
                if hasattr(route, "path")
                else (
                    route.url_path_for(route.name)
                    if hasattr(route, "name")
                    else str(route)
                )
            )
            for route in app.routes
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=api_config["api"]["host"],
        port=api_config["api"]["port"],
        reload=api_config["api"]["reload"],
    )
