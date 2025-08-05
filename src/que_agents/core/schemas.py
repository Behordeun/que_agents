from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel


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


# Personal Virtual Assistant models
class PVARequest(BaseModel):
    user_id: str
    message: str
    session_id: str = None


class PVAResponse(BaseModel):
    response: str
    intent: str
    entities: Dict[str, Any]
    confidence: float
    actions_taken: List[str]
    suggestions: List[str]
    timestamp: str


# Financial Trading Bot models
class TradingAnalysisRequest(BaseModel):
    symbol: str
    strategy_type: str = "momentum"


class TradingDecisionResponse(BaseModel):
    action: str
    symbol: str
    quantity: float
    confidence: float
    reasoning: str
    risk_score: float
    expected_return: float


class TradingCycleRequest(BaseModel):
    symbols: List[str] = None


class TradingCycleResponse(BaseModel):
    timestamp: str
    symbols_analyzed: int
    trades_executed: int
    decisions: List[Dict[str, Any]]
    portfolio_status: Dict[str, Any]
    average_confidence: float


class CampaignType(Enum):
    PRODUCT_LAUNCH = "product_launch"
    CUSTOMER_RETENTION = "customer_retention"
    LEAD_GENERATION = "lead_generation"
    BRAND_AWARENESS = "brand_awareness"
    SEASONAL_PROMOTION = "seasonal_promotion"


class ContentType(Enum):
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    BLOG_POST = "blog_post"
    AD_COPY = "ad_copy"
    LANDING_PAGE = "landing_page"


@dataclass
class CampaignRequest:
    """Campaign creation request"""

    campaign_type: CampaignType
    target_audience: str
    budget: float
    duration_days: int
    goals: List[str]
    channels: List[str]
    content_requirements: List[ContentType]


@dataclass
class ContentPiece:
    """Generated content piece"""

    content_type: ContentType
    platform: str
    title: str
    content: str
    hashtags: List[str]
    call_to_action: str
    estimated_reach: int


@dataclass
class CampaignPlan:
    """Complete campaign plan"""

    campaign_id: str
    strategy: str
    content_pieces: List[ContentPiece]
    schedule: List[Dict]
    budget_allocation: Dict[str, float]
    success_metrics: List[str]
    estimated_performance: Dict[str, Any]


@dataclass
class MarketConditions:
    """Current market conditions"""

    symbol: str
    current_price: float
    volume: float
    change_24h: float
    rsi: float
    macd: float
    moving_avg_20: float
    moving_avg_50: float
    volatility: float
    market_sentiment: str


@dataclass
class TradingDecision:
    """Trading decision result"""

    action: str  # buy, sell, hold
    symbol: str
    quantity: float
    confidence: float
    reasoning: str
    risk_score: float
    expected_return: float


@dataclass
class PortfolioStatus:
    """Current portfolio status"""

    total_value: float
    cash_balance: float
    holdings: Dict[str, float]
    performance_metrics: Dict[str, float]
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class Portfolio:
    """Portfolio data structure"""

    user_id: str
    total_value: float
    cash_balance: float
    holdings: Dict[str, float]  # {symbol: quantity}
    performance_metrics: Dict[str, float]  # {metric_name: value}
    unrealized_pnl: float
    realized_pnl: float
    last_updated: str


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


