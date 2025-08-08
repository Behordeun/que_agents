from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# Pydantic models
class CustomerSupportRequest(BaseModel):
    customer_id: int
    message: str
    priority: str = "medium"
    category: Optional[str] = None
    session_id: Optional[str] = None


class CustomerSupportResponse(BaseModel):
    response: str
    confidence: float
    escalate: bool
    suggested_actions: List[str]
    knowledge_sources: List[str]
    sentiment: str
    timestamp: str
    resolution_category: Optional[str] = None
    estimated_resolution_time: Optional[str] = None


class MarketingCampaignRequest(BaseModel):
    campaign_type: str
    target_audience: str
    budget: float
    duration_days: int
    goals: List[str]
    channels: List[str]
    content_requirements: List[str]
    industry: Optional[str] = None
    brand_voice: Optional[str] = "professional"


class MarketingCampaignResponse(BaseModel):
    success: bool
    campaign_id: Optional[int] = None
    message: str
    campaign_plan: Optional[dict] = None
    schedule: Optional[List[dict]] = None
    next_steps: Optional[List[str]] = None
    timestamp: str


class ContentGenerationRequest(BaseModel):
    platform: str
    content_type: str
    campaign_theme: str
    target_audience: str
    key_messages: List[str]
    brand_voice: str = "professional"
    tone: str = "engaging"
    include_hashtags: bool = True


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    agents: Dict[str, str]
    system_metrics: Optional[Dict[str, Any]] = None


# Personal Virtual Assistant models
class PVARequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class PVAResponse(BaseModel):
    response: str
    intent: str
    entities: Dict[str, Any]
    confidence: float
    actions_taken: List[str]
    suggestions: List[str]
    timestamp: str
    session_id: Optional[str] = None


# Financial Trading Bot models
class TradingAnalysisRequest(BaseModel):
    symbol: str
    strategy_type: str = "momentum"
    timeframe: str = "1d"
    risk_tolerance: str = "medium"


class TradingDecisionResponse(BaseModel):
    action: str
    symbol: str
    quantity: float
    confidence: float
    reasoning: str
    risk_score: float
    expected_return: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class TradingCycleRequest(BaseModel):
    symbols: Optional[List[str]] = None
    strategy_filters: Optional[List[str]] = None


class TradingCycleResponse(BaseModel):
    timestamp: str
    symbols_analyzed: int
    trades_executed: int
    decisions: List[Dict[str, Any]]
    portfolio_status: Dict[str, Any]
    average_confidence: float
    market_summary: Optional[Dict[str, Any]] = None


# Enhanced Enums
class CampaignType(Enum):
    PRODUCT_LAUNCH = "product_launch"
    CUSTOMER_RETENTION = "customer_retention"
    LEAD_GENERATION = "lead_generation"
    BRAND_AWARENESS = "brand_awareness"
    SEASONAL_PROMOTION = "seasonal_promotion"
    EVENT_PROMOTION = "event_promotion"
    THOUGHT_LEADERSHIP = "thought_leadership"
    USER_ACQUISITION = "user_acquisition"


class ContentType(Enum):
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    BLOG_POST = "blog_post"
    AD_COPY = "ad_copy"
    LANDING_PAGE = "landing_page"
    VIDEO = "video"
    PODCAST = "podcast"
    INFOGRAPHIC = "infographic"
    WHITEPAPER = "whitepaper"
    CASE_STUDY = "case_study"
    WEBINAR = "webinar"
    NEWSLETTER = "newsletter"
    PRESS_RELEASE = "press_release"
    IMAGE = "image"


class CampaignStatus(Enum):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class PostStatus(Enum):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    FAILED = "failed"
    PAUSED = "paused"


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# Enhanced Campaign Models
@dataclass
class CampaignRequest:
    """Enhanced campaign creation request"""

    campaign_type: CampaignType
    target_audience: str
    budget: float
    duration_days: int
    goals: List[str]
    channels: List[str]
    content_requirements: List[ContentType]
    industry: Optional[str] = None
    brand_voice: str = "professional"
    geographic_targeting: Optional[List[str]] = None
    competitor_analysis: bool = False


@dataclass
class ContentPiece:
    """Enhanced content piece with optimization features"""

    content_type: ContentType
    platform: str
    title: str
    content: str
    hashtags: List[str]
    call_to_action: str
    estimated_reach: int
    variations: List[Dict[str, Any]] = field(default_factory=list)
    optimization_score: float = 0.0
    target_demographics: Optional[Dict[str, Any]] = None
    scheduling_priority: Priority = Priority.MEDIUM
    estimated_engagement: Optional[int] = None
    seo_keywords: List[str] = field(default_factory=list)


@dataclass
class CampaignPlan:
    """Comprehensive campaign plan with enhanced features"""

    campaign_id: str
    strategy: str
    content_pieces: List[ContentPiece]
    schedule: List[Dict[str, Any]]
    budget_allocation: Dict[str, float]
    success_metrics: List[str]
    estimated_performance: Dict[str, Any]
    risk_assessment: Optional[Dict[str, Any]] = None
    optimization_roadmap: List[Dict[str, Any]] = field(default_factory=list)
    industry: Optional[str] = None
    brand_voice: str = "professional"
    competitive_analysis: Optional[Dict[str, Any]] = None
    target_segments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CampaignMetrics:
    """Campaign performance metrics"""

    campaign_id: str
    impressions: int
    clicks: int
    conversions: int
    cost_per_click: float
    cost_per_acquisition: float
    return_on_ad_spend: float
    engagement_rate: float
    reach: int
    frequency: float
    click_through_rate: float
    conversion_rate: float
    timestamp: datetime
    channel_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation structure"""

    action: str
    description: str
    priority: Priority
    estimated_effort: str
    expected_impact: str
    implementation_time: str
    confidence_level: str
    risk_level: str = "low"


@dataclass
class CampaignInsights:
    """Campaign insights and analytics"""

    campaign_id: str
    performance_summary: Dict[str, Any]
    channel_breakdown: Dict[str, Any]
    audience_insights: Dict[str, Any]
    content_performance: Dict[str, Any]
    optimization_opportunities: List[OptimizationRecommendation]
    competitive_intelligence: Optional[Dict[str, Any]] = None
    trend_analysis: Optional[Dict[str, Any]] = None


# Enhanced Financial Trading Models
@dataclass
class MarketConditions:
    """Enhanced market conditions with more indicators"""

    symbol: str
    current_price: float
    volume: float
    change_24h: float
    rsi: float
    macd: float
    moving_avg_20: float
    moving_avg_50: float
    moving_avg_200: float
    volatility: float
    market_sentiment: str
    bollinger_bands: Dict[str, float] = field(default_factory=dict)
    support_resistance: Dict[str, float] = field(default_factory=dict)
    news_sentiment: Optional[str] = None
    sector_performance: Optional[float] = None


@dataclass
class TradingDecision:
    """Enhanced trading decision with risk management"""

    action: str  # buy, sell, hold
    symbol: str
    quantity: float
    confidence: float
    reasoning: str
    risk_score: float
    expected_return: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_horizon: str = "short_term"
    strategy_used: str = "momentum"


@dataclass
class PortfolioStatus:
    """Enhanced portfolio status with detailed metrics"""

    total_value: float
    cash_balance: float
    holdings: Dict[str, float]
    performance_metrics: Dict[str, float]
    unrealized_pnl: float
    realized_pnl: float
    sector_allocation: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    beta: Optional[float] = None


@dataclass
class Portfolio:
    """Enhanced portfolio data structure"""

    user_id: str
    total_value: float
    cash_balance: float
    holdings: Dict[str, float]  # {symbol: quantity}
    performance_metrics: Dict[str, float]  # {metric_name: value}
    unrealized_pnl: float
    realized_pnl: float
    last_updated: str
    risk_profile: str = "moderate"
    investment_goals: List[str] = field(default_factory=list)
    time_horizon: str = "long_term"


# Enhanced Customer Support Models
@dataclass
class CustomerContext:
    """Enhanced customer context with more details"""

    customer_id: int
    name: str
    email: str
    tier: str
    company: str
    recent_interactions: List[Dict[str, Any]]
    open_tickets: List[Dict[str, Any]]
    purchase_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    satisfaction_score: Optional[float] = None
    lifetime_value: Optional[float] = None
    risk_score: Optional[float] = None


@dataclass
class AgentResponse:
    """Enhanced agent response structure"""

    message: str
    confidence: float
    escalate: bool
    suggested_actions: List[str]
    knowledge_sources: List[str]
    sentiment: str
    response_type: str = "standard"
    priority: Priority = Priority.MEDIUM
    estimated_resolution_time: Optional[str] = None
    follow_up_required: bool = False
    tags: List[str] = field(default_factory=list)


# Knowledge Base Models
@dataclass
class KnowledgeItem:
    """Knowledge base item structure"""

    id: str
    title: str
    content: str
    category: str
    tags: List[str]
    relevance_score: float
    source: str
    last_updated: datetime
    usage_count: int = 0
    effectiveness_score: Optional[float] = None


@dataclass
class SearchResult:
    """Search result structure"""

    items: List[KnowledgeItem]
    total_results: int
    search_time: float
    query: str
    filters_applied: Dict[str, Any] = field(default_factory=dict)


# Market Intelligence Models
@dataclass
class MarketIntelligence:
    """Market intelligence data structure"""

    industry: str
    market_size: float
    growth_rate: float
    key_trends: List[str]
    competitive_landscape: Dict[str, Any]
    consumer_insights: Dict[str, Any]
    seasonal_patterns: Dict[str, Any] = field(default_factory=dict)
    regulatory_environment: Optional[str] = None


@dataclass
class CompetitorAnalysis:
    """Competitor analysis structure"""

    competitor_name: str
    market_share: float
    strengths: List[str]
    weaknesses: List[str]
    marketing_strategies: List[str]
    pricing_strategy: str
    target_audience: str
    recent_campaigns: List[Dict[str, Any]] = field(default_factory=list)


# Audience Intelligence Models
@dataclass
class AudienceSegment:
    """Audience segment definition"""

    segment_name: str
    size: int
    demographics: Dict[str, Any]
    psychographics: Dict[str, Any]
    behavior_patterns: Dict[str, Any]
    preferred_channels: List[str]
    engagement_metrics: Dict[str, float]
    conversion_likelihood: float
    lifetime_value_potential: float


@dataclass
class AudienceInsights:
    """Comprehensive audience insights"""

    primary_audience: str
    segments: List[AudienceSegment]
    behavioral_trends: Dict[str, Any]
    channel_preferences: Dict[str, float]
    content_preferences: Dict[str, float]
    optimal_timing: Dict[str, List[str]]
    geographic_distribution: Dict[str, float] = field(default_factory=dict)
    device_usage: Dict[str, float] = field(default_factory=dict)
    knowledge_base_insights: List[KnowledgeItem] = field(default_factory=list)


# API Response Models
class CampaignCreationResponse(BaseModel):
    campaign_id: int
    status: str
    campaign_plan: Dict[str, Any]
    schedule: List[Dict[str, Any]]
    optimization_roadmap: List[Dict[str, Any]]
    next_steps: List[str]
    estimated_performance: Dict[str, Any]


class CampaignAnalysisResponse(BaseModel):
    campaign_id: int
    analysis: str
    performance_metrics: Dict[str, Any]
    insights: Dict[str, Any]
    recommendations: List[str]
    confidence_level: str


class OptimizationResponse(BaseModel):
    campaign_id: int
    optimization_recommendations: str
    priority_actions: List[Dict[str, Any]]
    estimated_impact: Dict[str, Any]
    implementation_timeline: str
    confidence_level: str


class DashboardResponse(BaseModel):
    campaign_id: int
    campaign_overview: Dict[str, Any]
    performance_summary: Dict[str, Any]
    channel_breakdown: Dict[str, Any]
    timeline_analysis: Dict[str, Any]
    roi_analysis: Dict[str, Any]
    recommendations: List[str]
    next_steps: List[str]
    last_updated: str


# Utility Models
@dataclass
class APIError:
    """API error structure"""

    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None

    @staticmethod
    def current_timestamp():
        return datetime.now().isoformat()

    timestamp: str = field(default_factory=current_timestamp)


@dataclass
class PaginationInfo:
    """Pagination information"""

    page: int
    per_page: int
    total_items: int
    total_pages: int
    has_next: bool
    has_prev: bool


@dataclass
class FilterOptions:
    """Filter options for queries"""

    date_range: Optional[Dict[str, str]] = None
    status: Optional[List[str]] = None
    campaign_type: Optional[List[str]] = None
    channels: Optional[List[str]] = None
    performance_threshold: Optional[Dict[str, float]] = None


# Configuration Models
@dataclass
class AgentConfig:
    """Agent configuration structure"""

    agent_type: str
    model_name: str
    temperature: float
    max_tokens: int
    system_prompt: str
    tools_enabled: List[str] = field(default_factory=list)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    fallback_strategies: List[str] = field(default_factory=list)


@dataclass
class SystemHealth:
    """System health monitoring"""

    status: str
    timestamp: datetime
    agents_status: Dict[str, str]
    database_status: str
    api_response_time: float
    memory_usage: float
    cpu_usage: float
    active_sessions: int
    error_rate: float = 0.0
    uptime: Optional[str] = None


@dataclass
class UserContext:
    """User context information"""

    user_id: str
    preferences: Dict[str, Any]
    learned_behaviors: Dict[str, Any]
    active_reminders: List[Dict]
    smart_devices: List[Dict]


@dataclass
class IntentResult:
    """Intent recognition result"""

    intent: str
    confidence: float
    entities: Dict[str, Any]
