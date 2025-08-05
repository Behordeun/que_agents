# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-01 23:53:31
# @Description: This module implements a marketing agent for autonomous campaign management

import json
from datetime import datetime
from typing import Any, Dict, List

import yaml
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from src.que_agents.core.database import (
    AudienceSegment,
    CampaignMetrics,
    MarketingCampaign,
    MarketingPost,
    get_session,
)
from src.que_agents.core.llm_factory import LLMFactory
from src.que_agents.core.schemas import (
    CampaignPlan,
    CampaignRequest,
    CampaignType,
    ContentPiece,
    ContentType,
)
from src.que_agents.knowledge_base.kb_manager import search_knowledge_base

# Load agent configuration
with open("configs/agent_config.yaml", "r") as f:
    agent_config = yaml.safe_load(f)


class MarketingAgent:
    """Marketing Agent for autonomous campaign management"""

    def __init__(self):
        config = agent_config["marketing_agent"]
        self.llm = LLMFactory.get_llm(
            agent_type="marketing",
            model_name=config["model_name"],
            temperature=config["temperature"],
            max_tokens=800,
        )
        # Platform-specific constraints
        self.platform_limits = {
            "twitter": {"max_chars": 280, "hashtag_limit": 3},
            "linkedin": {"max_chars": 3000, "hashtag_limit": 5},
            "facebook": {"max_chars": 2000, "hashtag_limit": 5},
            "instagram": {"max_chars": 2200, "hashtag_limit": 10},
            "email": {"subject_max": 50, "body_max": 2000},
        }

        # Initialize prompt templates
        self.campaign_prompt = self._create_campaign_prompt()
        self.content_prompt = self._create_content_prompt()
        self.analysis_prompt = self._create_analysis_prompt()

        # Create chains
        self.campaign_chain = self._create_campaign_chain()
        self.content_chain = self._create_content_chain()
        self.analysis_chain = self._create_analysis_chain()

    def _create_campaign_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for campaign strategy"""
        system_message = """You are an expert marketing strategist specializing in digital marketing campaigns. Your role is to:

1. Analyze campaign requirements and create comprehensive marketing strategies
2. Identify target audience segments and their characteristics
3. Recommend optimal channel mix and budget allocation
4. Define success metrics and KPIs
5. Create detailed campaign timelines and schedules

CAMPAIGN STRATEGY GUIDELINES:
- Focus on data-driven decisions and measurable outcomes
- Consider audience behavior and platform-specific best practices
- Optimize for engagement, conversion, and ROI
- Include A/B testing recommendations
- Account for seasonal trends and market conditions

Campaign Request: {campaign_request}
Market Research Data: {market_data}
Audience Insights: {audience_data}

Create a comprehensive campaign strategy that includes:
1. Strategic overview and positioning
2. Target audience analysis
3. Channel recommendations with rationale
4. Budget allocation across channels
5. Content strategy and themes
6. Timeline and key milestones
7. Success metrics and KPIs
8. Risk mitigation strategies"""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    "Please create a marketing campaign strategy based on the provided requirements.",
                ),
            ]
        )

    def _create_content_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for content generation"""
        system_message = """You are a creative marketing copywriter specializing in multi-platform content creation. Your role is to:

1. Create engaging, platform-optimized content
2. Maintain brand voice and messaging consistency
3. Include compelling calls-to-action
4. Optimize for platform-specific algorithms
5. Generate relevant hashtags and keywords

CONTENT CREATION GUIDELINES:
- Write in an engaging, conversational tone
- Include emotional triggers and persuasive elements
- Optimize for each platform's unique characteristics
- Use action-oriented language
- Include social proof when relevant
- Ensure compliance with platform policies

Platform: {platform}
Content Type: {content_type}
Campaign Theme: {campaign_theme}
Target Audience: {target_audience}
Key Messages: {key_messages}
Platform Constraints: {platform_constraints}

Create compelling content that:
1. Captures attention within the first few words
2. Communicates value proposition clearly
3. Includes a strong call-to-action
4. Uses appropriate hashtags and keywords
5. Fits platform character/length limits
6. Aligns with campaign objectives"""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    "Please create optimized content for the specified platform and campaign.",
                ),
            ]
        )

    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for campaign analysis"""
        system_message = """You are a marketing analytics expert specializing in campaign performance analysis. Your role is to:

1. Analyze campaign metrics and performance data
2. Identify trends, patterns, and insights
3. Provide actionable recommendations for optimization
4. Benchmark against industry standards
5. Predict future performance and outcomes

ANALYSIS GUIDELINES:
- Focus on actionable insights and recommendations
- Compare performance against goals and benchmarks
- Identify top-performing content and channels
- Suggest optimization strategies
- Consider statistical significance and data quality

Campaign Data: {campaign_data}
Performance Metrics: {performance_metrics}
Industry Benchmarks: {benchmarks}
Campaign Goals: {campaign_goals}

Provide a comprehensive analysis that includes:
1. Overall performance summary
2. Channel-specific performance breakdown
3. Content performance analysis
4. Audience engagement insights
5. ROI and conversion analysis
6. Optimization recommendations
7. Future strategy suggestions"""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    "Please analyze the campaign performance and provide optimization recommendations.",
                ),
            ]
        )

    def _create_campaign_chain(self):
        """Create campaign strategy chain"""
        return self.campaign_prompt | self.llm | StrOutputParser()

    def _create_content_chain(self):
        """Create content generation chain"""
        return self.content_prompt | self.llm | StrOutputParser()

    def _create_analysis_chain(self):
        """Create analysis chain"""
        return self.analysis_prompt | self.llm | StrOutputParser()

    def get_audience_insights(self, target_audience: str) -> Dict[str, Any]:
        """Get audience insights from database and knowledge base"""
        session = get_session()
        try:
            # Search for audience segments
            segments = (
                session.query(AudienceSegment)
                .filter(AudienceSegment.name.ilike(f"%{target_audience}%"))
                .all()
            )

            # Search knowledge base for audience data
            kb_results = search_knowledge_base(
                f"audience {target_audience}", category="data", limit=3
            )

            return {
                "segments": [
                    {
                        "name": s.name,
                        "criteria": s.criteria,
                        "size": s.estimated_size,
                        "characteristics": s.characteristics,
                    }
                    for s in segments
                ],
                "knowledge_base": kb_results,
            }
        finally:
            session.close()

    def get_market_data(self, campaign_type: CampaignType) -> Dict[str, Any]:
        """Get relevant market data and benchmarks"""
        # Search knowledge base for market data
        kb_results = search_knowledge_base(f"marketing {campaign_type.value}", limit=5)

        # Default benchmarks (in a real system, this would come from external APIs)
        benchmarks = {
            "email_marketing": {
                "open_rate": 0.22,
                "click_rate": 0.035,
                "conversion_rate": 0.02,
            },
            "social_media": {
                "engagement_rate": 0.045,
                "reach_rate": 0.15,
                "conversion_rate": 0.015,
            },
            "paid_ads": {"ctr": 0.025, "cpc": 1.50, "conversion_rate": 0.025},
        }

        return {"benchmarks": benchmarks, "knowledge_base": kb_results}

    def create_campaign_strategy(self, request: CampaignRequest) -> str:
        """Create a comprehensive campaign strategy"""
        # Get supporting data
        audience_data = self.get_audience_insights(request.target_audience)
        market_data = self.get_market_data(request.campaign_type)

        # Prepare context
        campaign_request_str = f"""
Campaign Type: {request.campaign_type.value}
Target Audience: {request.target_audience}
Budget: ${request.budget:,.2f}
Duration: {request.duration_days} days
Goals: {', '.join(request.goals)}
Channels: {', '.join(request.channels)}
Content Requirements: {', '.join([ct.value for ct in request.content_requirements])}
"""

        # Generate strategy
        strategy = self.campaign_chain.invoke(
            {
                "campaign_request": campaign_request_str,
                "market_data": json.dumps(market_data, indent=2),
                "audience_data": json.dumps(audience_data, indent=2),
            }
        )

        return strategy

    def generate_content(
        self,
        platform: str,
        content_type: ContentType,
        campaign_theme: str,
        target_audience: str,
        key_messages: List[str],
    ) -> ContentPiece:
        """Generate platform-optimized content"""
        # Get platform constraints
        constraints = self.platform_limits.get(platform, {})

        # Generate content
        content = self.content_chain.invoke(
            {
                "platform": platform,
                "content_type": content_type.value,
                "campaign_theme": campaign_theme,
                "target_audience": target_audience,
                "key_messages": ", ".join(key_messages),
                "platform_constraints": json.dumps(constraints),
            }
        )

        # Parse content (in a real system, you'd use structured output)
        lines = content.split("\n")
        title = lines[0] if lines else "Generated Content"
        main_content = "\n".join(lines[1:]) if len(lines) > 1 else content

        # Extract hashtags (simple extraction)
        hashtags = [word for word in content.split() if word.startswith("#")]

        # Generate estimated reach (simplified calculation)
        base_reach = {
            "twitter": 1000,
            "linkedin": 500,
            "facebook": 800,
            "instagram": 1200,
            "email": 2000,
        }.get(platform, 500)

        return ContentPiece(
            content_type=content_type,
            platform=platform,
            title=title,
            content=main_content,
            hashtags=hashtags[: constraints.get("hashtag_limit", 5)],
            call_to_action="Learn more",  # Would be extracted from content
            estimated_reach=base_reach,
        )

    def create_campaign_plan(self, request: CampaignRequest) -> CampaignPlan:
        """Create a complete campaign plan"""
        # Generate strategy
        strategy = self.create_campaign_strategy(request)

        # Generate content pieces
        content_pieces = []
        for channel in request.channels:
            for content_type in request.content_requirements:
                content_piece = self.generate_content(
                    platform=channel,
                    content_type=content_type,
                    campaign_theme=request.campaign_type.value,
                    target_audience=request.target_audience,
                    key_messages=request.goals,
                )
                content_pieces.append(content_piece)

        # Create schedule (simplified)
        schedule = []
        days_per_post = max(1, request.duration_days // len(content_pieces))
        for i, content in enumerate(content_pieces):
            schedule.append(
                {
                    "day": i * days_per_post + 1,
                    "content_id": i,
                    "platform": content.platform,
                    "content_type": content.content_type.value,
                    "estimated_reach": content.estimated_reach,
                }
            )

        # Budget allocation (simplified)
        budget_per_channel = request.budget / len(request.channels)
        budget_allocation = dict.fromkeys(request.channels, budget_per_channel)

        # Success metrics
        success_metrics = [
            "Reach and impressions",
            "Engagement rate",
            "Click-through rate",
            "Conversion rate",
            "Cost per acquisition",
            "Return on ad spend",
        ]

        # Estimated performance
        estimated_performance = {
            "total_reach": sum(cp.estimated_reach for cp in content_pieces),
            "estimated_engagement": sum(
                cp.estimated_reach * 0.045 for cp in content_pieces
            ),
            "estimated_conversions": sum(
                cp.estimated_reach * 0.02 for cp in content_pieces
            ),
            "estimated_roi": 2.5,
        }

        return CampaignPlan(
            campaign_id=f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            strategy=strategy,
            content_pieces=content_pieces,
            schedule=schedule,
            budget_allocation=budget_allocation,
            success_metrics=success_metrics,
            estimated_performance=estimated_performance,
        )

    def analyze_campaign_performance(self, campaign_id: int) -> str:
        """Analyze campaign performance and provide recommendations"""
        session = get_session()
        try:
            # Get campaign data
            campaign = (
                session.query(MarketingCampaign)
                .filter(MarketingCampaign.id == campaign_id)
                .first()
            )

            if not campaign:
                return "Campaign not found."

            # Get metrics
            metrics = (
                session.query(CampaignMetrics)
                .filter(CampaignMetrics.campaign_id == campaign_id)
                .all()
            )

            # Get posts
            _posts = (
                session.query(MarketingPost)
                .filter(MarketingPost.campaign_id == campaign_id)
                .all()
            )

            # Prepare data for analysis
            campaign_data = {
                "name": campaign.name,
                "type": campaign.campaign_type,
                "budget": float(campaign.budget),
                "start_date": (
                    campaign.start_date.isoformat() if campaign.start_date else None
                ),
                "end_date": (
                    campaign.end_date.isoformat() if campaign.end_date else None
                ),
                "status": campaign.status,
            }

            performance_metrics = [
                {
                    "metric_name": m.metric_name,
                    "metric_value": float(m.metric_value),
                    "date": m.date.isoformat() if m.date else None,
                }
                for m in metrics
            ]

            # Get benchmarks
            market_data = self.get_market_data(CampaignType(campaign.campaign_type))

            # Generate analysis
            analysis = self.analysis_chain.invoke(
                {
                    "campaign_data": json.dumps(campaign_data, indent=2),
                    "performance_metrics": json.dumps(performance_metrics, indent=2),
                    "benchmarks": json.dumps(market_data["benchmarks"], indent=2),
                    "campaign_goals": campaign.target_audience or "General audience",
                }
            )

            return analysis

        finally:
            session.close()

    def optimize_campaign(self, campaign_id: int) -> Dict[str, Any]:
        """Provide campaign optimization recommendations"""
        analysis = self.analyze_campaign_performance(campaign_id)

        # Extract optimization recommendations (simplified)
        recommendations = [
            "Increase budget allocation to top-performing channels",
            "A/B test different content variations",
            "Optimize posting times based on audience activity",
            "Refine targeting parameters for better reach",
            "Improve call-to-action messaging",
        ]

        return {
            "analysis": analysis,
            "recommendations": recommendations,
            "priority_actions": recommendations[:3],
            "estimated_improvement": "15-25% increase in performance",
        }


def test_marketing_agent():
    """Test the marketing agent with sample scenarios"""
    agent = MarketingAgent()

    print("=== Marketing Agent Test ===\n")

    # Test campaign creation
    print("1. Creating Product Launch Campaign:")
    request = CampaignRequest(
        campaign_type=CampaignType.PRODUCT_LAUNCH,
        target_audience="tech professionals",
        budget=10000.0,
        duration_days=30,
        goals=["increase brand awareness", "generate leads", "drive product adoption"],
        channels=["linkedin", "twitter", "email"],
        content_requirements=[ContentType.SOCIAL_MEDIA, ContentType.EMAIL],
    )

    campaign_plan = agent.create_campaign_plan(request)
    print(f"Campaign ID: {campaign_plan.campaign_id}")
    print(f"Strategy: {campaign_plan.strategy[:200]}...")
    print(f"Content Pieces: {len(campaign_plan.content_pieces)}")
    print(f"Estimated Reach: {campaign_plan.estimated_performance['total_reach']:,}")
    print("-" * 80)

    # Test content generation
    print("2. Generating Social Media Content:")
    content = agent.generate_content(
        platform="linkedin",
        content_type=ContentType.SOCIAL_MEDIA,
        campaign_theme="AI Analytics Platform Launch",
        target_audience="data scientists",
        key_messages=["powerful analytics", "easy integration", "real-time insights"],
    )
    print(f"Platform: {content.platform}")
    print(f"Title: {content.title}")
    print(f"Content: {content.content[:200]}...")
    print(f"Hashtags: {', '.join(content.hashtags)}")
    print("-" * 80)

    # Test campaign analysis
    print("3. Analyzing Campaign Performance:")
    analysis = agent.analyze_campaign_performance(1)  # Assuming campaign ID 1 exists
    print(f"Analysis: {analysis[:300]}...")
    print("-" * 80)


if __name__ == "__main__":
    test_marketing_agent()
