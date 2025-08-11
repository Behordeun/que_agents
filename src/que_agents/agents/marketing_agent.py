# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-05 16:00:00
# @Description: This module implements an enhanced marketing agent for autonomous campaign management with knowledge base integration

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

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
from src.que_agents.error_trace.errorlogger import system_logger
from src.que_agents.knowledge_base.kb_manager import search_agent_knowledge_base

system_logger.info("Initializing Marketing Agent ...")


DEFAULT_CALL_TO_ACTION = "Learn more"
GENERATED_CONTENT = "Generated content"
RETRY_CAMPAIGN_CREATION = "Retry campaign creation"
CAMPAIGN_STRATEGY_CREATION_STATUS = "Campaign strategy created successfully"
MONITOR_INITIAL_PERFORMANCE = "Monitor initial performance"

# Load agent configuration
try:
    with open("./configs/agent_config.yaml", "r") as f:
        agent_config = yaml.safe_load(f)
    system_logger.info("Agent configuration loaded successfully")
except Exception as e:
    system_logger.error(
        error=f"Error loading agent configuration: {e}",
        exc_info=True,
        additional_info={"file": "agent_config.yaml"},
    )
    # Fallback configuration
    agent_config = {"marketing_agent": {"model_name": "gpt-4", "temperature": 0.7}}
    system_logger.warning("Using fallback agent configuration")


class MarketingAgent:
    """Enhanced Marketing Agent for autonomous campaign management with AI-powered insights"""

    def __init__(self):
        system_logger._add_session_separator()
        system_logger.info("Initializing Marketing Agent")

        try:
            config = agent_config["marketing_agent"]
            self.llm = LLMFactory.get_llm(
                agent_type="marketing",
                model_name=config["model_name"],
                temperature=config["temperature"],
                max_tokens=800,
            )

            # Enhanced platform-specific constraints and features
            self.platform_limits = {
                "twitter": {
                    "max_chars": 280,
                    "hashtag_limit": 3,
                    "optimal_posting_times": ["9:00", "12:00", "18:00"],
                    "best_content_types": ["images", "videos", "polls"],
                    "engagement_multiplier": 1.2,
                },
                "linkedin": {
                    "max_chars": 3000,
                    "hashtag_limit": 5,
                    "optimal_posting_times": ["8:00", "12:00", "17:00"],
                    "best_content_types": [
                        "articles",
                        "professional_updates",
                        "videos",
                    ],
                    "engagement_multiplier": 0.8,
                },
                "facebook": {
                    "max_chars": 2000,
                    "hashtag_limit": 5,
                    "optimal_posting_times": ["9:00", "13:00", "15:00"],
                    "best_content_types": ["images", "videos", "events"],
                    "engagement_multiplier": 1.0,
                },
                "instagram": {
                    "max_chars": 2200,
                    "hashtag_limit": 10,
                    "optimal_posting_times": ["11:00", "14:00", "20:00"],
                    "best_content_types": ["images", "stories", "reels"],
                    "engagement_multiplier": 1.5,
                },
                "email": {
                    "subject_max": 50,
                    "body_max": 2000,
                    "optimal_sending_times": ["10:00", "14:00"],
                    "best_content_types": ["newsletters", "promotions", "updates"],
                    "engagement_multiplier": 2.0,
                },
                "youtube": {
                    "title_max": 100,
                    "description_max": 5000,
                    "hashtag_limit": 15,
                    "optimal_posting_times": ["14:00", "16:00", "18:00"],
                    "best_content_types": ["tutorials", "demos", "testimonials"],
                    "engagement_multiplier": 3.0,
                },
                "tiktok": {
                    "max_chars": 300,
                    "hashtag_limit": 5,
                    "optimal_posting_times": ["9:00", "12:00", "19:00"],
                    "best_content_types": ["short_videos", "trends", "challenges"],
                    "engagement_multiplier": 2.5,
                },
            }

            # Enhanced campaign types and strategies
            self.campaign_strategies = {
                "brand_awareness": {
                    "primary_metrics": ["reach", "impressions", "brand_mention"],
                    "recommended_channels": ["facebook", "instagram", "youtube"],
                    "content_focus": ["storytelling", "brand_values", "visual_content"],
                },
                "lead_generation": {
                    "primary_metrics": ["leads", "cost_per_lead", "conversion_rate"],
                    "recommended_channels": ["linkedin", "email", "google_ads"],
                    "content_focus": ["educational", "whitepapers", "webinars"],
                },
                "product_launch": {
                    "primary_metrics": ["awareness", "engagement", "sign_ups"],
                    "recommended_channels": ["twitter", "linkedin", "email", "youtube"],
                    "content_focus": ["features", "benefits", "demos"],
                },
                "customer_retention": {
                    "primary_metrics": ["engagement", "loyalty", "repeat_purchases"],
                    "recommended_channels": ["email", "in_app", "social_media"],
                    "content_focus": ["tips", "updates", "exclusive_content"],
                },
            }

            # Industry benchmarks (enhanced)
            self.industry_benchmarks = {
                "technology": {
                    "email_open_rate": 0.22,
                    "email_click_rate": 0.035,
                    "social_engagement": 0.045,
                    "conversion_rate": 0.025,
                },
                "healthcare": {
                    "email_open_rate": 0.25,
                    "email_click_rate": 0.038,
                    "social_engagement": 0.035,
                    "conversion_rate": 0.03,
                },
                "finance": {
                    "email_open_rate": 0.20,
                    "email_click_rate": 0.032,
                    "social_engagement": 0.025,
                    "conversion_rate": 0.022,
                },
                "retail": {
                    "email_open_rate": 0.18,
                    "email_click_rate": 0.025,
                    "social_engagement": 0.055,
                    "conversion_rate": 0.035,
                },
            }

            # Initialize enhanced prompt templates
            self.campaign_prompt = self._create_enhanced_campaign_prompt()
            self.content_prompt = self._create_enhanced_content_prompt()
            self.analysis_prompt = self._create_enhanced_analysis_prompt()
            self.optimization_prompt = self._create_optimization_prompt()
            self.audience_prompt = self._create_audience_analysis_prompt()

            # Create enhanced chains
            self.campaign_chain = self._create_campaign_chain()
            self.content_chain = self._create_content_chain()
            self.analysis_chain = self._create_analysis_chain()
            self.optimization_chain = self._create_optimization_chain()
            self.audience_chain = self._create_audience_chain()

            system_logger.info("Marketing Agent initialized successfully")
        except Exception as e:
            system_logger.error(
                error=f"Error initializing Marketing Agent: {e}",
                exc_info=True,
                additional_info={"config": agent_config},
            )
            raise RuntimeError("Marketing Agent initialization failed") from e

    def get_marketing_knowledge(self, query: str) -> List[Dict]:
        """Get marketing knowledge from knowledge base"""
        try:
            results = search_agent_knowledge_base("marketing", query, limit=3)
            system_logger.info(
                "Knowledge base query successful",
                additional_info={"query": query, "results_count": len(results)},
            )
            return results
        except Exception as e:
            system_logger.error(
                error=f"Error searching marketing knowledge: {e}",
                exc_info=True,
                additional_info={"query": query},
            )
            return []

    def get_enhanced_campaign_context(
        self, request: CampaignRequest, industry: Optional[str] = None
    ) -> str:
        """Get enhanced context from knowledge base for campaign planning"""
        try:
            # Search for campaign-specific knowledge
            campaign_knowledge = self.get_marketing_knowledge(
                f"{request.campaign_type.value} campaign strategy {request.target_audience}"
            )

            # Search for industry-specific knowledge
            industry_knowledge = []
            if industry:
                industry_knowledge = self.get_marketing_knowledge(
                    f"{industry} marketing best practices"
                )

            # Search for channel-specific knowledge
            channel_knowledge = self.get_marketing_knowledge(
                f"{' '.join(request.channels)} marketing optimization"
            )

            enhanced_context = ""
            if campaign_knowledge:
                enhanced_context += "Campaign Strategy Knowledge:\n"
                for kb_item in campaign_knowledge:
                    enhanced_context += (
                        f"- {kb_item['title']}: {kb_item['content'][:200]}...\n"
                    )

            if industry_knowledge:
                enhanced_context += f"\n{industry} Industry Insights:\n"
                for kb_item in industry_knowledge:
                    enhanced_context += (
                        f"- {kb_item['title']}: {kb_item['content'][:150]}...\n"
                    )

            if channel_knowledge:
                enhanced_context += "\nChannel Optimization Tips:\n"
                for kb_item in channel_knowledge:
                    enhanced_context += (
                        f"- {kb_item['title']}: {kb_item['content'][:150]}...\n"
                    )

            return enhanced_context
        except Exception as e:
            system_logger.error(
                error=f"Error getting enhanced campaign context: {e}",
                exc_info=True,
                additional_info={
                    "campaign_type": request.campaign_type.value,
                    "target_audience": request.target_audience,
                },
            )
            return ""

    def _create_enhanced_campaign_prompt(self) -> ChatPromptTemplate:
        """Create enhanced prompt template for campaign strategy with knowledge base integration"""
        system_message = """You are a world-class marketing strategist with expertise in digital marketing, data analytics, and customer psychology. Your role is to create comprehensive, data-driven marketing strategies that deliver exceptional ROI.

CORE COMPETENCIES:
- Multi-channel campaign orchestration
- Advanced audience segmentation and targeting
- Performance optimization and attribution modeling
- Creative strategy and brand positioning
- Conversion funnel optimization and customer journey mapping

STRATEGIC FRAMEWORK:
1. SITUATION ANALYSIS
   - Market landscape and competitive positioning
   - Customer behavior and preference analysis
   - Channel effectiveness and attribution modeling

2. STRATEGY DEVELOPMENT
   - Clear value proposition and messaging hierarchy
   - Audience segmentation with persona development
   - Channel mix optimization based on customer journey
   - Content strategy aligned with funnel stages

3. TACTICAL EXECUTION
   - Platform-specific content and creative requirements
   - Budget allocation with performance forecasting
   - Timeline with critical path milestones
   - A/B testing roadmap for continuous optimization

4. MEASUREMENT & OPTIMIZATION
   - KPI framework with leading and lagging indicators
   - Attribution modeling and customer lifetime value
   - Real-time optimization triggers and thresholds

ENHANCED CONSIDERATIONS:
- Leverage knowledge base insights for industry best practices
- Apply behavioral psychology principles for engagement
- Consider seasonal trends and market conditions
- Integrate emerging platforms and technologies
- Focus on sustainable growth and customer retention

Campaign Requirements: {campaign_request}
Industry Context: {industry_context}
Market Intelligence: {market_data}
Audience Research: {audience_data}
Knowledge Base Insights: {enhanced_context}
Competitive Analysis: {competitive_data}

Create a comprehensive marketing strategy that includes:
1. Executive Summary with key strategic recommendations
2. Situation Analysis and Market Opportunity
3. Target Audience Strategy with detailed personas
4. Multi-Channel Strategy and Channel Mix Rationale
5. Creative Strategy and Content Framework
6. Budget Allocation and ROI Projections
7. Implementation Timeline with Key Milestones
8. Success Metrics and Performance Tracking
9. Risk Assessment and Mitigation Strategies
10. Optimization Framework and Testing Roadmap"""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    "Create a comprehensive marketing campaign strategy based on the provided requirements and insights.",
                ),
            ]
        )

    def _create_enhanced_content_prompt(self) -> ChatPromptTemplate:
        """Create enhanced prompt template for content generation"""
        system_message = """You are an elite creative director and copywriter specializing in high-converting, platform-optimized content. Your expertise spans psychology-driven messaging, viral content mechanics, and conversion optimization.

CONTENT MASTERY AREAS:
- Persuasive copywriting with psychological triggers
- Platform algorithm optimization
- Visual storytelling and multimedia integration
- Conversion-focused call-to-action development
- Brand voice consistency across channels

CONTENT CREATION PRINCIPLES:
1. ATTENTION: Hook readers within first 3 seconds
2. INTEREST: Maintain engagement through storytelling
3. DESIRE: Build emotional connection and value proposition
4. ACTION: Clear, compelling call-to-action

PLATFORM OPTIMIZATION:
- Algorithm-friendly content structure
- Optimal posting times and frequency
- Platform-specific visual and format requirements
- Engagement mechanics (polls, questions, hashtags)
- Community building and conversation starters

PSYCHOLOGICAL TRIGGERS:
- Social proof and authority positioning
- Scarcity and urgency principles
- Loss aversion and fear of missing out
- Reciprocity and value-first approach
- Consistency and commitment psychology

Content Specifications:
Platform: {platform}
Content Type: {content_type}
Campaign Theme: {campaign_theme}
Target Audience: {target_audience}
Key Messages: {key_messages}
Platform Constraints: {platform_constraints}
Brand Voice: {brand_voice}
Competition Analysis: {competitor_content}
Knowledge Base Insights: {content_knowledge}

Generate content that includes:
1. Attention-grabbing headline/opening
2. Engaging body content with clear value
3. Strategic call-to-action with urgency
4. Relevant hashtags and keywords
5. Visual content suggestions
6. Engagement optimization elements
7. A/B testing variations
8. Performance prediction metrics"""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    "Create high-converting, platform-optimized content based on the specifications.",
                ),
            ]
        )

    def _create_enhanced_analysis_prompt(self) -> ChatPromptTemplate:
        """Create enhanced prompt template for campaign analysis"""
        system_message = """You are a senior marketing analytics consultant with expertise in performance measurement, attribution modeling, and data-driven optimization. Your role is to extract actionable insights from complex marketing data.

ANALYTICAL EXPERTISE:
- Multi-touch attribution and customer journey analysis
- Statistical significance testing and confidence intervals
- Cohort analysis and customer lifetime value modeling
- Marketing mix modeling and media effectiveness
- Predictive analytics and forecasting

ANALYSIS FRAMEWORK:
1. PERFORMANCE ASSESSMENT
   - Actual vs. planned performance analysis
   - Channel and campaign contribution analysis
   - Customer acquisition cost and lifetime value
   - Return on ad spend and marketing ROI

2. INSIGHT GENERATION
   - Trend identification and pattern recognition
   - Correlation analysis and causal relationships
   - Segment performance and audience insights
   - Creative and messaging effectiveness

3. OPTIMIZATION OPPORTUNITIES
   - Budget reallocation recommendations
   - Targeting and audience refinement
   - Creative and messaging improvements
   - Channel mix optimization

4. PREDICTIVE MODELING
   - Performance forecasting and scenario planning
   - Budget optimization and allocation modeling
   - Customer behavior prediction
   - Market trend anticipation

DATA SOURCES:
Campaign Performance: {campaign_data}
Channel Metrics: {channel_metrics}
Audience Analytics: {audience_analytics}
Conversion Data: {conversion_data}
Industry Benchmarks: {benchmarks}
Knowledge Base Insights: {analysis_insights}
Competitive Intelligence: {competitive_data}

Provide comprehensive analysis including:
1. Executive Summary with key findings
2. Overall Performance Assessment vs. Goals
3. Channel Performance Deep Dive
4. Audience Segment Analysis
5. Content Performance Insights
6. Conversion Funnel Analysis
7. ROI and Financial Impact Assessment
8. Optimization Recommendations (prioritized)
9. Predictive Insights and Forecasting
10. Strategic Recommendations for Future Campaigns"""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    "Analyze the campaign performance data and provide comprehensive insights and recommendations.",
                ),
            ]
        )

    def _create_optimization_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for campaign optimization"""
        system_message = """You are a performance marketing optimization specialist with expertise in real-time campaign improvement and ROI maximization.

OPTIMIZATION FOCUS AREAS:
- Budget reallocation for maximum ROI
- Audience targeting refinement
- Creative testing and iteration
- Channel mix optimization
- Conversion rate improvement

Current Performance: {current_performance}
Optimization Goals: {optimization_goals}
Available Budget: {available_budget}
Time Constraints: {time_constraints}

Provide specific, actionable optimization recommendations with expected impact and implementation timeline."""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    "Provide specific optimization recommendations for this campaign.",
                ),
            ]
        )

    def _create_audience_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for audience analysis"""
        system_message = """You are an audience research specialist with expertise in customer segmentation, persona development, and behavioral analysis.

AUDIENCE ANALYSIS FRAMEWORK:
- Demographic and psychographic profiling
- Behavioral pattern identification
- Channel preference mapping
- Content consumption analysis
- Purchase journey mapping

Audience Data: {audience_data}
Behavioral Metrics: {behavioral_metrics}
Channel Engagement: {channel_engagement}

Provide detailed audience insights and targeting recommendations."""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    "Analyze the audience data and provide targeting recommendations.",
                ),
            ]
        )

    def _create_campaign_chain(self):
        """Create enhanced campaign strategy chain"""
        return self.campaign_prompt | self.llm | StrOutputParser()

    def _create_content_chain(self):
        """Create enhanced content generation chain"""
        return self.content_prompt | self.llm | StrOutputParser()

    def _create_analysis_chain(self):
        """Create enhanced analysis chain"""
        return self.analysis_prompt | self.llm | StrOutputParser()

    def _create_optimization_chain(self):
        """Create optimization chain"""
        return self.optimization_prompt | self.llm | StrOutputParser()

    def _create_audience_chain(self):
        """Create audience analysis chain"""
        return self.audience_prompt | self.llm | StrOutputParser()

    def get_enhanced_audience_insights(
        self, target_audience: str, industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get enhanced audience insights from database and knowledge base with error handling"""
        session = get_session()
        try:
            # Search for audience segments in database
            segments = (
                session.query(AudienceSegment)
                .filter(AudienceSegment.name.ilike(f"%{target_audience}%"))
                .all()
            )

            # Create default segments if none found
            if not segments:
                default_segments_data = [
                    {
                        "name": f"{target_audience} - Early Adopters",
                        "criteria": {
                            "age_range": "25-40",
                            "tech_savvy": True,
                            "income": "high",
                        },
                        "characteristics": {
                            "engagement": "high",
                            "conversion_rate": 0.05,
                            "estimated_size": 10000,
                        },
                    },
                    {
                        "name": f"{target_audience} - Mainstream",
                        "criteria": {
                            "age_range": "30-55",
                            "tech_savvy": False,
                            "income": "medium",
                        },
                        "characteristics": {
                            "engagement": "medium",
                            "conversion_rate": 0.025,
                            "estimated_size": 50000,
                        },
                    },
                ]

                segments = []
                for segment_data in default_segments_data:
                    try:
                        segment = AudienceSegment(
                            name=segment_data["name"],
                            criteria=segment_data["criteria"],
                            characteristics=segment_data["characteristics"],
                        )
                        session.add(segment)
                        segments.append(segment)
                    except Exception as segment_error:
                        system_logger.error(f"Error creating segment: {segment_error}")

                        # Create a fallback segment object without saving to DB
                        class FallbackSegment:
                            def __init__(self, data):
                                self.name = data["name"]
                                self.criteria = data["criteria"]
                                self.characteristics = data["characteristics"]

                        segments.append(FallbackSegment(segment_data))

                try:
                    session.commit()
                except Exception as commit_error:
                    session.rollback()
                    system_logger.error(
                        f"Error saving segments to database: {commit_error}"
                    )

            # Search knowledge base for audience insights
            try:
                audience_kb = self.get_marketing_knowledge(
                    f"audience segmentation {target_audience}"
                )
            except Exception as kb_error:
                system_logger.error(f"Error accessing knowledge base: {kb_error}")
                audience_kb = []

            # Search for industry-specific audience data
            industry_audience_kb = []
            if industry:
                try:
                    industry_audience_kb = self.get_marketing_knowledge(
                        f"{industry} customer behavior"
                    )
                except Exception as industry_kb_error:
                    system_logger.error(
                        f"Error accessing industry knowledge base: {industry_kb_error}"
                    )

            # Enhanced audience analysis with error handling
            try:
                enhanced_insights = self._analyze_audience_behavior(
                    target_audience, segments
                )
            except Exception as analysis_error:
                system_logger.error(f"Error in audience analysis: {analysis_error}")
                enhanced_insights = {
                    "engagement_score": 0.5,
                    "behavioral_patterns": {},
                    "channel_preferences": {},
                    "content_preferences": {},
                }

            return {
                "segments": [
                    {
                        "name": s.name,
                        "criteria": getattr(s, "criteria", {}),
                        "size": getattr(s, "characteristics", {}).get(
                            "estimated_size", 10000
                        ),
                        "characteristics": getattr(s, "characteristics", {}),
                        "potential_reach": getattr(s, "characteristics", {}).get(
                            "estimated_size", 10000
                        ),
                        "engagement_score": enhanced_insights.get(
                            "engagement_score", 0.5
                        ),
                    }
                    for s in segments
                ],
                "knowledge_base_insights": audience_kb,
                "industry_insights": industry_audience_kb,
                "behavioral_patterns": enhanced_insights.get("behavioral_patterns", {}),
                "channel_preferences": enhanced_insights.get("channel_preferences", {}),
                "content_preferences": enhanced_insights.get("content_preferences", {}),
            }
        except Exception as e:
            system_logger.error(
                error=f"Error getting audience insights: {e}",
                exc_info=True,
                additional_info={"target_audience": target_audience},
            )
            # Return fallback data
            return {
                "segments": [
                    {
                        "name": f"{target_audience} - Default Segment",
                        "criteria": {
                            "age_range": "25-55",
                            "interests": target_audience,
                        },
                        "size": 25000,
                        "characteristics": {
                            "engagement": "medium",
                            "conversion_rate": 0.03,
                        },
                        "potential_reach": 25000,
                        "engagement_score": 0.5,
                    }
                ],
                "knowledge_base_insights": [],
                "industry_insights": [],
                "behavioral_patterns": {"primary_channels": ["social_media", "email"]},
                "channel_preferences": {"social_media": 0.6, "email": 0.4},
                "content_preferences": {
                    "educational": 0.5,
                    "promotional": 0.3,
                    "entertainment": 0.2,
                },
            }
        finally:
            session.close()

    def _analyze_audience_behavior(
        self, target_audience: str, _segments: List
    ) -> Dict[str, Any]:
        """Analyze audience behavior patterns"""
        audience_lower = target_audience.lower()

        # Behavioral analysis based on audience type
        behavior_patterns = {
            "tech": {
                "engagement_score": 0.7,
                "preferred_times": ["9:00-11:00", "14:00-16:00", "20:00-22:00"],
                "content_types": ["tutorials", "case_studies", "demos"],
                "channels": ["linkedin", "twitter", "youtube"],
            },
            "business": {
                "engagement_score": 0.6,
                "preferred_times": ["8:00-10:00", "12:00-14:00", "17:00-19:00"],
                "content_types": ["whitepapers", "webinars", "industry_reports"],
                "channels": ["linkedin", "email", "industry_publications"],
            },
            "consumer": {
                "engagement_score": 0.5,
                "preferred_times": ["11:00-13:00", "15:00-17:00", "19:00-21:00"],
                "content_types": ["videos", "infographics", "social_posts"],
                "channels": ["facebook", "instagram", "tiktok"],
            },
        }

        # Match audience type
        for key, patterns in behavior_patterns.items():
            if key in audience_lower:
                return {
                    "behavioral_patterns": patterns,
                    "engagement_score": patterns["engagement_score"],
                    "channel_preferences": dict.fromkeys(patterns["channels"], 0.8),
                    "content_preferences": dict.fromkeys(
                        patterns["content_types"], 0.9
                    ),
                }

        # Default patterns
        return {
            "behavioral_patterns": behavior_patterns["consumer"],
            "engagement_score": 0.5,
            "channel_preferences": {"facebook": 0.6, "email": 0.7},
            "content_preferences": {"social_posts": 0.6, "videos": 0.8},
        }

    def get_enhanced_market_data(
        self, campaign_type: CampaignType, industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get enhanced market data with knowledge base insights"""
        try:
            # Search knowledge base for market data
            market_kb = self.get_marketing_knowledge(
                f"market trends {campaign_type.value}"
            )

            # Search for industry-specific data
            industry_kb = []
            if industry:
                industry_kb = self.get_marketing_knowledge(
                    f"{industry} marketing trends"
                )

            # Get industry benchmarks
            industry_benchmarks = self.industry_benchmarks.get(
                industry.lower() if industry else "technology",
                self.industry_benchmarks["technology"],
            )

            # Enhanced market analysis
            market_trends = self._analyze_market_trends(
                campaign_type, industry if industry is not None else ""
            )
            competitive_landscape = self._analyze_competitive_landscape(campaign_type)

            return {
                "benchmarks": industry_benchmarks,
                "knowledge_base_insights": market_kb,
                "industry_insights": industry_kb,
                "market_trends": market_trends,
                "competitive_landscape": competitive_landscape,
                "growth_opportunities": self._identify_growth_opportunities(
                    campaign_type, industry if industry is not None else ""
                ),
                "risk_factors": self._identify_risk_factors(
                    campaign_type, industry if industry is not None else ""
                ),
            }
        except Exception as e:
            system_logger.error(f"Error getting enhanced market data: {e}")
            return {"benchmarks": self.industry_benchmarks["technology"]}

    def _analyze_market_trends(
        self, campaign_type: CampaignType, _industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze current market trends"""
        # Simulate market trend analysis
        base_trends = {
            "growth_rate": 0.15,
            "seasonality": "Q4 peak",
            "emerging_channels": ["tiktok", "clubhouse", "linkedin_audio"],
            "declining_channels": ["facebook_organic"],
            "content_trends": [
                "short_form_video",
                "interactive_content",
                "personalization",
            ],
        }

        # Adjust based on campaign type
        if campaign_type == CampaignType.PRODUCT_LAUNCH:
            base_trends["optimal_duration"] = "6-8 weeks"
            base_trends["budget_allocation"] = {
                "paid_media": 0.6,
                "content": 0.3,
                "influencer": 0.1,
            }
        elif campaign_type == CampaignType.BRAND_AWARENESS:
            base_trends["optimal_duration"] = "12-16 weeks"
            base_trends["budget_allocation"] = {
                "paid_media": 0.4,
                "content": 0.4,
                "pr": 0.2,
            }

        return base_trends

    def _analyze_competitive_landscape(
        self, _campaign_type: CampaignType
    ) -> Dict[str, Any]:
        """Analyze competitive landscape"""
        return {
            "market_saturation": "medium",
            "key_competitors": ["Competitor A", "Competitor B", "Competitor C"],
            "competitive_advantage_opportunities": [
                "unique_value_proposition",
                "superior_customer_service",
                "innovative_technology",
            ],
            "market_gaps": ["underserved_segments", "emerging_use_cases"],
            "differentiation_strategies": [
                "focus_on_specific_industry",
                "superior_user_experience",
                "competitive_pricing",
            ],
        }

    def _identify_growth_opportunities(
        self, campaign_type: CampaignType, _industry: Optional[str] = None
    ) -> List[str]:
        """Identify growth opportunities"""
        opportunities = [
            "Emerging social media platforms",
            "Voice search optimization",
            "AI-powered personalization",
            "Interactive content experiences",
            "Community building initiatives",
        ]

        if campaign_type == CampaignType.PRODUCT_LAUNCH:
            opportunities.extend(
                [
                    "Influencer partnerships",
                    "Product demo events",
                    "Early adopter programs",
                ]
            )

        return opportunities

    def _identify_risk_factors(
        self, _campaign_type: CampaignType, industry: Optional[str] = None
    ) -> List[str]:
        """Identify potential risk factors"""
        risks = [
            "Platform algorithm changes",
            "Increased competition",
            "Economic downturn impact",
            "Privacy regulation changes",
            "Seasonal demand fluctuations",
        ]

        if industry == "healthcare":
            risks.append("Regulatory compliance requirements")
        elif industry == "finance":
            risks.append("Financial services regulations")

        return risks

    def create_enhanced_campaign_strategy(
        self, request: CampaignRequest, industry: Optional[str] = None
    ) -> str:
        """Create a comprehensive campaign strategy with knowledge base insights"""
        try:
            # Get enhanced supporting data
            audience_data = self.get_enhanced_audience_insights(
                request.target_audience, industry
            )

            # FIXED: Safe campaign type handling
            campaign_type_value = self._get_safe_campaign_type_string(
                request.campaign_type
            )

            market_data = self.get_enhanced_market_data(request.campaign_type, industry)
            enhanced_context = self.get_enhanced_campaign_context(request, industry)

            # Prepare comprehensive context
            campaign_request_str = f"""
    Campaign Type: {campaign_type_value}
    Target Audience: {request.target_audience}
    Industry: {industry or 'General'}
    Budget: ${request.budget:,.2f}
    Duration: {request.duration_days} days
    Goals: {', '.join(request.goals)}
    Channels: {', '.join(request.channels)}
    Content Requirements: {', '.join([self._get_safe_content_type_string(ct) for ct in request.content_requirements])}
    Priority Metrics: {self.campaign_strategies.get(campaign_type_value, {}).get('primary_metrics', [])}
    """

            # Generate enhanced strategy
            strategy = self.campaign_chain.invoke(
                {
                    "campaign_request": campaign_request_str,
                    "industry_context": industry or "Technology",
                    "market_data": json.dumps(market_data, indent=2),
                    "audience_data": json.dumps(audience_data, indent=2),
                    "enhanced_context": enhanced_context,
                    "competitive_data": json.dumps(
                        market_data.get("competitive_landscape", {}), indent=2
                    ),
                }
            )

            system_logger.info(
                "Campaign strategy created successfully",
                additional_info={
                    "campaign_type": campaign_type_value,
                    "target_audience": request.target_audience,
                },
            )
            return strategy
        except Exception as e:
            # FIXED: Safe error logging
            campaign_type_value = self._get_safe_campaign_type_string(
                request.campaign_type
            )

            system_logger.error(
                error=f"Error creating enhanced campaign strategy: {e}",
                exc_info=True,
                additional_info={
                    "campaign_type": campaign_type_value,
                    "target_audience": request.target_audience,
                    "budget": request.budget,
                },
            )
            # Fallback to basic strategy
            return self._create_basic_strategy(request)

    def _get_safe_content_type_string(self, content_type) -> str:
        """Safely extract content type string from enum or string"""
        if content_type is None:
            return "social_media"

        # If it's already a string
        if isinstance(content_type, str):
            return content_type

        # If it has a .value attribute (enum)
        if hasattr(content_type, "value"):
            return str(content_type.value)

        # Fallback to string conversion
        return str(content_type)

    def _create_basic_strategy(self, request: CampaignRequest) -> str:
        """Fallback basic strategy creation with safe type handling"""
        # FIXED: Safe campaign type access
        campaign_type_str = self._get_safe_campaign_type_string(request.campaign_type)

        # FIXED: Safe content requirements processing
        content_requirements_str = ", ".join(
            [
                self._get_safe_content_type_string(ct)
                for ct in request.content_requirements
            ]
        )

        return f"""
    CAMPAIGN STRATEGY OVERVIEW

    Campaign Type: {campaign_type_str}
    Target Audience: {request.target_audience}
    Budget: ${request.budget:,.2f}
    Duration: {request.duration_days} days

    RECOMMENDED APPROACH:
    1. Multi-channel approach across {', '.join(request.channels)}
    2. Content mix including {content_requirements_str}
    3. Focus on {', '.join(request.goals)}
    4. Budget allocation: Equal distribution across channels
    5. Weekly performance reviews and optimizations

    This strategy provides a solid foundation for achieving your campaign objectives.
    """

    def generate_enhanced_content(
        self,
        platform: str,
        content_type: ContentType,
        campaign_theme: str,
        target_audience: str,
        key_messages: List[str],
        brand_voice: str = "professional",
        competitor_analysis: Optional[List[Dict]] = None,
    ) -> ContentPiece:
        """Generate enhanced platform-optimized content with knowledge base insights"""
        try:
            # Get platform constraints and optimization data
            platform_data = self.platform_limits.get(platform, {})

            # Get content-specific knowledge
            content_knowledge = self.get_marketing_knowledge(
                f"{platform} {content_type.value} best practices"
            )

            # Prepare enhanced context
            content_context = ""
            if content_knowledge:
                content_context = "Content Best Practices:\n"
                for kb_item in content_knowledge:
                    content_context += (
                        f"- {kb_item['title']}: {kb_item['content'][:150]}...\n"
                    )

            # Generate content using enhanced prompt
            content = self.content_chain.invoke(
                {
                    "platform": platform,
                    "content_type": content_type.value,
                    "campaign_theme": campaign_theme,
                    "target_audience": target_audience,
                    "key_messages": ", ".join(key_messages),
                    "platform_constraints": json.dumps(platform_data),
                    "brand_voice": brand_voice,
                    "competitor_content": json.dumps(competitor_analysis or {}),
                    "content_knowledge": content_context,
                }
            )

            # Enhanced content parsing
            parsed_content = self._parse_generated_content(
                content, platform, content_type
            )

            # Calculate enhanced reach estimation
            estimated_reach = self._calculate_enhanced_reach(
                platform, target_audience, len(key_messages), platform_data
            )

            # Generate content variations for A/B testing
            variations = self._generate_content_variations(parsed_content, platform)

            system_logger.info(
                "Content generated successfully",
                additional_info={
                    "platform": platform,
                    "content_type": content_type.value,
                    "estimated_reach": estimated_reach,
                },
            )
            return ContentPiece(
                content_type=content_type,
                platform=platform,
                title=parsed_content.get("title", GENERATED_CONTENT),
                content=parsed_content.get("content", ""),
                call_to_action=parsed_content.get("cta", DEFAULT_CALL_TO_ACTION),
                hashtags=parsed_content.get("hashtags", []),
                estimated_reach=estimated_reach,
                variations=variations,
                optimization_score=self._calculate_content_score(
                    parsed_content, platform_data
                ),
            )

        except Exception as e:
            system_logger.error(
                error=e,
                exc_info=True,
                additional_info={
                    "method": "generate_enhanced_content",
                    "platform": platform,
                    "content_type": content_type.value,
                    "campaign_theme": campaign_theme,
                },
            )
            return self._generate_fallback_content(
                platform, content_type, campaign_theme
            )

    def _parse_generated_content(
        self, content: str, platform: str, _content_type: ContentType
    ) -> Dict[str, Any]:
        """Parse and structure generated content"""
        lines = content.split("\n")

        # Extract title/headline
        title = lines[0].strip() if lines else GENERATED_CONTENT
        if title.startswith("Title:") or title.startswith("Headline:"):
            title = title.split(":", 1)[1].strip()

        cta = DEFAULT_CALL_TO_ACTION
        content_lines = []
        hashtags = []

        for line in lines[1:]:
            line = line.strip()
            if line.startswith("CTA:") or line.startswith("Call to Action:"):
                cta = line.split(":", 1)[1].strip()
            elif line.startswith("#"):
                hashtags.extend(
                    [tag.strip() for tag in line.split() if tag.startswith("#")]
                )
            elif line and not line.startswith(
                ("Title:", "Headline:", "CTA:", "Hashtags:")
            ):
                content_lines.append(line)

        main_content = "\n".join(content_lines).strip()

        # Extract hashtags from content if not explicitly provided
        if not hashtags:
            hashtag_pattern = r"#\w+"
            hashtags = re.findall(hashtag_pattern, content)

        # Limit hashtags based on platform
        platform_data = self.platform_limits.get(platform, {})
        hashtag_limit = platform_data.get("hashtag_limit", 5)
        hashtags = hashtags[:hashtag_limit]

        return {
            "title": title,
            "content": main_content,
            "cta": cta,
            "hashtags": hashtags,
        }

    def _calculate_enhanced_reach(
        self,
        platform: str,
        target_audience: str,
        message_count: int,
        platform_data: Dict,
    ) -> int:
        """Calculate enhanced reach estimation"""
        # Base reach calculations
        base_reach = {
            "twitter": 1500,
            "linkedin": 800,
            "facebook": 1200,
            "instagram": 2000,
            "email": 3000,
            "youtube": 5000,
            "tiktok": 3500,
        }.get(platform, 1000)

        # Audience size multiplier
        audience_multipliers = {
            "tech": 1.2,
            "business": 1.0,
            "consumer": 1.5,
            "healthcare": 0.8,
            "finance": 0.9,
        }

        audience_multiplier = 1.0
        for key, multiplier in audience_multipliers.items():
            if key in target_audience.lower():
                audience_multiplier = multiplier
                break

        # Platform engagement multiplier
        engagement_multiplier = platform_data.get("engagement_multiplier", 1.0)

        # Message quality multiplier
        message_multiplier = min(1.5, 1.0 + (message_count * 0.1))

        estimated_reach = int(
            base_reach
            * audience_multiplier
            * engagement_multiplier
            * message_multiplier
        )
        return estimated_reach

    def _generate_content_variations(
        self, content: Dict[str, Any], _platform: str
    ) -> List[Dict[str, Any]]:
        """Generate A/B testing variations"""
        variations = []

        # Title variations
        original_title = content.get("title", "")
        title_variations = [
            f"🚀 {original_title}",
            f"{original_title} - Don't Miss Out!",
            f"BREAKING: {original_title}",
        ]

        # CTA variations
        cta_variations = [
            "Get started today",
            "Try it free",
            "Book a demo",
            "Download now",
        ]

        for i, (title_var, cta_var) in enumerate(
            zip(title_variations[:2], cta_variations[:2])
        ):
            variations.append(
                {
                    "variant_id": f"var_{i+1}",
                    "title": title_var,
                    "content": content.get("content", ""),
                    "cta": cta_var,
                    "hashtags": content.get("hashtags", []),
                }
            )

        return variations

    def _calculate_content_score(
        self, content: Dict[str, Any], platform_data: Dict
    ) -> float:
        """Calculate content optimization score"""
        score = 0.5  # Base score

        # Title/headline quality
        title = content.get("title", "")
        if len(title) > 10 and len(title) < 60:
            score += 0.1
        if any(
            word in title.lower() for word in ["new", "free", "exclusive", "limited"]
        ):
            score += 0.1

        # Content length optimization
        content_text = content.get("content", "")
        max_chars = platform_data.get("max_chars", 1000)
        if 0.7 * max_chars <= len(content_text) <= 0.9 * max_chars:
            score += 0.1

        # CTA presence and quality
        cta = content.get("cta", "")
        if cta and any(
            word in cta.lower() for word in ["learn", "get", "try", "download", "book"]
        ):
            score += 0.1

        # Hashtag optimization
        hashtags = content.get("hashtags", [])
        hashtag_limit = platform_data.get("hashtag_limit", 5)
        if len(hashtags) > 0 and len(hashtags) <= hashtag_limit:
            score += 0.1

        return min(1.0, score)

    def _generate_fallback_content(
        self, platform: str, content_type, campaign_theme: str
    ) -> ContentPiece:
        """Generate fallback content when enhanced generation fails with safe type handling"""
        from src.que_agents.core.schemas import ContentType

        # FIXED: Safe content type handling
        if isinstance(content_type, str):
            try:
                content_type_enum = ContentType(content_type)
            except ValueError:
                content_type_enum = ContentType.SOCIAL_MEDIA
        elif hasattr(content_type, "value"):
            content_type_enum = content_type
        else:
            content_type_enum = ContentType.SOCIAL_MEDIA

        return ContentPiece(
            content_type=content_type_enum,
            platform=platform,
            title=f"Exciting {campaign_theme} Update",
            content=f"""
    🚀 Exciting news about {campaign_theme}!

    We're thrilled to share something amazing with you. Our latest innovation is designed to transform your experience and deliver exceptional value.

    Ready to learn more? Let's connect and explore the possibilities together.

    #Innovation #Technology #Growth
    """,
            call_to_action="Learn more",
            hashtags=["#Innovation", "#Technology", "#Growth"],
            estimated_reach=1000,
            variations=[],
            optimization_score=0.6,
        )

    def create_enhanced_campaign_plan(
        self,
        request: CampaignRequest,
        industry: Optional[List[str]] = None,
        brand_voice: str = "professional",
    ) -> CampaignPlan:
        """Create a comprehensive campaign plan with enhanced features"""
        try:
            # FIXED: Safe campaign_type handling
            campaign_type_str = self._get_safe_campaign_type_string(
                request.campaign_type
            )

            system_logger.info(
                "Creating enhanced campaign plan",
                additional_info={
                    "campaign_type": campaign_type_str,
                    "target_audience": request.target_audience,
                    "budget": request.budget,
                    "duration_days": request.duration_days,
                },
            )

            industry_str = self._extract_industry_string(industry)
            strategy = self.create_enhanced_campaign_strategy(request, industry_str)

            content_pieces = self._generate_content_pieces(
                request, brand_voice, campaign_type_str
            )

            schedule = self._create_optimized_schedule(
                content_pieces, request.duration_days
            )

            budget_allocation = self._create_enhanced_budget_allocation(
                request, content_pieces
            )

            success_metrics = self._define_success_metrics(campaign_type_str, request.goals)

            industry_str = self._extract_industry_string(industry, default="technology")
            estimated_performance = self._calculate_enhanced_performance(
                content_pieces, request, industry_str
            )

            risk_assessment = self._assess_campaign_risks(request, industry_str)

            optimization_roadmap = self._create_optimization_roadmap(
                request.duration_days
            )

            industry_value = self._extract_industry_value(industry)

            campaign_plan = self._build_campaign_plan(
                strategy,
                content_pieces,
                schedule,
                budget_allocation,
                success_metrics,
                estimated_performance,
                risk_assessment,
                optimization_roadmap,
                industry_value,
                brand_voice,
            )

            system_logger.info(
                "Campaign plan created successfully",
                additional_info={
                    "campaign_id": campaign_plan.campaign_id,
                    "content_pieces": len(content_pieces),
                    "schedule_items": len(schedule),
                },
            )
            return campaign_plan

        except Exception as e:
            # FIXED: Safe error logging with campaign type
            campaign_type_str = self._get_safe_campaign_type_string(
                request.campaign_type
            )

            system_logger.error(
                error=f"Error creating enhanced campaign plan: {e}",
                exc_info=True,
                additional_info={
                    "campaign_type": campaign_type_str,
                    "target_audience": request.target_audience,
                },
            )
            return self._create_basic_campaign_plan(request)

    def _get_safe_campaign_type_string(self, campaign_type) -> str:
        """Safely extract campaign type string from enum or string"""
        if campaign_type is None:
            return "unknown"

        # If it's already a string
        if isinstance(campaign_type, str):
            return campaign_type

        # If it has a .value attribute (enum)
        if hasattr(campaign_type, "value"):
            return str(campaign_type.value)

        # Fallback to string conversion
        return str(campaign_type)

    def _extract_industry_string(self, industry, default=None):
        """Helper to extract industry string from list or str, with optional default."""
        if isinstance(industry, list):
            return industry[0] if industry else default
        return industry if industry is not None else default

    def _extract_industry_value(self, industry):
        """Helper to extract industry value for campaign plan."""
        if isinstance(industry, list) and industry:
            return industry[0]
        elif isinstance(industry, str) or industry is None:
            return industry
        else:
            return str(industry)

    def _generate_content_pieces(self, request, brand_voice, campaign_type_str=None):
        """Helper to generate content pieces for campaign plan."""
        content_pieces = []

        # Use the safe campaign type string
        if campaign_type_str is None:
            campaign_type_str = (
                request.campaign_type.value
                if hasattr(request.campaign_type, "value")
                else str(request.campaign_type)
            )

        for channel in request.channels:
            for content_type in request.content_requirements:
                content_piece = self.generate_enhanced_content(
                    platform=channel,
                    content_type=content_type,
                    campaign_theme=campaign_type_str,
                    target_audience=request.target_audience,
                    key_messages=request.goals,
                    brand_voice=brand_voice,
                )
                content_pieces.append(content_piece)
        return content_pieces

    def _build_campaign_plan(
        self,
        strategy,
        content_pieces,
        schedule,
        budget_allocation,
        success_metrics,
        estimated_performance,
        risk_assessment,
        optimization_roadmap,
        industry_value,
        brand_voice,
    ):
        """Helper to build CampaignPlan object."""
        return CampaignPlan(
            campaign_id=f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            strategy=strategy,
            content_pieces=content_pieces,
            schedule=schedule,
            budget_allocation=budget_allocation,
            success_metrics=success_metrics,
            estimated_performance=estimated_performance,
            risk_assessment=risk_assessment,
            optimization_roadmap=optimization_roadmap,
            industry=industry_value,
            brand_voice=brand_voice,
        )

    def _create_optimized_schedule(
        self, content_pieces: List[ContentPiece], duration_days: int
    ) -> List[Dict[str, Any]]:
        """Create an optimized posting schedule"""
        schedule = []

        # Group content by platform for optimal timing
        platform_content = {}
        for i, content in enumerate(content_pieces):
            if content.platform not in platform_content:
                platform_content[content.platform] = []
            platform_content[content.platform].append((i, content))

        current_day = 1

        for platform, content_list in platform_content.items():
            platform_data = self.platform_limits.get(platform, {})
            optimal_times = platform_data.get("optimal_posting_times", ["12:00"])

            # Distribute content across duration
            days_between_posts = max(1, duration_days // len(content_list))

            for i, (content_id, content) in enumerate(content_list):
                post_day = min(duration_days, current_day + (i * days_between_posts))
                optimal_time = optimal_times[i % len(optimal_times)]

                schedule.append(
                    {
                        "day": post_day,
                        "time": optimal_time,
                        "content_id": content_id,
                        "platform": content.platform,
                        "content_type": content.content_type.value,
                        "estimated_reach": content.estimated_reach,
                        "optimization_score": getattr(
                            content, "optimization_score", 0.6
                        ),
                    }
                )

        return sorted(schedule, key=lambda x: (x["day"], x["time"]))

    def _create_enhanced_budget_allocation(
        self, request: CampaignRequest, content_pieces: List[ContentPiece]
    ) -> Dict[str, float]:
        """Create enhanced budget allocation based on performance potential"""
        total_budget = request.budget

        # Calculate platform performance scores
        platform_scores = {}
        for content in content_pieces:
            platform = content.platform
            if platform not in platform_scores:
                platform_scores[platform] = []

            # Score based on reach and optimization
            score = (content.estimated_reach / 1000) * getattr(
                content, "optimization_score", 0.6
            )
            platform_scores[platform].append(score)

        # Average scores per platform
        platform_avg_scores = {
            platform: sum(scores) / len(scores)
            for platform, scores in platform_scores.items()
        }

        # Allocate budget proportionally
        total_score = sum(platform_avg_scores.values())
        budget_allocation = {}

        for platform, score in platform_avg_scores.items():
            allocation_percentage = score / total_score
            budget_allocation[platform] = total_budget * allocation_percentage

        return budget_allocation

    def _define_success_metrics(self, campaign_type, goals: List[str]) -> List[str]:
        """Define comprehensive success metrics with safe type handling"""
        # Base metrics
        base_metrics = [
            "Reach and impressions",
            "Engagement rate",
            "Click-through rate",
            "Conversion rate",
            "Cost per acquisition",
            "Return on ad spend",
        ]

        # FIXED: Safe campaign type handling
        campaign_type_str = self._get_safe_campaign_type_string(campaign_type)

        # Campaign-specific metrics
        campaign_specific = self.campaign_strategies.get(campaign_type_str, {}).get(
            "primary_metrics", []
        )

        # Goal-specific metrics
        goal_metrics = []
        for goal in goals:
            if "awareness" in goal.lower():
                goal_metrics.extend(["Brand awareness lift", "Share of voice"])
            elif "lead" in goal.lower():
                goal_metrics.extend(["Lead quality score", "Sales qualified leads"])
            elif "retention" in goal.lower():
                goal_metrics.extend(["Customer lifetime value", "Retention rate"])

        # Combine and deduplicate
        all_metrics = list(set(base_metrics + campaign_specific + goal_metrics))
        return all_metrics

    def _calculate_enhanced_performance(
        self,
        content_pieces: List[ContentPiece],
        request: CampaignRequest,
        industry: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate enhanced performance estimates"""
        total_reach = sum(content.estimated_reach for content in content_pieces)

        # Industry-specific conversion rates
        industry_data = self.industry_benchmarks.get(
            industry.lower() if industry else "technology",
            self.industry_benchmarks["technology"],
        )

        # Calculate estimates
        estimated_engagement = total_reach * industry_data.get(
            "social_engagement", 0.045
        )
        estimated_clicks = estimated_engagement * 0.1  # 10% of engaged users click
        estimated_conversions = estimated_clicks * industry_data.get(
            "conversion_rate", 0.025
        )
        estimated_revenue = estimated_conversions * 100  # Average order value
        estimated_roi = (estimated_revenue - request.budget) / request.budget

        return {
            "total_reach": int(total_reach),
            "estimated_engagement": int(estimated_engagement),
            "estimated_clicks": int(estimated_clicks),
            "estimated_conversions": int(estimated_conversions),
            "estimated_revenue": estimated_revenue,
            "estimated_roi": estimated_roi,
            "confidence_interval": "±15%",
            "optimization_potential": "25-40% improvement with optimization",
        }

    def _assess_campaign_risks(
        self, request: CampaignRequest, industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Assess campaign risks and mitigation strategies with safe type handling"""
        # FIXED: Safe campaign type handling
        campaign_type_str = self._get_safe_campaign_type_string(request.campaign_type)

        # Convert string back to enum for _identify_risk_factors if needed
        try:
            from src.que_agents.core.schemas import CampaignType

            campaign_type_enum = CampaignType(campaign_type_str)
        except (ValueError, AttributeError):
            # Use the original if conversion fails
            campaign_type_enum = request.campaign_type

        risks = self._identify_risk_factors(campaign_type_enum, industry)

        risk_assessment = {
            "high_risk": [],
            "medium_risk": [],
            "low_risk": [],
            "mitigation_strategies": {},
        }

        # Categorize risks
        for risk in risks:
            if "algorithm" in risk.lower() or "regulation" in risk.lower():
                risk_assessment["high_risk"].append(risk)
                risk_assessment["mitigation_strategies"][
                    risk
                ] = "Diversify channels and maintain compliance monitoring"
            elif "competition" in risk.lower() or "economic" in risk.lower():
                risk_assessment["medium_risk"].append(risk)
                risk_assessment["mitigation_strategies"][
                    risk
                ] = "Competitive differentiation and flexible budget allocation"
            else:
                risk_assessment["low_risk"].append(risk)
                risk_assessment["mitigation_strategies"][
                    risk
                ] = "Regular monitoring and quick response protocols"

        return risk_assessment

    def _safe_enum_to_string(self, enum_or_string, fallback: str = "unknown") -> str:
        """Universal safe conversion from enum or string to string"""
        if enum_or_string is None:
            return fallback

        if isinstance(enum_or_string, str):
            return enum_or_string

        if hasattr(enum_or_string, "value"):
            return str(enum_or_string.value)

        return str(enum_or_string)

    def _create_optimization_roadmap(self, duration_days: int) -> List[Dict[str, Any]]:
        """Create optimization roadmap with milestones"""
        roadmap = []

        # Week 1: Initial optimization
        roadmap.append(
            {
                "week": 1,
                "focus": "Performance baseline establishment",
                "actions": [
                    MONITOR_INITIAL_PERFORMANCE,
                    "Identify top performers",
                    "Flag underperformers",
                ],
                "kpis": ["Reach", "Engagement", "CTR"],
            }
        )

        # Week 2: First optimization
        if duration_days > 7:
            roadmap.append(
                {
                    "week": 2,
                    "focus": "Content and targeting optimization",
                    "actions": [
                        "A/B testing implementation",
                        "Audience refinement",
                        "Budget reallocation",
                    ],
                    "kpis": ["Conversion rate", "CPA", "ROAS"],
                }
            )

        # Week 3+: Advanced optimization
        if duration_days > 14:
            roadmap.append(
                {
                    "week": 3,
                    "focus": "Advanced optimization and scaling",
                    "actions": [
                        "Lookalike audience creation",
                        "Creative refresh",
                        "Channel expansion",
                    ],
                    "kpis": ["Customer LTV", "Brand awareness", "Market share"],
                }
            )

        return roadmap

    def _create_basic_campaign_plan(self, request: CampaignRequest) -> CampaignPlan:
        """Create basic campaign plan as fallback"""
        # Safe campaign type access
        campaign_type_str = (
            request.campaign_type.value
            if hasattr(request.campaign_type, "value")
            else str(request.campaign_type)
        )

        # Generate basic content pieces
        content_pieces = []
        for channel in request.channels:
            content_piece = self._generate_fallback_content(
                channel, ContentType.SOCIAL_MEDIA, campaign_type_str
            )
            content_pieces.append(content_piece)

        return CampaignPlan(
            campaign_id=f"basic_campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            strategy="Basic campaign strategy focused on multi-channel approach",
            content_pieces=content_pieces,
            schedule=[
                {"day": i + 1, "platform": content.platform}
                for i, content in enumerate(content_pieces)
            ],
            budget_allocation=dict.fromkeys(
                request.channels, request.budget / len(request.channels)
            ),
            success_metrics=["Reach", "Engagement", "Conversions"],
            estimated_performance={"total_reach": 5000, "estimated_roi": 2.0},
        )

    def analyze_enhanced_campaign_performance(
        self, campaign_id, _include_predictive: bool = True
    ) -> str:
        """Analyze campaign performance with enhanced insights and predictions"""
        session = get_session()
        try:
            # Convert string campaign_id to int if needed
            if isinstance(campaign_id, str):
                try:
                    campaign_id_int = int(campaign_id.replace('campaign_', '').replace('fallback_', ''))
                except (ValueError, AttributeError):
                    return f"Invalid campaign ID format: {campaign_id}. Please use a numeric campaign ID."
            else:
                campaign_id_int = campaign_id
            
            # Get campaign data
            campaign = (
                session.query(MarketingCampaign)
                .filter(MarketingCampaign.id == campaign_id_int)
                .first()
            )

            if not campaign:
                return "Campaign not found. Please verify the campaign ID."

            # Get comprehensive metrics
            metrics = (
                session.query(CampaignMetrics)
                .filter(CampaignMetrics.campaign_id == campaign_id_int)
                .all()
            )

            # Get posts and content performance
            posts = (
                session.query(MarketingPost)
                .filter(MarketingPost.campaign_id == campaign_id_int)
                .all()
            )

            # Prepare enhanced data for analysis
            campaign_data = {
                "name": campaign.name,
                "type": campaign.campaign_type,
                "budget": float(getattr(campaign, "budget", 0)),
                "start_date": (
                    campaign.start_date.isoformat()
                    if campaign.start_date is not None
                    else None
                ),
                "end_date": (
                    campaign.end_date.isoformat()
                    if campaign.end_date is not None
                    else None
                ),
                "status": campaign.status,
                "target_audience": campaign.target_audience,
            }

            # Enhanced metrics analysis
            performance_data = self._analyze_performance_metrics(metrics)
            channel_analysis = self._analyze_channel_performance(posts)
            audience_insights = self._analyze_audience_performance(
                campaign_data["target_audience"]
            )

            # Get industry benchmarks for comparison
            market_data = self.get_enhanced_market_data(
                CampaignType(campaign.campaign_type),
                getattr(campaign, "industry", None),
            )

            # Generate comprehensive analysis
            analysis = self.analysis_chain.invoke(
                {
                    "campaign_data": json.dumps(campaign_data, indent=2),
                    "channel_metrics": json.dumps(performance_data, indent=2),
                    "audience_analytics": json.dumps(audience_insights, indent=2),
                    "conversion_data": json.dumps(channel_analysis, indent=2),
                    "benchmarks": json.dumps(
                        market_data.get("benchmarks", {}), indent=2
                    ),
                    "analysis_insights": "Enhanced analysis with AI-powered insights",
                }
            )

            system_logger.info(
                "Campaign analysis completed",
                additional_info={
                    "campaign_id": campaign_id_int,
                    "campaign_name": campaign.name,
                },
            )
            return analysis

        except Exception as e:
            system_logger.error(
                error=f"Error analyzing campaign performance: {e}",
                exc_info=True,
                additional_info={"campaign_id": campaign_id},
            )
            return f"Error analyzing campaign {campaign_id}. Please try again or contact support."
        finally:
            session.close()

    def _analyze_performance_metrics(self, metrics: List) -> Dict[str, Any]:
        """Analyze performance metrics with trends"""
        if not metrics:
            system_logger.warning("No metrics data available for analysis.")
            return {"error": "No metrics data available"}

        metrics_data = {}
        for metric in metrics:
            metric_name = metric.metric_name
            if metric_name not in metrics_data:
                metrics_data[metric_name] = []

            metrics_data[metric_name].append(
                {
                    "value": float(metric.metric_value),
                    "date": metric.date.isoformat() if metric.date else None,
                }
            )

        # Calculate trends
        analyzed_metrics = {}
        for metric_name, values in metrics_data.items():
            if len(values) >= 2:
                recent_value = values[-1]["value"]
                previous_value = values[-2]["value"]
                trend = (recent_value - previous_value) / previous_value * 100

                analyzed_metrics[metric_name] = {
                    "current_value": recent_value,
                    "trend_percentage": trend,
                    "trend_direction": "increasing" if trend > 0 else "decreasing",
                    "total_data_points": len(values),
                }

        return analyzed_metrics

    def _analyze_channel_performance(self, posts: List) -> Dict[str, Any]:
        """Analyze channel-specific performance"""
        if not posts:
            system_logger.warning("No posts data available for channel analysis.")
            return {"error": "No posts data available"}

        channel_performance = {}
        for post in posts:
            platform = getattr(post, "platform", "unknown")
            if platform not in channel_performance:
                channel_performance[platform] = {
                    "post_count": 0,
                    "total_engagement": 0,
                    "total_reach": 0,
                }

            channel_performance[platform]["post_count"] += 1
            # In a real implementation, you'd have actual engagement and reach data
            channel_performance[platform]["total_engagement"] += getattr(
                post, "engagement", 0
            )
            channel_performance[platform]["total_reach"] += getattr(post, "reach", 0)

        # Calculate averages
        for platform, data in channel_performance.items():
            if data["post_count"] > 0:
                data["avg_engagement"] = data["total_engagement"] / data["post_count"]
                data["avg_reach"] = data["total_reach"] / data["post_count"]

        return channel_performance

    def _analyze_audience_performance(self, target_audience: str) -> Dict[str, Any]:
        """Analyze audience-specific performance"""
        # This would typically involve more complex audience analysis
        # For now, return simulated insights
        return {
            "primary_segment": target_audience,
            "engagement_by_demographic": {
                "25-34": 0.045,
                "35-44": 0.038,
                "45-54": 0.032,
            },
            "top_performing_content_types": ["videos", "infographics", "case_studies"],
            "optimal_posting_times": ["10:00", "14:00", "18:00"],
            "audience_growth": 0.15,
        }

    def optimize_campaign_enhanced(
        self, campaign_id, optimization_goals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Provide enhanced campaign optimization with AI-powered recommendations"""
        try:
            # Convert string campaign_id to int if needed
            if isinstance(campaign_id, str):
                try:
                    campaign_id_int = int(campaign_id.replace('campaign_', '').replace('fallback_', ''))
                except (ValueError, AttributeError):
                    return {"error": f"Invalid campaign ID format: {campaign_id}"}
            else:
                campaign_id_int = campaign_id
                
            # Get comprehensive analysis
            analysis = self.analyze_enhanced_campaign_performance(campaign_id_int)

            # Get current campaign data
            session = get_session()
            campaign = (
                session.query(MarketingCampaign)
                .filter(MarketingCampaign.id == campaign_id_int)
                .first()
            )
            session.close()

            if not campaign:
                system_logger.warning(
                    f"Campaign {campaign_id} not found for optimization."
                )
                return {"error": "Campaign not found"}

            # Prepare optimization context
            current_performance = {
                "budget_utilization": 0.75,  # 75% of budget used
                "current_roi": 2.3,
                "top_performing_channel": "linkedin",
                "underperforming_channels": ["facebook", "twitter"],
            }

            optimization_goals = optimization_goals or [
                "increase_roi",
                "reduce_cpa",
                "improve_engagement",
            ]

            # Generate optimization recommendations
            optimization_analysis = self.optimization_chain.invoke(
                {
                    "current_performance": json.dumps(current_performance, indent=2),
                    "optimization_goals": ", ".join(optimization_goals),
                    "available_budget": str(
                        float(getattr(campaign, "budget", 0)) * 0.25
                    ),  # 25% remaining budget
                    "time_constraints": "2 weeks remaining",
                }
            )

            # Generate specific action items
            action_items = self._generate_optimization_actions(
                campaign, current_performance
            )

            # Estimate optimization impact
            impact_estimation = self._estimate_optimization_impact(
                current_performance, action_items
            )

            return {
                "analysis": analysis[:500] + "..." if len(analysis) > 500 else analysis,
                "optimization_recommendations": optimization_analysis,
                "priority_actions": action_items[:5],  # Top 5 actions
                "all_recommendations": action_items,
                "estimated_impact": impact_estimation,
                "implementation_timeline": "1-2 weeks",
                "confidence_level": "High",
                "next_review_date": (datetime.now() + timedelta(weeks=1)).isoformat(),
            }

        except Exception as e:
            system_logger.error(f"Error in enhanced campaign optimization: {e}")
            return {
                "error": "Optimization analysis failed",
                "fallback_recommendations": [
                    "Review top-performing content and create similar pieces",
                    "Reallocate budget from underperforming to high-performing channels",
                    "A/B test different call-to-action messages",
                    "Optimize posting times based on audience activity",
                ],
            }

    def _generate_optimization_actions(
        self, _campaign, performance_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate specific optimization action items"""
        actions = []

        # Budget reallocation
        if performance_data.get("underperforming_channels"):
            actions.append(
                {
                    "action": "Budget Reallocation",
                    "description": f"Shift 20% budget from {', '.join(performance_data['underperforming_channels'])} to {performance_data.get('top_performing_channel', 'top performer')}",
                    "priority": "High",
                    "estimated_effort": "Low",
                    "expected_impact": "15-25% ROI improvement",
                }
            )

        # Content optimization
        actions.append(
            {
                "action": "Content Refresh",
                "description": "Create 3 new content variations based on top-performing posts",
                "priority": "Medium",
                "estimated_effort": "Medium",
                "expected_impact": "10-20% engagement improvement",
            }
        )

        # Audience targeting
        actions.append(
            {
                "action": "Audience Refinement",
                "description": "Create lookalike audiences based on highest-converting segments",
                "priority": "High",
                "estimated_effort": "Low",
                "expected_impact": "20-30% conversion rate improvement",
            }
        )

        # Timing optimization
        actions.append(
            {
                "action": "Posting Schedule Optimization",
                "description": "Adjust posting times to peak audience activity periods",
                "priority": "Medium",
                "estimated_effort": "Low",
                "expected_impact": "5-15% reach improvement",
            }
        )

        # Creative testing
        actions.append(
            {
                "action": "A/B Testing Implementation",
                "description": "Test 3 different call-to-action approaches",
                "priority": "Medium",
                "estimated_effort": "Medium",
                "expected_impact": "10-25% click-through rate improvement",
            }
        )

        return actions

    def _estimate_optimization_impact(
        self, current_performance: Dict[str, Any], actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Estimate the impact of optimization actions"""
        current_roi = current_performance.get("current_roi", 2.0)

        # Calculate cumulative impact
        total_roi_improvement = 0
        total_engagement_improvement = 0

        for action in actions:
            impact_str = action.get("expected_impact", "0%")
            if "ROI" in impact_str:
                # Extract percentage and add to total
                percentage = float(impact_str.split("-")[0].replace("%", "")) / 100
                total_roi_improvement += percentage
            elif "engagement" in impact_str:
                percentage = float(impact_str.split("-")[0].replace("%", "")) / 100
                total_engagement_improvement += percentage

        new_roi = current_roi * (1 + total_roi_improvement)

        return {
            "current_roi": current_roi,
            "projected_roi": new_roi,
            "roi_improvement": f"{(new_roi - current_roi) / current_roi * 100:.1f}%",
            "engagement_improvement": f"{total_engagement_improvement * 100:.1f}%",
            "implementation_risk": "Low",
            "payback_period": "2-3 weeks",
            "confidence_interval": "±10%",
        }

    def get_campaign_insights_dashboard(self, campaign_id) -> Dict[str, Any]:
        """Generate comprehensive campaign insights dashboard"""
        session = get_session()
        try:
            # Convert string campaign_id to int if needed
            if isinstance(campaign_id, str):
                try:
                    campaign_id_int = int(campaign_id.replace('campaign_', '').replace('fallback_', ''))
                except (ValueError, AttributeError):
                    return {"error": f"Invalid campaign ID format: {campaign_id}"}
            else:
                campaign_id_int = campaign_id
                
            campaign = (
                session.query(MarketingCampaign)
                .filter(MarketingCampaign.id == campaign_id_int)
                .first()
            )

            if not campaign:
                system_logger.warning(
                    f"Campaign {campaign_id} not found for dashboard."
                )
                return {"error": "Campaign not found"}

            # Get all related data
            metrics = (
                session.query(CampaignMetrics)
                .filter(CampaignMetrics.campaign_id == campaign_id_int)
                .all()
            )

            posts = (
                session.query(MarketingPost)
                .filter(MarketingPost.campaign_id == campaign_id_int)
                .all()
            )

            # Performance summary
            performance_summary = self._create_performance_summary(campaign, metrics)

            # Channel breakdown
            channel_breakdown = self._create_channel_breakdown(posts)

            # Timeline analysis
            timeline_analysis = self._create_timeline_analysis(metrics)

            # ROI analysis
            roi_analysis = self._create_roi_analysis(campaign, metrics)

            return {
                "campaign_overview": {
                    "name": campaign.name,
                    "type": campaign.campaign_type,
                    "status": campaign.status,
                    "budget": float(getattr(campaign, "budget", 0)),
                    "duration": self._calculate_campaign_duration(campaign),
                    "target_audience": campaign.target_audience,
                },
                "performance_summary": performance_summary,
                "channel_breakdown": channel_breakdown,
                "timeline_analysis": timeline_analysis,
                "roi_analysis": roi_analysis,
                "recommendations": self._generate_dashboard_recommendations(
                    campaign, metrics
                ),
                "next_steps": self._generate_next_steps(campaign),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            system_logger.error(f"Error generating campaign insights dashboard: {e}")
            return {"error": "Failed to generate dashboard"}
        finally:
            session.close()

    def _create_performance_summary(self, _campaign, metrics: List) -> Dict[str, Any]:
        """Create performance summary for dashboard"""
        if not metrics:
            return {"status": "No performance data available"}

        # Calculate key metrics
        total_impressions = sum(
            float(m.metric_value) for m in metrics if m.metric_name == "impressions"
        )
        total_clicks = sum(
            float(m.metric_value) for m in metrics if m.metric_name == "clicks"
        )
        total_conversions = sum(
            float(m.metric_value) for m in metrics if m.metric_name == "conversions"
        )

        ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        conversion_rate = (
            (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
        )

        return {
            "total_impressions": int(total_impressions),
            "total_clicks": int(total_clicks),
            "total_conversions": int(total_conversions),
            "click_through_rate": round(ctr, 2),
            "conversion_rate": round(conversion_rate, 2),
            "estimated_reach": int(
                total_impressions * 0.8
            ),  # Assuming 80% unique reach
            "engagement_rate": (
                round(total_clicks / total_impressions * 100, 2)
                if total_impressions > 0
                else 0
            ),
        }

    def _create_channel_breakdown(self, posts: List) -> Dict[str, Any]:
        """Create channel performance breakdown"""
        if not posts:
            return {"status": "No channel data available"}

        channel_data = {}
        for post in posts:
            platform = getattr(post, "platform", "unknown")
            if platform not in channel_data:
                channel_data[platform] = {
                    "posts": 0,
                    "total_engagement": 0,
                    "total_reach": 0,
                    "performance_score": 0,
                }

            channel_data[platform]["posts"] += 1
            channel_data[platform]["total_engagement"] += getattr(post, "engagement", 0)
            channel_data[platform]["total_reach"] += getattr(post, "reach", 0)

        # Calculate performance scores
        for platform, data in channel_data.items():
            if data["posts"] > 0:
                data["avg_engagement"] = data["total_engagement"] / data["posts"]
                data["avg_reach"] = data["total_reach"] / data["posts"]
                data["performance_score"] = (
                    data["avg_engagement"] + data["avg_reach"]
                ) / 2

        return channel_data

    def _create_timeline_analysis(self, metrics: List) -> Dict[str, Any]:
        """Create timeline performance analysis"""
        if not metrics:
            return {"status": "No timeline data available"}

        # Group metrics by date
        daily_metrics = {}
        for metric in metrics:
            date_key = metric.date.strftime("%Y-%m-%d") if metric.date else "unknown"
            if date_key not in daily_metrics:
                daily_metrics[date_key] = {
                    "impressions": 0,
                    "clicks": 0,
                    "conversions": 0,
                    "engagement": 0,
                }

            if metric.metric_name in daily_metrics[date_key]:
                daily_metrics[date_key][metric.metric_name] += float(
                    metric.metric_value
                )

        # Calculate trends
        dates = sorted([d for d in daily_metrics.keys() if d != "unknown"])
        if len(dates) >= 2:
            recent_performance = daily_metrics[dates[-1]]
            previous_performance = daily_metrics[dates[-2]]

            trends = {}
            for metric_name in ["impressions", "clicks", "conversions"]:
                if previous_performance[metric_name] > 0:
                    trend = (
                        (
                            recent_performance[metric_name]
                            - previous_performance[metric_name]
                        )
                        / previous_performance[metric_name]
                        * 100
                    )
                    trends[f"{metric_name}_trend"] = round(trend, 2)

            return {
                "daily_data": daily_metrics,
                "trends": trends,
                "total_days": len(dates),
            }

        return {"daily_data": daily_metrics, "trends": {}, "total_days": len(dates)}

    def _create_roi_analysis(self, campaign, metrics: List) -> Dict[str, Any]:
        """Create ROI analysis"""
        budget = float(campaign.budget)

        # Calculate revenue from conversions (assuming average order value)
        total_conversions = sum(
            float(m.metric_value) for m in metrics if m.metric_name == "conversions"
        )
        avg_order_value = 150  # This would come from actual data
        total_revenue = total_conversions * avg_order_value

        roi = ((total_revenue - budget) / budget * 100) if budget > 0 else 0
        roas = (total_revenue / budget) if budget > 0 else 0

        return {
            "total_spend": budget,
            "total_revenue": total_revenue,
            "roi_percentage": round(roi, 2),
            "roas": round(roas, 2),
            "cost_per_conversion": (
                round(budget / total_conversions, 2) if total_conversions > 0 else 0
            ),
            "profit": total_revenue - budget,
            "break_even_point": budget / avg_order_value if avg_order_value > 0 else 0,
        }

    def _calculate_campaign_duration(self, campaign) -> int:
        """Calculate campaign duration in days"""
        if campaign.start_date and campaign.end_date:
            return (campaign.end_date - campaign.start_date).days
        elif campaign.start_date:
            return (datetime.now().date() - campaign.start_date).days
        else:
            return 0

    def _generate_dashboard_recommendations(
        self, _campaign, metrics: List
    ) -> List[str]:
        """Generate recommendations for dashboard"""
        recommendations = []

        # Analyze performance and generate recommendations
        total_conversions = sum(
            float(m.metric_value) for m in metrics if m.metric_name == "conversions"
        )

        if total_conversions < 10:
            recommendations.append(
                "Consider optimizing targeting to increase conversions"
            )

        recommendations.extend(
            [
                "Monitor daily performance trends closely",
                "Test different content formats for better engagement",
                "Consider expanding to high-performing channels",
                "Implement retargeting campaigns for better ROI",
            ]
        )

        return recommendations[:5]  # Return top 5 recommendations

    def _create_fallback_campaign_response(
        self, request: CampaignRequest
    ) -> Dict[str, Any]:
        """Create a fallback campaign response when database operations fail"""

        # Safe campaign type access
        campaign_type_str = (
            request.campaign_type.value
            if hasattr(request.campaign_type, "value")
            else str(request.campaign_type)
        )

        # Generate basic campaign content without database
        content_pieces = []
        for i, channel in enumerate(request.channels[:3]):  # Limit to 3 pieces
            content_pieces.append(
                {
                    "platform": channel,
                    "title": f"{campaign_type_str.replace('_', ' ').title()} - {channel.title()} Content",
                    "content": f"Engaging {campaign_type_str.replace('_', ' ')} content for {request.target_audience} on {channel}. This campaign aims to drive awareness and engagement with our target demographic.",
                    "hashtags": [
                        "#marketing",
                        f"#{campaign_type_str}",
                        "#engagement",
                    ],
                    "call_to_action": DEFAULT_CALL_TO_ACTION,
                    "estimated_reach": 5000 * (i + 1),
                }
            )

        # Generate schedule
        schedule = []
        for i, piece in enumerate(content_pieces):
            schedule.append(
                {
                    "day": i + 1,
                    "platform": piece["platform"],
                    "content_title": piece["title"],
                    "action": "publish_content",
                    "estimated_reach": piece["estimated_reach"],
                }
            )

        return {
            "campaign_id": f"fallback_{int(datetime.now().timestamp())}",
            "campaign_plan": {
                "strategy": f"Comprehensive {campaign_type_str.replace('_', ' ')} campaign targeting {request.target_audience}. This campaign focuses on multi-channel engagement across {', '.join(request.channels)} to maximize reach and conversion potential within the ${request.budget:,.0f} budget allocation.",
                "content_pieces_count": len(content_pieces),
                "budget_allocation": {
                    channel: request.budget / len(request.channels)
                    for channel in request.channels
                },
                "success_metrics": [
                    f"Target reach: {sum(piece['estimated_reach'] for piece in content_pieces):,}",
                    "Estimated engagement rate: 3.5%",
                    f"Projected conversions: {int(sum(piece['estimated_reach'] for piece in content_pieces) * 0.02)}",
                    "ROI target: 250%",
                ],
                "estimated_performance": {
                    "total_reach": sum(
                        piece["estimated_reach"] for piece in content_pieces
                    ),
                    "estimated_clicks": int(
                        sum(piece["estimated_reach"] for piece in content_pieces) * 0.05
                    ),
                    "estimated_conversions": int(
                        sum(piece["estimated_reach"] for piece in content_pieces) * 0.02
                    ),
                    "estimated_roi": 2.5,
                    "confidence_level": "medium",
                },
            },
            "content_pieces": content_pieces,
            "schedule": schedule,
            "optimization_roadmap": [
                f"{MONITOR_INITIAL_PERFORMANCE} metrics",
                "A/B test content variations",
                "Optimize based on engagement data",
                "Scale successful content types",
                "Adjust targeting based on performance",
            ],
            "status": "created_successfully_fallback",
            "next_steps": [
                "Review generated content pieces",
                "Approve campaign schedule",
                "Set up tracking and analytics",
                "Launch campaign execution",
                "Monitor performance metrics",
            ],
            "metadata": {
                "creation_method": "fallback_generation",
                "database_status": "unavailable",
                "generated_at": datetime.now().isoformat(),
                "campaign_duration": f"{request.duration_days} days",
                "budget_per_day": f"${request.budget / request.duration_days:.2f}",
            },
        }

    def _generate_next_steps(self, campaign) -> List[str]:
        """Generate next steps for campaign management"""
        next_steps = [
            "Review performance metrics weekly",
            "Optimize underperforming content",
            "Scale successful campaigns",
            "Prepare end-of-campaign report",
            "Plan follow-up campaigns based on learnings",
        ]

        return next_steps

    def _check_and_fix_database_schema(self, session) -> bool:
        """Check and fix database schema issues"""
        try:
            from sqlalchemy import inspect, text

            engine = session.get_bind()
            inspector = inspect(engine)

            # Check if marketing_posts table exists and has required columns
            if inspector.has_table("marketing_posts"):
                columns = inspector.get_columns("marketing_posts")
                column_names = [col["name"] for col in columns]

                # Add missing columns if needed
                if "title" not in column_names:
                    try:
                        session.execute(
                            text(
                                "ALTER TABLE marketing_posts ADD COLUMN title VARCHAR(255)"
                            )
                        )
                        session.commit()
                        system_logger.info(
                            "Added missing 'title' column to marketing_posts"
                        )
                    except Exception as e:
                        system_logger.warning(f"Could not add title column: {e}")
                        session.rollback()

                if "content_type" not in column_names:
                    try:
                        session.execute(
                            text(
                                "ALTER TABLE marketing_posts ADD COLUMN content_type VARCHAR(100)"
                            )
                        )
                        session.commit()
                        system_logger.info(
                            "Added missing 'content_type' column to marketing_posts"
                        )
                    except Exception as e:
                        system_logger.warning(f"Could not add content_type column: {e}")
                        session.rollback()

                return True
            return False
        except Exception as e:
            system_logger.error(f"Error checking database schema: {e}")
            return False

    def create_campaign_from_request(
        self, request: CampaignRequest, industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a complete campaign with database persistence - Refactored for lower cognitive complexity"""
        session = get_session()
        try:
            self._check_and_fix_database_schema(session)
            campaign_plan = self.create_enhanced_campaign_plan(
                request, [industry] if industry is not None else None
            )
            campaign = self._save_campaign_to_db(session, request, campaign_plan)
            content_pieces_saved = self._save_content_pieces_to_db(
                session, campaign, campaign_plan
            )
            response = self._build_campaign_creation_response(
                campaign, campaign_plan, content_pieces_saved
            )
            system_logger.info(
                "Campaign created and saved to database",
                additional_info={
                    "campaign_id": campaign.id,
                    "content_pieces": content_pieces_saved,
                    "target_audience": campaign.target_audience,
                },
            )
            return response
        except Exception as e:
            session.rollback()
            system_logger.error(
                error=f"Error creating campaign: {e}",
                exc_info=True,
                additional_info={
                    "campaign_type": getattr(
                        request.campaign_type, "value", str(request.campaign_type)
                    ),
                    "target_audience": request.target_audience,
                    "error_type": type(e).__name__,
                },
            )
            return self._build_campaign_creation_fallback_response(request, e)
        finally:
            session.close()

    def _save_campaign_to_db(self, session, request, campaign_plan):
        campaign_data = {
            "name": f"{request.campaign_type.value.replace('_', ' ').title()} Campaign",
            "campaign_type": request.campaign_type.value,
            "target_audience": str(request.target_audience),
            "budget": float(request.budget),
            "start_date": datetime.now().date(),
            "end_date": (datetime.now() + timedelta(days=request.duration_days)).date(),
            "status": "active",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        if hasattr(campaign_plan, "strategy") and campaign_plan.strategy:
            strategy_text = campaign_plan.strategy
            if len(strategy_text) > 10000:
                strategy_text = strategy_text[:10000] + "... [truncated]"
            campaign_data["strategy"] = strategy_text
        campaign = MarketingCampaign(**campaign_data)
        session.add(campaign)
        session.commit()
        return campaign

    def _save_content_pieces_to_db(self, session, campaign, campaign_plan):
        content_pieces_saved = 0
        for i, content in enumerate(getattr(campaign_plan, "content_pieces", [])):
            post_data = self._build_post_data(content, campaign.id, i)
            try:
                post = MarketingPost(**post_data)
                session.add(post)
                session.flush()
                content_pieces_saved += 1
            except Exception as post_error:
                system_logger.warning(
                    f"Failed to save full post data, trying minimal: {post_error}"
                )
                session.rollback()
                minimal_post_data = {
                    "campaign_id": campaign.id,
                    "platform": str(content.platform),
                    "content": str(content.content)[:1000],
                    "status": "scheduled",
                }
                try:
                    minimal_post = MarketingPost(**minimal_post_data)
                    session.add(minimal_post)
                    session.flush()
                    content_pieces_saved += 1
                except Exception as minimal_error:
                    system_logger.error(
                        f"Failed to save even minimal post: {minimal_error}"
                    )
                    session.rollback()
        if content_pieces_saved > 0:
            session.commit()
            system_logger.info(f"Saved {content_pieces_saved} content pieces")
        return content_pieces_saved

    def _build_post_data(self, content, campaign_id, i):
        post_data = {
            "campaign_id": campaign_id,
            "platform": str(content.platform),
            "content": str(content.content),
            "call_to_action": (
                str(content.call_to_action)[:255]
                if getattr(content, "call_to_action", None)
                else DEFAULT_CALL_TO_ACTION
            ),
            "estimated_reach": int(getattr(content, "estimated_reach", 0)),
            "scheduled_time": datetime.now() + timedelta(hours=i * 2),
            "status": "scheduled",
        }
        if hasattr(content, "title") and content.title:
            post_data["title"] = str(content.title)[:255]
        if hasattr(content, "content_type") and content.content_type:
            content_type_value = (
                content.content_type.value
                if hasattr(content.content_type, "value")
                else str(content.content_type)
            )
            post_data["content_type"] = content_type_value
        try:
            if isinstance(content.hashtags, list):
                post_data["hashtags"] = json.dumps(content.hashtags)
            else:
                post_data["hashtags"] = json.dumps([])
        except Exception:
            post_data["hashtags"] = json.dumps([])
        return post_data

    def _build_campaign_creation_response(
        self, campaign, campaign_plan, content_pieces_saved
    ):
        return {
            "campaign_id": campaign.id,
            "campaign_plan": {
                "strategy": (
                    getattr(
                        campaign_plan,
                        "strategy",
                        CAMPAIGN_STRATEGY_CREATION_STATUS,
                    )[:500]
                    + "..."
                    if hasattr(campaign_plan, "strategy")
                    and len(getattr(campaign_plan, "strategy", "")) > 500
                    else getattr(
                        campaign_plan,
                        "strategy",
                        CAMPAIGN_STRATEGY_CREATION_STATUS,
                    )
                ),
                "content_pieces_count": len(
                    getattr(campaign_plan, "content_pieces", [])
                ),
                "budget_allocation": getattr(campaign_plan, "budget_allocation", {}),
                "success_metrics": getattr(campaign_plan, "success_metrics", []),
                "estimated_performance": getattr(
                    campaign_plan, "estimated_performance", {}
                ),
            },
            "schedule": getattr(campaign_plan, "schedule", []),
            "optimization_roadmap": getattr(campaign_plan, "optimization_roadmap", []),
            "status": "created_successfully",
            "next_steps": [
                "Review and approve content pieces",
                "Set up tracking and analytics",
                "Begin campaign execution",
                MONITOR_INITIAL_PERFORMANCE,
            ],
            "database_info": {
                "campaign_saved": True,
                "content_pieces_saved": content_pieces_saved,
                "total_content_pieces": len(
                    getattr(campaign_plan, "content_pieces", [])
                ),
                "save_success_rate": f"{content_pieces_saved}/{len(getattr(campaign_plan, 'content_pieces', []))}",
            },
        }

    def _build_campaign_creation_fallback_response(self, request, error):
        return {
            "error": "Failed to create campaign in database",
            "message": str(error),
            "campaign_id": None,
            "campaign_plan": {
                "strategy": f"Basic {getattr(request.campaign_type, 'value', str(request.campaign_type)).replace('_', ' ')} campaign targeting {request.target_audience}",
                "content_pieces_count": len(request.channels),
                "budget_allocation": {
                    channel: request.budget / len(request.channels)
                    for channel in request.channels
                },
                "success_metrics": ["reach", "engagement", "conversions"],
                "estimated_performance": {
                    "total_reach": 10000,
                    "estimated_roi": 2.0,
                    "confidence_interval": "±20%",
                },
            },
            "schedule": [
                {"day": i + 1, "platform": channel, "action": "content_posting"}
                for i, channel in enumerate(request.channels)
            ],
            "optimization_roadmap": [
                MONITOR_INITIAL_PERFORMANCE,
                "Optimize underperforming content",
                "Scale successful elements",
            ],
            "status": "failed_to_save_to_database",
            "next_steps": [
                "Check database connection",
                RETRY_CAMPAIGN_CREATION,
                "Contact support if issue persists",
            ],
        }

    def create_marketing_campaign(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create marketing campaign with comprehensive error handling"""
        try:
            system_logger.info(
                "Creating marketing campaign",
                additional_info={
                    "campaign_type": request.get("campaign_type"),
                    "target_audience": request.get("target_audience"),
                    "budget": request.get("budget"),
                },
            )

            # Convert request to proper format
            campaign_request = CampaignRequest(
                campaign_type=CampaignType(request["campaign_type"]),
                target_audience=str(request["target_audience"]),  # Ensure string
                budget=float(request["budget"]),
                duration_days=int(request["duration_days"]),
                goals=request.get("goals", ["increase_awareness"]),
                channels=request.get("channels", ["social_media"]),
                content_requirements=request.get(
                    "content_requirements", ["social_media"]
                ),
            )

            # Try to create campaign with database persistence
            try:
                result = self.create_campaign_from_request(
                    campaign_request, industry=request.get("industry")
                )

                if "error" in result:
                    # If database creation failed, create fallback response
                    system_logger.warning(
                        f"Database campaign creation failed: {result['error']}"
                    )
                    return self._create_fallback_campaign_response(campaign_request)

                return result

            except Exception as db_error:
                system_logger.error(f"Database error in campaign creation: {db_error}")
                # Return fallback response
                return self._create_fallback_campaign_response(campaign_request)

        except Exception as e:
            system_logger.error(
                error=f"Error in create_marketing_campaign: {e}",
                exc_info=True,
                additional_info={
                    "context": "Create Marketing Campaign",
                    "campaign_type": request.get("campaign_type", "unknown"),
                    "target_audience": request.get("target_audience", "unknown"),
                },
            )

            # Return comprehensive error response
            return {
                "error": "Failed to create campaign",
                "message": "Campaign creation encountered technical issues. Please try again.",
                "campaign_id": None,
                "campaign_plan": {
                    "strategy": f"Unable to generate complete strategy due to technical issues. Basic {request.get('campaign_type', 'marketing')} campaign targeting {request.get('target_audience', 'general audience')} with ${request.get('budget', 0)} budget.",
                    "content_pieces_count": 0,
                    "budget_allocation": {"analysis_pending": True},
                    "success_metrics": ["reach", "engagement", "conversions"],
                    "estimated_performance": {"status": "analysis_pending"},
                },
                "schedule": [],
                "optimization_roadmap": [
                    "Resolve technical issues",
                    RETRY_CAMPAIGN_CREATION,
                ],
                "status": "creation_failed",
                "next_steps": [
                    "Check system status",
                    "Verify database connectivity",
                    "Contact support if issue persists",
                    RETRY_CAMPAIGN_CREATION,
                ],
                "technical_details": {
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat(),
                    "recommendation": "Please try again in a few minutes",
                },
            }

    def generate_marketing_content(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate marketing content (API compatibility method)"""
        try:
            from src.que_agents.core.schemas import ContentType

            # Convert string content_type to enum if needed
            content_type_str = request.get("content_type", "social_media")
            try:
                content_type = ContentType(content_type_str)
            except ValueError:
                content_type = ContentType.SOCIAL_MEDIA

            # Generate content using existing method
            content_piece = self.generate_enhanced_content(
                platform=request.get("platform", "social_media"),
                content_type=content_type,
                campaign_theme=request.get("campaign_theme", "marketing"),
                target_audience=request.get("target_audience", "general audience"),
                key_messages=request.get("key_messages", ["engaging", "innovative"]),
                brand_voice=request.get("brand_voice", "professional"),
            )

            # Convert ContentPiece to dict for API response
            return {
                "title": getattr(content_piece, "title", GENERATED_CONTENT),
                "content": getattr(content_piece, "content", ""),
                "call_to_action": getattr(
                    content_piece, "call_to_action", DEFAULT_CALL_TO_ACTION
                ),
                "hashtags": getattr(content_piece, "hashtags", []),
                "platform": getattr(content_piece, "platform", request.get("platform")),
                "estimated_reach": getattr(content_piece, "estimated_reach", 1000),
                "optimization_score": getattr(content_piece, "optimization_score", 0.6),
            }

        except Exception as e:
            system_logger.error(
                f"Error in generate_marketing_content: {e}", exc_info=True
            )
            return {
                "error": str(e),
                "fallback_content": {
                    "title": f"Exciting {request.get('campaign_theme', 'Marketing')} Update",
                    "content": f"Great content about {request.get('campaign_theme', 'our initiative')} for {request.get('target_audience', 'our audience')}!",
                    "call_to_action": DEFAULT_CALL_TO_ACTION,
                    "hashtags": ["#Marketing", "#Innovation"],
                    "platform": request.get("platform", "social_media"),
                    "estimated_reach": 1500,
                    "optimization_score": 0.7,
                },
            }

    def analyze_campaign_performance(self, campaign_id: int) -> Dict[str, Any]:
        """Analyze campaign performance (API compatibility method)"""
        try:
            # Use the existing enhanced analysis method
            analysis_text = self.analyze_enhanced_campaign_performance(campaign_id)

            # Get dashboard data for structured response
            dashboard_data = self.get_campaign_insights_dashboard(campaign_id)

            return {
                "campaign_id": campaign_id,
                "analysis": analysis_text,
                "dashboard_data": dashboard_data,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            system_logger.error(
                f"Error in analyzing campaign performance: {e}",
                additional_info={"campaign_id": campaign_id},
                exc_info=True,
            )
            return {
                "error": "Failed to analyze campaign performance",
                "message": str(e),
                "status": "failed",
                "campaign_id": campaign_id,
            }

    def optimize_campaign(
        self, campaign_id: int, optimization_goals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Optimize campaign (API compatibility method)"""
        try:
            # Use the existing enhanced optimization method
            optimization_result = self.optimize_campaign_enhanced(
                campaign_id, optimization_goals
            )

            return {
                "campaign_id": campaign_id,
                "optimization_result": optimization_result,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            system_logger.error(
                error=f"Error in optimize_campaign: {e}",
                exc_info=True,
                additional_info={"campaign_id": campaign_id},
            )
            return {
                "error": "Failed to optimize campaign",
                "message": str(e),
                "status": "failed",
                "campaign_id": campaign_id,
            }

    def get_campaign_insights(self, campaign_id: int) -> Dict[str, Any]:
        """Get campaign insights (API compatibility method)"""
        try:
            # Use the existing dashboard method
            insights = self.get_campaign_insights_dashboard(campaign_id)

            return {
                "campaign_id": campaign_id,
                "insights": insights,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            system_logger.error(f"Error in get_campaign_insights: {e}")
            return {
                "error": "Failed to get campaign insights",
                "message": str(e),
                "status": "failed",
                "campaign_id": campaign_id,
            }

    def get_audience_analysis(
        self, target_audience: str, industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get audience analysis (API compatibility method)"""
        try:
            # Use the existing enhanced audience insights method
            audience_insights = self.get_enhanced_audience_insights(
                target_audience, industry
            )

            return {
                "target_audience": target_audience,
                "industry": industry,
                "audience_insights": audience_insights,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            system_logger.error(f"Error in get_audience_analysis: {e}")
            return {
                "error": "Failed to get audience analysis",
                "message": str(e),
                "status": "failed",
                "target_audience": target_audience,
            }

    def get_market_intelligence(
        self, campaign_type: str, industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get market intelligence (API compatibility method)"""
        try:
            # Convert string to enum
            campaign_type_enum = CampaignType(campaign_type)

            # Use the existing enhanced market data method
            market_data = self.get_enhanced_market_data(campaign_type_enum, industry)

            return {
                "campaign_type": campaign_type,
                "industry": industry,
                "market_data": market_data,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            system_logger.error(f"Error in get_market_intelligence: {e}")
            return {
                "error": "Failed to get market intelligence",
                "message": str(e),
                "status": "failed",
                "campaign_type": campaign_type,
            }

    def get_content_suggestions(
        self,
        platform: str,
        industry: Optional[str] = None,
        content_type: str = "social_media",
    ) -> Dict[str, Any]:
        """Get content suggestions (API compatibility method)"""
        try:
            # Get platform-specific knowledge
            platform_knowledge = self.get_marketing_knowledge(
                f"{platform} content best practices"
            )

            # Get industry-specific knowledge
            industry_knowledge = []
            if industry:
                industry_knowledge = self.get_marketing_knowledge(
                    f"{industry} content strategy"
                )

            # Generate content suggestions based on platform
            platform_data = self.platform_limits.get(platform, {})
            suggested_content_types = platform_data.get("best_content_types", ["posts"])
            optimal_times = platform_data.get("optimal_posting_times", ["12:00"])

            return {
                "platform": platform,
                "industry": industry,
                "content_type": content_type,
                "suggestions": {
                    "recommended_content_types": suggested_content_types,
                    "optimal_posting_times": optimal_times,
                    "platform_knowledge": platform_knowledge,
                    "industry_knowledge": industry_knowledge,
                    "character_limits": {
                        "max_chars": platform_data.get("max_chars", 1000),
                        "hashtag_limit": platform_data.get("hashtag_limit", 5),
                    },
                },
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            system_logger.error(f"Error in get_content_suggestions: {e}")
            return {
                "error": "Failed to get content suggestions",
                "message": str(e),
                "status": "failed",
                "platform": platform,
            }

    def get_campaign_templates(
        self, campaign_type: str, industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get campaign templates (API compatibility method)"""
        try:
            # Get campaign strategy data
            strategy_data = self.campaign_strategies.get(campaign_type, {})

            # Get industry benchmarks
            industry_benchmarks = self.industry_benchmarks.get(
                industry.lower() if industry else "technology",
                self.industry_benchmarks["technology"],
            )

            # Create template structure
            template = {
                "campaign_type": campaign_type,
                "industry": industry,
                "recommended_channels": strategy_data.get("recommended_channels", []),
                "content_focus": strategy_data.get("content_focus", []),
                "primary_metrics": strategy_data.get("primary_metrics", []),
                "industry_benchmarks": industry_benchmarks,
                "suggested_budget_allocation": {
                    "paid_media": 0.6,
                    "content_creation": 0.3,
                    "tools_and_analytics": 0.1,
                },
                "timeline_template": {
                    "planning_phase": "Week 1-2",
                    "content_creation": "Week 2-3",
                    "campaign_launch": "Week 4",
                    "optimization_phase": "Week 5+",
                },
            }

            return {
                "template": template,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            system_logger.error(f"Error in get_campaign_templates: {e}")
            return {
                "error": "Failed to get campaign templates",
                "message": str(e),
                "status": "failed",
                "campaign_type": campaign_type,
            }


def test_enhanced_marketing_agent():
    """Comprehensive test of the enhanced marketing agent"""
    system_logger.info("Starting Marketing Agent test suite")
    agent = MarketingAgent()

    print("=== Enhanced Marketing Agent Test ===\n")

    # Test campaign creation with different scenarios
    test_campaigns = [
        {
            "request": CampaignRequest(
                campaign_type=CampaignType.PRODUCT_LAUNCH,
                target_audience="tech-savvy professionals",
                budget=50000.0,
                duration_days=30,
                goals=["increase awareness", "generate leads", "drive trials"],
                channels=["linkedin", "twitter", "email"],
                content_requirements=[
                    ContentType.SOCIAL_MEDIA,
                    ContentType.VIDEO,
                    ContentType.EMAIL,
                ],
            ),
            "industry": "technology",
            "description": "B2B SaaS Product Launch",
        },
        {
            "request": CampaignRequest(
                campaign_type=CampaignType.BRAND_AWARENESS,
                target_audience="young consumers aged 18-35",
                budget=25000.0,
                duration_days=45,
                goals=["brand recognition", "social engagement"],
                channels=["instagram", "tiktok", "facebook"],
                content_requirements=[ContentType.SOCIAL_MEDIA, ContentType.IMAGE],
            ),
            "industry": "retail",
            "description": "Consumer Brand Awareness",
        },
        {
            "request": CampaignRequest(
                campaign_type=CampaignType.LEAD_GENERATION,
                target_audience="healthcare professionals",
                budget=15000.0,
                duration_days=21,
                goals=["qualified leads", "webinar signups"],
                channels=["linkedin", "email"],
                content_requirements=[ContentType.EMAIL, ContentType.BLOG_POST],
            ),
            "industry": "healthcare",
            "description": "Healthcare Lead Generation",
        },
    ]

    for i, test_case in enumerate(test_campaigns, 1):
        system_logger.info(
            f"Running test case {i}: {test_case['description']}",
            additional_info=test_case,
        )
        print(f"Test Case {i}: {test_case['description']}")
        print("=" * 60)

        # Create campaign
        result = agent.create_campaign_from_request(
            test_case["request"], test_case["industry"]
        )

        if "error" not in result:
            campaign_id = result["campaign_id"]
            print(f"✅ Campaign Created Successfully (ID: {campaign_id})")
            print(f"Content Pieces: {result['campaign_plan']['content_pieces_count']}")
            print(
                f"Estimated ROI: {result['campaign_plan']['estimated_performance'].get('estimated_roi', 'N/A')}"
            )
            print(
                f"Total Reach: {result['campaign_plan']['estimated_performance'].get('total_reach', 'N/A'):,}"
            )

            # Test campaign analysis
            print("\n📊 Performance Analysis:")
            analysis = agent.analyze_enhanced_campaign_performance(campaign_id)
            print(analysis[:300] + "..." if len(analysis) > 300 else analysis)

            # Test optimization
            print("\n🎯 Optimization Recommendations:")
            optimization = agent.optimize_campaign_enhanced(campaign_id)
            if "error" not in optimization:
                print(f"Priority Actions: {len(optimization['priority_actions'])}")
                print(
                    f"Estimated Impact: {optimization['estimated_impact'].get('roi_improvement', 'N/A')}"
                )

            # Test dashboard
            print("\n📈 Campaign Dashboard:")
            dashboard = agent.get_campaign_insights_dashboard(campaign_id)
            if "error" not in dashboard:
                overview = dashboard["campaign_overview"]
                print(f"Status: {overview['status']}")
                print(f"Budget: ${overview['budget']:,.2f}")
                print(f"Duration: {overview['duration']} days")
        else:
            system_logger.error(
                f"Test case {i} failed: {result['error']}", additional_info=test_case
            )
            print(f"❌ Campaign Creation Failed: {result['error']}")

        print("\n" + "=" * 80 + "\n")

    # Test knowledge base integration
    print("🧠 Knowledge Base Integration Test:")
    knowledge = agent.get_marketing_knowledge(
        "social media optimization best practices"
    )
    print(f"Knowledge base results: {len(knowledge)} items found")
    if knowledge:
        print(f"First result: {knowledge[0]['title']}")

    # Test audience insights
    print("\n👥 Enhanced Audience Insights Test:")
    audience_insights = agent.get_enhanced_audience_insights(
        "tech professionals", "technology"
    )
    print(f"Audience segments: {len(audience_insights['segments'])}")
    print(f"Knowledge insights: {len(audience_insights['knowledge_base_insights'])}")

    system_logger.info("Marketing Agent testing completed")
    print("\n✅ Enhanced Marketing Agent Testing Complete!")


if __name__ == "__main__":
    test_enhanced_marketing_agent()
