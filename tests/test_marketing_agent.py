from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.que_agents.agents.marketing_agent import MarketingAgent
from src.que_agents.core.schemas import CampaignRequest, CampaignType, ContentType


@pytest.fixture
def agent():
    with patch("src.que_agents.agents.marketing_agent.LLMFactory.get_llm") as mock_llm:
        mock_llm.return_value = MagicMock()
        return MarketingAgent()


@pytest.fixture
def sample_request():
    return CampaignRequest(
        campaign_type=CampaignType.BRAND_AWARENESS,
        target_audience="tech professionals",
        budget=10000,
        duration_days=30,
        goals=["awareness", "engagement"],
        channels=["twitter", "linkedin"],
        content_requirements=[ContentType.SOCIAL_MEDIA, ContentType.VIDEO],
    )


def test_agent_instantiation(agent):
    assert isinstance(agent, MarketingAgent)
    assert hasattr(agent, "platform_limits")
    assert hasattr(agent, "campaign_strategies")
    assert hasattr(agent, "industry_benchmarks")


def test_get_marketing_knowledge_success(agent):
    with patch(
        "src.que_agents.agents.marketing_agent.search_agent_knowledge_base"
    ) as mock_search:
        mock_search.return_value = [{"title": "T1", "content": "C1"}]
        result = agent.get_marketing_knowledge("query")
        assert isinstance(result, list)
        assert result[0]["title"] == "T1"


def test_get_marketing_knowledge_error(agent):
    with patch(
        "src.que_agents.agents.marketing_agent.search_agent_knowledge_base",
        side_effect=Exception("fail"),
    ):
        result = agent.get_marketing_knowledge("query")
        assert result == []


def test_get_enhanced_campaign_context(agent, sample_request):
    with patch.object(
        agent,
        "get_marketing_knowledge",
        return_value=[{"title": "T", "content": "C" * 300}],
    ):
        context = agent.get_enhanced_campaign_context(sample_request, "technology")
        assert "Campaign Strategy Knowledge" in context
        assert "technology Industry Insights" in context
        assert "Channel Optimization Tips" in context


def test_get_enhanced_campaign_context_no_industry(agent, sample_request):
    with patch.object(agent, "get_marketing_knowledge", return_value=[]):
        context = agent.get_enhanced_campaign_context(sample_request)
        assert isinstance(context, str)


def test_get_enhanced_audience_insights_default(agent):
    with patch("src.que_agents.agents.marketing_agent.get_session") as mock_session:
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = []
        mock_session.return_value = mock_db
        with patch.object(agent, "get_marketing_knowledge", return_value=[]):
            result = agent.get_enhanced_audience_insights("tech")
            assert "segments" in result
            assert isinstance(result["segments"], list)


def test_get_enhanced_audience_insights_with_segments(agent):
    mock_segment = MagicMock()
    mock_segment.name = "Tech Segment"
    mock_segment.criteria = {"age": "25-40"}
    mock_segment.characteristics = {"engagement": "high"}

    with patch("src.que_agents.agents.marketing_agent.get_session") as mock_session:
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_segment]
        mock_session.return_value = mock_db
        with patch.object(agent, "get_marketing_knowledge", return_value=[]):
            result = agent.get_enhanced_audience_insights("tech", "technology")
            assert len(result["segments"]) > 0


def test_get_enhanced_audience_insights_error(agent):
    # Test the fallback behavior by mocking the session to return a failing query
    with patch("src.que_agents.agents.marketing_agent.get_session") as mock_session:
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("db error")
        mock_session.return_value = mock_db
        result = agent.get_enhanced_audience_insights("tech")
        assert "segments" in result
        assert len(result["segments"]) > 0  # Should return fallback data


def test_analyze_audience_behavior(agent):
    result = agent._analyze_audience_behavior("tech professionals", [])
    assert "behavioral_patterns" in result
    assert "engagement_score" in result
    assert "channel_preferences" in result
    assert "content_preferences" in result


def test_analyze_audience_behavior_business(agent):
    result = agent._analyze_audience_behavior("business executives", [])
    assert result["engagement_score"] == 0.6


def test_analyze_audience_behavior_consumer(agent):
    result = agent._analyze_audience_behavior("general consumers", [])
    assert result["engagement_score"] == 0.5


def test_get_enhanced_market_data(agent):
    with patch.object(agent, "get_marketing_knowledge", return_value=[]):
        data = agent.get_enhanced_market_data(
            CampaignType.BRAND_AWARENESS, "technology"
        )
        assert "benchmarks" in data
        assert "market_trends" in data
        assert "competitive_landscape" in data
        assert "growth_opportunities" in data
        assert "risk_factors" in data


def test_get_enhanced_market_data_error(agent):
    with patch.object(agent, "get_marketing_knowledge", side_effect=Exception("error")):
        data = agent.get_enhanced_market_data(CampaignType.BRAND_AWARENESS)
        assert "benchmarks" in data


def test_analyze_market_trends(agent):
    trends = agent._analyze_market_trends(CampaignType.PRODUCT_LAUNCH, "tech")
    assert "growth_rate" in trends
    assert "optimal_duration" in trends
    assert "budget_allocation" in trends


def test_analyze_competitive_landscape(agent):
    landscape = agent._analyze_competitive_landscape(CampaignType.BRAND_AWARENESS)
    assert "market_saturation" in landscape
    assert "key_competitors" in landscape
    assert "competitive_advantage_opportunities" in landscape


def test_identify_growth_opportunities(agent):
    opportunities = agent._identify_growth_opportunities(
        CampaignType.PRODUCT_LAUNCH, "tech"
    )
    assert isinstance(opportunities, list)
    assert len(opportunities) > 0
    assert "Influencer partnerships" in opportunities


def test_identify_risk_factors(agent):
    risks = agent._identify_risk_factors(CampaignType.BRAND_AWARENESS, "healthcare")
    assert isinstance(risks, list)
    assert "Regulatory compliance requirements" in risks


def test_create_enhanced_campaign_strategy_success(agent, sample_request):
    with (
        patch.object(
            agent, "get_enhanced_audience_insights", return_value={"segments": []}
        ),
        patch.object(
            agent,
            "get_enhanced_market_data",
            return_value={"competitive_landscape": {}, "benchmarks": {}},
        ),
        patch.object(agent, "get_enhanced_campaign_context", return_value="context"),
    ):
        # Mock the chain invoke method directly
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "strategy"
        agent.campaign_chain = mock_chain

        strategy = agent.create_enhanced_campaign_strategy(sample_request, "technology")
        assert isinstance(strategy, str)


def test_create_enhanced_campaign_strategy_error(agent, sample_request):
    with patch.object(
        agent, "get_enhanced_audience_insights", side_effect=Exception("error")
    ):
        strategy = agent.create_enhanced_campaign_strategy(sample_request)
        assert isinstance(strategy, str)
        assert "CAMPAIGN STRATEGY OVERVIEW" in strategy


def test_get_safe_campaign_type_string(agent):
    assert (
        agent._get_safe_campaign_type_string(CampaignType.BRAND_AWARENESS)
        == "brand_awareness"
    )
    assert agent._get_safe_campaign_type_string("custom_type") == "custom_type"
    assert agent._get_safe_campaign_type_string(None) == "unknown"


def test_get_safe_content_type_string(agent):
    assert (
        agent._get_safe_content_type_string(ContentType.SOCIAL_MEDIA) == "social_media"
    )
    assert agent._get_safe_content_type_string("custom_content") == "custom_content"
    assert agent._get_safe_content_type_string(None) == "social_media"


def test_create_basic_strategy(agent, sample_request):
    strategy = agent._create_basic_strategy(sample_request)
    assert "CAMPAIGN STRATEGY OVERVIEW" in strategy
    assert "10,000" in strategy  # Budget is formatted with commas


def test_generate_enhanced_content_success(agent):
    with patch.object(agent, "get_marketing_knowledge", return_value=[]):
        # Mock the chain invoke method directly
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Title: Test\nCTA: Try now!\n#hashtag"
        agent.content_chain = mock_chain

        content = agent.generate_enhanced_content(
            "twitter", ContentType.SOCIAL_MEDIA, "theme", "tech", ["msg"]
        )
        assert content.title == "Test"
        assert "Try now!" in content.call_to_action
        assert isinstance(content.hashtags, list)


def test_generate_enhanced_content_error(agent):
    # Mock the chain invoke method to raise exception
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("error")
    agent.content_chain = mock_chain

    content = agent.generate_enhanced_content(
        "twitter", ContentType.SOCIAL_MEDIA, "theme", "tech", ["msg"]
    )
    assert hasattr(content, "title")
    assert hasattr(content, "content")


def test_parse_generated_content(agent):
    content_text = (
        "Title: Test Title\nThis is content\nCTA: Click here\n#test #marketing"
    )
    parsed = agent._parse_generated_content(
        content_text, "twitter", ContentType.SOCIAL_MEDIA
    )
    assert parsed["title"] == "Test Title"
    assert parsed["cta"] == "Click here"
    assert "#test" in parsed["hashtags"]


def test_calculate_enhanced_reach(agent):
    reach = agent._calculate_enhanced_reach(
        "twitter", "tech professionals", 3, {"engagement_multiplier": 1.2}
    )
    assert isinstance(reach, int)
    assert reach > 0


def test_generate_content_variations(agent):
    content = {
        "title": "Test",
        "content": "Content",
        "cta": "Click",
        "hashtags": ["#test"],
    }
    variations = agent._generate_content_variations(content, "twitter")
    assert isinstance(variations, list)
    assert len(variations) > 0


def test_calculate_content_score(agent):
    content = {
        "title": "Great new product",
        "content": "A" * 100,
        "cta": "Learn more",
        "hashtags": ["#test"],
    }
    platform_data = {"max_chars": 280, "hashtag_limit": 3}
    score = agent._calculate_content_score(content, platform_data)
    assert 0 <= score <= 1


def test_generate_fallback_content(agent):
    content = agent._generate_fallback_content(
        "twitter", ContentType.SOCIAL_MEDIA, "test"
    )
    assert hasattr(content, "title")
    assert hasattr(content, "content")
    assert hasattr(content, "call_to_action")


def test_get_marketing_trends_success(agent):
    with patch.object(
        agent,
        "get_marketing_knowledge",
        return_value=[{"title": "Trend", "content": "Content"}],
    ):
        trends = agent.get_marketing_trends("brand_awareness", "tech")
        assert "trends" in trends
        assert trends["data_source"] == "knowledge_base"


def test_get_marketing_trends_fallback(agent):
    with patch.object(agent, "get_marketing_knowledge", side_effect=Exception("fail")):
        trends = agent.get_marketing_trends("brand_awareness", "tech")
        assert "trends" in trends
        assert trends["data_source"] == "agent_trends"
        assert len(trends["trends"]) == 3


def test_get_audience_segments_success(agent):
    with patch.object(
        agent,
        "get_enhanced_audience_insights",
        return_value={
            "segments": [
                {"name": "seg", "criteria": {}, "engagement_score": 0.1, "size": 100}
            ]
        },
    ):
        segs = agent.get_audience_segments()
        assert "segments" in segs
        assert segs["total_segments"] == 1


def test_get_audience_segments_error(agent):
    with patch.object(
        agent, "get_enhanced_audience_insights", side_effect=Exception("error")
    ):
        segs = agent.get_audience_segments()
        assert "segments" in segs
        assert segs["total_segments"] == 1  # Fallback


def test_get_campaign_list_empty(agent):
    with patch("src.que_agents.agents.marketing_agent.get_session") as mock_session:
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.limit.return_value.all.return_value = (
            []
        )
        mock_db.query.return_value.limit.return_value.all.return_value = []
        mock_session.return_value = mock_db
        result = agent.get_campaign_list()
        assert "campaigns" in result
        assert result["total_count"] == 0


def test_get_campaign_list_with_campaigns(agent):
    mock_campaign = MagicMock()
    mock_campaign.id = 1
    mock_campaign.name = "Test Campaign"
    mock_campaign.campaign_type = "brand_awareness"
    mock_campaign.status = "active"
    mock_campaign.budget = 1000
    mock_campaign.target_audience = "tech"
    mock_campaign.start_date = datetime.now().date()
    mock_campaign.end_date = None
    mock_campaign.created_at = datetime.now()

    with patch("src.que_agents.agents.marketing_agent.get_session") as mock_session:
        mock_db = MagicMock()
        mock_db.query.return_value.limit.return_value.all.return_value = [mock_campaign]
        mock_session.return_value = mock_db
        result = agent.get_campaign_list()
        assert result["total_count"] == 1
        assert result["campaigns"][0]["name"] == "Test Campaign"


def test_get_campaign_list_error(agent):
    # Test the fallback behavior by mocking the session to return a failing query
    with patch("src.que_agents.agents.marketing_agent.get_session") as mock_session:
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("db error")
        mock_session.return_value = mock_db
        result = agent.get_campaign_list()
        assert "campaigns" in result
        assert result["total_count"] == 0


def test_create_enhanced_campaign_plan_success(agent, sample_request):
    with (
        patch.object(
            agent, "create_enhanced_campaign_strategy", return_value="strategy"
        ),
        patch.object(agent, "_generate_content_pieces", return_value=[]),
        patch.object(agent, "_create_optimized_schedule", return_value=[]),
        patch.object(agent, "_create_enhanced_budget_allocation", return_value={}),
        patch.object(agent, "_define_success_metrics", return_value=[]),
        patch.object(agent, "_calculate_enhanced_performance", return_value={}),
        patch.object(agent, "_assess_campaign_risks", return_value={}),
        patch.object(agent, "_create_optimization_roadmap", return_value=[]),
        patch.object(agent, "_extract_industry_value", return_value="technology"),
        patch.object(agent, "_build_campaign_plan") as mock_build,
    ):
        mock_build.return_value = MagicMock(
            campaign_id="id", content_pieces=[], schedule=[]
        )
        plan = agent.create_enhanced_campaign_plan(
            sample_request, industry=["technology"]
        )
        assert hasattr(plan, "campaign_id")


def test_create_enhanced_campaign_plan_error(agent, sample_request):
    with patch.object(
        agent, "create_enhanced_campaign_strategy", side_effect=Exception("error")
    ):
        plan = agent.create_enhanced_campaign_plan(sample_request)
        assert hasattr(plan, "campaign_id")


def test_extract_industry_string(agent):
    assert agent._extract_industry_string(["tech", "finance"]) == "tech"
    assert agent._extract_industry_string("healthcare") == "healthcare"
    assert agent._extract_industry_string(None, "default") == "default"
    assert agent._extract_industry_string([], "default") == "default"


def test_extract_industry_value(agent):
    assert agent._extract_industry_value(["tech"]) == "tech"
    assert agent._extract_industry_value("finance") == "finance"
    assert agent._extract_industry_value(None) is None


def test_generate_content_pieces(agent, sample_request):
    mock_content = MagicMock()
    mock_content.platform = "twitter"
    mock_content.content_type = ContentType.SOCIAL_MEDIA

    with patch.object(agent, "generate_enhanced_content", return_value=mock_content):
        pieces = agent._generate_content_pieces(sample_request, "professional")
        assert len(pieces) > 0


def test_generate_content_pieces_error(agent, sample_request):
    with patch.object(
        agent, "generate_enhanced_content", side_effect=Exception("error")
    ):
        pieces = agent._generate_content_pieces(sample_request, "professional")
        assert len(pieces) > 0  # Should have fallback content


def test_create_optimized_schedule(agent):
    mock_content = MagicMock()
    mock_content.platform = "twitter"
    mock_content.content_type = ContentType.SOCIAL_MEDIA
    mock_content.estimated_reach = 1000

    schedule = agent._create_optimized_schedule([mock_content], 30)
    assert isinstance(schedule, list)
    assert len(schedule) > 0


def test_create_enhanced_budget_allocation(agent, sample_request):
    mock_content = MagicMock()
    mock_content.platform = "twitter"
    mock_content.estimated_reach = 1000
    mock_content.optimization_score = 0.8

    allocation = agent._create_enhanced_budget_allocation(
        sample_request, [mock_content]
    )
    assert isinstance(allocation, dict)
    assert "twitter" in allocation


def test_define_success_metrics(agent):
    metrics = agent._define_success_metrics(
        "brand_awareness", ["awareness", "engagement"]
    )
    assert isinstance(metrics, list)
    assert len(metrics) > 0
    assert "Brand awareness lift" in metrics


def test_calculate_enhanced_performance(agent, sample_request):
    mock_content = MagicMock()
    mock_content.estimated_reach = 1000

    performance = agent._calculate_enhanced_performance(
        [mock_content], sample_request, "technology"
    )
    assert "total_reach" in performance
    assert "estimated_roi" in performance


def test_assess_campaign_risks(agent, sample_request):
    risks = agent._assess_campaign_risks(sample_request, "healthcare")
    assert "high_risk" in risks
    assert "medium_risk" in risks
    assert "low_risk" in risks
    assert "mitigation_strategies" in risks


def test_create_optimization_roadmap(agent):
    roadmap = agent._create_optimization_roadmap(30)
    assert isinstance(roadmap, list)
    assert len(roadmap) >= 2  # Should have multiple weeks


def test_create_basic_campaign_plan(agent, sample_request):
    plan = agent._create_basic_campaign_plan(sample_request)
    assert hasattr(plan, "campaign_id")
    assert hasattr(plan, "strategy")
    assert hasattr(plan, "content_pieces")


def test_analyze_enhanced_campaign_performance_not_found(agent):
    with patch("src.que_agents.agents.marketing_agent.get_session") as mock_session:
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_session.return_value = mock_db
        result = agent.analyze_enhanced_campaign_performance("campaign_999")
        assert "not found" in result.lower() or "verify" in result.lower()


def test_analyze_enhanced_campaign_performance_invalid_id(agent):
    result = agent.analyze_enhanced_campaign_performance("invalid_id")
    assert "no numeric id" in result.lower()


def test_analyze_enhanced_campaign_performance_success(agent):
    mock_campaign = MagicMock()
    mock_campaign.name = "Test Campaign"
    mock_campaign.campaign_type = "brand_awareness"
    mock_campaign.budget = 1000
    mock_campaign.start_date = datetime.now().date()
    mock_campaign.end_date = None
    mock_campaign.status = "active"
    mock_campaign.target_audience = "tech"

    with (
        patch("src.que_agents.agents.marketing_agent.get_session") as mock_session,
        patch.object(
            agent, "get_enhanced_market_data", return_value={"benchmarks": {}}
        ),
    ):
        # Mock the analysis chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Analysis complete"
        agent.analysis_chain = mock_chain

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_campaign
        )
        mock_db.query.return_value.filter.return_value.all.return_value = []
        mock_session.return_value = mock_db
        result = agent.analyze_enhanced_campaign_performance("123")
        assert isinstance(result, str)


def test_analyze_enhanced_campaign_performance_db_exception(agent):
    # Test the fallback behavior by mocking the session to return a failing query
    with patch("src.que_agents.agents.marketing_agent.get_session") as mock_session:
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("db fail")
        mock_session.return_value = mock_db
        result = agent.analyze_enhanced_campaign_performance("123")
        assert "error" in result.lower()


def test_analyze_performance_metrics(agent):
    mock_metric = MagicMock()
    mock_metric.metric_name = "impressions"
    mock_metric.metric_value = 1000
    mock_metric.date = datetime.now()

    result = agent._analyze_performance_metrics([mock_metric, mock_metric])
    assert "impressions" in result


def test_analyze_performance_metrics_empty(agent):
    result = agent._analyze_performance_metrics([])
    assert "error" in result


def test_analyze_channel_performance(agent):
    mock_post = MagicMock()
    mock_post.platform = "twitter"
    mock_post.engagement = 100
    mock_post.reach = 1000

    result = agent._analyze_channel_performance([mock_post])
    assert "twitter" in result


def test_analyze_channel_performance_empty(agent):
    result = agent._analyze_channel_performance([])
    assert "error" in result


def test_analyze_audience_performance(agent):
    result = agent._analyze_audience_performance("tech professionals")
    assert "primary_segment" in result
    assert "engagement_by_demographic" in result


def test_optimize_campaign_enhanced_success(agent):
    mock_campaign = MagicMock()
    mock_campaign.budget = 1000

    with (
        patch.object(
            agent, "analyze_enhanced_campaign_performance", return_value="Analysis"
        ),
        patch("src.que_agents.agents.marketing_agent.get_session") as mock_session,
    ):
        # Mock the optimization chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Optimization"
        agent.optimization_chain = mock_chain

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_campaign
        )
        mock_session.return_value = mock_db
        result = agent.optimize_campaign_enhanced("123")
        assert "optimization_recommendations" in result


def test_optimize_campaign_enhanced_not_found(agent):
    with patch("src.que_agents.agents.marketing_agent.get_session") as mock_session:
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_session.return_value = mock_db
        result = agent.optimize_campaign_enhanced("123")
        assert "error" in result


def test_optimize_campaign_enhanced_error(agent):
    with patch.object(
        agent, "analyze_enhanced_campaign_performance", side_effect=Exception("error")
    ):
        result = agent.optimize_campaign_enhanced("123")
        assert "error" in result
        assert "fallback_recommendations" in result


def test_generate_optimization_actions(agent):
    performance_data = {
        "underperforming_channels": ["facebook"],
        "top_performing_channel": "twitter",
    }
    actions = agent._generate_optimization_actions(None, performance_data)
    assert isinstance(actions, list)
    assert len(actions) > 0


def test_estimate_optimization_impact(agent):
    current_performance = {"current_roi": 2.0}
    actions = [{"expected_impact": "15-25% ROI improvement"}]
    impact = agent._estimate_optimization_impact(current_performance, actions)
    assert "current_roi" in impact
    assert "projected_roi" in impact


def test_get_campaign_insights_dashboard_success(agent):
    mock_campaign = MagicMock()
    mock_campaign.name = "Test"
    mock_campaign.campaign_type = "brand_awareness"
    mock_campaign.status = "active"
    mock_campaign.budget = 1000
    mock_campaign.target_audience = "tech"
    mock_campaign.start_date = datetime.now().date()
    mock_campaign.end_date = None

    with patch("src.que_agents.agents.marketing_agent.get_session") as mock_session:
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_campaign
        )
        mock_db.query.return_value.filter.return_value.all.return_value = []
        mock_session.return_value = mock_db
        result = agent.get_campaign_insights_dashboard("123")
        assert "campaign_overview" in result


def test_get_campaign_insights_dashboard_not_found(agent):
    with patch("src.que_agents.agents.marketing_agent.get_session") as mock_session:
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_session.return_value = mock_db
        result = agent.get_campaign_insights_dashboard("123")
        assert "error" in result


def test_get_campaign_insights_dashboard_error(agent):
    # Test the fallback behavior by mocking the session to return a failing query
    with patch("src.que_agents.agents.marketing_agent.get_session") as mock_session:
        mock_db = MagicMock()
        mock_db.query.side_effect = Exception("error")
        mock_session.return_value = mock_db
        result = agent.get_campaign_insights_dashboard("123")
        assert "error" in result


def test_create_performance_summary(agent):
    mock_metric = MagicMock()
    mock_metric.metric_name = "impressions"
    mock_metric.metric_value = 1000

    result = agent._create_performance_summary(None, [mock_metric])
    assert "total_impressions" in result


def test_create_performance_summary_empty(agent):
    result = agent._create_performance_summary(None, [])
    assert "status" in result


def test_create_channel_breakdown(agent):
    mock_post = MagicMock()
    mock_post.platform = "twitter"
    mock_post.engagement = 100
    mock_post.reach = 1000

    result = agent._create_channel_breakdown([mock_post])
    assert "twitter" in result


def test_create_timeline_analysis(agent):
    mock_metric = MagicMock()
    mock_metric.metric_name = "impressions"
    mock_metric.metric_value = 1000
    mock_metric.date = datetime.now()

    result = agent._create_timeline_analysis([mock_metric])
    assert "daily_data" in result


def test_create_roi_analysis(agent):
    mock_campaign = MagicMock()
    mock_campaign.budget = 1000

    mock_metric = MagicMock()
    mock_metric.metric_name = "conversions"
    mock_metric.metric_value = 10

    result = agent._create_roi_analysis(mock_campaign, [mock_metric])
    assert "total_spend" in result
    assert "roi_percentage" in result


def test_calculate_campaign_duration(agent):
    mock_campaign = MagicMock()
    mock_campaign.start_date = datetime.now().date()
    mock_campaign.end_date = (datetime.now() + timedelta(days=30)).date()

    duration = agent._calculate_campaign_duration(mock_campaign)
    assert duration == 30


def test_generate_dashboard_recommendations(agent):
    recommendations = agent._generate_dashboard_recommendations(None, [])
    assert isinstance(recommendations, list)
    assert len(recommendations) <= 5


def test_generate_next_steps(agent):
    next_steps = agent._generate_next_steps(None)
    assert isinstance(next_steps, list)
    assert len(next_steps) > 0


def test_create_marketing_campaign_success(agent):
    request_data = {
        "campaign_type": "brand_awareness",
        "target_audience": "tech professionals",
        "budget": 10000,
        "duration_days": 30,
        "goals": ["awareness"],
        "channels": ["twitter"],
        "content_requirements": ["social_media"],
    }

    with patch.object(
        agent, "create_campaign_from_request", return_value={"campaign_id": "123"}
    ):
        result = agent.create_marketing_campaign(request_data)
        assert "campaign_id" in result


def test_create_marketing_campaign_error(agent):
    request_data = {
        "campaign_type": "invalid_type",
        "target_audience": "tech",
        "budget": 1000,
        "duration_days": 30,
    }

    result = agent.create_marketing_campaign(request_data)
    assert "error" in result


def test_generate_marketing_content_success(agent):
    request_data = {
        "platform": "twitter",
        "content_type": "social_media",
        "campaign_theme": "product launch",
        "target_audience": "tech professionals",
    }

    mock_content = MagicMock()
    mock_content.title = "Test Title"
    mock_content.content = "Test Content"
    mock_content.call_to_action = "Learn more"
    mock_content.hashtags = ["#test"]
    mock_content.platform = "twitter"
    mock_content.estimated_reach = 1000
    mock_content.optimization_score = 0.8

    with patch.object(agent, "generate_enhanced_content", return_value=mock_content):
        result = agent.generate_marketing_content(request_data)
        assert "title" in result
        assert result["title"] == "Test Title"


def test_generate_marketing_content_error(agent):
    request_data = {"platform": "twitter"}

    with patch.object(
        agent, "generate_enhanced_content", side_effect=Exception("error")
    ):
        result = agent.generate_marketing_content(request_data)
        assert "error" in result
        assert "fallback_content" in result


def test_analyze_campaign_performance_api(agent):
    with patch.object(
        agent, "analyze_enhanced_campaign_performance", return_value="Analysis"
    ):
        with patch.object(
            agent, "get_campaign_insights_dashboard", return_value={"data": "test"}
        ):
            result = agent.analyze_campaign_performance(123)
            assert "campaign_id" in result
            assert result["status"] == "success"


def test_optimize_campaign_api(agent):
    with patch.object(
        agent, "optimize_campaign_enhanced", return_value={"recommendations": []}
    ):
        result = agent.optimize_campaign(123)
        assert "campaign_id" in result
        assert result["status"] == "success"


def test_get_campaign_insights_api(agent):
    with patch.object(
        agent, "get_campaign_insights_dashboard", return_value={"insights": "test"}
    ):
        result = agent.get_campaign_insights(123)
        assert "campaign_id" in result
        assert result["status"] == "success"


def test_get_audience_analysis_api(agent):
    with patch.object(
        agent, "get_enhanced_audience_insights", return_value={"segments": []}
    ):
        result = agent.get_audience_analysis("tech professionals")
        assert "target_audience" in result
        assert result["status"] == "success"


def test_get_market_intelligence_api(agent):
    with patch.object(agent, "get_enhanced_market_data", return_value={"trends": []}):
        result = agent.get_market_intelligence("brand_awareness")
        assert "campaign_type" in result
        assert result["status"] == "success"


def test_get_content_suggestions_api(agent):
    with patch.object(agent, "get_marketing_knowledge", return_value=[]):
        result = agent.get_content_suggestions("twitter")
        assert "platform" in result
        assert "suggestions" in result
        assert result["status"] == "success"


def test_get_campaign_templates_api(agent):
    result = agent.get_campaign_templates("brand_awareness")
    assert "template" in result
    assert result["status"] == "success"


def test_create_campaign_from_request_success(agent, sample_request):
    mock_campaign = MagicMock()
    mock_campaign.id = 123
    mock_campaign.name = "Test Campaign"

    mock_plan = MagicMock()
    mock_plan.strategy = "Test strategy"
    mock_plan.content_pieces = []
    mock_plan.schedule = []
    mock_plan.budget_allocation = {}
    mock_plan.success_metrics = []
    mock_plan.estimated_performance = {}
    mock_plan.optimization_roadmap = []

    with (
        patch("src.que_agents.agents.marketing_agent.get_session") as mock_session,
        patch.object(agent, "create_enhanced_campaign_plan", return_value=mock_plan),
        patch.object(agent, "_save_campaign_to_db", return_value=mock_campaign),
        patch.object(agent, "_save_content_pieces_to_db", return_value=2),
    ):
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        result = agent.create_campaign_from_request(sample_request)
        assert "campaign_id" in result
        assert result["status"] == "created_successfully"


def test_create_campaign_from_request_error(agent, sample_request):
    # Test the fallback behavior by mocking the campaign plan creation to fail
    with patch.object(
        agent, "create_enhanced_campaign_plan", side_effect=Exception("plan error")
    ):
        result = agent.create_campaign_from_request(sample_request)
        assert "error" in result


def test_safe_enum_to_string(agent):
    assert agent._safe_enum_to_string(CampaignType.BRAND_AWARENESS) == "brand_awareness"
    assert agent._safe_enum_to_string("string_value") == "string_value"
    assert agent._safe_enum_to_string(None, "fallback") == "fallback"


def test_check_and_fix_database_schema(agent):
    mock_session = MagicMock()
    mock_engine = MagicMock()
    mock_inspector = MagicMock()

    with (
        patch("sqlalchemy.inspect", return_value=mock_inspector),
        patch.object(mock_session, "get_bind", return_value=mock_engine),
    ):
        mock_inspector.has_table.return_value = True
        mock_inspector.get_columns.return_value = [{"name": "id"}, {"name": "content"}]
        result = agent._check_and_fix_database_schema(mock_session)
        assert isinstance(result, bool)
