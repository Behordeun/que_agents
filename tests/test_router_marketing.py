"""
Unit tests for Marketing Router to improve code coverage.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.que_agents.router.marketing import (
    MarketingAgentService,
    get_marketing_service,
    router,
)


@pytest.fixture
def agent_manager():
    return MagicMock()


@pytest.fixture
def service(agent_manager):
    return MarketingAgentService(agent_manager)


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.create_marketing_campaign.return_value = {
        "campaign_id": 123,
        "campaign_plan": {"strategy": "digital marketing"},
        "content_pieces": [{"type": "email", "content": "Test email"}],
        "estimated_reach": 10000,
        "budget_allocation": {"email": 5000, "social": 5000},
    }
    agent.generate_marketing_content.return_value = {
        "content": "Generated marketing content",
        "content_type": "email",
        "target_audience": "professionals",
        "tone": "professional",
    }
    agent.analyze_campaign_performance.return_value = {
        "campaign_id": 123,
        "performance_metrics": {"clicks": 1000, "conversions": 50},
        "roi": 2.5,
        "recommendations": ["Increase budget"],
    }
    return agent


class TestMarketingAgentService:
    """Test MarketingAgentService functionality"""

    def test_get_agent_success(self, service, agent_manager, mock_agent):
        """Test successful agent retrieval"""
        agent_manager.get_agent.return_value = mock_agent
        agent = service.get_agent("test_token")
        assert agent == mock_agent
        agent_manager.get_agent.assert_called_once_with("marketing", "test_token")

    def test_get_agent_not_found(self, service, agent_manager):
        """Test agent not found scenario"""
        agent_manager.get_agent.return_value = None
        with patch("src.que_agents.router.marketing.system_logger") as mock_logger:
            agent = service.get_agent("test_token")
            assert agent is None
            mock_logger.error.assert_called_once()

    def test_create_campaign_success(self, service, mock_agent):
        """Test successful campaign creation"""
        service.get_agent = MagicMock(return_value=mock_agent)
        request = {
            "campaign_type": "email",
            "target_audience": "professionals",
            "budget": 10000.0,
            "duration_days": 30,
        }

        result = service.create_campaign(request, "test_token")

        assert result["success"] is True
        assert result["campaign_id"] == 123
        mock_agent.create_marketing_campaign.assert_called_once()

    def test_create_campaign_missing_fields(self, service):
        """Test campaign creation with missing required fields"""
        service.get_agent = MagicMock(return_value=MagicMock())
        request = {"campaign_type": "email"}  # Missing required fields

        with pytest.raises(HTTPException) as exc:
            service.create_campaign(request, "test_token")
        assert exc.value.status_code == 400

    def test_create_campaign_agent_unavailable(self, service):
        """Test campaign creation with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)
        request = {
            "campaign_type": "email",
            "target_audience": "professionals",
            "budget": 10000.0,
            "duration_days": 30,
        }

        with pytest.raises(HTTPException) as exc:
            service.create_campaign(request, "test_token")
        assert exc.value.status_code == 503

    def test_create_campaign_agent_error(self, service, mock_agent):
        """Test campaign creation with agent error"""
        mock_agent.create_marketing_campaign.return_value = {"error": "Agent error"}
        service.get_agent = MagicMock(return_value=mock_agent)
        request = {
            "campaign_type": "email",
            "target_audience": "professionals",
            "budget": 10000.0,
            "duration_days": 30,
        }

        with pytest.raises(HTTPException) as exc:
            service.create_campaign(request, "test_token")
        assert exc.value.status_code == 400

    def test_create_campaign_technical_issues(self, service, mock_agent):
        """Test campaign creation with technical issues"""
        mock_agent.create_marketing_campaign.return_value = {
            "error": "technical issues",
            "message": "Technical issues occurred",
        }
        service.get_agent = MagicMock(return_value=mock_agent)
        request = {
            "campaign_type": "email",
            "target_audience": "professionals",
            "budget": 10000.0,
            "duration_days": 30,
        }

        result = service.create_campaign(request, "test_token")

        assert result["success"] is False
        assert result["error"] == "technical_issue"

    def test_create_campaign_exception(self, service, mock_agent):
        """Test campaign creation with exception"""
        mock_agent.create_marketing_campaign.side_effect = Exception("Unexpected error")
        service.get_agent = MagicMock(return_value=mock_agent)
        request = {
            "campaign_type": "email",
            "target_audience": "professionals",
            "budget": 10000.0,
            "duration_days": 30,
        }

        with patch("src.que_agents.router.marketing.system_logger"):
            with pytest.raises(HTTPException) as exc:
                service.create_campaign(request, "test_token")
            assert exc.value.status_code == 500

    def test_generate_content_success(self, service, mock_agent):
        """Test successful content generation"""
        service.get_agent = MagicMock(return_value=mock_agent)
        request = {
            "platform": "social_media",
            "content_type": "post",
            "campaign_theme": "technology",
        }

        result = service.generate_content(request, "test_token")

        assert result["success"] is True
        assert "content" in result
        mock_agent.generate_marketing_content.assert_called_once()

    def test_generate_content_agent_unavailable(self, service):
        """Test content generation with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)
        request = {"platform": "social_media"}

        result = service.generate_content(request, "test_token")

        assert result["success"] is True
        assert result["generation_method"] == "fallback"

    def test_generate_content_agent_error(self, service, mock_agent):
        """Test content generation with agent error"""
        mock_agent.generate_marketing_content.side_effect = Exception("Content error")
        service.get_agent = MagicMock(return_value=mock_agent)
        request = {"platform": "social_media"}

        with patch("src.que_agents.router.marketing.system_logger"):
            result = service.generate_content(request, "test_token")

            assert result["success"] is True
            assert result["generation_method"] == "fallback"

    def test_generate_content_agent_missing_method(self, service, mock_agent):
        """Test content generation when agent missing method"""
        del mock_agent.generate_marketing_content
        service.get_agent = MagicMock(return_value=mock_agent)
        request = {"platform": "social_media"}

        with patch("src.que_agents.router.marketing.system_logger"):
            result = service.generate_content(request, "test_token")

            assert result["success"] is True
            assert result["generation_method"] == "fallback"

    def test_generate_fallback_content_social(self, service):
        """Test fallback content generation for social media"""
        request = {
            "platform": "social_media",
            "campaign_theme": "technology",
            "target_audience": "developers",
        }

        result = service._generate_fallback_content(request)

        assert result["success"] is True
        assert result["generation_method"] == "fallback"
        assert "content" in result

    def test_generate_fallback_content_email(self, service):
        """Test fallback content generation for email"""
        request = {
            "platform": "email",
            "campaign_theme": "marketing",
            "target_audience": "professionals",
        }

        result = service._generate_fallback_content(request)

        assert result["success"] is True
        assert "content" in result
        assert "subject_line" in result["content"]

    def test_generate_fallback_content_blog(self, service):
        """Test fallback content generation for blog"""
        request = {
            "platform": "blog",
            "campaign_theme": "business",
            "target_audience": "entrepreneurs",
        }

        result = service._generate_fallback_content(request)

        assert result["success"] is True
        assert "content" in result
        assert "title" in result["content"]

    def test_generate_fallback_content_generic(self, service):
        """Test fallback content generation for generic platform"""
        request = {
            "platform": "unknown",
            "campaign_theme": "general",
            "target_audience": "users",
        }

        result = service._generate_fallback_content(request)

        assert result["success"] is True
        assert "content" in result

    def test_analyze_campaign_performance_success(self, service, mock_agent):
        """Test successful campaign performance analysis"""
        service.get_agent = MagicMock(return_value=mock_agent)

        result = service.analyze_campaign_performance("123", "test_token")

        assert "performance_metrics" in result
        assert result["data_source"] == "agent_analytics"
        mock_agent.analyze_campaign_performance.assert_called_once_with("123")

    def test_analyze_campaign_performance_agent_unavailable(self, service):
        """Test campaign performance analysis with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)

        result = service.analyze_campaign_performance("123", "test_token")

        assert "campaign_id" in result
        assert result["data_source"] == "fallback_analytics"

    def test_analyze_campaign_performance_agent_error(self, service, mock_agent):
        """Test campaign performance analysis with agent error"""
        mock_agent.analyze_campaign_performance.side_effect = Exception(
            "Analysis error"
        )
        service.get_agent = MagicMock(return_value=mock_agent)

        with patch("src.que_agents.router.marketing.system_logger"):
            result = service.analyze_campaign_performance("123", "test_token")

            assert result["data_source"] == "fallback_analytics"

    def test_analyze_campaign_performance_agent_unavailable_result(
        self, service, mock_agent
    ):
        """Test campaign performance analysis with unavailable result"""
        mock_agent.analyze_campaign_performance.return_value = {"status": "unavailable"}
        service.get_agent = MagicMock(return_value=mock_agent)

        result = service.analyze_campaign_performance("123", "test_token")

        assert result["data_source"] == "fallback_analytics"

    def test_get_campaign_list_success(self, service, mock_agent):
        """Test successful campaign list retrieval"""
        mock_agent.get_campaign_list.return_value = {
            "campaigns": [{"id": 1, "name": "Test Campaign"}],
            "total": 1,
        }
        service.get_agent = MagicMock(return_value=mock_agent)

        result = service.get_campaign_list("active", 10, "test_token")

        assert "campaigns" in result
        assert result["data_source"] == "agent_data"
        mock_agent.get_campaign_list.assert_called_once_with("active", 10)

    def test_get_campaign_list_agent_unavailable(self, service):
        """Test campaign list with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)

        result = service.get_campaign_list("active", 5, "test_token")

        assert "campaigns" in result
        assert result["data_source"] == "fallback_data"

    def test_get_campaign_list_agent_error(self, service, mock_agent):
        """Test campaign list with agent error"""
        mock_agent.get_campaign_list.side_effect = Exception("List error")
        service.get_agent = MagicMock(return_value=mock_agent)

        with patch("src.que_agents.router.marketing.system_logger"):
            result = service.get_campaign_list("active", 5, "test_token")

            assert result["data_source"] == "fallback_data"

    def test_get_content_templates_success(self, service, mock_agent):
        """Test successful content templates retrieval"""
        mock_agent.get_content_templates.return_value = {
            "templates": [{"id": 1, "name": "Template 1"}]
        }
        service.get_agent = MagicMock(return_value=mock_agent)

        result = service.get_content_templates("email", "test_token")

        assert "templates" in result
        mock_agent.get_content_templates.assert_called_once_with("email")

    def test_get_content_templates_agent_unavailable(self, service):
        """Test content templates with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)

        result = service.get_content_templates("email", "test_token")

        assert "templates" in result
        assert result["data_source"] == "fallback_templates"

    def test_get_content_templates_agent_error(self, service, mock_agent):
        """Test content templates with agent error"""
        mock_agent.get_content_templates.side_effect = Exception("Template error")
        service.get_agent = MagicMock(return_value=mock_agent)

        with patch("src.que_agents.router.marketing.system_logger"):
            result = service.get_content_templates("email", "test_token")

            assert result["data_source"] == "fallback_templates"

    def test_generate_fallback_performance_data(self, service):
        """Test fallback performance data generation"""
        result = service._generate_fallback_performance_data("test_campaign")

        assert result["campaign_id"] == "test_campaign"
        assert "performance_metrics" in result
        assert "engagement_metrics" in result
        assert result["data_source"] == "fallback_analytics"

    def test_generate_fallback_performance_data_with_error(self, service):
        """Test fallback performance data with error"""
        result = service._generate_fallback_performance_data(
            "test_campaign", error="Test error"
        )

        assert "error_note" in result
        assert "Test error" in result["error_note"]

    def test_enhance_performance_data(self, service):
        """Test performance data enhancement"""
        data = {"campaign_id": "123", "metrics": {"clicks": 100}}

        result = service._enhance_performance_data(data)

        assert result["campaign_id"] == "123"
        assert result["data_source"] == "agent_analytics"
        assert result["analysis_depth"] == "comprehensive"

    def test_generate_fallback_campaign_list(self, service):
        """Test fallback campaign list generation"""
        result = service._generate_fallback_campaign_list("active", 5)

        assert "campaigns" in result
        assert result["data_source"] == "fallback_data"
        assert len(result["campaigns"]) <= 5

    def test_enhance_campaign_list(self, service):
        """Test campaign list enhancement"""
        campaigns = {"campaigns": [{"id": 1}], "total": 1}

        result = service._enhance_campaign_list(campaigns)

        assert result["campaigns"] == [{"id": 1}]
        assert result["data_source"] == "agent_data"

    def test_generate_fallback_templates_specific_type(self, service):
        """Test fallback templates for specific content type"""
        result = service._generate_fallback_templates("email")

        assert result["content_type"] == "email"
        assert "templates" in result
        assert result["data_source"] == "fallback_templates"

    def test_generate_fallback_templates_all_types(self, service):
        """Test fallback templates for all content types"""
        result = service._generate_fallback_templates()

        assert "templates" in result
        assert "content_types" in result
        assert result["data_source"] == "fallback_templates"

    def test_generate_social_content(self, service):
        """Test social media content generation"""
        result = service._generate_social_content(
            "technology", "developers", "professional"
        )

        assert "title" in result
        assert "content" in result
        assert "hashtags" in result
        assert result["platform"] == "social_media"

    def test_generate_email_content(self, service):
        """Test email content generation"""
        result = service._generate_email_content(
            "marketing", "professionals", "friendly"
        )

        assert "subject_line" in result
        assert "content" in result
        assert result["platform"] == "email"

    def test_generate_blog_content(self, service):
        """Test blog content generation"""
        result = service._generate_blog_content(
            "business", "entrepreneurs", "authoritative"
        )

        assert "title" in result
        assert "content" in result
        assert result["platform"] == "blog"

    def test_generate_generic_content(self, service):
        """Test generic content generation"""
        result = service._generate_generic_content("general", "users", "casual")

        assert "title" in result
        assert "content" in result
        assert result["platform"] == "generic"


class TestRouterDependencies:
    """Test router dependencies and endpoints"""

    def test_get_marketing_service(self):
        """Test service dependency creation"""
        with patch("src.que_agents.router.marketing.agent_manager") as mock_manager:
            service = get_marketing_service()
            assert isinstance(service, MarketingAgentService)
            assert service.agent_manager == mock_manager

    def test_router_exists(self):
        """Test router is properly configured"""
        assert router is not None
        assert hasattr(router, "routes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
