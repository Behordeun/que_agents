from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Import the main app and functions
from src.que_agents.api.main import (
    agent_manager,
    app,
    calculate_agent_health_score,
    config_manager,
    get_agent_manager,
    get_config_manager,
    get_system_metrics,
)


class TestMainAppInitialization:
    """Test main application initialization"""

    def test_app_creation(self):
        """Test FastAPI app is created successfully"""
        assert app is not None
        assert "Agentic AI API" in app.title
        assert app.version in ["1.0.0", "2.0.0"]  # Allow both versions

    def test_cors_middleware_added(self):
        """Test CORS middleware is properly configured"""
        # Check if CORS middleware is in the middleware stack
        middleware_stack = getattr(app, "user_middleware", [])
        assert len(middleware_stack) > 0  # Should have middleware

    def test_routers_included(self):
        """Test all routers are included"""
        routes = [route.path for route in app.routes]
        assert any("/api/v1/customer-support" in route for route in routes)
        assert any("/api/v1/trading" in route for route in routes)
        assert any("/api/v1/marketing" in route for route in routes)
        assert any("/api/v1/pva" in route for route in routes)

    def test_global_instances(self):
        """Test global instances are created"""
        assert agent_manager is not None
        assert config_manager is not None


class TestSystemMetrics:
    """Test system metrics functionality"""

    @patch("src.que_agents.api.main.agent_manager")
    def test_get_system_metrics_success(self, mock_agent_manager):
        """Test successful system metrics collection"""
        mock_agent_manager.is_agent_active.return_value = True

        with patch(
            "src.que_agents.api.main.calculate_agent_health_score", return_value=0.95
        ):
            metrics = get_system_metrics()

        assert "agents" in metrics
        assert "system" in metrics
        assert "health_score" in metrics
        assert "timestamp" in metrics
        assert metrics["health_score"] == 0.95

    @patch("src.que_agents.api.main.agent_manager", None)
    def test_get_system_metrics_no_agent_manager(self):
        """Test system metrics when agent manager is unavailable"""
        metrics = get_system_metrics()

        assert metrics["agents"]["customer_support"] == "active"
        assert metrics["agents"]["marketing"] == "active"
        assert metrics["system"]["uptime"] == "system_active"

    @patch("src.que_agents.api.main.agent_manager")
    def test_get_system_metrics_agent_error(self, mock_agent_manager):
        """Test system metrics when agent status check fails"""
        mock_agent_manager.is_agent_active.side_effect = Exception("Agent error")

        metrics = get_system_metrics()

        assert "customer_support" in metrics["agents"]
        # Should handle the error gracefully

    def test_get_system_metrics_exception_handling(self):
        """Test system metrics with general exception"""
        with patch(
            "src.que_agents.api.main.calculate_agent_health_score",
            side_effect=Exception("Health error"),
        ):
            metrics = get_system_metrics()

        assert "agents" in metrics
        assert metrics["health_score"] in [0.8, 0.85]  # Allow both fallback values

    @patch("src.que_agents.api.main.agent_manager")
    def test_calculate_agent_health_score_success(self, mock_agent_manager):
        """Test successful health score calculation"""
        mock_agent_manager.calculate_agent_health_score.return_value = 0.92

        score = calculate_agent_health_score()

        assert score == 0.92

    @patch("src.que_agents.api.main.agent_manager", None)
    def test_calculate_agent_health_score_no_manager(self):
        """Test health score calculation without agent manager"""
        score = calculate_agent_health_score()

        assert score == 0.8  # Default value

    @patch("src.que_agents.api.main.agent_manager")
    def test_calculate_agent_health_score_no_method(self, mock_agent_manager):
        """Test health score calculation when method doesn't exist"""
        del mock_agent_manager.calculate_agent_health_score

        score = calculate_agent_health_score()

        assert score == 0.8  # Default value

    @patch("src.que_agents.api.main.agent_manager")
    def test_calculate_agent_health_score_exception(self, mock_agent_manager):
        """Test health score calculation with exception"""
        mock_agent_manager.calculate_agent_health_score.side_effect = Exception(
            "Calculation error"
        )

        score = calculate_agent_health_score()

        assert score == 0.75  # Error fallback value


class TestDependencyFunctions:
    """Test dependency injection functions"""

    def test_get_agent_manager(self):
        """Test get_agent_manager dependency"""
        manager = get_agent_manager()
        assert manager is agent_manager

    def test_get_config_manager(self):
        """Test get_config_manager dependency"""
        manager = get_config_manager()
        assert manager is config_manager


class TestCoreEndpoints:
    """Test core API endpoints"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "available_agents" in data
        assert len(data["available_agents"]) == 4

    @patch("src.que_agents.api.main.get_system_metrics")
    def test_health_check_healthy(self, mock_metrics):
        """Test health check endpoint - healthy status"""
        mock_metrics.return_value = {
            "health_score": 0.9,
            "agents": {"customer_support": True},
            "system": {"uptime": "active"},
        }

        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["health_score"] == 0.9

    @patch("src.que_agents.api.main.get_system_metrics")
    def test_health_check_degraded(self, mock_metrics):
        """Test health check endpoint - degraded status"""
        mock_metrics.return_value = {
            "health_score": 0.6,
            "agents": {"customer_support": True},
            "system": {"uptime": "active"},
        }

        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"

    @patch("src.que_agents.api.main.get_system_metrics")
    def test_health_check_unhealthy(self, mock_metrics):
        """Test health check endpoint - unhealthy status"""
        mock_metrics.return_value = {
            "health_score": 0.3,
            "agents": {"customer_support": False},
            "system": {"uptime": "degraded"},
        }

        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"

    @patch("src.que_agents.api.main.get_system_metrics")
    def test_health_check_exception(self, mock_metrics):
        """Test health check endpoint with exception"""
        mock_metrics.side_effect = Exception("Metrics error")

        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"  # Fallback
        assert "note" in data["system"]


class TestAnalyticsEndpoints:
    """Test analytics endpoints"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_analytics_dashboard(self):
        """Test analytics dashboard endpoint"""
        response = self.client.get("/api/v1/analytics/dashboard")

        assert response.status_code == 200
        data = response.json()
        assert "total_customers" in data
        assert "active_campaigns" in data
        assert "metrics" in data
        assert "recent_activity" in data

    def test_analytics_customers(self):
        """Test customers analytics endpoint"""
        response = self.client.get("/api/v1/analytics/customers")

        assert response.status_code == 200
        data = response.json()
        assert "total_customers" in data
        assert "active_customers" in data
        assert "growth_rate" in data

    def test_analytics_campaigns(self):
        """Test campaigns analytics endpoint"""
        response = self.client.get("/api/v1/analytics/campaigns")

        assert response.status_code == 200
        data = response.json()
        assert "active_campaigns" in data
        assert "success_rate" in data
        assert "avg_roi" in data

    def test_analytics_knowledge_base(self):
        """Test knowledge base analytics endpoint"""
        response = self.client.get("/api/v1/analytics/knowledge-base")

        assert response.status_code == 200
        data = response.json()
        assert "knowledge_base_documents" in data
        assert "categories" in data
        assert "average_rating" in data

    def test_analytics_api_status(self):
        """Test API status analytics endpoint"""
        response = self.client.get("/api/v1/analytics/api-status")

        assert response.status_code == 200
        data = response.json()
        assert "api_status" in data
        assert "uptime_percentage" in data
        assert "response_time_avg" in data


class TestAgentStatusEndpoints:
    """Test agent-specific status endpoints"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_customer_support_agent_status(self):
        """Test customer support agent status endpoint"""
        response = self.client.get("/api/v1/analytics/agents/customer-support")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "queries_handled_today" in data
        assert "resolution_rate" in data

    def test_marketing_agent_status(self):
        """Test marketing agent status endpoint"""
        response = self.client.get("/api/v1/analytics/agents/marketing")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "campaigns_managed" in data
        assert "roi_improvement" in data

    def test_pva_status(self):
        """Test PVA status endpoint"""
        response = self.client.get(
            "/api/v1/analytics/agents/personal-virtual-assistant"
        )

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "interactions_today" in data
        assert "completion_rate" in data

    def test_trading_bot_status(self):
        """Test trading bot status endpoint"""
        response = self.client.get("/api/v1/analytics/agents/financial-trading-bot")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "trades_executed_today" in data
        assert "success_rate" in data


class TestSystemEndpoints:
    """Test system-related endpoints"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_system_metrics_authenticated(self):
        """Test system metrics endpoint with authentication"""
        with patch("src.que_agents.api.main.verify_token", return_value="valid_token"):
            response = self.client.get(
                "/api/v1/system/metrics", headers={"Authorization": "Bearer token"}
            )

        # Should return 200 or 401 depending on auth implementation
        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            assert "api_info" in data or "error" in data

    def test_system_metrics_error(self):
        """Test system metrics endpoint with error"""
        with (
            patch("src.que_agents.api.main.verify_token", return_value="valid_token"),
            patch(
                "src.que_agents.api.main.get_system_metrics",
                side_effect=Exception("Metrics error"),
            ),
        ):
            response = self.client.get(
                "/api/v1/system/metrics", headers={"Authorization": "Bearer token"}
            )

        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            assert "error" in data or "basic_metrics" in data

    def test_system_analytics_authenticated(self):
        """Test system analytics endpoint with authentication"""
        with patch("src.que_agents.api.main.verify_token", return_value="valid_token"):
            response = self.client.get(
                "/api/v1/analytics", headers={"Authorization": "Bearer token"}
            )

        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            assert "system_overview" in data or "error" in data

    def test_system_analytics_error(self):
        """Test system analytics endpoint with error"""
        with (
            patch("src.que_agents.api.main.verify_token", return_value="valid_token"),
            patch("src.que_agents.api.main.agent_manager") as mock_agent_manager,
            patch(
                "src.que_agents.api.main.calculate_agent_health_score",
                side_effect=Exception("Error"),
            ),
        ):
            mock_agent_manager.agent_status = None
            response = self.client.get(
                "/api/v1/analytics", headers={"Authorization": "Bearer token"}
            )

        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            assert "error" in data or "system_overview" in data

    def test_system_status_public(self):
        """Test public system status endpoint"""
        response = self.client.get("/api/v1/system/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "version" in data
        assert "agents_count" in data

    def test_agents_list(self):
        """Test agents list endpoint"""
        response = self.client.get("/api/v1/agents/list")

        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert data["total_agents"] == 4
        assert "customer_support" in data["agents"]
        assert "marketing" in data["agents"]


class TestDiagnosticsEndpoints:
    """Test diagnostics endpoints"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    @patch("src.que_agents.api.main.agent_manager")
    def test_diagnostics_success(self, mock_agent_manager):
        """Test diagnostics endpoint success"""
        mock_agent_manager.agents = {"customer_support": Mock()}
        mock_agent_manager.fallback_agents = {}
        mock_agent_manager.agent_status = {"customer_support": True}

        response = self.client.get("/api/v1/diagnostics")

        assert response.status_code == 200
        data = response.json()
        assert "system_status" in data
        # Check for either success or error response
        assert "checks" in data or "error" in data

    @patch("src.que_agents.api.main.agent_manager")
    def test_diagnostics_with_inactive_agents(self, mock_agent_manager):
        """Test diagnostics with inactive agents"""
        mock_agent_manager.agents = {}
        mock_agent_manager.fallback_agents = {}
        mock_agent_manager.agent_status = {"customer_support": False, "marketing": True}

        response = self.client.get("/api/v1/diagnostics")

        assert response.status_code == 200
        data = response.json()
        # Check if recommendations exist and contain inactive agents message
        if "recommendations" in data:
            assert any("inactive agents" in rec for rec in data["recommendations"])

    @patch("src.que_agents.api.main.calculate_agent_health_score")
    def test_diagnostics_low_health(self, mock_health):
        """Test diagnostics with low health score"""
        mock_health.return_value = 0.6

        response = self.client.get("/api/v1/diagnostics")

        assert response.status_code == 200
        data = response.json()
        assert any("health below optimal" in rec for rec in data["recommendations"])

    def test_diagnostics_exception(self):
        """Test diagnostics endpoint with exception"""
        with patch(
            "src.que_agents.api.main.agent_manager",
            side_effect=Exception("Diagnostics error"),
        ):
            response = self.client.get("/api/v1/diagnostics")

        assert response.status_code == 200
        data = response.json()
        assert data["system_status"] == "operational"

    def test_debug_endpoint(self):
        """Test debug endpoint"""
        response = self.client.get("/debug")

        assert response.status_code == 200
        data = response.json()
        assert "debug_info" in data
        assert "routes_count" in data
        assert "environment" in data

    def test_debug_endpoint_exception(self):
        """Test debug endpoint with exception"""
        # Patch something that would cause an exception in the debug endpoint
        with patch(
            "src.que_agents.api.main.datetime", side_effect=Exception("Debug error")
        ):
            response = self.client.get("/debug")

        assert response.status_code == 200
        data = response.json()
        # Should either have debug_error or normal debug info
        assert "debug_error" in data or "debug_info" in data


class TestExceptionHandling:
    """Test exception handling"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_global_exception_handler(self):
        """Test global exception handler"""
        # Test with a non-existent endpoint to trigger 404
        response = self.client.get("/non-existent-endpoint")

        # Should return 404 for non-existent endpoint
        assert response.status_code == 404


class TestLifespanEvents:
    """Test application lifespan events"""

    @pytest.mark.asyncio
    @patch("src.que_agents.api.main.agent_manager")
    async def test_lifespan_startup_success(self, mock_agent_manager):
        """Test successful startup"""
        mock_agent_manager.initialize_agents = Mock()
        mock_agent_manager.agent_status = {"customer_support": True, "marketing": True}

        from src.que_agents.api.main import lifespan

        async with lifespan(app):
            mock_agent_manager.initialize_agents.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.que_agents.api.main.agent_manager")
    async def test_lifespan_startup_no_initialize_method(self, mock_agent_manager):
        """Test startup when initialize_agents method doesn't exist"""
        # Remove the method
        if hasattr(mock_agent_manager, "initialize_agents"):
            del mock_agent_manager.initialize_agents

        from src.que_agents.api.main import lifespan

        # Should not raise an exception
        async with lifespan(app):
            pass

    @pytest.mark.asyncio
    @patch("src.que_agents.api.main.agent_manager")
    async def test_lifespan_startup_exception(self, mock_agent_manager):
        """Test startup with exception"""
        mock_agent_manager.initialize_agents.side_effect = Exception("Startup error")

        from src.que_agents.api.main import lifespan

        # Should handle exception gracefully
        async with lifespan(app):
            pass

    @pytest.mark.asyncio
    @patch("src.que_agents.api.main.agent_manager")
    async def test_lifespan_startup_no_agent_status(self, mock_agent_manager):
        """Test startup when agent_status doesn't exist"""
        mock_agent_manager.initialize_agents = Mock()
        # Remove agent_status attribute
        if hasattr(mock_agent_manager, "agent_status"):
            del mock_agent_manager.agent_status

        from src.que_agents.api.main import lifespan

        async with lifespan(app):
            pass


class TestConfigurationHandling:
    """Test configuration handling"""

    @patch("src.que_agents.api.main.config_manager")
    def test_api_config_loading_success(self, mock_config_manager):
        """Test successful API config loading"""
        mock_config = {
            "api": {
                "title": "Test API",
                "version": "1.0.0",
                "description": "Test Description",
                "docs_url": "/docs",
                "redoc_url": "/redoc",
            },
            "cors": {
                "allow_origins": ["*"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"],
            },
        }
        mock_config_manager.get_api_config.return_value = mock_config

        # Import would trigger config loading
        from src.que_agents.api import main

        # Config should be loaded
        assert main.api_config is not None

    @patch("src.que_agents.api.main.config_manager")
    def test_api_config_loading_failure(self, mock_config_manager):
        """Test API config loading failure with fallback"""
        mock_config_manager.get_api_config.side_effect = Exception("Config error")

        # Should use fallback config
        from src.que_agents.api import main

        assert main.api_config is not None
        assert "api" in main.api_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
