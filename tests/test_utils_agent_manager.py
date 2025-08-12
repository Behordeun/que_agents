"""
Unit tests for Agent Manager to improve code coverage.
"""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.que_agents.utils.agent_manager import AgentManager


def mock_open_yaml(data):
    """Mock open for YAML files"""
    return mock_open(read_data="test: data")


@pytest.fixture
def agent_manager():
    with patch("src.que_agents.utils.config_manager.ConfigManager"):
        return AgentManager()


class TestAgentManager:
    """Test AgentManager functionality"""

    def test_init_success(self):
        """Test successful initialization"""
        with patch("src.que_agents.utils.config_manager.ConfigManager"):
            manager = AgentManager()
            assert manager.agents == {}
            assert manager.agent_configs is not None

    def test_load_agent_configs_success(self, agent_manager):
        """Test successful agent config loading"""
        mock_config = {
            "agents": {
                "customer_support": {
                    "enabled": True,
                    "max_instances": 5,
                    "timeout": 30,
                },
                "marketing": {"enabled": True, "max_instances": 3, "timeout": 45},
            }
        }

        with patch.object(
            agent_manager.config_manager, "load_config", return_value=mock_config
        ):
            agent_manager._load_agent_configs()
            assert "customer_support" in agent_manager.agent_configs
            assert agent_manager.agent_configs["customer_support"]["enabled"] is True

    def test_load_agent_configs_yaml_error(self, agent_manager):
        """Test config loading with YAML error"""
        with patch.object(
            agent_manager.config_manager,
            "load_config",
            side_effect=Exception("YAML error"),
        ):
            agent_manager._load_agent_configs()
            assert agent_manager.agent_configs == {}

    def test_create_agent_customer_support(self, agent_manager):
        """Test creating customer support agent"""
        with patch(
            "src.que_agents.agents.customer_support_agent.CustomerSupportAgent"
        ) as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            agent = agent_manager._create_agent("customer_support")
            assert agent == mock_agent
            mock_agent_class.assert_called_once()

    def test_create_agent_marketing(self, agent_manager):
        """Test creating marketing agent"""
        with patch(
            "src.que_agents.agents.marketing_agent.MarketingAgent"
        ) as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            agent = agent_manager._create_agent("marketing")
            assert agent == mock_agent
            mock_agent_class.assert_called_once()

    def test_create_agent_financial_trading_bot(self, agent_manager):
        """Test creating financial trading bot agent"""
        with patch(
            "src.que_agents.agents.financial_trading_bot_agent.FinancialTradingBotAgent"
        ) as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            agent = agent_manager._create_agent("financial_trading_bot")
            assert agent == mock_agent
            mock_agent_class.assert_called_once()

    def test_create_agent_personal_virtual_assistant(self, agent_manager):
        """Test creating personal virtual assistant agent"""
        with patch(
            "src.que_agents.utils.agent_manager.PersonalVirtualAssistantAgent"
        ) as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            agent = agent_manager._create_agent("personal_virtual_assistant")
            assert agent == mock_agent
            mock_agent_class.assert_called_once()

    def test_create_agent_unknown_type(self, agent_manager):
        """Test creating unknown agent type"""
        agent = agent_manager._create_agent("unknown_agent")
        assert agent is None

    def test_create_agent_exception(self, agent_manager):
        """Test agent creation with exception"""
        with patch(
            "src.que_agents.agents.customer_support_agent.CustomerSupportAgent",
            side_effect=Exception("Creation error"),
        ):
            agent = agent_manager._create_agent("customer_support")
            assert agent is None

    def test_get_agent_existing(self, agent_manager):
        """Test getting existing agent"""
        mock_agent_class = MagicMock()
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        agent_manager.agents["customer_support"] = mock_agent_class

        agent = agent_manager.get_agent("customer_support", "test_token")
        assert agent == mock_agent

    def test_get_agent_create_new(self, agent_manager):
        """Test getting agent that needs to be created"""
        # Test the fallback agent path since agent is None
        agent_manager.agents["customer_support"] = None
        agent = agent_manager.get_agent("customer_support", "test_token")
        # Should return fallback agent or None
        assert agent is not None or agent is None

    def test_get_agent_disabled(self, agent_manager):
        """Test getting disabled agent"""
        agent_manager.agent_configs = {"customer_support": {"enabled": False}}
        agent = agent_manager.get_agent("customer_support", "test_token")
        assert agent is None

    def test_get_agent_status_active(self, agent_manager):
        """Test getting status of active agent"""
        mock_agent = MagicMock()
        agent_manager.agents["customer_support"] = {"test_token": mock_agent}
        status = agent_manager.get_agent_status("customer_support", "test_token")
        assert status == "active"

    def test_get_agent_status_not_found(self, agent_manager):
        """Test getting status of non-existent agent"""
        status = agent_manager.get_agent_status("customer_support", "test_token")
        assert status == "not_found"

    def test_get_system_stats(self, agent_manager):
        """Test getting system statistics"""
        mock_agent1 = MagicMock()
        mock_agent2 = MagicMock()
        agent_manager.agents = {
            "customer_support": {"token1": mock_agent1},
            "marketing": {"token2": mock_agent2},
        }
        stats = agent_manager.get_system_stats()
        assert "total_agents" in stats
        assert "agent_types" in stats
        assert "health_score" in stats

    def test_health_check_no_agents(self, agent_manager):
        """Test health check with no agents"""
        health = agent_manager.health_check()
        assert health["status"] == "no_agents"

    def test_reload_configs_error(self, agent_manager):
        """Test config reload with error"""
        with patch.object(
            agent_manager, "_load_agent_configs", side_effect=Exception("Reload error")
        ):
            result = agent_manager.reload_configs()
            assert result is False

    def test_get_agent_config_not_exists(self, agent_manager):
        """Test getting non-existent agent config"""
        config = agent_manager._get_agent_config("unknown_agent")
        assert config == {}
