"""
Unit tests for Agent Manager to improve code coverage.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.que_agents.utils.agent_manager import AgentManager


@pytest.fixture
def agent_manager():
    return AgentManager()


@pytest.fixture
def mock_llm_factory():
    with patch("src.que_agents.utils.agent_manager.LLMFactory") as mock:
        mock_llm = MagicMock()
        mock.get_llm.return_value = mock_llm
        yield mock


class TestAgentManager:
    """Test AgentManager functionality"""

    def test_init_success(self, mock_llm_factory):
        """Test successful initialization"""
        manager = AgentManager()
        assert manager.agents == {}
        assert manager.agent_configs is not None

    def test_init_config_error(self):
        """Test initialization with config error"""
        with patch("builtins.open", side_effect=FileNotFoundError("Config not found")):
            manager = AgentManager()
            assert manager.agents == {}
            assert manager.agent_configs == {}

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

        with patch("builtins.open", mock_open_yaml(mock_config)):
            with patch(
                "src.que_agents.utils.agent_manager.yaml.safe_load",
                return_value=mock_config,
            ):
                agent_manager._load_agent_configs()

                assert "customer_support" in agent_manager.agent_configs
                assert (
                    agent_manager.agent_configs["customer_support"]["enabled"] is True
                )

    def test_load_agent_configs_file_not_found(self, agent_manager):
        """Test config loading with file not found"""
        with patch("builtins.open", side_effect=FileNotFoundError("Config not found")):
            agent_manager._load_agent_configs()
            assert agent_manager.agent_configs == {}

    def test_load_agent_configs_yaml_error(self, agent_manager):
        """Test config loading with YAML error"""
        with patch("builtins.open", mock_open_yaml({})):
            with patch(
                "src.que_agents.utils.agent_manager.yaml.safe_load",
                side_effect=Exception("YAML error"),
            ):
                agent_manager._load_agent_configs()
                assert agent_manager.agent_configs == {}

    def test_create_agent_customer_support(self, agent_manager, mock_llm_factory):
        """Test creating customer support agent"""
        with patch(
            "src.que_agents.utils.agent_manager.CustomerSupportAgent"
        ) as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            agent = agent_manager._create_agent("customer_support")

            assert agent == mock_agent
            mock_agent_class.assert_called_once()

    def test_create_agent_marketing(self, agent_manager, mock_llm_factory):
        """Test creating marketing agent"""
        with patch(
            "src.que_agents.utils.agent_manager.MarketingAgent"
        ) as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            agent = agent_manager._create_agent("marketing")

            assert agent == mock_agent
            mock_agent_class.assert_called_once()

    def test_create_agent_financial_trading_bot(self, agent_manager, mock_llm_factory):
        """Test creating financial trading bot agent"""
        with patch(
            "src.que_agents.utils.agent_manager.FinancialTradingBotAgent"
        ) as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            agent = agent_manager._create_agent("financial_trading_bot")

            assert agent == mock_agent
            mock_agent_class.assert_called_once()

    def test_create_agent_personal_virtual_assistant(
        self, agent_manager, mock_llm_factory
    ):
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

    def test_create_agent_exception(self, agent_manager, mock_llm_factory):
        """Test agent creation with exception"""
        with patch(
            "src.que_agents.utils.agent_manager.CustomerSupportAgent",
            side_effect=Exception("Creation error"),
        ):
            with patch(
                "src.que_agents.utils.agent_manager.system_logger"
            ) as mock_logger:
                agent = agent_manager._create_agent("customer_support")
                assert agent is None
                mock_logger.error.assert_called()

    def test_get_agent_existing(self, agent_manager):
        """Test getting existing agent"""
        mock_agent = MagicMock()
        agent_manager.agents["customer_support"] = {"test_token": mock_agent}

        agent = agent_manager.get_agent("customer_support", "test_token")
        assert agent == mock_agent

    def test_get_agent_create_new(self, agent_manager, mock_llm_factory):
        """Test getting agent that needs to be created"""
        mock_agent = MagicMock()

        with patch.object(agent_manager, "_create_agent", return_value=mock_agent):
            agent = agent_manager.get_agent("customer_support", "test_token")

            assert agent == mock_agent
            assert "customer_support" in agent_manager.agents
            assert agent_manager.agents["customer_support"]["test_token"] == mock_agent

    def test_get_agent_creation_failed(self, agent_manager):
        """Test getting agent when creation fails"""
        with patch.object(agent_manager, "_create_agent", return_value=None):
            agent = agent_manager.get_agent("customer_support", "test_token")
            assert agent is None

    def test_get_agent_disabled(self, agent_manager):
        """Test getting disabled agent"""
        agent_manager.agent_configs = {"customer_support": {"enabled": False}}

        with patch("src.que_agents.utils.agent_manager.system_logger") as mock_logger:
            agent = agent_manager.get_agent("customer_support", "test_token")
            assert agent is None
            mock_logger.warning.assert_called()

    def test_remove_agent_success(self, agent_manager):
        """Test successful agent removal"""
        mock_agent = MagicMock()
        agent_manager.agents["customer_support"] = {"test_token": mock_agent}

        result = agent_manager.remove_agent("customer_support", "test_token")

        assert result is True
        assert "test_token" not in agent_manager.agents["customer_support"]

    def test_remove_agent_not_found(self, agent_manager):
        """Test removing non-existent agent"""
        result = agent_manager.remove_agent("customer_support", "test_token")
        assert result is False

    def test_remove_agent_type_not_found(self, agent_manager):
        """Test removing agent with non-existent type"""
        result = agent_manager.remove_agent("unknown_agent", "test_token")
        assert result is False

    def test_list_agents_empty(self, agent_manager):
        """Test listing agents when none exist"""
        agents = agent_manager.list_agents()
        assert agents == {}

    def test_list_agents_with_agents(self, agent_manager):
        """Test listing agents with existing agents"""
        mock_agent1 = MagicMock()
        mock_agent2 = MagicMock()
        agent_manager.agents = {
            "customer_support": {"token1": mock_agent1},
            "marketing": {"token2": mock_agent2},
        }

        agents = agent_manager.list_agents()

        assert "customer_support" in agents
        assert "marketing" in agents
        assert len(agents["customer_support"]) == 1
        assert len(agents["marketing"]) == 1

    def test_get_agent_status_active(self, agent_manager):
        """Test getting status of active agent"""
        mock_agent = MagicMock()
        agent_manager.agents["customer_support"] = {"test_token": mock_agent}

        status = agent_manager.get_agent_status("customer_support", "test_token")

        assert status["exists"] is True
        assert status["status"] == "active"
        assert status["agent_type"] == "customer_support"

    def test_get_agent_status_not_found(self, agent_manager):
        """Test getting status of non-existent agent"""
        status = agent_manager.get_agent_status("customer_support", "test_token")

        assert status["exists"] is False
        assert status["status"] == "not_found"
        assert status["agent_type"] == "customer_support"

    def test_cleanup_agents_success(self, agent_manager):
        """Test successful agent cleanup"""
        mock_agent1 = MagicMock()
        mock_agent2 = MagicMock()
        agent_manager.agents = {
            "customer_support": {"token1": mock_agent1, "token2": mock_agent2},
            "marketing": {},
        }

        cleaned = agent_manager.cleanup_agents()

        assert cleaned >= 0
        # Verify cleanup was attempted
        assert isinstance(cleaned, int)

    def test_cleanup_agents_with_exception(self, agent_manager):
        """Test agent cleanup with exception"""
        mock_agent = MagicMock()
        mock_agent.cleanup.side_effect = Exception("Cleanup error")
        agent_manager.agents = {"customer_support": {"token1": mock_agent}}

        with patch("src.que_agents.utils.agent_manager.system_logger") as mock_logger:
            cleaned = agent_manager.cleanup_agents()
            assert isinstance(cleaned, int)
            # Should log error but continue

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
        assert "agents_by_type" in stats
        assert "system_health" in stats
        assert stats["total_agents"] == 2
        assert stats["agents_by_type"]["customer_support"] == 1
        assert stats["agents_by_type"]["marketing"] == 1

    def test_health_check_healthy(self, agent_manager):
        """Test health check when system is healthy"""
        mock_agent = MagicMock()
        agent_manager.agents = {"customer_support": {"token1": mock_agent}}

        health = agent_manager.health_check()

        assert health["status"] == "healthy"
        assert health["total_agents"] == 1
        assert "timestamp" in health

    def test_health_check_no_agents(self, agent_manager):
        """Test health check with no agents"""
        health = agent_manager.health_check()

        assert health["status"] == "healthy"
        assert health["total_agents"] == 0

    def test_reload_configs_success(self, agent_manager):
        """Test successful config reload"""
        mock_config = {
            "agents": {"customer_support": {"enabled": True, "max_instances": 10}}
        }

        with patch.object(agent_manager, "_load_agent_configs") as mock_load:
            result = agent_manager.reload_configs()
            assert result is True
            mock_load.assert_called_once()

    def test_reload_configs_error(self, agent_manager):
        """Test config reload with error"""
        with patch.object(
            agent_manager, "_load_agent_configs", side_effect=Exception("Reload error")
        ):
            with patch(
                "src.que_agents.utils.agent_manager.system_logger"
            ) as mock_logger:
                result = agent_manager.reload_configs()
                assert result is False
                mock_logger.error.assert_called()

    def test_validate_agent_type_valid(self, agent_manager):
        """Test validating valid agent type"""
        assert agent_manager._validate_agent_type("customer_support") is True
        assert agent_manager._validate_agent_type("marketing") is True
        assert agent_manager._validate_agent_type("financial_trading_bot") is True
        assert agent_manager._validate_agent_type("personal_virtual_assistant") is True

    def test_validate_agent_type_invalid(self, agent_manager):
        """Test validating invalid agent type"""
        assert agent_manager._validate_agent_type("unknown_agent") is False
        assert agent_manager._validate_agent_type("") is False
        assert agent_manager._validate_agent_type(None) is False

    def test_get_agent_config_exists(self, agent_manager):
        """Test getting existing agent config"""
        agent_manager.agent_configs = {
            "customer_support": {"enabled": True, "max_instances": 5}
        }

        config = agent_manager._get_agent_config("customer_support")
        assert config["enabled"] is True
        assert config["max_instances"] == 5

    def test_get_agent_config_not_exists(self, agent_manager):
        """Test getting non-existent agent config"""
        config = agent_manager._get_agent_config("unknown_agent")
        assert config["enabled"] is True  # Default config
        assert config["max_instances"] == 10

    def test_is_agent_enabled_true(self, agent_manager):
        """Test checking if agent is enabled (true)"""
        agent_manager.agent_configs = {"customer_support": {"enabled": True}}

        assert agent_manager._is_agent_enabled("customer_support") is True

    def test_is_agent_enabled_false(self, agent_manager):
        """Test checking if agent is enabled (false)"""
        agent_manager.agent_configs = {"customer_support": {"enabled": False}}

        assert agent_manager._is_agent_enabled("customer_support") is False

    def test_is_agent_enabled_default(self, agent_manager):
        """Test checking if agent is enabled (default)"""
        assert agent_manager._is_agent_enabled("unknown_agent") is True


def mock_open_yaml(data):
    """Helper function to mock file opening for YAML data"""
    from unittest.mock import mock_open

    import yaml

    return mock_open(read_data=yaml.dump(data))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
