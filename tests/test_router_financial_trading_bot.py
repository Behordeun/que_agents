"""
Unit tests for Financial Trading Bot Router to improve code coverage.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.que_agents.router.financial_trading_bot import (
    FinancialTradingBotService,
    get_trading_service,
    router,
)


@pytest.fixture
def agent_manager():
    return MagicMock()


@pytest.fixture
def service(agent_manager):
    return FinancialTradingBotService(agent_manager)


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    # Mock the decision object
    mock_decision = MagicMock()
    mock_decision.action = "buy"
    mock_decision.symbol = "AAPL"
    mock_decision.quantity = 100
    mock_decision.confidence = 0.85
    mock_decision.reasoning = "Strong fundamentals"
    mock_decision.risk_score = 0.3
    mock_decision.expected_return = 0.15

    agent.make_trading_decision.return_value = mock_decision
    return agent


class TestFinancialTradingBotService:
    """Test FinancialTradingBotService functionality"""

    def test_get_agent_success(self, service, agent_manager, mock_agent):
        """Test successful agent retrieval"""
        agent_manager.get_agent.return_value = mock_agent
        agent = service.get_agent("test_token")
        assert agent == mock_agent
        agent_manager.get_agent.assert_called_once_with(
            "financial_trading_bot", "test_token"
        )

    def test_get_agent_not_found(self, service, agent_manager):
        """Test agent not found scenario"""
        agent_manager.get_agent.return_value = None
        with patch(
            "src.que_agents.router.financial_trading_bot.system_logger"
        ) as mock_logger:
            agent = service.get_agent("test_token")
            assert agent is None
            mock_logger.error.assert_called_once()

    def test_run_trading_cycle_operation_success(self, service, mock_agent):
        """Test successful trading cycle operation"""
        service.get_agent = MagicMock(return_value=mock_agent)
        mock_agent.run_trading_cycle.return_value = {
            "trades_executed": 3,
            "decisions": [{"symbol": "AAPL", "action": "buy"}],
        }

        result = service.run_trading_cycle_operation(["AAPL", "GOOGL"], "test_token")

        assert "trades_executed" in result or "decisions" in result
        mock_agent.run_trading_cycle.assert_called_once_with(["AAPL", "GOOGL"])

    def test_run_trading_cycle_operation_agent_unavailable(self, service):
        """Test trading cycle with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)

        with pytest.raises(HTTPException) as exc:
            service.run_trading_cycle_operation(["AAPL"], "test_token")
        assert exc.value.status_code == 503

    def test_run_trading_cycle_operation_fallback(self, service, mock_agent):
        """Test trading cycle with fallback when agent doesn't have method"""
        service.get_agent = MagicMock(return_value=mock_agent)
        # Remove the method to test fallback
        del mock_agent.run_trading_cycle

        result = service.run_trading_cycle_operation(["AAPL"], "test_token")

        assert "fallback_mode" in result
        assert result["fallback_mode"] is True

    def test_run_trading_cycle_operation_error(self, service, mock_agent):
        """Test trading cycle with error"""
        mock_agent.run_trading_cycle.side_effect = Exception("Cycle error")
        service.get_agent = MagicMock(return_value=mock_agent)

        with patch("src.que_agents.router.financial_trading_bot.system_logger"):
            with pytest.raises(HTTPException) as exc:
                service.run_trading_cycle_operation(["AAPL"], "test_token")
            assert exc.value.status_code == 500

    def test_get_portfolio_status_data_success(self, service, mock_agent):
        """Test successful portfolio status retrieval"""
        mock_portfolio = MagicMock()
        mock_portfolio.total_value = 15000.0
        mock_portfolio.cash_balance = 3000.0
        mock_agent.get_portfolio_status.return_value = mock_portfolio
        service.get_agent = MagicMock(return_value=mock_agent)

        result = service.get_portfolio_status_data("test_token")

        assert "portfolio_value" in result
        assert result["portfolio_value"] == 15000.0

    def test_get_portfolio_status_data_agent_unavailable(self, service):
        """Test portfolio status with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)

        result = service.get_portfolio_status_data("test_token")

        assert "status" in result
        assert result["status"] == "agent_unavailable"

    def test_get_portfolio_status_data_fallback(self, service, mock_agent):
        """Test portfolio status with fallback data"""
        service.get_agent = MagicMock(return_value=mock_agent)
        # Remove the method to test fallback
        del mock_agent.get_portfolio_status

        result = service.get_portfolio_status_data("test_token")

        assert "portfolio_value" in result
        assert result["portfolio_value"] == 10000.0  # Fallback value

    def test_get_market_data_for_symbol_success(self, service, mock_agent):
        """Test successful market data retrieval"""
        mock_agent.get_market_data.return_value = {
            "symbol": "AAPL",
            "current_price": 150.0,
            "change_24h": 2.5,
            "volume": 1000000,
            "rsi": 65.0,
            "macd": 0.5,
            "moving_avg_20": 148.0,
            "moving_avg_50": 145.0,
            "volatility": 0.2,
            "market_sentiment": "bullish",
        }
        service.get_agent = MagicMock(return_value=mock_agent)

        result = service.get_market_data_for_symbol("AAPL", "test_token")

        assert result["symbol"] == "AAPL"
        assert result["current_price"] == 150.0

    def test_get_market_data_for_symbol_agent_unavailable(self, service):
        """Test market data with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)

        result = service.get_market_data_for_symbol("AAPL", "test_token")

        assert result["symbol"] == "AAPL"
        assert "data_source" in result
        assert result["data_source"] == "fallback_generator"

    def test_get_market_data_for_symbol_error(self, service, mock_agent):
        """Test market data with error"""
        mock_agent.get_market_data.side_effect = Exception("Market data error")
        service.get_agent = MagicMock(return_value=mock_agent)

        with patch("src.que_agents.router.financial_trading_bot.system_logger"):
            result = service.get_market_data_for_symbol("AAPL", "test_token")

            assert result["symbol"] == "AAPL"
            assert "error_note" in result

    def test_get_performance_report_data_success(self, service, mock_agent):
        """Test successful performance report retrieval"""
        mock_agent.get_performance_report.return_value = {
            "portfolio_value": 12000.0,
            "total_return": 5.0,
            "trade_statistics": {"total_trades": 50},
        }
        service.get_agent = MagicMock(return_value=mock_agent)

        result = service.get_performance_report_data("test_token")

        assert "portfolio_value" in result
        assert "report_type" in result
        assert result["report_type"] == "enhanced"

    def test_get_performance_report_data_agent_unavailable(self, service):
        """Test performance report with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)

        result = service.get_performance_report_data("test_token")

        assert "status" in result
        assert result["status"] == "fallback_mode"

    def test_get_performance_report_data_error(self, service, mock_agent):
        """Test performance report with error"""
        mock_agent.get_performance_report.side_effect = Exception("Report error")
        service.get_agent = MagicMock(return_value=mock_agent)

        with patch("src.que_agents.router.financial_trading_bot.system_logger"):
            result = service.get_performance_report_data("test_token")

            assert "error_note" in result

    def test_generate_fallback_trading_cycle(self, service):
        """Test fallback trading cycle generation"""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        result = service._generate_fallback_trading_cycle(symbols)

        assert result["symbols_analyzed"] == 3
        assert result["fallback_mode"] is True
        assert len(result["decisions"]) == 3

    def test_add_unavailable_status(self, service):
        """Test adding unavailable status to data"""
        data = {"portfolio_value": 10000.0}

        result = service._add_unavailable_status(data)

        assert result["status"] == "agent_unavailable"
        assert "note" in result

    def test_format_portfolio_status_dict(self, service):
        """Test formatting portfolio status from dictionary"""
        portfolio_dict = {
            "total_value": 15000.0,
            "cash_balance": 3000.0,
            "holdings": {"AAPL": {"shares": 10}},
        }

        result = service._format_portfolio_status(portfolio_dict)

        assert result["portfolio_value"] == 15000.0
        assert result["cash_balance"] == 3000.0
        assert result["status"] == "active"

    def test_format_portfolio_status_object(self, service):
        """Test formatting portfolio status from object"""
        portfolio_obj = MagicMock()
        portfolio_obj.total_value = 20000.0
        portfolio_obj.cash_balance = 5000.0
        portfolio_obj.holdings = {"GOOGL": {"shares": 5}}

        result = service._format_portfolio_status(portfolio_obj)

        assert result["portfolio_value"] == 20000.0
        assert result["cash_balance"] == 5000.0

    def test_generate_fallback_market_data(self, service):
        """Test fallback market data generation"""
        result = service._generate_fallback_market_data("AAPL")

        assert result["symbol"] == "AAPL"
        assert "current_price" in result
        assert "data_source" in result
        assert result["data_source"] == "fallback_generator"

    def test_generate_fallback_market_data_with_error(self, service):
        """Test fallback market data generation with error"""
        result = service._generate_fallback_market_data("AAPL", error="Test error")

        assert result["symbol"] == "AAPL"
        assert "error_note" in result
        assert "Test error" in result["error_note"]

    def test_get_default_market_value(self, service):
        """Test getting default market values"""
        assert service._get_default_market_value("current_price", "AAPL") == 150.0
        assert service._get_default_market_value("rsi", "AAPL") == 50.0
        assert service._get_default_market_value("unknown_field", "AAPL") == "unknown"

    def test_market_obj_to_dict(self, service):
        """Test converting market object to dictionary"""
        market_obj = MagicMock()
        market_obj.symbol = "AAPL"
        market_obj.current_price = 155.0
        market_obj.volume = 2000000

        result = service._market_obj_to_dict(market_obj)

        assert result["symbol"] == "AAPL"
        assert result["current_price"] == 155.0
        assert result["volume"] == 2000000

    def test_generate_fallback_performance_report(self, service):
        """Test fallback performance report generation"""
        result = service._generate_fallback_performance_report()

        assert "portfolio_value" in result
        assert "trade_statistics" in result
        assert result["status"] == "fallback_mode"

    def test_generate_fallback_performance_report_with_error(self, service):
        """Test fallback performance report with error"""
        result = service._generate_fallback_performance_report(error="Test error")

        assert "error_note" in result
        assert "Test error" in result["error_note"]

    def test_enhance_performance_report(self, service):
        """Test enhancing performance report"""
        report = {"portfolio_value": 10000.0, "total_return": 2.5}

        result = service._enhance_performance_report(report)

        assert result["portfolio_value"] == 10000.0
        assert result["report_type"] == "enhanced"
        assert result["data_source"] == "agent"


class TestRouterDependencies:
    """Test router dependencies and endpoints"""

    def test_get_trading_service(self):
        """Test service dependency creation"""
        with patch("src.que_agents.api.main.agent_manager") as mock_manager:
            service = get_trading_service()
            assert isinstance(service, FinancialTradingBotService)
            assert service.agent_manager == mock_manager

    def test_router_exists(self):
        """Test router is properly configured"""
        assert router is not None
        assert hasattr(router, "routes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
