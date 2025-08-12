"""
Unit tests for Financial Calculations utility to improve code coverage.
"""

from unittest.mock import patch

import pytest

from src.que_agents.utils.financial_calculations import (
    RiskManagement,
    TradingCalculations,
)


@pytest.fixture
def sample_prices():
    return [100, 102, 98, 105, 103, 107, 104, 109, 106, 112]


@pytest.fixture
def sample_portfolio():
    return {
        "AAPL": {"value": 5000, "shares": 50},
        "GOOGL": {"value": 3000, "shares": 10},
        "MSFT": {"value": 2000, "shares": 20},
    }


class TestTradingCalculations:
    """Test TradingCalculations functionality"""

    def test_calculate_portfolio_metrics_basic(self):
        """Test basic portfolio metrics calculation"""
        holdings = {"AAPL": {"value": 5000}, "GOOGL": {"value": 3000}}

        result = TradingCalculations.calculate_portfolio_metrics(
            portfolio_value=10000.0, cash_balance=2000.0, holdings=holdings
        )

        assert result["total_value"] == 10000.0
        assert result["cash_balance"] == 2000.0
        assert result["invested_amount"] == 8000.0
        assert result["cash_percentage"] == 20.0
        assert result["number_of_positions"] == 2

    def test_calculate_portfolio_metrics_with_history(self, sample_prices):
        """Test portfolio metrics with historical data"""
        holdings = {"AAPL": {"value": 5000}}

        result = TradingCalculations.calculate_portfolio_metrics(
            portfolio_value=10000.0,
            cash_balance=5000.0,
            holdings=holdings,
            historical_values=sample_prices,
        )

        assert "volatility" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result

    def test_calculate_portfolio_metrics_error(self):
        """Test portfolio metrics calculation with error"""
        with patch("src.que_agents.utils.financial_calculations.system_logger"):
            result = TradingCalculations.calculate_portfolio_metrics(
                portfolio_value="invalid",  # Invalid type
                cash_balance=1000.0,
                holdings={},
            )

            assert "error" in result

    def test_calculate_returns_success(self, sample_prices):
        """Test successful returns calculation"""
        returns = TradingCalculations.calculate_returns(sample_prices)

        assert len(returns) == len(sample_prices) - 1
        assert isinstance(returns[0], float)

    def test_calculate_returns_insufficient_data(self):
        """Test returns calculation with insufficient data"""
        returns = TradingCalculations.calculate_returns([100])
        assert returns == []

    def test_calculate_volatility_success(self):
        """Test successful volatility calculation"""
        returns = [0.02, -0.01, 0.03, -0.02, 0.04]
        volatility = TradingCalculations.calculate_volatility(returns)

        assert isinstance(volatility, float)
        assert volatility >= 0

    def test_calculate_volatility_insufficient_data(self):
        """Test volatility calculation with insufficient data"""
        volatility = TradingCalculations.calculate_volatility([0.02])
        assert volatility == 0.0

    def test_calculate_volatility_not_annualized(self):
        """Test volatility calculation without annualization"""
        returns = [0.02, -0.01, 0.03, -0.02, 0.04]
        volatility = TradingCalculations.calculate_volatility(returns, annualized=False)

        assert isinstance(volatility, float)
        assert volatility >= 0

    def test_calculate_sharpe_ratio_success(self):
        """Test successful Sharpe ratio calculation"""
        returns = [0.02, -0.01, 0.03, -0.02, 0.04]
        sharpe = TradingCalculations.calculate_sharpe_ratio(returns)

        assert isinstance(sharpe, float)

    def test_calculate_sharpe_ratio_insufficient_data(self):
        """Test Sharpe ratio calculation with insufficient data"""
        sharpe = TradingCalculations.calculate_sharpe_ratio([0.02])
        assert sharpe == 0.0

    def test_calculate_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio calculation with zero standard deviation"""
        returns = [0.05, 0.05, 0.05, 0.05]  # Constant returns
        sharpe = TradingCalculations.calculate_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_calculate_max_drawdown_success(self, sample_prices):
        """Test successful max drawdown calculation"""
        drawdown = TradingCalculations.calculate_max_drawdown(sample_prices)

        assert isinstance(drawdown, float)
        assert drawdown >= 0

    def test_calculate_max_drawdown_insufficient_data(self):
        """Test max drawdown calculation with insufficient data"""
        drawdown = TradingCalculations.calculate_max_drawdown([100])
        assert drawdown == 0.0

    def test_calculate_position_size_success(self):
        """Test successful position size calculation"""
        size = TradingCalculations.calculate_position_size(
            portfolio_value=10000.0,
            risk_percentage=2.0,
            entry_price=100.0,
            stop_loss_price=95.0,
        )

        assert isinstance(size, int)
        assert size >= 0

    def test_calculate_position_size_invalid_params(self):
        """Test position size calculation with invalid parameters"""
        size = TradingCalculations.calculate_position_size(
            portfolio_value=10000.0,
            risk_percentage=2.0,
            entry_price=100.0,
            stop_loss_price=105.0,  # Stop loss higher than entry
        )

        assert size == 0

    def test_calculate_risk_reward_ratio_success(self):
        """Test successful risk-reward ratio calculation"""
        ratio = TradingCalculations.calculate_risk_reward_ratio(
            entry_price=100.0, stop_loss_price=95.0, target_price=110.0
        )

        assert isinstance(ratio, float)
        assert ratio > 0

    def test_calculate_risk_reward_ratio_invalid_params(self):
        """Test risk-reward ratio calculation with invalid parameters"""
        ratio = TradingCalculations.calculate_risk_reward_ratio(
            entry_price=100.0,
            stop_loss_price=105.0,  # Stop loss higher than entry
            target_price=110.0,
        )

        assert ratio is None

    def test_calculate_technical_indicators_success(self, sample_prices):
        """Test successful technical indicators calculation"""
        # Extend sample prices to meet minimum requirements
        extended_prices = sample_prices * 5  # 50 data points

        indicators = TradingCalculations.calculate_technical_indicators(extended_prices)

        assert "current_price" in indicators
        assert "sma_20" in indicators
        assert "sma_50" in indicators
        assert "rsi" in indicators

    def test_calculate_technical_indicators_insufficient_data(self):
        """Test technical indicators calculation with insufficient data"""
        indicators = TradingCalculations.calculate_technical_indicators([100, 102])

        assert "error" in indicators

    def test_calculate_technical_indicators_error(self):
        """Test technical indicators calculation with error"""
        with patch("src.que_agents.utils.financial_calculations.system_logger"):
            indicators = TradingCalculations.calculate_technical_indicators("invalid")

            assert "error" in indicators

    def test_calculate_rsi_success(self, sample_prices):
        """Test successful RSI calculation"""
        # Extend sample prices to meet minimum requirements
        extended_prices = sample_prices * 2  # 20 data points

        rsi = TradingCalculations.calculate_rsi(extended_prices)

        assert isinstance(rsi, float)
        assert 0 <= rsi <= 100

    def test_calculate_rsi_insufficient_data(self):
        """Test RSI calculation with insufficient data"""
        rsi = TradingCalculations.calculate_rsi([100, 102])
        assert rsi == 50.0  # Neutral RSI

    def test_calculate_rsi_zero_loss(self):
        """Test RSI calculation with zero average loss"""
        # Prices that only go up
        prices = [
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
        ]
        rsi = TradingCalculations.calculate_rsi(prices)
        assert rsi == 100.0


class TestRiskManagement:
    """Test RiskManagement functionality"""

    def test_assess_portfolio_risk_success(self, sample_portfolio):
        """Test successful portfolio risk assessment"""
        result = RiskManagement.assess_portfolio_risk(
            portfolio_value=10000.0,
            positions=sample_portfolio,
            market_conditions="neutral",
        )

        assert "risk_level" in result
        assert "risk_score" in result
        assert "concentration_ratio" in result
        assert "recommendations" in result

    def test_assess_portfolio_risk_high_concentration(self):
        """Test portfolio risk assessment with high concentration"""
        positions = {"AAPL": {"value": 8000}, "GOOGL": {"value": 2000}}

        result = RiskManagement.assess_portfolio_risk(
            portfolio_value=10000.0, positions=positions, market_conditions="neutral"
        )

        assert result["risk_level"] == "high"
        assert "High concentration" in str(result["risk_factors"])

    def test_assess_portfolio_risk_limited_diversification(self):
        """Test portfolio risk assessment with limited diversification"""
        positions = {"AAPL": {"value": 3000}, "GOOGL": {"value": 3000}}

        result = RiskManagement.assess_portfolio_risk(
            portfolio_value=6000.0, positions=positions, market_conditions="neutral"
        )

        assert "Limited diversification" in str(result["risk_factors"])

    def test_assess_portfolio_risk_volatile_market(self):
        """Test portfolio risk assessment with volatile market"""
        positions = {"AAPL": {"value": 5000}}

        result = RiskManagement.assess_portfolio_risk(
            portfolio_value=5000.0, positions=positions, market_conditions="volatile"
        )

        assert "Volatile market" in str(result["risk_factors"])

    def test_assess_portfolio_risk_error(self):
        """Test portfolio risk assessment with error"""
        with patch("src.que_agents.utils.financial_calculations.system_logger"):
            result = RiskManagement.assess_portfolio_risk(
                portfolio_value="invalid",  # Invalid type
                positions={},
                market_conditions="neutral",
            )

            assert result["risk_level"] == "unknown"
            assert "error" in result

    def test_extract_position_values_dict_format(self):
        """Test position values extraction with dictionary format"""
        positions = {"AAPL": {"value": 5000}, "GOOGL": {"value": 3000}}

        values, total = RiskManagement._extract_position_values(positions)

        assert values == [5000, 3000]
        assert total == 8000

    def test_extract_position_values_numeric_format(self):
        """Test position values extraction with numeric format"""
        positions = {"AAPL": 5000, "GOOGL": 3000}

        values, total = RiskManagement._extract_position_values(positions)

        assert values == [5000, 3000]
        assert total == 8000

    def test_determine_risk_level_low(self):
        """Test risk level determination for low risk"""
        risk_level, factors = RiskManagement._determine_risk_level(
            concentration_ratio=15.0, num_positions=10, market_conditions="neutral"
        )

        assert risk_level == "low"
        assert len(factors) == 0

    def test_determine_risk_level_medium(self):
        """Test risk level determination for medium risk"""
        risk_level, factors = RiskManagement._determine_risk_level(
            concentration_ratio=25.0, num_positions=3, market_conditions="neutral"
        )

        assert risk_level == "medium"
        assert len(factors) > 0

    def test_determine_risk_level_high(self):
        """Test risk level determination for high risk"""
        risk_level, factors = RiskManagement._determine_risk_level(
            concentration_ratio=35.0, num_positions=2, market_conditions="volatile"
        )

        assert risk_level == "high"
        assert len(factors) > 0

    def test_calculate_risk_score(self):
        """Test risk score calculation"""
        assert RiskManagement._calculate_risk_score("low") == 0.2
        assert RiskManagement._calculate_risk_score("medium") == 0.5
        assert RiskManagement._calculate_risk_score("high") == 0.8
        assert RiskManagement._calculate_risk_score("unknown") == 0.5
        assert RiskManagement._calculate_risk_score("invalid") == 0.5

    def test_get_risk_recommendations_high_risk(self):
        """Test risk recommendations for high risk"""
        recommendations = RiskManagement._get_risk_recommendations(
            "high", ["High concentration in single position"]
        )

        assert len(recommendations) > 0
        assert any("reducing position sizes" in rec for rec in recommendations)

    def test_get_risk_recommendations_medium_risk(self):
        """Test risk recommendations for medium risk"""
        recommendations = RiskManagement._get_risk_recommendations(
            "medium", ["Limited diversification"]
        )

        assert len(recommendations) > 0
        assert any("diversification" in rec for rec in recommendations)

    def test_get_risk_recommendations_low_risk(self):
        """Test risk recommendations for low risk"""
        recommendations = RiskManagement._get_risk_recommendations("low", [])

        assert len(recommendations) > 0
        assert any("Continue current" in rec for rec in recommendations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
