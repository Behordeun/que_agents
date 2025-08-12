from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.que_agents.agents.financial_trading_bot_agent import FinancialTradingBotAgent
from src.que_agents.core.schemas import (
    MarketConditions,
    PortfolioStatus,
    TradingDecision,
)


class TestFinancialTradingBotAgentInitialization:
    """Test agent initialization and configuration"""

    @patch("src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load")
    @patch("src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm")
    @patch("builtins.open")
    def test_init_with_default_config(self, mock_open, mock_llm_factory, mock_yaml):
        """Test agent initialization with default configuration"""
        mock_yaml.return_value = {
            "financial_trading_bot_agent": {
                "model_name": "gpt-4",
                "temperature": 0.3,
                "max_tokens": 800,
            }
        }
        mock_llm = Mock()
        mock_llm_factory.return_value = mock_llm

        with patch.object(
            FinancialTradingBotAgent,
            "initialize_default_trading_strategy",
            return_value=1,
        ):
            agent = FinancialTradingBotAgent(portfolio_id=1)

        assert agent.portfolio_id == 1
        assert agent.max_position_size == 0.1
        assert agent.stop_loss_threshold == 0.05
        assert agent.take_profit_threshold == 0.15
        assert agent.min_confidence_threshold == 0.6
        assert len(agent.supported_symbols) == 10

    @patch("src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load")
    @patch("src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm")
    @patch("builtins.open")
    def test_init_with_fallback_config_key(
        self, mock_open, mock_llm_factory, mock_yaml
    ):
        """Test initialization with fallback config key"""
        mock_yaml.return_value = {
            "financial_trading_bot": {"model_name": "gpt-3.5-turbo", "temperature": 0.5}
        }
        mock_llm_factory.return_value = Mock()

        with patch.object(
            FinancialTradingBotAgent,
            "initialize_default_trading_strategy",
            return_value=1,
        ):
            FinancialTradingBotAgent()

        mock_llm_factory.assert_called_once()

    @patch("src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load")
    @patch("src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm")
    @patch("builtins.open")
    def test_init_strategy_creation_failure(
        self, mock_open, mock_llm_factory, mock_yaml
    ):
        """Test initialization when strategy creation fails"""
        mock_yaml.return_value = {
            "financial_trading_bot_agent": {"model_name": "gpt-4", "temperature": 0.3}
        }
        mock_llm_factory.return_value = Mock()

        with patch.object(
            FinancialTradingBotAgent,
            "initialize_default_trading_strategy",
            side_effect=Exception("DB Error"),
        ):
            agent = FinancialTradingBotAgent()

        assert agent.default_strategy_id is None


class TestMarketDataAndAnalysis:
    """Test market data retrieval and analysis"""

    @pytest.fixture
    def agent(self):
        with (
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load"
            ) as mock_yaml,
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm"
            ) as mock_llm_factory,
            patch("builtins.open"),
            patch.object(
                FinancialTradingBotAgent,
                "initialize_default_trading_strategy",
                return_value=1,
            ),
        ):

            mock_yaml.return_value = {
                "financial_trading_bot_agent": {
                    "model_name": "gpt-4",
                    "temperature": 0.3,
                }
            }
            mock_llm_factory.return_value = Mock()
            return FinancialTradingBotAgent()

    def test_get_market_data_known_symbol(self, agent):
        """Test market data retrieval for known symbols"""
        market_data = agent.get_market_data("AAPL")

        assert market_data.symbol == "AAPL"
        assert isinstance(market_data.current_price, float)
        assert market_data.current_price > 0
        assert isinstance(market_data.rsi, float)
        assert 0 <= market_data.rsi <= 100
        assert isinstance(market_data.macd, float)
        assert market_data.market_sentiment in ["bullish", "bearish", "neutral"]

    def test_get_market_data_unknown_symbol(self, agent):
        """Test market data retrieval for unknown symbols"""
        market_data = agent.get_market_data("UNKNOWN")

        assert market_data.symbol == "UNKNOWN"
        assert isinstance(market_data.current_price, float)
        assert market_data.current_price > 0  # Should be positive with variation

    def test_analyze_market_success(self, agent):
        """Test successful market analysis"""
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Market analysis: AAPL shows bullish momentum"
        agent.analysis_chain = mock_chain

        analysis = agent.analyze_market("AAPL")

        assert "Market analysis" in analysis
        mock_chain.invoke.assert_called_once()

    def test_analyze_market_llm_failure(self, agent):
        """Test market analysis when LLM fails"""
        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("LLM Error")
        agent.analysis_chain = mock_chain

        analysis = agent.analyze_market("AAPL")

        assert "Market analysis for AAPL" in analysis
        assert "Current price" in analysis

    def test_analyze_market_with_knowledge_success(self, agent):
        """Test enhanced market analysis with knowledge base"""
        with (
            patch.object(agent, "get_trading_knowledge") as mock_kb,
            patch.object(agent, "get_market_data") as mock_market,
        ):

            mock_kb.return_value = [
                {"title": "AAPL Strategy", "content": "Buy on dips"}
            ]
            mock_market.return_value = MarketConditions(
                symbol="AAPL",
                current_price=150.0,
                volume=1000000,
                change_24h=2.5,
                rsi=45.0,
                macd=1.2,
                moving_avg_20=148.0,
                moving_avg_50=145.0,
                moving_avg_200=140.0,
                volatility=0.25,
                market_sentiment="bullish",
            )

            mock_chain = Mock()
            mock_chain.invoke.return_value = "Enhanced analysis with knowledge"
            agent.analysis_chain = mock_chain

            analysis = agent.analyze_market_with_knowledge("AAPL")

            assert "Enhanced analysis" in analysis
            mock_kb.assert_called()

    def test_analyze_market_with_knowledge_fallback(self, agent):
        """Test enhanced analysis fallback to basic analysis"""
        with (
            patch.object(agent, "get_market_data") as mock_market,
            patch.object(
                agent, "analyze_market", return_value="Basic analysis"
            ) as mock_basic,
        ):

            mock_market.return_value = MarketConditions(
                symbol="AAPL",
                current_price=150.0,
                volume=1000000,
                change_24h=2.5,
                rsi=45.0,
                macd=1.2,
                moving_avg_20=148.0,
                moving_avg_50=145.0,
                moving_avg_200=140.0,
                volatility=0.25,
                market_sentiment="bullish",
            )

            # Mock the chain to raise an exception
            mock_chain = Mock()
            mock_chain.invoke.side_effect = Exception("LLM Error")
            agent.analysis_chain = mock_chain

            analysis = agent.analyze_market_with_knowledge("AAPL")

            assert analysis == "Basic analysis"
            mock_basic.assert_called_once_with("AAPL")


class TestKnowledgeBaseIntegration:
    """Test knowledge base integration"""

    @pytest.fixture
    def agent(self):
        with (
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load"
            ) as mock_yaml,
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm"
            ) as mock_llm_factory,
            patch("builtins.open"),
            patch.object(
                FinancialTradingBotAgent,
                "initialize_default_trading_strategy",
                return_value=1,
            ),
        ):

            mock_yaml.return_value = {
                "financial_trading_bot_agent": {
                    "model_name": "gpt-4",
                    "temperature": 0.3,
                }
            }
            mock_llm_factory.return_value = Mock()
            return FinancialTradingBotAgent()

    @patch(
        "src.que_agents.agents.financial_trading_bot_agent.search_agent_knowledge_base"
    )
    def test_get_trading_knowledge_success(self, mock_search, agent):
        """Test successful knowledge base search"""
        mock_search.return_value = [
            {"title": "Risk Management", "content": "Always use stop losses"},
            {
                "title": "Technical Analysis",
                "content": "RSI below 30 indicates oversold",
            },
        ]

        results = agent.get_trading_knowledge("risk management")

        assert len(results) == 2
        assert results[0]["title"] == "Risk Management"
        mock_search.assert_called_once_with(
            "financial_trading_bot", "risk management", limit=3
        )

    @patch(
        "src.que_agents.agents.financial_trading_bot_agent.search_agent_knowledge_base"
    )
    def test_get_trading_knowledge_failure(self, mock_search, agent):
        """Test knowledge base search failure"""
        mock_search.side_effect = Exception("KB Connection Error")

        results = agent.get_trading_knowledge("risk management")

        assert results == []


class TestTradingDecisions:
    """Test trading decision making"""

    @pytest.fixture
    def agent(self):
        with (
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load"
            ) as mock_yaml,
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm"
            ) as mock_llm_factory,
            patch("builtins.open"),
            patch.object(
                FinancialTradingBotAgent,
                "initialize_default_trading_strategy",
                return_value=1,
            ),
        ):

            mock_yaml.return_value = {
                "financial_trading_bot_agent": {
                    "model_name": "gpt-4",
                    "temperature": 0.3,
                }
            }
            mock_llm_factory.return_value = Mock()
            return FinancialTradingBotAgent()

    @pytest.fixture
    def sample_portfolio_status(self):
        return PortfolioStatus(
            total_value=10000.0,
            cash_balance=5000.0,
            holdings={"AAPL": 10.0},
            performance_metrics={"total_return": 0.05},
            unrealized_pnl=500.0,
            realized_pnl=0.0,
        )

    @pytest.fixture
    def sample_market_conditions(self):
        return MarketConditions(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            change_24h=2.5,
            rsi=45.0,
            macd=1.2,
            moving_avg_20=148.0,
            moving_avg_50=145.0,
            moving_avg_200=140.0,
            volatility=0.25,
            market_sentiment="bullish",
        )

    def test_make_trading_decision_success(
        self, agent, sample_portfolio_status, sample_market_conditions
    ):
        """Test successful trading decision making"""
        with (
            patch.object(agent, "analyze_market", return_value="Bullish analysis"),
            patch.object(
                agent, "get_portfolio_status", return_value=sample_portfolio_status
            ),
            patch.object(
                agent, "get_market_data", return_value=sample_market_conditions
            ),
        ):

            mock_chain = Mock()
            mock_chain.invoke.return_value = "BUY 5 shares with high confidence"
            agent.decision_chain = mock_chain

            decision = agent.make_trading_decision("AAPL")

            assert isinstance(decision, TradingDecision)
            assert decision.symbol == "AAPL"
            assert decision.action in ["buy", "sell", "hold"]
            assert 0 <= decision.confidence <= 1
            assert decision.quantity >= 0

    def test_make_enhanced_trading_decision_success(
        self, agent, sample_portfolio_status, sample_market_conditions
    ):
        """Test enhanced trading decision with knowledge base"""
        with (
            patch.object(
                agent,
                "get_trading_knowledge",
                return_value=[{"title": "Strategy", "content": "Buy on momentum"}],
            ),
            patch.object(
                agent, "analyze_market_with_knowledge", return_value="Enhanced analysis"
            ),
            patch.object(
                agent, "get_portfolio_status", return_value=sample_portfolio_status
            ),
            patch.object(
                agent, "get_market_data", return_value=sample_market_conditions
            ),
        ):

            mock_chain = Mock()
            mock_chain.invoke.return_value = "BUY 3 shares based on enhanced analysis"
            agent.decision_chain = mock_chain

            decision = agent.make_enhanced_trading_decision("AAPL")

            assert isinstance(decision, TradingDecision)
            assert decision.symbol == "AAPL"

    def test_make_trading_decision_llm_failure(
        self, agent, sample_portfolio_status, sample_market_conditions
    ):
        """Test trading decision when LLM fails"""
        with (
            patch.object(agent, "analyze_market", return_value="Analysis"),
            patch.object(
                agent, "get_portfolio_status", return_value=sample_portfolio_status
            ),
            patch.object(
                agent, "get_market_data", return_value=sample_market_conditions
            ),
        ):

            mock_chain = Mock()
            mock_chain.invoke.side_effect = Exception("LLM Error")
            agent.decision_chain = mock_chain

            decision = agent.make_trading_decision("AAPL")

            assert isinstance(decision, TradingDecision)
            assert "Fallback decision" in decision.reasoning

    def test_parse_trading_decision_buy(
        self, agent, sample_market_conditions, sample_portfolio_status
    ):
        """Test parsing buy decision"""
        decision_text = "I recommend to BUY 10 shares with high confidence"

        decision = agent._parse_trading_decision(
            decision_text, "AAPL", sample_market_conditions, sample_portfolio_status
        )

        assert decision.action == "buy"
        assert decision.quantity > 0

    def test_parse_trading_decision_sell(
        self, agent, sample_market_conditions, sample_portfolio_status
    ):
        """Test parsing sell decision"""
        decision_text = "I recommend to SELL shares due to bearish signals"

        decision = agent._parse_trading_decision(
            decision_text, "AAPL", sample_market_conditions, sample_portfolio_status
        )

        assert decision.action == "sell"

    def test_parse_trading_decision_hold(
        self, agent, sample_market_conditions, sample_portfolio_status
    ):
        """Test parsing hold decision"""
        decision_text = "Market conditions are unclear, recommend to wait"

        decision = agent._parse_trading_decision(
            decision_text, "AAPL", sample_market_conditions, sample_portfolio_status
        )

        assert decision.action == "hold"
        assert decision.quantity == 0


class TestConfidenceCalculation:
    """Test confidence calculation methods"""

    @pytest.fixture
    def agent(self):
        with (
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load"
            ) as mock_yaml,
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm"
            ) as mock_llm_factory,
            patch("builtins.open"),
            patch.object(
                FinancialTradingBotAgent,
                "initialize_default_trading_strategy",
                return_value=1,
            ),
        ):

            mock_yaml.return_value = {
                "financial_trading_bot_agent": {
                    "model_name": "gpt-4",
                    "temperature": 0.3,
                }
            }
            mock_llm_factory.return_value = Mock()
            return FinancialTradingBotAgent()

    @pytest.fixture
    def bullish_market_conditions(self):
        return MarketConditions(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            change_24h=2.5,
            rsi=35.0,
            macd=1.5,
            moving_avg_20=148.0,
            moving_avg_50=145.0,
            moving_avg_200=140.0,
            volatility=0.20,
            market_sentiment="bullish",
        )

    def test_calculate_confidence_buy_bullish(self, agent, bullish_market_conditions):
        """Test confidence calculation for buy in bullish market"""
        confidence = agent._calculate_confidence(bullish_market_conditions, "buy")

        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be higher for aligned signals

    def test_rsi_confidence_buy_oversold(self, agent, bullish_market_conditions):
        """Test RSI confidence for buy when oversold"""
        bullish_market_conditions.rsi = 25.0  # Oversold
        confidence = agent._rsi_confidence(bullish_market_conditions, "buy")

        assert confidence == 0.2

    def test_rsi_confidence_sell_overbought(self, agent, bullish_market_conditions):
        """Test RSI confidence for sell when overbought"""
        bullish_market_conditions.rsi = 75.0  # Overbought
        confidence = agent._rsi_confidence(bullish_market_conditions, "sell")

        assert confidence == 0.2

    def test_moving_avg_confidence_buy_uptrend(self, agent, bullish_market_conditions):
        """Test moving average confidence for buy in uptrend"""
        confidence = agent._moving_avg_confidence(bullish_market_conditions, "buy")

        assert confidence > 0  # Should be positive for uptrend

    def test_macd_confidence_buy_positive(self, agent, bullish_market_conditions):
        """Test MACD confidence for buy with positive MACD"""
        confidence = agent._macd_confidence(bullish_market_conditions, "buy")

        assert confidence == 0.1

    def test_sentiment_confidence_aligned(self, agent, bullish_market_conditions):
        """Test sentiment confidence when aligned"""
        confidence = agent._sentiment_confidence(bullish_market_conditions, "buy")

        assert confidence == 0.1


class TestRiskCalculation:
    """Test risk calculation methods"""

    @pytest.fixture
    def agent(self):
        with (
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load"
            ) as mock_yaml,
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm"
            ) as mock_llm_factory,
            patch("builtins.open"),
            patch.object(
                FinancialTradingBotAgent,
                "initialize_default_trading_strategy",
                return_value=1,
            ),
        ):

            mock_yaml.return_value = {
                "financial_trading_bot_agent": {
                    "model_name": "gpt-4",
                    "temperature": 0.3,
                }
            }
            mock_llm_factory.return_value = Mock()
            return FinancialTradingBotAgent()

    @pytest.fixture
    def high_volatility_market(self):
        return MarketConditions(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            change_24h=2.5,
            rsi=45.0,
            macd=1.2,
            moving_avg_20=148.0,
            moving_avg_50=145.0,
            moving_avg_200=140.0,
            volatility=0.35,
            market_sentiment="neutral",
        )

    def test_calculate_risk_score_high_volatility(self, agent, high_volatility_market):
        """Test risk score calculation with high volatility"""
        risk_score = agent._calculate_risk_score(high_volatility_market, "buy", 10.0)

        assert 0 <= risk_score <= 1
        assert risk_score > 0.3  # Should be higher due to volatility

    def test_calculate_risk_score_large_position(self, agent, high_volatility_market):
        """Test risk score with large position size"""
        risk_score = agent._calculate_risk_score(high_volatility_market, "buy", 100.0)

        assert risk_score > 0.5  # Should be higher for large positions

    def test_calculate_risk_score_sentiment_mismatch(self, agent):
        """Test risk score when sentiment mismatches action"""
        bearish_market = MarketConditions(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            change_24h=-2.5,
            rsi=45.0,
            macd=-1.2,
            moving_avg_20=148.0,
            moving_avg_50=145.0,
            moving_avg_200=140.0,
            volatility=0.25,
            market_sentiment="bearish",
        )

        risk_score = agent._calculate_risk_score(bearish_market, "buy", 10.0)

        assert risk_score > 0.5  # Should be higher for mismatched sentiment

    def test_calculate_expected_return_bullish_buy(self, agent):
        """Test expected return calculation for bullish buy"""
        bullish_market = MarketConditions(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            change_24h=2.5,
            rsi=45.0,
            macd=1.2,
            moving_avg_20=148.0,
            moving_avg_50=145.0,
            moving_avg_200=140.0,
            volatility=0.25,
            market_sentiment="bullish",
        )

        expected_return = agent._calculate_expected_return(bullish_market, "buy")

        assert expected_return > 0.02  # Should be higher than base return

    def test_calculate_expected_return_hold(self, agent):
        """Test expected return for hold action"""
        market = MarketConditions(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            change_24h=0.0,
            rsi=50.0,
            macd=0.0,
            moving_avg_20=150.0,
            moving_avg_50=150.0,
            moving_avg_200=150.0,
            volatility=0.20,
            market_sentiment="neutral",
        )

        expected_return = agent._calculate_expected_return(market, "hold")

        assert expected_return == 0.0


class TestFallbackMethods:
    """Test fallback methods when LLM is unavailable"""

    @pytest.fixture
    def agent(self):
        with (
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load"
            ) as mock_yaml,
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm"
            ) as mock_llm_factory,
            patch("builtins.open"),
            patch.object(
                FinancialTradingBotAgent,
                "initialize_default_trading_strategy",
                return_value=1,
            ),
        ):

            mock_yaml.return_value = {
                "financial_trading_bot_agent": {
                    "model_name": "gpt-4",
                    "temperature": 0.3,
                }
            }
            mock_llm_factory.return_value = Mock()
            return FinancialTradingBotAgent()

    def test_generate_fallback_analysis(self, agent):
        """Test fallback market analysis generation"""
        market_data = MarketConditions(
            symbol="AAPL",
            current_price=155.0,
            volume=2000000,
            change_24h=3.2,
            rsi=65.0,
            macd=1.8,
            moving_avg_20=152.0,
            moving_avg_50=148.0,
            moving_avg_200=145.0,
            volatility=0.30,
            market_sentiment="bullish",
        )

        analysis = agent._generate_fallback_analysis(market_data)

        assert "Market analysis for AAPL" in analysis
        assert "Current price $155.00" in analysis
        assert "RSI 65.0" in analysis
        assert "trending bullish" in analysis

    def test_generate_fallback_decision_buy_signal(self, agent):
        """Test fallback decision for buy signal"""
        market_conditions = MarketConditions(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            change_24h=2.5,
            rsi=25.0,
            macd=1.5,
            moving_avg_20=152.0,
            moving_avg_50=148.0,
            moving_avg_200=145.0,
            volatility=0.25,
            market_sentiment="bullish",
        )

        portfolio_status = PortfolioStatus(
            total_value=10000.0,
            cash_balance=5000.0,
            holdings={},
            performance_metrics={},
            unrealized_pnl=0.0,
            realized_pnl=0.0,
        )

        decision = agent._generate_fallback_decision(
            "AAPL", market_conditions, portfolio_status
        )

        assert decision.action == "buy"
        assert decision.confidence == 0.7
        assert "RSI oversold + bullish sentiment" in decision.reasoning

    def test_generate_fallback_decision_sell_signal(self, agent):
        """Test fallback decision for sell signal"""
        market_conditions = MarketConditions(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            change_24h=-2.5,
            rsi=75.0,
            macd=-1.5,
            moving_avg_20=148.0,
            moving_avg_50=152.0,
            moving_avg_200=155.0,
            volatility=0.25,
            market_sentiment="bearish",
        )

        portfolio_status = PortfolioStatus(
            total_value=10000.0,
            cash_balance=2000.0,
            holdings={"AAPL": 20.0},
            performance_metrics={},
            unrealized_pnl=0.0,
            realized_pnl=0.0,
        )

        decision = agent._generate_fallback_decision(
            "AAPL", market_conditions, portfolio_status
        )

        assert decision.action == "sell"
        assert decision.confidence == 0.7
        assert "RSI overbought + bearish sentiment" in decision.reasoning

    def test_generate_fallback_decision_hold(self, agent):
        """Test fallback decision for hold"""
        market_conditions = MarketConditions(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            change_24h=0.5,
            rsi=50.0,
            macd=0.1,
            moving_avg_20=150.0,
            moving_avg_50=150.0,
            moving_avg_200=150.0,
            volatility=0.20,
            market_sentiment="neutral",
        )

        portfolio_status = PortfolioStatus(
            total_value=10000.0,
            cash_balance=5000.0,
            holdings={"AAPL": 10.0},
            performance_metrics={},
            unrealized_pnl=0.0,
            realized_pnl=0.0,
        )

        decision = agent._generate_fallback_decision(
            "AAPL", market_conditions, portfolio_status
        )

        assert decision.action == "hold"
        assert "Mixed signals" in decision.reasoning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestPortfolioManagement:
    """Test portfolio management functionality"""

    @pytest.fixture
    def agent(self):
        with (
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load"
            ) as mock_yaml,
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm"
            ) as mock_llm_factory,
            patch("builtins.open"),
            patch.object(
                FinancialTradingBotAgent,
                "initialize_default_trading_strategy",
                return_value=1,
            ),
        ):

            mock_yaml.return_value = {
                "financial_trading_bot_agent": {
                    "model_name": "gpt-4",
                    "temperature": 0.3,
                }
            }
            mock_llm_factory.return_value = Mock()
            return FinancialTradingBotAgent(portfolio_id=1)

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_get_portfolio_status_existing_portfolio(self, mock_get_session, agent):
        """Test getting portfolio status for existing portfolio"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session

        mock_portfolio = Mock()
        mock_portfolio.id = 1
        mock_portfolio.cash_balance = 5000.0
        mock_portfolio.holdings = {"AAPL": 10.0, "GOOGL": 5.0}
        mock_portfolio.performance_metrics = {"total_return": 0.05}

        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_portfolio
        )

        with patch.object(
            agent, "_calculate_holdings_value", return_value=(12000.0, 7000.0)
        ):
            portfolio_status = agent.get_portfolio_status()

        assert portfolio_status.total_value == 12000.0
        assert portfolio_status.cash_balance == 5000.0
        assert portfolio_status.holdings == {"AAPL": 10.0, "GOOGL": 5.0}

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_get_portfolio_status_no_portfolio(self, mock_get_session, agent):
        """Test getting portfolio status when no portfolio exists"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        with patch.object(agent, "_create_default_portfolio", return_value=None):
            portfolio_status = agent.get_portfolio_status()

        assert portfolio_status.total_value == 10000.0
        assert portfolio_status.cash_balance == 10000.0
        assert portfolio_status.holdings == {}

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_get_portfolio_status_database_error(self, mock_get_session, agent):
        """Test portfolio status retrieval with database error"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        mock_session.query.side_effect = Exception("Database connection error")

        portfolio_status = agent.get_portfolio_status()

        assert portfolio_status.total_value == 10000.0
        assert portfolio_status.cash_balance == 10000.0
        mock_session.rollback.assert_called_once()

    def test_create_default_portfolio_success(self, agent):
        """Test successful default portfolio creation"""
        mock_session = Mock()

        agent._create_default_portfolio(mock_session)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_create_default_portfolio_failure(self, agent):
        """Test default portfolio creation failure"""
        mock_session = Mock()
        mock_session.add.side_effect = Exception("Database error")

        portfolio = agent._create_default_portfolio(mock_session)

        assert portfolio is None
        mock_session.rollback.assert_called_once()

    def test_calculate_holdings_value_success(self, agent):
        """Test holdings value calculation"""
        holdings = {"AAPL": 10.0, "GOOGL": 2.0}
        cash_balance = 5000.0

        with patch.object(agent, "get_market_data") as mock_market:
            mock_market.side_effect = [
                MarketConditions(
                    symbol="AAPL",
                    current_price=150.0,
                    volume=1000000,
                    change_24h=2.5,
                    rsi=45.0,
                    macd=1.2,
                    moving_avg_20=148.0,
                    moving_avg_50=145.0,
                    moving_avg_200=140.0,
                    volatility=0.25,
                    market_sentiment="bullish",
                ),
                MarketConditions(
                    symbol="GOOGL",
                    current_price=2800.0,
                    volume=500000,
                    change_24h=1.8,
                    rsi=55.0,
                    macd=0.8,
                    moving_avg_20=2750.0,
                    moving_avg_50=2700.0,
                    moving_avg_200=2650.0,
                    volatility=0.30,
                    market_sentiment="neutral",
                ),
            ]

            total_value, holdings_value = agent._calculate_holdings_value(
                holdings, cash_balance
            )

        expected_holdings_value = (10.0 * 150.0) + (2.0 * 2800.0)  # 1500 + 5600 = 7100
        expected_total_value = (
            cash_balance + expected_holdings_value
        )  # 5000 + 7100 = 12100

        assert holdings_value == expected_holdings_value
        assert total_value == expected_total_value

    def test_calculate_holdings_value_market_data_error(self, agent):
        """Test holdings value calculation with market data error"""
        holdings = {"UNKNOWN": 10.0}
        cash_balance = 5000.0

        with patch.object(
            agent, "get_market_data", side_effect=Exception("Market data error")
        ):
            total_value, holdings_value = agent._calculate_holdings_value(
                holdings, cash_balance
            )

        # Should use fallback price of 100.0
        expected_holdings_value = 10.0 * 100.0  # 1000
        expected_total_value = cash_balance + expected_holdings_value  # 6000

        assert holdings_value == expected_holdings_value
        assert total_value == expected_total_value


class TestTradeExecution:
    """Test trade execution functionality"""

    @pytest.fixture
    def agent(self):
        with (
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load"
            ) as mock_yaml,
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm"
            ) as mock_llm_factory,
            patch("builtins.open"),
            patch.object(
                FinancialTradingBotAgent,
                "initialize_default_trading_strategy",
                return_value=1,
            ),
        ):

            mock_yaml.return_value = {
                "financial_trading_bot_agent": {
                    "model_name": "gpt-4",
                    "temperature": 0.3,
                }
            }
            mock_llm_factory.return_value = Mock()
            return FinancialTradingBotAgent(portfolio_id=1)

    @pytest.fixture
    def buy_decision(self):
        return TradingDecision(
            action="buy",
            symbol="AAPL",
            quantity=10.0,
            confidence=0.8,
            reasoning="Strong buy signal",
            risk_score=0.3,
            expected_return=0.05,
        )

    @pytest.fixture
    def sell_decision(self):
        return TradingDecision(
            action="sell",
            symbol="AAPL",
            quantity=5.0,
            confidence=0.7,
            reasoning="Take profit",
            risk_score=0.2,
            expected_return=0.03,
        )

    @pytest.fixture
    def hold_decision(self):
        return TradingDecision(
            action="hold",
            symbol="AAPL",
            quantity=0.0,
            confidence=0.5,
            reasoning="Wait for better signals",
            risk_score=0.1,
            expected_return=0.0,
        )

    def test_execute_trade_hold_action(self, agent, hold_decision):
        """Test executing hold action"""
        result = agent.execute_trade(hold_decision)

        assert result is True

    def test_execute_trade_low_confidence(self, agent, buy_decision):
        """Test executing trade with low confidence"""
        buy_decision.confidence = 0.4  # Below threshold

        result = agent.execute_trade(buy_decision)

        assert result is False

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_execute_trade_no_portfolio(self, mock_get_session, agent, buy_decision):
        """Test executing trade when portfolio doesn't exist"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = agent.execute_trade(buy_decision)

        assert result is False

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_execute_trade_buy_success(self, mock_get_session, agent, buy_decision):
        """Test successful buy trade execution"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session

        mock_portfolio = Mock()
        mock_portfolio.id = 1
        mock_portfolio.cash_balance = 5000.0
        mock_portfolio.holdings = {}
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_portfolio
        )

        with (
            patch.object(agent, "get_market_data") as mock_market,
            patch.object(agent, "_get_or_create_strategy", return_value=1),
            patch.object(agent, "_log_trade"),
        ):

            mock_market.return_value = MarketConditions(
                symbol="AAPL",
                current_price=150.0,
                volume=1000000,
                change_24h=2.5,
                rsi=45.0,
                macd=1.2,
                moving_avg_20=148.0,
                moving_avg_50=145.0,
                moving_avg_200=140.0,
                volatility=0.25,
                market_sentiment="bullish",
            )

            result = agent.execute_trade(buy_decision)

        assert result is True

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_execute_trade_sell_success(self, mock_get_session, agent, sell_decision):
        """Test successful sell trade execution"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session

        mock_portfolio = Mock()
        mock_portfolio.id = 1
        mock_portfolio.cash_balance = 1000.0
        mock_portfolio.holdings = {"AAPL": 10.0}
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_portfolio
        )

        with (
            patch.object(agent, "get_market_data") as mock_market,
            patch.object(agent, "_get_or_create_strategy", return_value=1),
            patch.object(agent, "_log_trade"),
        ):

            mock_market.return_value = MarketConditions(
                symbol="AAPL",
                current_price=150.0,
                volume=1000000,
                change_24h=2.5,
                rsi=45.0,
                macd=1.2,
                moving_avg_20=148.0,
                moving_avg_50=145.0,
                moving_avg_200=140.0,
                volatility=0.25,
                market_sentiment="bullish",
            )

            result = agent.execute_trade(sell_decision)

        assert result is True

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_execute_trade_database_error(self, mock_get_session, agent, buy_decision):
        """Test trade execution with database error"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        mock_session.query.side_effect = Exception("Database error")

        result = agent.execute_trade(buy_decision)

        assert result is False
        mock_session.rollback.assert_called_once()

    def test_execute_buy_insufficient_cash(self, agent):
        """Test buy execution with insufficient cash"""
        mock_portfolio = Mock()
        mock_portfolio.cash_balance = 100.0  # Not enough for trade

        holdings = {}
        decision = TradingDecision(
            action="buy",
            symbol="AAPL",
            quantity=10.0,
            confidence=0.8,
            reasoning="Buy signal",
            risk_score=0.3,
            expected_return=0.05,
        )

        result = agent._execute_buy(mock_portfolio, holdings, decision, 1500.0, 15.0)

        assert result is False

    def test_execute_buy_sufficient_cash(self, agent):
        """Test buy execution with sufficient cash"""
        mock_portfolio = Mock()
        mock_portfolio.cash_balance = 2000.0

        holdings = {}
        decision = TradingDecision(
            action="buy",
            symbol="AAPL",
            quantity=10.0,
            confidence=0.8,
            reasoning="Buy signal",
            risk_score=0.3,
            expected_return=0.05,
        )

        result = agent._execute_buy(mock_portfolio, holdings, decision, 1500.0, 15.0)

        assert result is True
        assert holdings["AAPL"] == 10.0
        assert mock_portfolio.cash_balance == 485.0  # 2000 - 1500 - 15

    def test_execute_sell_insufficient_shares(self, agent):
        """Test sell execution with insufficient shares"""
        mock_portfolio = Mock()
        mock_portfolio.cash_balance = 1000.0

        holdings = {"AAPL": 3.0}  # Not enough shares
        decision = TradingDecision(
            action="sell",
            symbol="AAPL",
            quantity=5.0,
            confidence=0.7,
            reasoning="Sell signal",
            risk_score=0.2,
            expected_return=0.03,
        )

        result = agent._execute_sell(mock_portfolio, holdings, decision, 750.0, 7.5)

        assert result is False

    def test_execute_sell_sufficient_shares(self, agent):
        """Test sell execution with sufficient shares"""
        mock_portfolio = Mock()
        mock_portfolio.cash_balance = 1000.0

        holdings = {"AAPL": 10.0}
        decision = TradingDecision(
            action="sell",
            symbol="AAPL",
            quantity=5.0,
            confidence=0.7,
            reasoning="Sell signal",
            risk_score=0.2,
            expected_return=0.03,
        )

        result = agent._execute_sell(mock_portfolio, holdings, decision, 750.0, 7.5)

        assert result is True
        assert holdings["AAPL"] == 5.0
        assert mock_portfolio.cash_balance == 1742.5  # 1000 + 750 - 7.5

    def test_execute_sell_all_shares(self, agent):
        """Test selling all shares removes from holdings"""
        mock_portfolio = Mock()
        mock_portfolio.cash_balance = 1000.0

        holdings = {"AAPL": 5.0}
        decision = TradingDecision(
            action="sell",
            symbol="AAPL",
            quantity=5.0,
            confidence=0.7,
            reasoning="Sell all",
            risk_score=0.2,
            expected_return=0.03,
        )

        result = agent._execute_sell(mock_portfolio, holdings, decision, 750.0, 7.5)

        assert result is True
        assert "AAPL" not in holdings


class TestStrategyManagement:
    """Test trading strategy management"""

    @pytest.fixture
    def agent(self):
        with (
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load"
            ) as mock_yaml,
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm"
            ) as mock_llm_factory,
            patch("builtins.open"),
            patch.object(
                FinancialTradingBotAgent,
                "initialize_default_trading_strategy",
                return_value=1,
            ),
        ):

            mock_yaml.return_value = {
                "financial_trading_bot_agent": {
                    "model_name": "gpt-4",
                    "temperature": 0.3,
                }
            }
            mock_llm_factory.return_value = Mock()
            return FinancialTradingBotAgent()

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_initialize_default_trading_strategy_existing(
        self, mock_get_session, agent
    ):
        """Test initializing strategy when one already exists"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session

        mock_strategy = Mock()
        mock_strategy.id = 1
        mock_session.query.return_value.first.return_value = mock_strategy

        strategy_id = agent.initialize_default_trading_strategy()

        assert strategy_id == 1

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_initialize_default_trading_strategy_create_new(
        self, mock_get_session, agent
    ):
        """Test creating new default strategy"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        mock_session.query.return_value.first.return_value = None

        # Mock the created strategy
        mock_strategy = Mock()
        mock_strategy.id = 2

        with patch(
            "src.que_agents.agents.financial_trading_bot_agent.TradingStrategy",
            return_value=mock_strategy,
        ):
            strategy_id = agent.initialize_default_trading_strategy()

        assert strategy_id == 2
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_initialize_default_trading_strategy_error(self, mock_get_session, agent):
        """Test strategy initialization with database error"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        mock_session.query.side_effect = Exception("Database error")

        with pytest.raises(Exception):
            agent.initialize_default_trading_strategy()

        mock_session.rollback.assert_called_once()

    def test_get_or_create_strategy_existing(self, agent):
        """Test getting existing strategy"""
        mock_session = Mock()
        mock_strategy = Mock()
        mock_strategy.id = 1
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_strategy
        )

        strategy_id = agent._get_or_create_strategy(mock_session, 1)

        assert strategy_id == 1

    def test_get_or_create_strategy_none_provided(self, agent):
        """Test creating strategy when none provided"""
        mock_session = Mock()
        mock_strategy = Mock()
        mock_strategy.id = 1
        mock_session.query.return_value.first.return_value = mock_strategy

        strategy_id = agent._get_or_create_strategy(mock_session, None)

        assert strategy_id == 1

    def test_get_or_create_strategy_not_found(self, agent):
        """Test creating strategy when specified one not found"""
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.first.return_value = None

        mock_strategy = Mock()
        mock_strategy.id = 2

        with patch(
            "src.que_agents.agents.financial_trading_bot_agent.TradingStrategy",
            return_value=mock_strategy,
        ):
            strategy_id = agent._get_or_create_strategy(mock_session, 999)

        assert strategy_id == 2
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_log_trade(self, agent):
        """Test trade logging"""
        mock_session = Mock()
        strategy_id = 1
        decision = TradingDecision(
            action="buy",
            symbol="AAPL",
            quantity=10.0,
            confidence=0.8,
            reasoning="Buy signal",
            risk_score=0.3,
            expected_return=0.05,
        )
        current_price = 150.0
        trade_value = 1500.0
        fees = 15.0
        market_conditions = MarketConditions(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            change_24h=2.5,
            rsi=45.0,
            macd=1.2,
            moving_avg_20=148.0,
            moving_avg_50=145.0,
            moving_avg_200=140.0,
            volatility=0.25,
            market_sentiment="bullish",
        )

        with patch(
            "src.que_agents.agents.financial_trading_bot_agent.TradeLog"
        ) as mock_trade_log:
            agent._log_trade(
                mock_session,
                strategy_id,
                decision,
                current_price,
                trade_value,
                fees,
                market_conditions,
            )

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()


class TestTradingCycle:
    """Test trading cycle functionality"""

    @pytest.fixture
    def agent(self):
        with (
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load"
            ) as mock_yaml,
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm"
            ) as mock_llm_factory,
            patch("builtins.open"),
            patch.object(
                FinancialTradingBotAgent,
                "initialize_default_trading_strategy",
                return_value=1,
            ),
        ):

            mock_yaml.return_value = {
                "financial_trading_bot_agent": {
                    "model_name": "gpt-4",
                    "temperature": 0.3,
                }
            }
            mock_llm_factory.return_value = Mock()
            return FinancialTradingBotAgent()

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_run_trading_cycle_default_symbols(self, mock_get_session, agent):
        """Test running trading cycle with default symbols"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session

        mock_strategy = Mock()
        mock_strategy.id = 1
        mock_session.query.return_value.first.return_value = mock_strategy

        with (
            patch.object(agent, "make_enhanced_trading_decision") as mock_decision,
            patch.object(agent, "execute_trade", return_value=True),
            patch.object(agent, "get_portfolio_status") as mock_portfolio,
        ):

            mock_decision.return_value = TradingDecision(
                action="buy",
                symbol="AAPL",
                quantity=10.0,
                confidence=0.8,
                reasoning="Buy signal",
                risk_score=0.3,
                expected_return=0.05,
            )
            mock_portfolio.return_value = PortfolioStatus(
                total_value=10000.0,
                cash_balance=5000.0,
                holdings={},
                performance_metrics={},
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )

            results = agent.run_trading_cycle()

        assert results["symbols_analyzed"] == 5
        assert results["trades_executed"] >= 0
        assert len(results["decisions"]) == 5

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_run_trading_cycle_custom_symbols(self, mock_get_session, agent):
        """Test running trading cycle with custom symbols"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session

        mock_strategy = Mock()
        mock_strategy.id = 1
        mock_session.query.return_value.first.return_value = mock_strategy

        symbols = ["AAPL", "GOOGL"]

        with (
            patch.object(agent, "make_enhanced_trading_decision") as mock_decision,
            patch.object(agent, "execute_trade", return_value=False),
            patch.object(agent, "get_portfolio_status") as mock_portfolio,
        ):

            mock_decision.return_value = TradingDecision(
                action="hold",
                symbol="AAPL",
                quantity=0.0,
                confidence=0.5,
                reasoning="Hold signal",
                risk_score=0.2,
                expected_return=0.0,
            )
            mock_portfolio.return_value = PortfolioStatus(
                total_value=10000.0,
                cash_balance=5000.0,
                holdings={},
                performance_metrics={},
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )

            results = agent.run_trading_cycle(symbols)

        assert results["symbols_analyzed"] == 2
        assert results["trades_executed"] == 0

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_run_trading_cycle_no_existing_strategy(self, mock_get_session, agent):
        """Test trading cycle when no strategy exists"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        mock_session.query.return_value.first.return_value = None

        mock_strategy = Mock()
        mock_strategy.id = 2

        with (
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.TradingStrategy",
                return_value=mock_strategy,
            ),
            patch.object(agent, "make_enhanced_trading_decision") as mock_decision,
            patch.object(agent, "execute_trade", return_value=True),
            patch.object(agent, "get_portfolio_status") as mock_portfolio,
        ):

            mock_decision.return_value = TradingDecision(
                action="buy",
                symbol="AAPL",
                quantity=5.0,
                confidence=0.7,
                reasoning="Buy signal",
                risk_score=0.3,
                expected_return=0.04,
            )
            mock_portfolio.return_value = PortfolioStatus(
                total_value=10000.0,
                cash_balance=5000.0,
                holdings={},
                performance_metrics={},
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )

            results = agent.run_trading_cycle(["AAPL"])

        assert results["symbols_analyzed"] == 1
        mock_session.add.assert_called_once()

    def test_process_symbol_for_trading_cycle_success(self, agent):
        """Test processing symbol in trading cycle"""
        results = {"trades_executed": 0, "decisions": [], "total_confidence": 0.0}
        strategy_id = 1

        with (
            patch.object(agent, "make_enhanced_trading_decision") as mock_decision,
            patch.object(agent, "execute_trade", return_value=True),
        ):

            mock_decision.return_value = TradingDecision(
                action="buy",
                symbol="AAPL",
                quantity=10.0,
                confidence=0.8,
                reasoning="Strong buy",
                risk_score=0.3,
                expected_return=0.05,
            )

            agent._process_symbol_for_trading_cycle("AAPL", results, strategy_id)

        assert results["trades_executed"] == 1
        assert len(results["decisions"]) == 1
        assert results["decisions"][0]["executed"] is True

    def test_process_symbol_for_trading_cycle_low_confidence(self, agent):
        """Test processing symbol with low confidence decision"""
        results = {"trades_executed": 0, "decisions": [], "total_confidence": 0.0}
        strategy_id = 1

        with patch.object(agent, "make_enhanced_trading_decision") as mock_decision:
            mock_decision.return_value = TradingDecision(
                action="buy",
                symbol="AAPL",
                quantity=10.0,
                confidence=0.4,  # Below threshold
                reasoning="Weak buy",
                risk_score=0.5,
                expected_return=0.02,
            )

            agent._process_symbol_for_trading_cycle("AAPL", results, strategy_id)

        assert results["trades_executed"] == 0
        assert len(results["decisions"]) == 1
        assert results["decisions"][0]["executed"] is False

    def test_process_symbol_for_trading_cycle_error(self, agent):
        """Test processing symbol with error"""
        results = {"trades_executed": 0, "decisions": [], "total_confidence": 0.0}
        strategy_id = 1

        with (
            patch.object(
                agent,
                "make_enhanced_trading_decision",
                side_effect=Exception("Decision error"),
            ),
            patch.object(
                agent, "make_trading_decision", side_effect=Exception("Fallback error")
            ),
        ):
            agent._process_symbol_for_trading_cycle("AAPL", results, strategy_id)

        assert len(results["decisions"]) == 1
        assert results["decisions"][0]["action"] == "error"


class TestPerformanceReporting:
    """Test performance reporting functionality"""

    @pytest.fixture
    def agent(self):
        with (
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load"
            ) as mock_yaml,
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm"
            ) as mock_llm_factory,
            patch("builtins.open"),
            patch.object(
                FinancialTradingBotAgent,
                "initialize_default_trading_strategy",
                return_value=1,
            ),
        ):

            mock_yaml.return_value = {
                "financial_trading_bot_agent": {
                    "model_name": "gpt-4",
                    "temperature": 0.3,
                }
            }
            mock_llm_factory.return_value = Mock()
            return FinancialTradingBotAgent(portfolio_id=1)

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_get_performance_report_success(self, mock_get_session, agent):
        """Test successful performance report generation"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session

        # Mock portfolio
        mock_portfolio = Mock()
        mock_portfolio.id = 1
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_portfolio
        )

        # Mock trades
        mock_trade1 = Mock()
        mock_trade1.symbol = "AAPL"
        mock_trade1.trade_type = "buy"
        mock_trade1.quantity = 10.0
        mock_trade1.price = 150.0
        mock_trade1.executed_at = datetime.now()
        mock_trade1.confidence_score = 0.8

        mock_trade2 = Mock()
        mock_trade2.symbol = "GOOGL"
        mock_trade2.trade_type = "sell"
        mock_trade2.quantity = 5.0
        mock_trade2.price = 2800.0
        mock_trade2.executed_at = datetime.now()
        mock_trade2.confidence_score = 0.7

        mock_session.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
            mock_trade1,
            mock_trade2,
        ]

        with patch.object(agent, "get_portfolio_status") as mock_portfolio_status:
            mock_portfolio_status.return_value = PortfolioStatus(
                total_value=12000.0,
                cash_balance=5000.0,
                holdings={"AAPL": 10.0},
                performance_metrics={"total_return": 0.2},
                unrealized_pnl=2000.0,
                realized_pnl=0.0,
            )

            report = agent.get_performance_report()

        assert report["portfolio_value"] == 12000.0
        assert report["total_return_pct"] == 20.0
        assert report["trade_statistics"]["total_trades"] == 2
        assert report["trade_statistics"]["buy_trades"] == 1
        assert report["trade_statistics"]["sell_trades"] == 1
        assert len(report["recent_trades"]) == 2

    @patch("src.que_agents.agents.financial_trading_bot_agent.get_session")
    def test_get_performance_report_no_portfolio(self, mock_get_session, agent):
        """Test performance report when portfolio doesn't exist"""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        report = agent.get_performance_report()

        assert "error" in report
        assert report["error"] == "Portfolio not found"


class TestPromptTemplates:
    """Test prompt template creation"""

    @pytest.fixture
    def agent(self):
        with (
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.yaml.safe_load"
            ) as mock_yaml,
            patch(
                "src.que_agents.agents.financial_trading_bot_agent.LLMFactory.get_llm"
            ) as mock_llm_factory,
            patch("builtins.open"),
            patch.object(
                FinancialTradingBotAgent,
                "initialize_default_trading_strategy",
                return_value=1,
            ),
        ):

            mock_yaml.return_value = {
                "financial_trading_bot_agent": {
                    "model_name": "gpt-4",
                    "temperature": 0.3,
                }
            }
            mock_llm_factory.return_value = Mock()
            return FinancialTradingBotAgent()

    def test_create_analysis_prompt(self, agent):
        """Test analysis prompt template creation"""
        prompt = agent._create_analysis_prompt()

        assert prompt is not None
        assert hasattr(prompt, "format_messages")

    def test_create_decision_prompt(self, agent):
        """Test decision prompt template creation"""
        prompt = agent._create_decision_prompt()

        assert prompt is not None
        assert hasattr(prompt, "format_messages")

    def test_create_risk_prompt(self, agent):
        """Test risk prompt template creation"""
        prompt = agent._create_risk_prompt()

        assert prompt is not None
        assert hasattr(prompt, "format_messages")

    def test_create_analysis_chain(self, agent):
        """Test analysis chain creation"""
        chain = agent._create_analysis_chain()

        assert chain is not None
        assert hasattr(chain, "invoke")

    def test_create_decision_chain(self, agent):
        """Test decision chain creation"""
        chain = agent._create_decision_chain()

        assert chain is not None
        assert hasattr(chain, "invoke")

    def test_create_risk_chain(self, agent):
        """Test risk chain creation"""
        chain = agent._create_risk_chain()

        assert chain is not None
        assert hasattr(chain, "invoke")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
