# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-05 12:30:00
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-05 12:30:00
# @Description: This module implements a Financial Automated Trading Bot agent using LangChain and SQLAlchemy

import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from src.que_agents.core.database import (
    Portfolio,
    TradeLog,
    TradingStrategy,
    get_session,
)
from src.que_agents.core.llm_factory import LLMFactory
from src.que_agents.core.schemas import (
    MarketConditions,
    PortfolioStatus,
    TradingDecision,
)
from src.que_agents.error_trace.errorlogger import system_logger
from src.que_agents.knowledge_base.kb_manager import search_agent_knowledge_base

system_logger.info("Initializing Financial Trading Bot Agent ...")

PORTFOLIO_ISSUE = "Portfolio not found"

# Load agent configuration
with open("./configs/agent_config.yaml", "r") as f:
    agent_config = yaml.safe_load(f)


class FinancialTradingBotAgent:
    """Financial Automated Trading Bot Agent"""

    def __init__(self, portfolio_id: int = 1):
        # Try different config key names
        config_key = "financial_trading_bot_agent"
        if config_key not in agent_config:
            config_key = "financial_trading_bot"

        config = agent_config[config_key]
        self.llm = LLMFactory.get_llm(
            agent_type="financial_trading_bot",
            model_name=config["model_name"],
            temperature=config["temperature"],
            max_tokens=config.get("max_tokens", 800),
        )

        self.portfolio_id = portfolio_id

        # Trading parameters
        self.max_position_size = 0.1  # Max 10% of portfolio per position
        self.stop_loss_threshold = 0.05  # 5% stop loss
        self.take_profit_threshold = 0.15  # 15% take profit
        self.min_confidence_threshold = 0.6  # Minimum confidence to execute trade

        # Supported symbols (for simulation)
        self.supported_symbols = [
            "AAPL",
            "GOOGL",
            "MSFT",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "NFLX",
            "AMD",
            "INTC",
        ]

        # Initialize default strategy if needed
        try:
            self.default_strategy_id = self.initialize_default_trading_strategy()
        except Exception as e:
            system_logger.warning(
                f"Could not initialize default trading strategy: {e}. Will create on demand."
            )
            self.default_strategy_id = None

        # Initialize prompt templates
        self.analysis_prompt = self._create_analysis_prompt()
        self.decision_prompt = self._create_decision_prompt()
        self.risk_prompt = self._create_risk_prompt()

        # Create chains
        self.analysis_chain = self._create_analysis_chain()
        self.decision_chain = self._create_decision_chain()
        self.risk_chain = self._create_risk_chain()

    def get_trading_knowledge(self, query: str) -> List[Dict]:
        """Get trading-related knowledge from knowledge base"""
        try:
            kb_results = search_agent_knowledge_base(
                "financial_trading_bot", query, limit=3
            )
            return kb_results
        except Exception as e:
            system_logger.error(
                f"Error searching trading knowledge: {e}",
                additional_info={
                    "query": query,
                    "results": 0,
                },
                exc_info=True,
            )
            return []

    def analyze_market_with_knowledge(self, symbol: str) -> str:
        """Enhanced market analysis using knowledge base"""
        # Get basic market data
        market_data = self.get_market_data(symbol)

        # Search for relevant trading strategies
        strategy_knowledge = self.get_trading_knowledge(
            f"{symbol} trading strategy technical analysis"
        )
        risk_knowledge = self.get_trading_knowledge(
            f"risk management {market_data.market_sentiment}"
        )

        # Prepare enhanced context
        knowledge_context = ""
        if strategy_knowledge:
            knowledge_context += "Relevant Trading Strategies:\n"
            for kb_item in strategy_knowledge:
                knowledge_context += (
                    f"- {kb_item['title']}: {kb_item['content'][:200]}...\n"
                )

        if risk_knowledge:
            knowledge_context += "\nRisk Management Guidelines:\n"
            for kb_item in risk_knowledge:
                knowledge_context += (
                    f"- {kb_item['title']}: {kb_item['content'][:200]}...\n"
                )

        # Enhanced market data string
        market_data_str = f"""
    Symbol: {market_data.symbol}
    Current Price: ${market_data.current_price:.2f}
    24h Change: {market_data.change_24h:.2f}%
    Volume: {market_data.volume:,.0f}
    Market Sentiment: {market_data.market_sentiment}

    Knowledge Base Context:
    {knowledge_context}
    """

        technical_indicators_str = f"""
    RSI: {market_data.rsi:.2f}
    MACD: {market_data.macd:.2f}
    20-day MA: ${market_data.moving_avg_20:.2f}
    50-day MA: ${market_data.moving_avg_50:.2f}
    200-day MA: ${market_data.moving_avg_200:.2f}
    Volatility: {market_data.volatility:.2f}
    """

        # Get historical performance (simulated)
        historical_data_str = (
            "Recent performance shows moderate volatility with mixed signals."
        )

        try:
            analysis = self.analysis_chain.invoke(
                {
                    "symbol": symbol,
                    "market_data": market_data_str,
                    "technical_indicators": technical_indicators_str,
                    "historical_data": historical_data_str,
                }
            )
            return analysis
        except Exception as e:
            system_logger.error(
                f"Error in enhanced market analysis: {e}",
                additional_info={
                    "symbol": symbol,
                    "market_data": market_data_str,
                    "technical_indicators": technical_indicators_str,
                    "historical_data": historical_data_str,
                },
                exc_info=True,
            )
            return self.analyze_market(symbol)  # Fallback to basic analysis

    def make_enhanced_trading_decision(
        self, symbol: str, strategy_type: str = "momentum"
    ) -> TradingDecision:
        """Enhanced trading decision using knowledge base"""
        # Get relevant knowledge for decision making
        decision_knowledge = self.get_trading_knowledge(
            f"{strategy_type} strategy {symbol}"
        )

        # Use enhanced market analysis
        market_analysis = self.analyze_market_with_knowledge(symbol)

        # Get portfolio status
        portfolio_status = self.get_portfolio_status()

        # Get market conditions
        market_conditions = self.get_market_data(symbol)

        # Prepare context for decision making with knowledge
        portfolio_str = f"""
Total Value: ${portfolio_status.total_value:.2f}
Cash Balance: ${portfolio_status.cash_balance:.2f}
Current Holdings: {portfolio_status.holdings}
Performance: {portfolio_status.performance_metrics.get('total_return', 0):.2%}
"""

        risk_parameters_str = f"""
Max Position Size: {self.max_position_size:.1%}
Stop Loss: {self.stop_loss_threshold:.1%}
Take Profit: {self.take_profit_threshold:.1%}
Min Confidence: {self.min_confidence_threshold:.1%}
"""

        # Add knowledge context to decision making
        knowledge_context = ""
        if decision_knowledge:
            knowledge_context = "Relevant Decision Knowledge:\n"
            for kb_item in decision_knowledge:
                knowledge_context += (
                    f"- {kb_item['title']}: {kb_item['content'][:150]}...\n"
                )

        try:
            decision_text = self.decision_chain.invoke(
                {
                    "symbol": symbol,
                    "market_analysis": f"{market_analysis}\n\nKnowledge Context:\n{knowledge_context}",
                    "portfolio_status": portfolio_str,
                    "risk_parameters": risk_parameters_str,
                    "strategy_type": strategy_type,
                }
            )

            # Parse decision (simplified - in real implementation, use structured output)
            decision = self._parse_trading_decision(
                decision_text, symbol, market_conditions, portfolio_status
            )

            return decision

        except Exception as e:
            system_logger.error(
                f"Error making enhanced trading decision: {e}",
                additional_info={
                    "symbol": symbol,
                    "market_analysis": market_analysis,
                    "portfolio_status": portfolio_status,
                    "risk_parameters": risk_parameters_str,
                    "strategy_type": strategy_type,
                },
                exc_info=True,
            )
            # Fallback to basic decision making
            return self.make_trading_decision(symbol, strategy_type)

    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for market analysis"""
        system_message = """You are an expert financial market analyst specializing in algorithmic trading. Your role is to:

1. Analyze current market conditions and technical indicators
2. Identify trading opportunities and market trends
3. Assess market sentiment and volatility
4. Provide data-driven insights for trading decisions

ANALYSIS GUIDELINES:
- Focus on technical analysis using provided indicators
- Consider market sentiment and news impact
- Evaluate risk-reward ratios
- Identify support and resistance levels
- Assess trend strength and momentum

Market Data: {market_data}
Technical Indicators: {technical_indicators}
Historical Performance: {historical_data}

Provide a comprehensive market analysis including:
1. Current market trend and direction
2. Key support and resistance levels
3. Technical indicator signals (RSI, MACD, Moving Averages)
4. Market sentiment assessment
5. Volatility analysis
6. Potential trading opportunities
7. Risk factors to consider"""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "Please analyze the current market conditions for {symbol}"),
            ]
        )

    def _create_decision_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for trading decisions"""
        system_message = """You are an algorithmic trading decision engine. Your role is to make data-driven trading decisions based on:

1. Market analysis and technical indicators
2. Risk management parameters
3. Portfolio allocation and diversification
4. Expected returns and probability of success

DECISION CRITERIA:
- Only recommend trades with confidence > 60%
- Consider position sizing based on risk
- Ensure proper diversification
- Factor in transaction costs and slippage
- Maintain risk-reward ratio of at least 1:2

Market Analysis: {market_analysis}
Current Portfolio: {portfolio_status}
Risk Parameters: {risk_parameters}
Strategy Type: {strategy_type}

Make a trading decision and provide:
1. Action (buy/sell/hold)
2. Quantity/position size
3. Confidence level (0-1)
4. Detailed reasoning
5. Risk assessment
6. Expected return
7. Stop loss and take profit levels"""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "Make a trading decision for {symbol} based on the analysis"),
            ]
        )

    def _create_risk_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for risk assessment"""
        system_message = """You are a risk management specialist for algorithmic trading. Your role is to:

1. Assess the risk level of proposed trades
2. Ensure compliance with risk management rules
3. Calculate position sizing based on risk tolerance
4. Identify potential risk factors and mitigation strategies

RISK ASSESSMENT CRITERIA:
- Maximum position size: 10% of portfolio
- Stop loss: 5% maximum loss per trade
- Portfolio diversification requirements
- Volatility and correlation analysis
- Market condition risk factors

Proposed Trade: {proposed_trade}
Current Portfolio: {portfolio_status}
Market Conditions: {market_conditions}

Provide risk assessment including:
1. Risk score (0-1, where 1 is highest risk)
2. Position size recommendation
3. Stop loss and take profit levels
4. Risk factors identified
5. Risk mitigation strategies
6. Compliance with risk rules"""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "Assess the risk for this trading decision"),
            ]
        )

    def _create_analysis_chain(self):
        """Create market analysis chain"""
        return self.analysis_prompt | self.llm | StrOutputParser()

    def _create_decision_chain(self):
        """Create trading decision chain"""
        return self.decision_prompt | self.llm | StrOutputParser()

    def _create_risk_chain(self):
        """Create risk assessment chain"""
        return self.risk_prompt | self.llm | StrOutputParser()

    def get_market_data(self, symbol: str) -> MarketConditions:
        """Get current market data for a symbol (simulated)"""
        # In a real implementation, this would call actual market data APIs
        # For simulation, we'll generate realistic market data

        base_price = {
            "AAPL": 150.0,
            "GOOGL": 2800.0,
            "MSFT": 300.0,
            "AMZN": 3200.0,
            "TSLA": 800.0,
            "NVDA": 450.0,
            "META": 250.0,
            "NFLX": 400.0,
            "AMD": 100.0,
            "INTC": 50.0,
        }.get(symbol, 100.0)

        # Add some random variation
        price_variation = random.uniform(-0.05, 0.05)
        current_price = base_price * (1 + price_variation)

        # Generate technical indicators
        rsi = random.uniform(30, 70)
        macd = random.uniform(-2, 2)
        moving_avg_20 = current_price * random.uniform(0.98, 1.02)
        moving_avg_50 = current_price * random.uniform(0.95, 1.05)
        moving_avg_200 = current_price * random.uniform(
            0.90, 1.10
        )  # Add missing 200-day MA
        volatility = random.uniform(0.15, 0.35)
        volume = random.uniform(1000000, 10000000)
        change_24h = random.uniform(-5, 5)

        # Determine market sentiment
        sentiment_score = (rsi - 50) / 50 + (
            current_price - moving_avg_20
        ) / moving_avg_20
        if sentiment_score > 0.1:
            market_sentiment = "bullish"
        elif sentiment_score < -0.1:
            market_sentiment = "bearish"
        else:
            market_sentiment = "neutral"

        return MarketConditions(
            symbol=symbol,
            current_price=current_price,
            volume=volume,
            change_24h=change_24h,
            rsi=rsi,
            macd=macd,
            moving_avg_20=moving_avg_20,
            moving_avg_50=moving_avg_50,
            moving_avg_200=moving_avg_200,  # Include the 200-day moving average
            volatility=volatility,
            market_sentiment=market_sentiment,
        )

    def get_portfolio_status(self) -> PortfolioStatus:
        """Get current portfolio status with enhanced error handling"""
        session = get_session()
        try:
            portfolio = (
                session.query(Portfolio)
                .filter(Portfolio.id == self.portfolio_id)
                .first()
            )

            if not portfolio:
                portfolio = self._create_default_portfolio(session)
                if not portfolio:
                    return PortfolioStatus(
                        total_value=10000.0,
                        cash_balance=10000.0,
                        holdings={},
                        performance_metrics={"total_return": 0.0},
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                    )

            cash_balance = (
                float(portfolio.cash_balance)
                if portfolio.cash_balance is not None
                else 10000.0
            )
            holdings = portfolio.holdings or {}
            total_value, holdings_value = self._calculate_holdings_value(
                holdings, cash_balance
            )

            initial_value = 10000.0  # Starting portfolio value
            total_return = (
                (total_value - initial_value) / initial_value
                if initial_value > 0
                else 0.0
            )

            try:
                portfolio.total_value = total_value
                portfolio.performance_metrics = {
                    "total_return": total_return,
                    "total_value": total_value,
                    "unrealized_pnl": holdings_value,
                    "holdings_value": holdings_value,
                }
                session.commit()
            except Exception as update_error:
                system_logger.warning(
                    f"Failed to update portfolio in database: {update_error}"
                )
                session.rollback()

            return PortfolioStatus(
                total_value=total_value,
                cash_balance=cash_balance,
                holdings=holdings,
                performance_metrics=portfolio.performance_metrics
                or {"total_return": total_return},
                unrealized_pnl=holdings_value,
                realized_pnl=0.0,
            )

        except Exception as e:
            system_logger.error(f"Error in get_portfolio_status: {e}", exc_info=True)
            session.rollback()
            return PortfolioStatus(
                total_value=10000.0,
                cash_balance=10000.0,
                holdings={},
                performance_metrics={"total_return": 0.0},
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )
        finally:
            session.close()

    def _create_default_portfolio(self, session):
        """Helper to create a default portfolio"""
        try:
            portfolio = Portfolio(
                portfolio_name=f"Trading Bot Portfolio {self.portfolio_id}",
                cash_balance=10000.0,
                total_value=10000.0,
                holdings={},
                performance_metrics={"total_return": 0.0, "sharpe_ratio": 0.0},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            session.add(portfolio)
            session.commit()
            system_logger.info(f"Created new portfolio with ID: {self.portfolio_id}")
            return portfolio
        except Exception as create_error:
            system_logger.error(f"Failed to create portfolio: {create_error}")
            session.rollback()
            return None

    def _calculate_holdings_value(self, holdings, cash_balance):
        """Helper to calculate holdings value and total value"""
        holdings_value = 0.0
        total_value = cash_balance
        for symbol, quantity in holdings.items():
            if isinstance(quantity, (int, float)) and quantity > 0:
                try:
                    market_data = self.get_market_data(symbol)
                    symbol_value = quantity * market_data.current_price
                    holdings_value += symbol_value
                    total_value += symbol_value
                except Exception as market_error:
                    system_logger.warning(
                        f"Could not get market data for {symbol}: {market_error}"
                    )
                    fallback_prices = {"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 300.0}
                    fallback_price = fallback_prices.get(symbol, 100.0)
                    symbol_value = quantity * fallback_price
                    holdings_value += symbol_value
                    total_value += symbol_value
        return total_value, holdings_value

    def analyze_market(self, symbol: str) -> str:
        """Analyze market conditions for a symbol"""
        market_data = self.get_market_data(symbol)

        # Prepare market data for analysis
        market_data_str = f"""
    Symbol: {market_data.symbol}
    Current Price: ${market_data.current_price:.2f}
    24h Change: {market_data.change_24h:.2f}%
    Volume: {market_data.volume:,.0f}
    Market Sentiment: {market_data.market_sentiment}
    """

        technical_indicators_str = f"""
    RSI: {market_data.rsi:.2f}
    MACD: {market_data.macd:.2f}
    20-day MA: ${market_data.moving_avg_20:.2f}
    50-day MA: ${market_data.moving_avg_50:.2f}
    200-day MA: ${market_data.moving_avg_200:.2f}
    Volatility: {market_data.volatility:.2f}
    """

        # Get historical performance (simulated)
        historical_data_str = (
            "Recent performance shows moderate volatility with mixed signals."
        )

        try:
            analysis = self.analysis_chain.invoke(
                {
                    "symbol": symbol,
                    "market_data": market_data_str,
                    "technical_indicators": technical_indicators_str,
                    "historical_data": historical_data_str,
                }
            )
            return analysis
        except Exception as e:
            system_logger.error(
                f"Error analyzing market: {e}",
                additional_info={
                    "symbol": symbol,
                    "market_data": market_data_str,
                    "technical_indicators": technical_indicators_str,
                    "historical_data": historical_data_str,
                },
                exc_info=True,
            )
            # Enhanced fallback analysis when LLM is unavailable
            return self._generate_fallback_analysis(market_data)

    def make_trading_decision(
        self, symbol: str, strategy_type: str = "momentum"
    ) -> TradingDecision:
        """Make a trading decision for a symbol"""
        # Get market analysis
        market_analysis = self.analyze_market(symbol)

        # Get portfolio status
        portfolio_status = self.get_portfolio_status()

        # Get market conditions
        market_conditions = self.get_market_data(symbol)

        # Prepare context for decision making
        portfolio_str = f"""
Total Value: ${portfolio_status.total_value:.2f}
Cash Balance: ${portfolio_status.cash_balance:.2f}
Current Holdings: {portfolio_status.holdings}
Performance: {portfolio_status.performance_metrics.get('total_return', 0):.2%}
"""

        risk_parameters_str = f"""
Max Position Size: {self.max_position_size:.1%}
Stop Loss: {self.stop_loss_threshold:.1%}
Take Profit: {self.take_profit_threshold:.1%}
Min Confidence: {self.min_confidence_threshold:.1%}
"""

        try:
            decision_text = self.decision_chain.invoke(
                {
                    "symbol": symbol,
                    "market_analysis": market_analysis,
                    "portfolio_status": portfolio_str,
                    "risk_parameters": risk_parameters_str,
                    "strategy_type": strategy_type,
                }
            )

            # Parse decision (simplified - in real implementation, use structured output)
            decision = self._parse_trading_decision(
                decision_text, symbol, market_conditions, portfolio_status
            )

            return decision

        except Exception as e:
            system_logger.error(
                f"Error making trading decision: {e}",
                additional_info={
                    "symbol": symbol,
                    "market_analysis": market_analysis,
                    "portfolio_status": portfolio_str,
                    "risk_parameters": risk_parameters_str,
                    "strategy_type": strategy_type,
                },
                exc_info=True,
            )
            # Enhanced fallback decision when LLM is unavailable
            return self._generate_fallback_decision(
                symbol, market_conditions, portfolio_status
            )

    def _parse_trading_decision(
        self,
        decision_text: str,
        symbol: str,
        market_conditions: MarketConditions,
        portfolio_status: PortfolioStatus,
    ) -> TradingDecision:
        """Parse trading decision from LLM output"""
        decision_text_lower = decision_text.lower()

        # Determine action
        if "buy" in decision_text_lower and "sell" not in decision_text_lower:
            action = "buy"
        elif "sell" in decision_text_lower:
            action = "sell"
        else:
            action = "hold"

        # Calculate position size based on portfolio and risk
        max_position_value = portfolio_status.total_value * self.max_position_size

        if action == "buy":
            quantity = max_position_value / market_conditions.current_price
            # Ensure we have enough cash
            if (
                quantity * market_conditions.current_price
                > portfolio_status.cash_balance
            ):
                quantity = (
                    portfolio_status.cash_balance
                    / market_conditions.current_price
                    * 0.95
                )  # Leave some buffer
        elif action == "sell":
            current_holding = portfolio_status.holdings.get(symbol, 0)
            quantity = current_holding * 0.5  # Sell half position as default
        else:
            quantity = 0.0

        # Calculate confidence based on technical indicators
        confidence = self._calculate_confidence(market_conditions, action)

        # Calculate risk score
        risk_score = self._calculate_risk_score(market_conditions, action, quantity)

        # Calculate expected return (simplified)
        expected_return = self._calculate_expected_return(market_conditions, action)

        return TradingDecision(
            action=action,
            symbol=symbol,
            quantity=max(0, quantity),
            confidence=confidence,
            reasoning=(
                decision_text[:200] + "..."
                if len(decision_text) > 200
                else decision_text
            ),
            risk_score=risk_score,
            expected_return=expected_return,
        )

    def _calculate_confidence(
        self, market_conditions: MarketConditions, action: str
    ) -> float:
        """Calculate confidence score based on technical indicators"""
        confidence = 0.5  # Base confidence
        confidence += self._rsi_confidence(market_conditions, action)
        confidence += self._moving_avg_confidence(market_conditions, action)
        confidence += self._macd_confidence(market_conditions, action)
        confidence += self._sentiment_confidence(market_conditions, action)
        return min(1.0, max(0.0, confidence))

    def _rsi_confidence(
        self, market_conditions: MarketConditions, action: str
    ) -> float:
        if (action == "buy" and market_conditions.rsi < 40) or (
            action == "sell" and market_conditions.rsi > 60
        ):
            return 0.2
        return 0.0

    def _moving_avg_confidence(
        self, market_conditions: MarketConditions, action: str
    ) -> float:
        price = market_conditions.current_price
        ma_20 = market_conditions.moving_avg_20
        ma_50 = market_conditions.moving_avg_50
        ma_200 = market_conditions.moving_avg_200

        confidence_boost = 0.0

        # Check if price is above/below key moving averages
        if action == "buy":
            if price > ma_20 and ma_20 > ma_50:
                confidence_boost += 0.1
            if price > ma_200:  # Above long-term trend
                confidence_boost += 0.05
        elif action == "sell":
            if price < ma_20 and ma_20 < ma_50:
                confidence_boost += 0.1
            if price < ma_200:  # Below long-term trend
                confidence_boost += 0.05

        return confidence_boost

    def _macd_confidence(
        self, market_conditions: MarketConditions, action: str
    ) -> float:
        if (action == "buy" and market_conditions.macd > 0) or (
            action == "sell" and market_conditions.macd < 0
        ):
            return 0.1
        return 0.0

    def _sentiment_confidence(
        self, market_conditions: MarketConditions, action: str
    ) -> float:
        if (action == "buy" and market_conditions.market_sentiment == "bullish") or (
            action == "sell" and market_conditions.market_sentiment == "bearish"
        ):
            return 0.1
        return 0.0

    def _calculate_risk_score(
        self, market_conditions: MarketConditions, action: str, quantity: float
    ) -> float:
        """Calculate risk score for the trade"""
        risk_score = 0.3  # Base risk

        # Volatility risk
        risk_score += market_conditions.volatility * 0.5

        # Position size risk
        if quantity > 0:
            position_value = quantity * market_conditions.current_price
            portfolio_value = 10000.0  # Simplified
            position_ratio = position_value / portfolio_value
            risk_score += position_ratio * 0.3

        # Market sentiment risk
        if (market_conditions.market_sentiment == "bearish" and action == "buy") or (
            market_conditions.market_sentiment == "bullish" and action == "sell"
        ):
            risk_score += 0.2

        return min(1.0, max(0.0, risk_score))

    def _calculate_expected_return(
        self, market_conditions: MarketConditions, action: str
    ) -> float:
        """Calculate expected return for the trade"""
        if action == "hold":
            return 0.0

        # Simplified expected return calculation
        base_return = 0.02  # 2% base expected return

        # Adjust based on market sentiment
        if action == "buy" and market_conditions.market_sentiment == "bullish":
            base_return *= 1.5
        elif action == "sell" and market_conditions.market_sentiment == "bearish":
            base_return *= 1.5
        elif action == "buy" and market_conditions.market_sentiment == "bearish":
            base_return *= 0.5
        elif action == "sell" and market_conditions.market_sentiment == "bullish":
            base_return *= 0.5

        # Adjust for volatility (higher volatility = higher potential return but also risk)
        volatility_adjustment = market_conditions.volatility * 0.1
        base_return += volatility_adjustment

        return base_return

    def _generate_fallback_analysis(self, market_data: MarketConditions) -> str:
        """Generate fallback market analysis when LLM is unavailable"""
        # Technical analysis based on indicators
        trend = "neutral"
        if (
            market_data.current_price
            > market_data.moving_avg_20
            > market_data.moving_avg_50
        ):
            trend = "bullish"
        elif (
            market_data.current_price
            < market_data.moving_avg_20
            < market_data.moving_avg_50
        ):
            trend = "bearish"

        rsi_signal = "neutral"
        if market_data.rsi < 30:
            rsi_signal = "oversold (potential buy)"
        elif market_data.rsi > 70:
            rsi_signal = "overbought (potential sell)"

        macd_signal = "bullish" if market_data.macd > 0 else "bearish"

        if market_data.volatility > 0.25:
            volatility_level = "high"
        elif market_data.volatility > 0.15:
            volatility_level = "moderate"
        else:
            volatility_level = "low"

        return f"""Market analysis for {market_data.symbol}: Current price ${market_data.current_price:.2f}, RSI {market_data.rsi:.1f}, trending {market_data.market_sentiment}.
        
Technical Analysis:
- Trend: {trend} (price vs moving averages)
- RSI Signal: {rsi_signal}
- MACD Signal: {macd_signal}
- Volatility: {volatility_level} ({market_data.volatility:.2f})
- Volume: {market_data.volume:,.0f}
- 24h Change: {market_data.change_24h:.2f}%
        
Recommendation: Based on technical indicators, the stock shows {trend} momentum with {rsi_signal} RSI levels."""

    def _generate_fallback_decision(
        self,
        symbol: str,
        market_conditions: MarketConditions,
        portfolio_status: PortfolioStatus,
    ) -> TradingDecision:
        """Generate fallback trading decision when LLM is unavailable"""
        # Rule-based decision making
        action = "hold"
        confidence = 0.5
        reasoning = "Fallback decision due to LLM service unavailability. "

        # Simple momentum strategy
        if (
            market_conditions.rsi < 30
            and market_conditions.market_sentiment == "bullish"
        ):
            action = "buy"
            confidence = 0.7
            reasoning += "RSI oversold + bullish sentiment suggests buy opportunity."
        elif (
            market_conditions.rsi > 70
            and market_conditions.market_sentiment == "bearish"
        ):
            action = "sell"
            confidence = 0.7
            reasoning += "RSI overbought + bearish sentiment suggests sell opportunity."
        elif (
            market_conditions.current_price
            > market_conditions.moving_avg_20
            > market_conditions.moving_avg_50
        ):
            if market_conditions.macd > 0:
                action = "buy"
                confidence = 0.65
                reasoning += "Bullish trend with positive MACD suggests buy."
        elif (
            market_conditions.current_price
            < market_conditions.moving_avg_20
            < market_conditions.moving_avg_50
        ):
            if market_conditions.macd < 0:
                action = "sell"
                confidence = 0.65
                reasoning += "Bearish trend with negative MACD suggests sell."
        else:
            reasoning += "Mixed signals, holding position."

        # Calculate position size
        quantity = 0.0
        if action == "buy":
            max_position_value = portfolio_status.total_value * self.max_position_size
            quantity = min(
                max_position_value / market_conditions.current_price,
                portfolio_status.cash_balance / market_conditions.current_price * 0.95,
            )
        elif action == "sell":
            current_holding = portfolio_status.holdings.get(symbol, 0)
            quantity = current_holding * 0.5  # Sell half position

        # Calculate risk and return
        risk_score = self._calculate_risk_score(market_conditions, action, quantity)
        expected_return = self._calculate_expected_return(market_conditions, action)

        return TradingDecision(
            action=action,
            symbol=symbol,
            quantity=max(0, quantity),
            confidence=confidence,
            reasoning=reasoning,
            risk_score=risk_score,
            expected_return=expected_return,
        )

    def execute_trade(
        self, decision: TradingDecision, strategy_id: Optional[int] = None
    ) -> bool:
        """Execute a trading decision"""
        if decision.action == "hold" or decision.quantity <= 0:
            return True

        if decision.confidence < self.min_confidence_threshold:
            print(
                f"Trade confidence {decision.confidence:.2f} below threshold {self.min_confidence_threshold:.2f}"
            )
            return False

        session = get_session()
        portfolio = None  # Ensure portfolio is always defined
        try:
            # Get current portfolio
            portfolio = (
                session.query(Portfolio)
                .filter(Portfolio.id == self.portfolio_id)
                .first()
            )

            if not portfolio:
                print(PORTFOLIO_ISSUE)
                return False

            # Get or create strategy and update strategy_id
            strategy_id = self._get_or_create_strategy(session, strategy_id)

            # Get current market price
            market_conditions = self.get_market_data(decision.symbol)
            current_price = market_conditions.current_price

            # Calculate trade value
            trade_value = decision.quantity * current_price
            fees = trade_value * 0.001  # 0.1% trading fee

            holdings = portfolio.holdings or {}

            if decision.action == "buy":
                if not self._execute_buy(
                    portfolio, holdings, decision, trade_value, fees
                ):
                    return False
            elif decision.action == "sell" and not self._execute_sell(
                portfolio, holdings, decision, trade_value, fees
            ):
                return False

            # Update portfolio
            portfolio.holdings.clear()
            portfolio.holdings.update(holdings)

            # Log the trade with valid strategy_id
            self._log_trade(
                session,
                strategy_id,
                decision,
                current_price,
                trade_value,
                fees,
                market_conditions,
            )

            print(
                f"Trade executed: {decision.action.upper()} {decision.quantity:.2f} {decision.symbol} at ${current_price:.2f} (Strategy: {strategy_id})"
            )
            return True

        except Exception as e:
            system_logger.error(
                f"Error executing trade: {e}",
                additional_info={
                    "decision": decision.__dict__,
                    "portfolio": portfolio.__dict__ if portfolio is not None else None,
                    "strategy_id": strategy_id,
                },
                exc_info=True,
            )
            session.rollback()
            return False
        finally:
            session.close()

    def _get_or_create_strategy(self, session, strategy_id):
        """Helper to get or create a trading strategy and return its id"""
        if strategy_id is None:
            system_logger.warning("No strategy_id provided. Creating default strategy.")
            strategy = session.query(TradingStrategy).first()
            if not strategy:
                strategy = TradingStrategy(
                    strategy_type="momentum",  # Use correct column name
                    description="Default Momentum Strategy",
                    parameters={
                        "rsi_oversold": 30,
                        "rsi_overbought": 70,
                        "macd_signal": True,
                        "moving_avg_crossover": True,
                        "max_position_size": 0.1,
                        "stop_loss": 0.05,
                        "take_profit": 0.15,
                    },
                    risk_parameters={
                        "risk_tolerance": "medium",
                        "expected_return": 0.08,
                        "max_drawdown": 0.15,
                    },
                    is_active=True,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                session.add(strategy)
                session.commit()
            return getattr(strategy, "id", None)
        else:
            strategy = (
                session.query(TradingStrategy)
                .filter(TradingStrategy.id == strategy_id)
                .first()
            )
            if not strategy:
                system_logger.warning(
                    f"Strategy with id {strategy_id} not found. Creating default strategy."
                )
                strategy = TradingStrategy(
                    strategy_type="momentum",  # Use correct column name
                    description=f"Auto-created Strategy {strategy_id}",
                    parameters={
                        "rsi_oversold": 30,
                        "rsi_overbought": 70,
                        "macd_signal": True,
                        "moving_avg_crossover": True,
                        "max_position_size": 0.1,
                        "stop_loss": 0.05,
                        "take_profit": 0.15,
                    },
                    risk_parameters={
                        "risk_tolerance": "medium",
                        "expected_return": 0.08,
                        "max_drawdown": 0.15,
                    },
                    is_active=True,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                session.add(strategy)
                session.commit()
            return strategy.id

    def _log_trade(
        self,
        session,
        strategy_id,
        decision,
        current_price,
        trade_value,
        fees,
        market_conditions,
    ):
        """Helper to log the trade"""
        trade_log = TradeLog(
            strategy_id=strategy_id,
            symbol=decision.symbol,
            trade_type=decision.action,
            quantity=decision.quantity,
            price=current_price,
            total_value=trade_value,
            fees=fees,
            market_conditions={
                "rsi": market_conditions.rsi,
                "macd": market_conditions.macd,
                "volatility": market_conditions.volatility,
                "sentiment": market_conditions.market_sentiment,
            },
            confidence_score=decision.confidence,
        )
        session.add(trade_log)
        session.commit()

    def _execute_buy(self, portfolio, holdings, decision, trade_value, fees) -> bool:
        """Helper to execute buy logic"""
        total_cost = trade_value + fees
        if total_cost > float(portfolio.cash_balance):
            system_logger.warning(
                f"Insufficient cash: need ${total_cost:.2f}, have ${float(portfolio.cash_balance):.2f}"
            )
            return False
        portfolio.cash_balance -= total_cost
        current_holding = holdings.get(decision.symbol, 0)
        holdings[decision.symbol] = current_holding + decision.quantity
        return True

    def _execute_sell(self, portfolio, holdings, decision, trade_value, fees) -> bool:
        """Helper to execute sell logic"""
        current_holding = holdings.get(decision.symbol, 0)
        if decision.quantity > current_holding:
            print(
                f"Insufficient shares: trying to sell {decision.quantity}, have {current_holding}"
            )
            return False
        portfolio.cash_balance += trade_value - fees
        holdings[decision.symbol] = current_holding - decision.quantity
        if holdings[decision.symbol] <= 0:
            del holdings[decision.symbol]
        return True

    def run_trading_cycle(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run a complete trading cycle for specified symbols"""
        if symbols is None:
            symbols = self.supported_symbols[:5]  # Trade top 5 symbols

        results = {
            "timestamp": datetime.now().isoformat(),
            "symbols_analyzed": len(symbols),
            "trades_executed": 0,
            "decisions": [],
            "portfolio_status": None,
            "total_confidence": 0.0,
        }

        # Ensure we have a valid strategy for trading
        session = get_session()
        try:
            strategy = session.query(TradingStrategy).first()
            if not strategy:
                strategy = TradingStrategy(
                    strategy_name="Trading Cycle Strategy",
                    description="Strategy for automated trading cycle execution",
                    strategy_type="momentum",
                    parameters={
                        "rsi_oversold": 30,
                        "rsi_overbought": 70,
                        "macd_signal": True,
                        "moving_avg_crossover": True,
                        "max_position_size": 0.1,
                        "stop_loss": 0.05,
                        "take_profit": 0.15,
                    },
                    risk_parameters={
                        "risk_tolerance": "medium",
                        "expected_return": 0.08,
                        "max_drawdown": 0.15,
                    },
                    is_active=True,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                session.add(strategy)
                session.commit()
            strategy_id = strategy.id
        finally:
            session.close()

        for symbol in symbols:
            self._process_symbol_for_trading_cycle(symbol, results, strategy_id)

        # Get final portfolio status
        results["portfolio_status"] = self.get_portfolio_status().__dict__
        results["average_confidence"] = (
            results["total_confidence"] / len(symbols) if symbols else 0
        )

        return results

    def _process_symbol_for_trading_cycle(
        self, symbol: str, results: Dict[str, Any], strategy_id: int
    ):
        """Helper to process each symbol in trading cycle to reduce cognitive complexity"""
        decision = None  # Ensure decision is always defined
        try:
            # Make trading decision (use enhanced version if knowledge base is available)
            try:
                decision = self.make_enhanced_trading_decision(symbol)
            except Exception:
                decision = self.make_trading_decision(symbol)

            # Execute trade if conditions are met
            trade_executed = False
            if (
                decision.confidence >= self.min_confidence_threshold
                and decision.action != "hold"
            ):
                trade_executed = self.execute_trade(decision, strategy_id)
                if trade_executed:
                    results["trades_executed"] += 1

            # Record decision
            results["decisions"].append(
                {
                    "symbol": symbol,
                    "action": decision.action,
                    "quantity": decision.quantity,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning,
                    "executed": trade_executed,
                    "strategy_id": strategy_id,
                }
            )

            results["total_confidence"] += decision.confidence

        except Exception as e:
            system_logger.error(
                f"Error processing {symbol}: {e}",
                additional_info={
                    "decision": decision.__dict__ if decision is not None else None,
                    "strategy_id": strategy_id,
                },
                exc_info=True,
            )
            results["decisions"].append(
                {"symbol": symbol, "action": "error", "error": str(e)}
            )

    def initialize_default_trading_strategy(self) -> int:
        """Initialize a default trading strategy if none exists"""
        session = get_session()
        try:
            # Check if any strategy exists
            strategy = session.query(TradingStrategy).first()
            if strategy:
                return strategy.id

            # Create default strategy using only fields that exist in the database model
            default_strategy = TradingStrategy(
                strategy_type="momentum",  # Use the correct column name
                description="Automated trading strategy for momentum-based decisions",
                parameters={
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "macd_signal": True,
                    "moving_avg_crossover": True,
                    "max_position_size": 0.1,
                    "stop_loss": 0.05,
                    "take_profit": 0.15,
                    "min_confidence": 0.6,
                    "volatility_threshold": 0.3,
                },
                risk_parameters={
                    "risk_tolerance": "medium",
                    "expected_return": 0.08,
                    "max_drawdown": 0.15,
                    "max_position_size": 0.1,
                    "stop_loss_threshold": 0.05,
                    "take_profit_threshold": 0.15,
                },
                is_active=True,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            session.add(default_strategy)
            session.commit()

            system_logger.info(
                f"Created default trading strategy with ID: {default_strategy.id}",
                additional_info={
                    "strategy_type": default_strategy.strategy_type,
                    "parameters": default_strategy.parameters,
                },
            )

            return default_strategy.id

        except Exception as e:
            session.rollback()
            system_logger.error(
                f"Error creating default trading strategy: {e}", exc_info=True
            )
            raise
        finally:
            session.close()

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report"""
        session = get_session()
        try:
            # Get portfolio
            portfolio = (
                session.query(Portfolio)
                .filter(Portfolio.id == self.portfolio_id)
                .first()
            )

            if not portfolio:
                system_logger.error(
                    PORTFOLIO_ISSUE,
                    additional_info={"portfolio_id": self.portfolio_id},
                    exc_info=True,
                )
                return {"error": PORTFOLIO_ISSUE}

            # Get trade history
            trades = (
                session.query(TradeLog)
                .join(TradingStrategy)
                .filter(TradeLog.strategy_id.isnot(None))
                .order_by(TradeLog.executed_at.desc())
                .limit(50)
                .all()
            )

            # Calculate performance metrics
            portfolio_status = self.get_portfolio_status()
            initial_value = 10000.0
            total_return = (
                portfolio_status.total_value - initial_value
            ) / initial_value

            # Trade statistics
            total_trades = len(trades)
            buy_trades = len([t for t in trades if t.trade_type == "buy"])
            sell_trades = len([t for t in trades if t.trade_type == "sell"])

            # Recent trades
            recent_trades = [
                {
                    "symbol": t.symbol,
                    "type": t.trade_type,
                    "quantity": t.quantity,
                    "price": t.price,
                    "timestamp": t.executed_at.isoformat() if t.executed_at else None,
                    "confidence": t.confidence_score,
                }
                for t in trades[:10]
            ]

            return {
                "portfolio_value": portfolio_status.total_value,
                "cash_balance": portfolio_status.cash_balance,
                "total_return": total_return,
                "total_return_pct": total_return * 100,
                "holdings": portfolio_status.holdings,
                "trade_statistics": {
                    "total_trades": total_trades,
                    "buy_trades": buy_trades,
                    "sell_trades": sell_trades,
                },
                "recent_trades": recent_trades,
                "performance_metrics": portfolio_status.performance_metrics,
                "timestamp": datetime.now().isoformat(),
            }

        finally:
            session.close()


def test_financial_trading_bot_agent():
    """Test the Financial Trading Bot agent"""
    agent = FinancialTradingBotAgent()

    print("=== Financial Trading Bot Agent Test ===\n")

    # Test market analysis
    print("1. Market Analysis Test:")
    analysis = agent.analyze_market("AAPL")
    print(f"AAPL Analysis: {analysis[:200]}...\n")

    # Test enhanced market analysis with knowledge base
    print("1.1. Enhanced Market Analysis Test:")
    try:
        enhanced_analysis = agent.analyze_market_with_knowledge("AAPL")
        print(f"Enhanced AAPL Analysis: {enhanced_analysis[:200]}...\n")
    except Exception as e:
        print(f"Enhanced analysis failed, falling back to basic: {e}\n")

    # Test trading decision
    print("2. Trading Decision Test:")
    decision = agent.make_trading_decision("AAPL")
    print(f"Decision: {decision.action.upper()} {decision.quantity:.2f} AAPL")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Reasoning: {decision.reasoning[:100]}...\n")

    # Test enhanced trading decision
    print("2.1. Enhanced Trading Decision Test:")
    try:
        enhanced_decision = agent.make_enhanced_trading_decision("AAPL")
        print(
            f"Enhanced Decision: {enhanced_decision.action.upper()} {enhanced_decision.quantity:.2f} AAPL"
        )
        print(f"Enhanced Confidence: {enhanced_decision.confidence:.2f}")
        print(f"Enhanced Reasoning: {enhanced_decision.reasoning[:100]}...\n")
    except Exception as e:
        print(f"Enhanced decision failed, falling back to basic: {e}\n")

    # Test trading cycle
    print("3. Trading Cycle Test:")
    results = agent.run_trading_cycle(["AAPL", "GOOGL", "MSFT"])
    print(f"Symbols analyzed: {results['symbols_analyzed']}")
    print(f"Trades executed: {results['trades_executed']}")
    print(f"Average confidence: {results['average_confidence']:.2f}\n")

    # Test performance report
    print("4. Performance Report Test:")
    report = agent.get_performance_report()
    print(f"Portfolio value: ${report['portfolio_value']:.2f}")
    print(f"Total return: {report['total_return_pct']:.2f}%")
    print(f"Total trades: {report['trade_statistics']['total_trades']}")

    # Test knowledge base integration
    print("\n5. Knowledge Base Integration Test:")
    try:
        knowledge = agent.get_trading_knowledge("risk management strategies")
        print(f"Knowledge base results: {len(knowledge)} items found")
        if knowledge:
            print(f"First result: {knowledge[0]['title']}")
    except Exception as e:
        print(f"Knowledge base test failed: {e}")


if __name__ == "__main__":
    test_financial_trading_bot_agent()
