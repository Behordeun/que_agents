# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-10
# @Description: Financial Trading Bot API routes and handlers

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.que_agents.core.schemas import TradingAnalysisRequest, TradingDecisionResponse
from src.que_agents.error_trace.errorlogger import system_logger
from src.que_agents.utils.agent_manager import AgentManager
from src.que_agents.utils.auth import get_token_from_state


class FinancialTradingBotService:
    """Service class for financial trading bot operations"""

    FINANCIAL_TRADING_BOT_CONTEXT = "Financial Trading Bot"

    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        self.TRADING_BOT_UNAVAILABLE = "Financial Trading Bot not available"

    def get_agent(self, token: str):
        """Get financial trading bot agent"""
        agent = self.agent_manager.get_agent("financial_trading_bot", token)
        if agent is None:
            system_logger.error(
                "Financial trading bot is not available in AgentManager.",
                additional_info={"context": self.FINANCIAL_TRADING_BOT_CONTEXT},
            )
        return agent

    def analyze_and_make_decision(
        self,
        request: TradingAnalysisRequest,
        token: str = Depends(get_token_from_state),
    ) -> TradingDecisionResponse:
        """Analyze market and make trading decision"""
        try:
            agent = self.get_agent(token)
            if not agent:
                raise HTTPException(
                    status_code=503, detail=self.TRADING_BOT_UNAVAILABLE
                )

            decision = agent.make_trading_decision(
                symbol=request.symbol, strategy_type=request.strategy_type
            )

            return TradingDecisionResponse(
                action=decision.action,
                symbol=decision.symbol,
                quantity=decision.quantity,
                confidence=decision.confidence,
                reasoning=decision.reasoning,
                risk_score=decision.risk_score,
                expected_return=decision.expected_return,
            )
        except HTTPException:
            raise
        except Exception as e:
            system_logger.error(
                f"Error in trading analysis: {str(e)}",
                additional_info={
                    "context": "Trading Analysis",
                    "symbol": request.symbol,
                    "strategy_type": request.strategy_type,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error analyzing trading decision: {str(e)}",
            )

    def run_trading_cycle_operation(
        self,
        symbols: Optional[List[str]] = None,
        token: str = Depends(get_token_from_state),
    ) -> Dict[str, Any]:
        """Run trading cycle with specified symbols"""
        try:
            agent = self.get_agent(token)
            if not agent:
                raise HTTPException(
                    status_code=503, detail=self.TRADING_BOT_UNAVAILABLE
                )

            # Use default symbols if none provided
            if symbols is None:
                symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]

            # Use fallback or agent method
            if hasattr(agent, "run_trading_cycle"):
                result = agent.run_trading_cycle(symbols)
            else:
                result = self._generate_fallback_trading_cycle(symbols)

            return result

        except HTTPException:
            raise
        except Exception as e:
            system_logger.error(
                f"Error in trading cycle: {str(e)}",
                additional_info={
                    "context": "Run Trading Cycle",
                    "symbols": symbols,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail=f"Trading cycle failed: {str(e)}"
            )

    def _generate_fallback_trading_cycle(self, symbols: List[str]) -> Dict[str, Any]:
        """Generate fallback trading cycle results"""
        return {
            "timestamp": datetime.now().isoformat(),
            "symbols_analyzed": len(symbols),
            "trades_executed": 0,
            "decisions": [
                {
                    "symbol": symbol,
                    "action": "hold",
                    "confidence": 0.6
                    + (hash(symbol) % 20) / 100,  # Pseudo-random confidence
                    "executed": False,
                    "reasoning": f"Market conditions for {symbol} suggest holding position",
                }
                for symbol in symbols
            ],
            "portfolio_status": {
                "total_value": 10000.0,
                "cash_balance": 5000.0,
            },
            "average_confidence": 0.7,
            "market_conditions": "neutral",
            "risk_assessment": "moderate",
            "fallback_mode": True,
        }

    def get_portfolio_status_data(
        self, token: str = Depends(get_token_from_state)
    ) -> Dict[str, Any]:
        """Get portfolio status with comprehensive error handling"""
        fallback_data = {
            "portfolio_value": 10000.0,
            "cash_balance": 5000.0,
            "unrealized_pnl": 250.0,
            "realized_pnl": 150.0,
            "holdings": {
                "AAPL": {"shares": 10.0, "value": 1500.0, "avg_price": 150.0},
                "GOOGL": {"shares": 5.0, "value": 1350.0, "avg_price": 270.0},
                "MSFT": {"shares": 8.0, "value": 2400.0, "avg_price": 300.0},
            },
            "performance_metrics": {
                "total_return": 4.0,
                "daily_return": 0.5,
                "sharpe_ratio": 1.2,
                "max_drawdown": -2.5,
                "volatility": 15.2,
            },
            "sector_allocation": {
                "Technology": 80.0,
                "Healthcare": 15.0,
                "Finance": 5.0,
            },
            "risk_metrics": {
                "portfolio_beta": 1.1,
                "value_at_risk_95": -0.03,
                "expected_shortfall": -0.045,
                "volatility": 0.15,
            },
            "recent_trades": [
                {
                    "symbol": "AAPL",
                    "action": "buy",
                    "quantity": 2.0,
                    "price": 152.5,
                    "timestamp": (datetime.now()).isoformat(),
                },
                {
                    "symbol": "MSFT",
                    "action": "sell",
                    "quantity": 1.0,
                    "price": 305.0,
                    "timestamp": (datetime.now()).isoformat(),
                },
            ],
            "status": "fallback_data",
            "timestamp": datetime.now().isoformat(),
            "note": "Using simulated portfolio data",
        }

        try:
            agent = self.get_agent(token)
            if not agent:
                return self._add_unavailable_status(fallback_data)

            if hasattr(agent, "get_portfolio_status"):
                try:
                    portfolio_status = agent.get_portfolio_status()
                    return self._format_portfolio_status(portfolio_status)
                except AttributeError as ae:
                    system_logger.warning(f"Portfolio method missing: {ae}")
                    return fallback_data
                except Exception as e:
                    system_logger.error(
                        f"Error getting portfolio status from agent: {e}",
                        additional_info={
                            "context": "Get Portfolio Status",
                            "error_type": type(e).__name__,
                        },
                        exc_info=True,
                    )
                    return fallback_data

            return fallback_data

        except Exception as e:
            system_logger.error(
                f"Critical error in portfolio endpoint: {str(e)}",
                additional_info={
                    "context": "Get Portfolio Status",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return {
                "portfolio_value": 10000.0,
                "cash_balance": 10000.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "holdings": {},
                "performance_metrics": {"total_return": 0.0},
                "status": "error",
                "error_message": "Portfolio service temporarily unavailable",
                "timestamp": datetime.now().isoformat(),
            }

    def _add_unavailable_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add unavailable status to portfolio data"""
        data.update(
            {
                "status": "agent_unavailable",
                "note": "Trading bot agent not available, using fallback data",
            }
        )
        return data

    def _format_portfolio_status(self, portfolio_status) -> Dict[str, Any]:
        """Helper to format portfolio status into a standardized dict"""
        now = datetime.now().isoformat()

        if hasattr(portfolio_status, "__dict__"):
            result = self._from_obj_portfolio(portfolio_status, now)
        elif isinstance(portfolio_status, dict):
            result = self._from_dict_portfolio(portfolio_status, now)
        else:
            raise ValueError(
                f"Unexpected portfolio status type: {type(portfolio_status)}"
            )

        # Ensure all numeric values are properly formatted
        for key in (
            "portfolio_value",
            "cash_balance",
            "unrealized_pnl",
            "realized_pnl",
        ):
            try:
                result[key] = (
                    float(result.get(key, 0.0)) if result.get(key) is not None else 0.0
                )
            except (ValueError, TypeError):
                result[key] = 0.0

        return result

    def _from_obj_portfolio(self, obj, now: str) -> Dict[str, Any]:
        """Convert portfolio object to dictionary"""
        return {
            "portfolio_value": float(getattr(obj, "total_value", 10000.0)),
            "cash_balance": float(getattr(obj, "cash_balance", 5000.0)),
            "unrealized_pnl": float(getattr(obj, "unrealized_pnl", 0.0)),
            "realized_pnl": float(getattr(obj, "realized_pnl", 0.0)),
            "holdings": dict(getattr(obj, "holdings", {})),
            "performance_metrics": dict(
                getattr(obj, "performance_metrics", {"total_return": 0.0})
            ),
            "sector_allocation": (
                dict(getattr(obj, "sector_allocation", {}))
                if hasattr(obj, "sector_allocation")
                else {}
            ),
            "risk_metrics": (
                dict(getattr(obj, "risk_metrics", {}))
                if hasattr(obj, "risk_metrics")
                else {}
            ),
            "status": "active",
            "timestamp": now,
        }

    def _from_dict_portfolio(self, d: Dict[str, Any], now: str) -> Dict[str, Any]:
        """Convert portfolio dictionary to standardized format"""
        return {
            "portfolio_value": float(
                d.get("total_value", d.get("portfolio_value", 10000.0))
            ),
            "cash_balance": float(d.get("cash_balance", 5000.0)),
            "unrealized_pnl": float(d.get("unrealized_pnl", 0.0)),
            "realized_pnl": float(d.get("realized_pnl", 0.0)),
            "holdings": dict(d.get("holdings", {})),
            "performance_metrics": dict(
                d.get("performance_metrics", {"total_return": 0.0})
            ),
            "sector_allocation": (
                dict(d.get("sector_allocation", {})) if "sector_allocation" in d else {}
            ),
            "risk_metrics": (
                dict(d.get("risk_metrics", {})) if "risk_metrics" in d else {}
            ),
            "status": "active",
            "timestamp": now,
        }

    def get_market_data_for_symbol(
        self,
        symbol: str,
        token: str = Depends(get_token_from_state),
    ) -> Dict[str, Any]:
        """Get market data for symbol with enhanced error handling"""
        try:
            agent = self.get_agent(token)
            if not agent or not hasattr(agent, "get_market_data"):
                return self._generate_fallback_market_data(symbol)
            return self._retrieve_and_validate_market_data(agent, symbol)
        except Exception as e:
            system_logger.error(
                f"Error getting market data: {str(e)}",
                additional_info={
                    "context": "Get Market Data",
                    "symbol": symbol,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return self._generate_fallback_market_data(symbol, error=str(e))

    def _retrieve_and_validate_market_data(self, agent, symbol: str) -> Dict[str, Any]:
        """Helper to retrieve and validate market data from agent"""
        try:
            market_data = agent.get_market_data(symbol)
            required_fields = [
                "symbol",
                "current_price",
                "change_24h",
                "volume",
                "rsi",
                "macd",
                "moving_avg_20",
                "moving_avg_50",
                "volatility",
                "market_sentiment",
            ]
            if isinstance(market_data, dict):
                for field in required_fields:
                    if field not in market_data:
                        market_data[field] = self._get_default_market_value(
                            field, symbol
                        )
                return market_data
            else:
                return self._market_obj_to_dict(market_data)
        except Exception as e:
            system_logger.warning(f"Agent market data retrieval failed: {e}")
            return self._generate_fallback_market_data(symbol, error=str(e))

    def _generate_fallback_market_data(
        self, symbol: str, error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate fallback market data"""
        # Generate pseudo-random but realistic market data based on symbol
        base_price = 100 + (hash(symbol) % 500)
        change = ((hash(symbol + "change") % 200) - 100) / 10  # -10% to +10%

        data = {
            "symbol": symbol.upper(),
            "current_price": round(base_price + change, 2),
            "change_24h": round(change, 2),
            "volume": (hash(symbol + "volume") % 10000000) + 100000,
            "rsi": max(20, min(80, 50 + (hash(symbol + "rsi") % 40) - 20)),
            "macd": round(((hash(symbol + "macd") % 200) - 100) / 100, 3),
            "moving_avg_20": round(base_price * 0.98, 2),
            "moving_avg_50": round(base_price * 0.95, 2),
            "volatility": round(0.1 + (hash(symbol + "vol") % 30) / 100, 3),
            "market_sentiment": ["bullish", "bearish", "neutral"][hash(symbol) % 3],
            "timestamp": datetime.now().isoformat(),
            "data_source": "fallback_generator",
        }

        if error:
            data["error_note"] = f"Using fallback data due to: {error}"

        return data

    def _get_default_market_value(self, field: str, symbol: str):
        """Get default value for missing market data field"""
        defaults = {
            "symbol": symbol.upper(),
            "current_price": 150.0,
            "change_24h": 2.5,
            "volume": 1000000,
            "rsi": 50.0,
            "macd": 0.0,
            "moving_avg_20": 148.0,
            "moving_avg_50": 145.0,
            "volatility": 0.2,
            "market_sentiment": "neutral",
        }
        return defaults.get(field, "unknown")

    def _market_obj_to_dict(self, market_obj) -> Dict[str, Any]:
        """Convert market data object to dictionary"""
        if hasattr(market_obj, "__dict__"):
            return {
                "symbol": getattr(market_obj, "symbol", "UNKNOWN"),
                "current_price": float(getattr(market_obj, "current_price", 150.0)),
                "change_24h": float(getattr(market_obj, "change_24h", 2.5)),
                "volume": int(getattr(market_obj, "volume", 1000000)),
                "rsi": float(getattr(market_obj, "rsi", 50.0)),
                "macd": float(getattr(market_obj, "macd", 0.0)),
                "moving_avg_20": float(getattr(market_obj, "moving_avg_20", 148.0)),
                "moving_avg_50": float(getattr(market_obj, "moving_avg_50", 145.0)),
                "volatility": float(getattr(market_obj, "volatility", 0.2)),
                "market_sentiment": str(
                    getattr(market_obj, "market_sentiment", "neutral")
                ),
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return self._generate_fallback_market_data(
                getattr(market_obj, "symbol", "UNKNOWN")
            )

    def get_performance_report_data(
        self, token: str = Depends(get_token_from_state)
    ) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            agent = self.get_agent(token)
            if not agent:
                return self._generate_fallback_performance_report()

            if hasattr(agent, "get_performance_report"):
                try:
                    report = agent.get_performance_report()
                    return self._enhance_performance_report(report)
                except Exception as e:
                    system_logger.warning(f"Agent performance report failed: {e}")
                    return self._generate_fallback_performance_report(error=str(e))
            else:
                return self._generate_fallback_performance_report()

        except Exception as e:
            system_logger.error(
                f"Error generating performance report: {str(e)}",
                additional_info={
                    "context": "Get Performance Report",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return self._generate_fallback_performance_report(error=str(e))

    def _generate_fallback_performance_report(
        self, error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate fallback performance report"""
        report = {
            "portfolio_value": 10000.0,
            "total_return": 2.5,
            "daily_return": 0.1,
            "weekly_return": 1.2,
            "monthly_return": 2.5,
            "trade_statistics": {
                "total_trades": 45,
                "winning_trades": 28,
                "losing_trades": 17,
                "win_rate": 62.2,
                "average_win": 125.50,
                "average_loss": -75.25,
            },
            "recent_trades": [
                {
                    "symbol": "AAPL",
                    "action": "buy",
                    "quantity": 10,
                    "price": 150.25,
                    "timestamp": datetime.now().isoformat(),
                    "pnl": 125.0,
                }
            ],
            "risk_metrics": {
                "max_drawdown": -3.2,
                "sharpe_ratio": 1.45,
                "volatility": 12.8,
                "beta": 1.1,
            },
            "status": "fallback_mode",
            "timestamp": datetime.now().isoformat(),
        }

        if error:
            report["error_note"] = f"Using fallback data due to: {error}"

        return report

    def _enhance_performance_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance performance report with additional metrics"""
        enhanced = report.copy()
        enhanced.update(
            {
                "timestamp": datetime.now().isoformat(),
                "report_type": "enhanced",
                "data_source": "agent",
            }
        )
        return enhanced


# Create router
router = APIRouter(tags=["Financial Trading Bot"])


# Dependency to get financial trading bot service
def get_trading_service() -> FinancialTradingBotService:
    """Get financial trading bot service instance"""
    from src.que_agents.api.main import agent_manager

    return FinancialTradingBotService(agent_manager)


# Financial Trading Bot endpoints
@router.post("/analyze", response_model=TradingDecisionResponse)
async def analyze_and_decide(
    request: TradingAnalysisRequest,
    service: FinancialTradingBotService = Depends(get_trading_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Analyze Market & Make Trading Decision**

    Perform comprehensive market analysis and generate AI-powered trading recommendations with risk assessment.

    **Request Body:**
    ```json
    {
        "symbol": "AAPL",
        "strategy_type": "momentum"
    }
    ```

    **Response:**
    ```json
    {
        "action": "buy",
        "symbol": "AAPL",
        "quantity": 10,
        "confidence": 0.85,
        "reasoning": "Strong momentum indicators with RSI at 45, MACD showing bullish crossover, and positive earnings outlook",
        "risk_score": 0.35,
        "expected_return": 0.08,
        "entry_price": 150.25,
        "stop_loss": 142.50,
        "take_profit": 165.00,
        "analysis_timestamp": "2024-01-15T10:30:00Z"
    }
    ```

    **Supported Strategies:**
    - `momentum`: Trend-following based on price momentum
    - `mean_reversion`: Contrarian approach targeting price reversals
    - `breakout`: Trading on price breakouts from key levels
    - `scalping`: Short-term high-frequency trading
    - `swing`: Medium-term position trading
    - `value`: Fundamental analysis-based decisions

    **Trading Actions:**
    - `buy`: Open long position
    - `sell`: Open short position
    - `hold`: Maintain current position
    - `close`: Exit existing position

    **Analysis Components:**
    - 📈 **Technical Indicators**: RSI, MACD, Moving Averages, Bollinger Bands
    - 📊 **Market Sentiment**: News analysis, social sentiment, institutional flow
    - 📉 **Risk Metrics**: Volatility, correlation, drawdown analysis
    - 💰 **Fundamental Data**: Earnings, revenue, financial ratios
    - 🌍 **Market Context**: Sector performance, economic indicators

    **Features:**
    - 🤖 AI-powered decision making
    - 🎯 Multi-factor analysis integration
    - 🛡️ Risk-adjusted recommendations
    - 📈 Confidence scoring system
    - 📊 Expected return calculations
    - ⚡ Real-time market data analysis

    **Risk Management:**
    - Automated stop-loss suggestions
    - Take-profit target recommendations
    - Position sizing based on risk tolerance
    - Portfolio correlation analysis

    **Status Codes:**
    - `200`: Analysis completed successfully
    - `400`: Invalid symbol or strategy type
    - `503`: Trading bot service unavailable
    - `500`: Market data or analysis error
    """
    service.get_agent(token)
    return service.analyze_and_make_decision(request)


class TradingCycleRequest(BaseModel):
    symbols: Optional[List[str]] = None


@router.post("/cycle")
async def run_trading_cycle(
    request: Optional[TradingCycleRequest] = None,
    service: FinancialTradingBotService = Depends(get_trading_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Run Trading Cycle**

    Execute a complete trading cycle analyzing multiple symbols and making automated trading decisions.

    **Request Body (Optional):**
    ```json
    {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    }
    ```

    **Response:**
    ```json
    {
        "timestamp": "2024-01-15T10:30:00Z",
        "symbols_analyzed": 5,
        "trades_executed": 2,
        "decisions": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "confidence": 0.82,
                "executed": true,
                "reasoning": "Strong momentum with bullish technical indicators",
                "quantity": 10,
                "entry_price": 150.25
            },
            {
                "symbol": "GOOGL",
                "action": "hold",
                "confidence": 0.65,
                "executed": false,
                "reasoning": "Mixed signals, awaiting clearer trend direction"
            }
        ],
        "portfolio_status": {
            "total_value": 125000.0,
            "cash_balance": 45000.0,
            "positions_count": 8
        },
        "performance_summary": {
            "cycle_pnl": 1250.75,
            "success_rate": 0.75,
            "average_confidence": 0.73
        },
        "market_conditions": "bullish",
        "risk_assessment": "moderate",
        "next_cycle_time": "2024-01-15T11:30:00Z"
    }
    ```

    **Cycle Process:**
    1. 📈 **Market Scan**: Analyze all symbols in watchlist
    2. 🤖 **Decision Making**: Apply AI algorithms to each symbol
    3. 🛡️ **Risk Assessment**: Evaluate portfolio impact
    4. ⚡ **Execution**: Execute approved trades
    5. 📉 **Reporting**: Generate cycle summary

    **Default Symbols:**
    If no symbols provided, analyzes: AAPL, GOOGL, MSFT, TSLA, NVDA, AMZN, META

    **Features:**
    - 🔄 Automated multi-symbol analysis
    - 🎯 Intelligent trade execution
    - 🛡️ Portfolio risk management
    - 📈 Performance tracking
    - ⏰ Scheduled cycle execution
    - 📊 Market condition assessment

    **Execution Criteria:**
    - Minimum confidence threshold (default: 0.7)
    - Risk limits and position sizing
    - Portfolio correlation constraints
    - Market volatility considerations

    **Risk Controls:**
    - Maximum position size limits
    - Daily loss limits
    - Correlation-based diversification
    - Volatility-adjusted position sizing

    **Use Cases:**
    - Automated trading execution
    - Portfolio rebalancing
    - Systematic strategy implementation
    - Risk-managed trading operations

    **Status Codes:**
    - `200`: Trading cycle completed successfully
    - `400`: Invalid symbol list or parameters
    - `503`: Trading bot service unavailable
    - `500`: Execution or market data error
    """
    symbols = request.symbols if request else None
    return service.run_trading_cycle_operation(symbols)


@router.get("/portfolio")
async def get_portfolio_status(
    service: FinancialTradingBotService = Depends(get_trading_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Get Portfolio Status**

    Retrieve comprehensive portfolio information including holdings, performance metrics, and risk analysis.

    **Response:**
    ```json
    {
        "portfolio_value": 125000.0,
        "cash_balance": 45000.0,
        "unrealized_pnl": 2500.75,
        "realized_pnl": 1850.25,
        "holdings": {
            "AAPL": {
                "shares": 50.0,
                "value": 7512.50,
                "avg_price": 148.25,
                "current_price": 150.25,
                "unrealized_pnl": 100.0,
                "weight": 0.06
            },
            "GOOGL": {
                "shares": 25.0,
                "value": 6750.0,
                "avg_price": 268.00,
                "current_price": 270.00,
                "unrealized_pnl": 50.0,
                "weight": 0.054
            }
        },
        "performance_metrics": {
            "total_return": 0.045,
            "daily_return": 0.008,
            "weekly_return": 0.025,
            "monthly_return": 0.045,
            "sharpe_ratio": 1.35,
            "max_drawdown": -0.025,
            "volatility": 0.18,
            "beta": 1.12
        },
        "sector_allocation": {
            "Technology": 0.65,
            "Healthcare": 0.20,
            "Finance": 0.15
        },
        "risk_metrics": {
            "portfolio_beta": 1.12,
            "value_at_risk_95": -0.032,
            "expected_shortfall": -0.048,
            "volatility": 0.18,
            "correlation_risk": 0.25
        },
        "recent_trades": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 10.0,
                "price": 150.25,
                "timestamp": "2024-01-15T09:30:00Z",
                "pnl": 125.0
            }
        ],
        "last_updated": "2024-01-15T10:30:00Z"
    }
    ```

    **Portfolio Components:**
    - 💰 **Valuation**: Total value, cash balance, P&L
    - 📈 **Holdings**: Individual positions with details
    - 🎯 **Performance**: Returns, ratios, and benchmarks
    - 🏢 **Allocation**: Sector and asset distribution
    - 🛡️ **Risk**: VaR, volatility, correlation metrics
    - 📉 **Activity**: Recent trades and transactions

    **Performance Metrics:**
    - **Total Return**: Overall portfolio performance
    - **Sharpe Ratio**: Risk-adjusted return measure
    - **Max Drawdown**: Largest peak-to-trough decline
    - **Volatility**: Standard deviation of returns
    - **Beta**: Market sensitivity coefficient

    **Risk Metrics:**
    - **Value at Risk (VaR)**: Potential loss at 95% confidence
    - **Expected Shortfall**: Average loss beyond VaR
    - **Portfolio Beta**: Overall market exposure
    - **Correlation Risk**: Diversification effectiveness

    **Holdings Details:**
    - Current market value and position size
    - Average cost basis and current price
    - Unrealized profit/loss per position
    - Portfolio weight and allocation

    **Features:**
    - 📈 Real-time portfolio valuation
    - 🛡️ Comprehensive risk analysis
    - 🎯 Performance attribution
    - 🏢 Sector diversification tracking
    - 📉 Trade history and P&L
    - ⏰ Time-weighted returns

    **Use Cases:**
    - Portfolio management dashboards
    - Risk monitoring and reporting
    - Performance analysis and attribution
    - Regulatory compliance reporting

    **Status Codes:**
    - `200`: Portfolio data retrieved successfully
    - `500`: Portfolio service or data unavailable
    """
    return service.get_portfolio_status_data()


@router.get("/market/{symbol}")
async def get_market_data(
    symbol: str,
    service: FinancialTradingBotService = Depends(get_trading_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Get Market Data**

    Retrieve comprehensive real-time and historical market data for a specific trading symbol.

    **Path Parameters:**
    - `symbol` (str): Trading symbol (e.g., AAPL, GOOGL, MSFT)

    **Response:**
    ```json
    {
        "symbol": "AAPL",
        "current_price": 150.25,
        "change_24h": 2.15,
        "change_percent": 1.45,
        "volume": 45678900,
        "market_cap": 2450000000000,
        "technical_indicators": {
            "rsi": 58.5,
            "macd": 0.85,
            "macd_signal": 0.72,
            "moving_avg_20": 148.75,
            "moving_avg_50": 145.20,
            "moving_avg_200": 142.80,
            "bollinger_upper": 152.30,
            "bollinger_lower": 145.60
        },
        "price_levels": {
            "support_1": 147.50,
            "support_2": 145.00,
            "resistance_1": 152.00,
            "resistance_2": 155.50
        },
        "volatility": 0.22,
        "market_sentiment": "bullish",
        "analyst_ratings": {
            "buy": 15,
            "hold": 8,
            "sell": 2,
            "average_target": 165.00
        },
        "fundamental_data": {
            "pe_ratio": 28.5,
            "eps": 5.25,
            "dividend_yield": 0.52,
            "book_value": 4.15
        },
        "trading_session": {
            "open": 148.10,
            "high": 151.20,
            "low": 147.85,
            "previous_close": 148.10
        },
        "timestamp": "2024-01-15T10:30:00Z"
    }
    ```

    **Market Data Components:**
    - 📈 **Price Action**: Current price, changes, volume
    - 📊 **Technical Indicators**: RSI, MACD, Moving Averages
    - 🎯 **Support/Resistance**: Key price levels
    - 📉 **Volatility**: Price movement measurements
    - 📰 **Sentiment**: Market mood and analyst opinions
    - 📋 **Fundamentals**: Financial ratios and metrics

    **Technical Indicators:**
    - **RSI**: Relative Strength Index (0-100)
    - **MACD**: Moving Average Convergence Divergence
    - **Moving Averages**: 20, 50, 200-day trends
    - **Bollinger Bands**: Volatility-based price channels

    **Price Levels:**
    - **Support**: Price levels where buying interest emerges
    - **Resistance**: Price levels where selling pressure increases
    - **Breakout Levels**: Key technical thresholds

    **Fundamental Metrics:**
    - **P/E Ratio**: Price-to-earnings valuation
    - **EPS**: Earnings per share
    - **Dividend Yield**: Annual dividend percentage
    - **Book Value**: Net asset value per share

    **Features:**
    - ⚡ Real-time price updates
    - 📈 Comprehensive technical analysis
    - 📊 Multi-timeframe indicators
    - 🎯 Key level identification
    - 📉 Volatility measurements
    - 📰 Sentiment analysis integration

    **Use Cases:**
    - Trading decision support
    - Technical analysis workflows
    - Market research and screening
    - Risk assessment and monitoring

    **Supported Symbols:**
    - US Stocks (NYSE, NASDAQ)
    - Major ETFs and indices
    - Popular cryptocurrency pairs
    - Forex major pairs

    **Status Codes:**
    - `200`: Market data retrieved successfully
    - `400`: Invalid or unsupported symbol
    - `404`: Symbol not found
    - `500`: Market data service unavailable
    """
    return service.get_market_data_for_symbol(symbol)


@router.get("/performance")
async def get_performance_report(
    service: FinancialTradingBotService = Depends(get_trading_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Get Performance Report**

    Generate comprehensive trading performance analysis with detailed metrics, statistics, and insights.

    **Response:**
    ```json
    {
        "portfolio_value": 125000.0,
        "total_return": 0.045,
        "daily_return": 0.008,
        "weekly_return": 0.025,
        "monthly_return": 0.045,
        "ytd_return": 0.045,
        "trade_statistics": {
            "total_trades": 156,
            "winning_trades": 98,
            "losing_trades": 58,
            "win_rate": 0.628,
            "average_win": 245.75,
            "average_loss": -125.50,
            "largest_win": 1250.00,
            "largest_loss": -485.25,
            "profit_factor": 1.92
        },
        "performance_metrics": {
            "sharpe_ratio": 1.35,
            "sortino_ratio": 1.68,
            "calmar_ratio": 1.25,
            "max_drawdown": -0.025,
            "volatility": 0.18,
            "beta": 1.12,
            "alpha": 0.015,
            "information_ratio": 0.85
        },
        "risk_metrics": {
            "value_at_risk_95": -0.032,
            "expected_shortfall": -0.048,
            "downside_deviation": 0.12,
            "maximum_consecutive_losses": 4,
            "recovery_factor": 1.8
        },
        "monthly_returns": [
            {"month": "2024-01", "return": 0.045, "trades": 23},
            {"month": "2023-12", "return": 0.032, "trades": 28}
        ],
        "recent_trades": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 10,
                "entry_price": 148.25,
                "exit_price": 150.75,
                "pnl": 250.0,
                "hold_time": "2 days",
                "timestamp": "2024-01-15T09:30:00Z"
            }
        ],
        "sector_performance": {
            "Technology": 0.052,
            "Healthcare": 0.038,
            "Finance": 0.041
        },
        "benchmark_comparison": {
            "sp500_return": 0.035,
            "outperformance": 0.010,
            "correlation": 0.78
        },
        "report_period": {
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-15T10:30:00Z",
            "days": 15
        },
        "timestamp": "2024-01-15T10:30:00Z"
    }
    ```

    **Performance Categories:**
    - 📈 **Returns**: Total, daily, weekly, monthly performance
    - 🎯 **Trade Stats**: Win rate, profit factor, trade analysis
    - 🛡️ **Risk Metrics**: Drawdown, volatility, VaR analysis
    - 📉 **Ratios**: Sharpe, Sortino, Calmar ratios
    - 🏢 **Sector Analysis**: Performance by sector
    - 📋 **Benchmarking**: Comparison to market indices

    **Key Performance Ratios:**
    - **Sharpe Ratio**: Risk-adjusted return measure
    - **Sortino Ratio**: Downside risk-adjusted returns
    - **Calmar Ratio**: Return vs maximum drawdown
    - **Information Ratio**: Active return vs tracking error
    - **Profit Factor**: Gross profit vs gross loss

    **Risk Analysis:**
    - **Maximum Drawdown**: Largest peak-to-trough decline
    - **Value at Risk**: Potential loss at confidence level
    - **Expected Shortfall**: Average loss beyond VaR
    - **Volatility**: Standard deviation of returns
    - **Beta**: Market sensitivity measure

    **Trade Analytics:**
    - Win/loss ratios and statistics
    - Average profit and loss per trade
    - Trade frequency and holding periods
    - Consecutive win/loss streaks
    - Profit factor and expectancy

    **Features:**
    - 📈 Comprehensive performance analysis
    - 🛡️ Advanced risk metrics
    - 🎯 Benchmark comparisons
    - 📅 Time-series performance tracking
    - 🏢 Sector attribution analysis
    - 📉 Trade-level statistics

    **Use Cases:**
    - Performance review and analysis
    - Risk management reporting
    - Strategy evaluation and optimization
    - Investor reporting and compliance

    **Status Codes:**
    - `200`: Performance report generated successfully
    - `500`: Performance analysis service unavailable
    """
    return service.get_performance_report_data()


@router.get("/holdings")
async def get_current_holdings(
    service: FinancialTradingBotService = Depends(get_trading_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Get Current Holdings**

    Retrieve detailed information about all current portfolio positions and cash balances.

    **Response:**
    ```json
    {
        "holdings": {
            "AAPL": {
                "shares": 50.0,
                "value": 7512.50,
                "avg_price": 148.25,
                "current_price": 150.25,
                "unrealized_pnl": 100.0,
                "weight": 0.06,
                "sector": "Technology",
                "purchase_date": "2024-01-10T09:30:00Z"
            },
            "GOOGL": {
                "shares": 25.0,
                "value": 6750.0,
                "avg_price": 268.00,
                "current_price": 270.00,
                "unrealized_pnl": 50.0,
                "weight": 0.054,
                "sector": "Technology",
                "purchase_date": "2024-01-12T14:15:00Z"
            }
        },
        "summary": {
            "total_positions": 8,
            "total_value": 125000.0,
            "cash_balance": 45000.0,
            "invested_amount": 80000.0,
            "unrealized_pnl": 2500.75,
            "total_return": 0.045
        },
        "allocation": {
            "by_sector": {
                "Technology": 0.65,
                "Healthcare": 0.20,
                "Finance": 0.15
            },
            "by_asset_class": {
                "Stocks": 0.85,
                "ETFs": 0.10,
                "Cash": 0.05
            }
        },
        "timestamp": "2024-01-15T10:30:00Z"
    }
    ```

    **Holdings Details:**
    - 📈 **Position Size**: Number of shares and market value
    - 💰 **Cost Basis**: Average purchase price and total cost
    - 📉 **Current Value**: Real-time market value and P&L
    - 🎯 **Portfolio Weight**: Percentage of total portfolio
    - 🏢 **Sector**: Industry classification
    - 📅 **Purchase Date**: When position was initiated

    **Portfolio Summary:**
    - Total number of positions held
    - Combined market value of all holdings
    - Available cash balance
    - Total invested capital
    - Unrealized profit/loss across all positions
    - Overall portfolio return percentage

    **Allocation Breakdown:**
    - **By Sector**: Technology, Healthcare, Finance, etc.
    - **By Asset Class**: Stocks, ETFs, Bonds, Cash
    - **By Geography**: Domestic vs International exposure
    - **By Market Cap**: Large, Mid, Small cap distribution

    **Features:**
    - ⚡ Real-time position valuation
    - 📈 Unrealized P&L calculations
    - 🎯 Portfolio weight analysis
    - 🏢 Sector diversification tracking
    - 💰 Cash management monitoring
    - 📅 Position aging analysis

    **Key Metrics:**
    - Position concentration risk
    - Sector allocation balance
    - Cash utilization efficiency
    - Unrealized gains/losses
    - Portfolio diversification score

    **Use Cases:**
    - Portfolio monitoring dashboards
    - Position management interfaces
    - Risk assessment and rebalancing
    - Tax planning and harvesting
    - Performance attribution analysis

    **Status Codes:**
    - `200`: Holdings retrieved successfully
    - `500`: Portfolio service unavailable
    """
    portfolio_data = service.get_portfolio_status_data()
    return {
        "holdings": portfolio_data.get("holdings", {}),
        "total_value": portfolio_data.get("portfolio_value", 0.0),
        "cash_balance": portfolio_data.get("cash_balance", 0.0),
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/risk-assessment")
async def get_risk_assessment(
    service: FinancialTradingBotService = Depends(get_trading_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Get Risk Assessment**

    Comprehensive portfolio risk analysis with metrics, warnings, and recommendations.

    **Response:**
    ```json
    {
        "overall_risk": "moderate",
        "risk_score": 0.45,
        "portfolio_beta": 1.12,
        "volatility": 0.18,
        "max_drawdown": -0.025,
        "value_at_risk": {
            "95_percent": -0.032,
            "99_percent": -0.048,
            "time_horizon": "1_day"
        },
        "risk_factors": {
            "concentration_risk": {
                "level": "medium",
                "description": "Top 3 positions represent 45% of portfolio",
                "recommendation": "Consider diversifying large positions"
            },
            "sector_concentration": {
                "level": "high",
                "description": "65% allocation to Technology sector",
                "recommendation": "Reduce technology exposure"
            },
            "correlation_risk": {
                "level": "medium",
                "description": "High correlation between major holdings",
                "recommendation": "Add uncorrelated assets"
            }
        },
        "stress_test_results": {
            "market_crash_scenario": {
                "portfolio_impact": -0.15,
                "worst_performers": ["AAPL", "GOOGL", "MSFT"]
            },
            "sector_rotation": {
                "portfolio_impact": -0.08,
                "affected_positions": ["Technology stocks"]
            }
        },
        "recommendations": [
            "Reduce position size in AAPL to below 10%",
            "Add defensive sectors (Utilities, Consumer Staples)",
            "Consider hedging with put options",
            "Increase cash allocation to 10-15%"
        ],
        "risk_limits": {
            "position_limit": 0.10,
            "sector_limit": 0.30,
            "daily_var_limit": 0.02,
            "max_drawdown_limit": 0.05
        },
        "compliance_status": {
            "position_limits": "compliant",
            "sector_limits": "violation",
            "risk_limits": "compliant"
        },
        "timestamp": "2024-01-15T10:30:00Z"
    }
    ```

    **Risk Categories:**
    - 🛡️ **Market Risk**: Beta, volatility, correlation exposure
    - 🎯 **Concentration Risk**: Position and sector concentration
    - 📉 **Drawdown Risk**: Maximum loss potential
    - 🔄 **Liquidity Risk**: Asset liquidity and trading volume
    - 🌍 **Systematic Risk**: Market-wide risk factors

    **Risk Levels:**
    - `low`: Minimal risk exposure (0.0-0.3)
    - `moderate`: Balanced risk profile (0.3-0.6)
    - `high`: Elevated risk levels (0.6-0.8)
    - `extreme`: Very high risk exposure (0.8-1.0)

    **Key Risk Metrics:**
    - **Portfolio Beta**: Market sensitivity (1.0 = market average)
    - **Volatility**: Standard deviation of returns
    - **Value at Risk**: Potential loss at confidence levels
    - **Maximum Drawdown**: Largest peak-to-trough decline
    - **Correlation**: Inter-asset relationship strength

    **Stress Testing:**
    - Market crash scenarios (-20%, -30%, -40%)
    - Sector rotation impacts
    - Interest rate shock effects
    - Currency fluctuation impacts
    - Liquidity crisis simulations

    **Risk Factors Analysis:**
    - Position concentration warnings
    - Sector allocation imbalances
    - Geographic concentration risks
    - Asset class diversification gaps
    - Correlation clustering issues

    **Features:**
    - 📈 Multi-factor risk analysis
    - 🛡️ Stress testing scenarios
    - ⚠️ Risk limit monitoring
    - 💡 Actionable recommendations
    - 📉 Historical risk tracking
    - 🎯 Compliance monitoring

    **Use Cases:**
    - Daily risk monitoring
    - Portfolio rebalancing decisions
    - Regulatory compliance reporting
    - Investment committee presentations

    **Status Codes:**
    - `200`: Risk assessment completed successfully
    - `500`: Risk analysis service unavailable
    """
    try:
        agent = service.get_agent(token)
        if not agent:
            return {
                "overall_risk": "moderate",
                "risk_score": 0.5,
                "recommendations": ["Monitor market conditions", "Diversify portfolio"],
                "status": "agent_unavailable",
                "timestamp": datetime.now().isoformat(),
            }

        portfolio_data = service.get_portfolio_status_data()
        risk_metrics = portfolio_data.get("risk_metrics", {})

        return {
            "overall_risk": "moderate",
            "risk_score": risk_metrics.get("value_at_risk_95", 0.5),
            "portfolio_beta": risk_metrics.get("portfolio_beta", 1.0),
            "volatility": risk_metrics.get("volatility", 0.15),
            "max_drawdown": portfolio_data.get("performance_metrics", {}).get(
                "max_drawdown", -2.5
            ),
            "recommendations": [
                "Maintain diversified portfolio",
                "Monitor volatility levels",
                "Review position sizes regularly",
            ],
            "status": "active",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        system_logger.error(
            f"Error in risk assessment: {str(e)}",
            additional_info={
                "context": "Get Risk Assessment",
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        return {
            "overall_risk": "unknown",
            "risk_score": 0.5,
            "error": "Risk assessment temporarily unavailable",
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/watchlist")
async def get_watchlist(
    service: FinancialTradingBotService = Depends(get_trading_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Get Trading Watchlist**

    Retrieve the curated list of symbols being monitored for trading opportunities.

    **Response:**
    ```json
    {
        "symbols": [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "sector": "Technology",
                "current_price": 150.25,
                "change_24h": 2.15,
                "change_percent": 1.45,
                "volume": 45678900,
                "market_cap": 2450000000000,
                "watchlist_reason": "Strong momentum and earnings growth",
                "added_date": "2024-01-10T09:00:00Z",
                "priority": "high",
                "alerts": [
                    {
                        "type": "price_target",
                        "condition": "above",
                        "value": 155.00,
                        "status": "active"
                    }
                ]
            },
            {
                "symbol": "GOOGL",
                "name": "Alphabet Inc.",
                "sector": "Technology",
                "current_price": 270.00,
                "change_24h": -1.50,
                "change_percent": -0.55,
                "volume": 28456700,
                "market_cap": 1750000000000,
                "watchlist_reason": "AI developments and cloud growth",
                "added_date": "2024-01-08T14:30:00Z",
                "priority": "medium",
                "alerts": []
            }
        ],
        "summary": {
            "total_symbols": 12,
            "high_priority": 4,
            "medium_priority": 6,
            "low_priority": 2,
            "active_alerts": 8
        },
        "sector_breakdown": {
            "Technology": 5,
            "Healthcare": 3,
            "Finance": 2,
            "Consumer": 2
        },
        "performance_summary": {
            "top_performer": {
                "symbol": "NVDA",
                "change_percent": 5.25
            },
            "worst_performer": {
                "symbol": "META",
                "change_percent": -2.15
            },
            "average_change": 0.85
        },
        "last_updated": "2024-01-15T10:30:00Z"
    }
    ```

    **Watchlist Features:**
    - 👁️ **Symbol Monitoring**: Real-time price and volume tracking
    - 📊 **Performance Tracking**: Daily changes and trends
    - ⚠️ **Alert System**: Price targets and technical alerts
    - 🏷️ **Priority Levels**: High, medium, low priority classification
    - 📅 **Historical Tracking**: Addition dates and reasons
    - 🏢 **Sector Analysis**: Industry diversification

    **Symbol Information:**
    - Company name and sector classification
    - Real-time price and volume data
    - Daily price changes and percentages
    - Market capitalization
    - Reason for watchlist inclusion
    - Priority level and alert status

    **Alert Types:**
    - **Price Targets**: Above/below specific prices
    - **Volume Spikes**: Unusual trading activity
    - **Technical Breakouts**: Chart pattern alerts
    - **Earnings Events**: Upcoming announcements
    - **News Catalysts**: Significant news events

    **Priority Levels:**
    - `high`: Immediate trading opportunities
    - `medium`: Potential opportunities to monitor
    - `low`: Long-term prospects and research

    **Watchlist Management:**
    - Dynamic symbol addition/removal
    - Priority level adjustments
    - Alert configuration and management
    - Performance-based ranking
    - Sector diversification tracking

    **Features:**
    - ⚡ Real-time market data integration
    - 📊 Performance ranking and sorting
    - ⚠️ Customizable alert system
    - 🏢 Sector diversification analysis
    - 📈 Historical performance tracking
    - 🎯 Opportunity scoring system

    **Use Cases:**
    - Trading opportunity identification
    - Market screening and research
    - Alert-based trading strategies
    - Portfolio diversification planning

    **Status Codes:**
    - `200`: Watchlist retrieved successfully
    - `500`: Watchlist service unavailable
    """
    try:
        agent = service.get_agent(token)
        if not agent:
            # Return default watchlist
            default_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
            return {
                "symbols": default_symbols,
                "count": len(default_symbols),
                "last_updated": datetime.now().isoformat(),
                "status": "fallback_data",
            }

        if hasattr(agent, "get_watchlist"):
            watchlist = agent.get_watchlist()
            return watchlist
        else:
            # Fallback watchlist
            default_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META"]
            return {
                "symbols": default_symbols,
                "count": len(default_symbols),
                "last_updated": datetime.now().isoformat(),
                "status": "default_watchlist",
            }

    except Exception as e:
        system_logger.error(
            f"Error getting watchlist: {str(e)}",
            additional_info={
                "context": "Get Watchlist",
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        return {
            "symbols": [],
            "count": 0,
            "error": "Watchlist temporarily unavailable",
            "timestamp": datetime.now().isoformat(),
        }
