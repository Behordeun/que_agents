# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-10
# @Description: Financial Trading Bot API routes and handlers

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from src.que_agents.core.schemas import TradingAnalysisRequest, TradingDecisionResponse
from src.que_agents.error_trace.errorlogger import system_logger
from src.que_agents.utils.agent_manager import AgentManager
from src.que_agents.utils.auth import get_verified_token


class FinancialTradingBotService:
    """Service class for financial trading bot operations"""

    FINANCIAL_TRADING_BOT_CONTEXT = "Financial Trading Bot"

    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        self.TRADING_BOT_UNAVAILABLE = "Financial Trading Bot not available"

    def get_agent(self, token: str):
        """Get financial trading bot agent"""
        agent = getattr(self.agent_manager, "financial_trading_bot", None)
        if agent is None:
            system_logger.error(
                "Financial trading bot is not available in AgentManager.",
                additional_info={"context": self.FINANCIAL_TRADING_BOT_CONTEXT},
            )
        # If the agent needs to use the token, you would pass it here, e.g.:
        # agent.set_token(token)
        return agent


    def analyze_and_make_decision(
        self,
        request: TradingAnalysisRequest,
        token: str = Depends(get_verified_token),
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
        self, symbols: Optional[List[str]] = None,
        token: str = Depends(get_verified_token),
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

    def get_portfolio_status_data(self, token: str = Depends(get_verified_token)) -> Dict[str, Any]:
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
                        exc_info=True
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
        token: str = Depends(get_verified_token),
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
                exc_info=True
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
            return self._generate_fallback_market_data(symbol)

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
            self,
            token: str = Depends(get_verified_token)
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
                    return self._generate_fallback_performance_report()
            else:
                return self._generate_fallback_performance_report()

        except Exception as e:
            system_logger.error(
                f"Error generating performance report: {str(e)}",
                additional_info={
                    "context": "Get Performance Report",
                    "error_type": type(e).__name__,
                },
                exc_info=True
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
def get_trading_service(
    agent_manager: AgentManager = Depends(),
) -> FinancialTradingBotService:
    """Get financial trading bot service instance"""
    return FinancialTradingBotService(agent_manager)


# Financial Trading Bot endpoints
@router.post("/analyze", response_model=TradingDecisionResponse)
async def analyze_and_decide(
    request: TradingAnalysisRequest,
    service: FinancialTradingBotService = Depends(get_trading_service),
    token: str = Depends(get_verified_token),
):
    """Analyze market and make trading decision"""
    service.get_agent(token)
    return service.analyze_and_make_decision(request)


@router.post("/cycle")
async def run_trading_cycle(
    symbols: Optional[List[str]] = None,
    service: FinancialTradingBotService = Depends(get_trading_service),
    token: str = Depends(get_verified_token),
):
    """Run trading cycle with optional symbol list"""
    return service.run_trading_cycle_operation(symbols)


@router.get("/portfolio")
async def get_portfolio_status(
    service: FinancialTradingBotService = Depends(get_trading_service),
    token: str = Depends(get_verified_token),
):
    """Get comprehensive portfolio status"""
    return service.get_portfolio_status_data()


@router.get("/market/{symbol}")
async def get_market_data(
    symbol: str,
    service: FinancialTradingBotService = Depends(get_trading_service),
    token: str = Depends(get_verified_token),
):
    """Get market data for specific symbol"""
    return service.get_market_data_for_symbol(symbol)


@router.get("/performance")
async def get_performance_report(
    service: FinancialTradingBotService = Depends(get_trading_service),
    token: str = Depends(get_verified_token),
):
    """Get comprehensive performance report"""
    return service.get_performance_report_data()


@router.get("/holdings")
async def get_current_holdings(
    service: FinancialTradingBotService = Depends(get_trading_service),
    token: str = Depends(get_verified_token),
):
    """Get current portfolio holdings"""
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
    token: str = Depends(get_verified_token),
):
    """Get current risk assessment"""
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
            exc_info=True
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
    token: str = Depends(get_verified_token),
):
    """Get trading bot watchlist"""
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
            exc_info=True
        )
        return {
            "symbols": [],
            "count": 0,
            "error": "Watchlist temporarily unavailable",
            "timestamp": datetime.now().isoformat(),
        }
