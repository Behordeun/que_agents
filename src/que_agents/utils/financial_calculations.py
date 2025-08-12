# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-10
# @Description: Trading utilities and calculations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.que_agents.error_trace.errorlogger import system_logger


class TradingCalculations:
    """Utility class for trading-related calculations"""

    @staticmethod
    def calculate_portfolio_metrics(
        portfolio_value: float,
        cash_balance: float,
        holdings: Dict[str, Any],
        historical_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        try:
            total_invested = portfolio_value - cash_balance
            cash_percentage = (
                (cash_balance / portfolio_value) * 100 if portfolio_value > 0 else 0
            )

            metrics = {
                "total_value": portfolio_value,
                "cash_balance": cash_balance,
                "invested_amount": total_invested,
                "cash_percentage": round(cash_percentage, 2),
                "invested_percentage": round(100 - cash_percentage, 2),
                "number_of_positions": len(holdings),
            }

            # Calculate additional metrics if historical data is available
            if historical_values and len(historical_values) > 1:
                returns = TradingCalculations.calculate_returns(historical_values)
                metrics.update(
                    {
                        "volatility": TradingCalculations.calculate_volatility(returns),
                        "sharpe_ratio": TradingCalculations.calculate_sharpe_ratio(
                            returns
                        ),
                        "max_drawdown": TradingCalculations.calculate_max_drawdown(
                            historical_values
                        ),
                    }
                )

            return metrics

        except Exception as e:
            system_logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {
                "total_value": portfolio_value,
                "cash_balance": cash_balance,
                "error": "Metrics calculation failed",
            }

    @staticmethod
    def calculate_returns(prices: List[float]) -> List[float]:
        """Calculate returns from price series"""
        if len(prices) < 2:
            return []

        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] != 0:
                ret = (prices[i] - prices[i - 1]) / prices[i - 1]
                returns.append(ret)

        return returns

    @staticmethod
    def calculate_volatility(returns: List[float], annualized: bool = True) -> float:
        """Calculate volatility from returns"""
        if len(returns) < 2:
            return 0.0

        try:
            returns_array = np.array(returns)
            volatility = np.std(returns_array)

            if annualized:
                # Assuming daily returns, annualize by multiplying by sqrt(252)
                volatility *= math.sqrt(252)

            return round(volatility * 100, 2)  # Return as percentage
        except Exception:
            return 0.0

    @staticmethod
    def calculate_sharpe_ratio(
        returns: List[float], risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0

        try:
            returns_array = np.array(returns)
            avg_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            if std_return == 0:
                return 0.0

            # Annualize the average return
            annualized_return = avg_return * 252
            annualized_std = std_return * math.sqrt(252)

            sharpe = (annualized_return - risk_free_rate) / annualized_std
            return round(sharpe, 2)
        except Exception:
            return 0.0

    @staticmethod
    def calculate_max_drawdown(prices: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(prices) < 2:
            return 0.0

        try:
            peak = prices[0]
            max_drawdown = 0.0

            for price in prices:
                if price > peak:
                    peak = price

                drawdown = (peak - price) / peak if peak != 0 else 0
                max_drawdown = max(max_drawdown, drawdown)

            return round(max_drawdown * 100, 2)  # Return as percentage
        except Exception:
            return 0.0

    @staticmethod
    def calculate_position_size(
        portfolio_value: float,
        risk_percentage: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> int:
        """Calculate position size based on risk management"""
        try:
            if stop_loss_price >= entry_price or entry_price <= 0:
                return 0

            risk_amount = portfolio_value * (risk_percentage / 100)
            risk_per_share = entry_price - stop_loss_price

            if risk_per_share <= 0:
                return 0

            position_size = int(risk_amount / risk_per_share)
            return max(0, position_size)
        except Exception:
            return 0

    @staticmethod
    def calculate_risk_reward_ratio(
        entry_price: float, stop_loss_price: float, target_price: float
    ) -> Optional[float]:
        """Calculate risk-reward ratio"""
        try:
            if (
                entry_price <= 0
                or stop_loss_price >= entry_price
                or target_price <= entry_price
            ):
                return None

            risk = entry_price - stop_loss_price
            reward = target_price - entry_price

            if risk <= 0:
                return None

            ratio = reward / risk
            return round(ratio, 2)
        except Exception:
            return None

    @staticmethod
    def calculate_technical_indicators(
        prices: List[float], volume: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Calculate basic technical indicators"""
        try:
            if len(prices) < 20:
                return {"error": "Insufficient data for technical analysis"}

            indicators = {}

            # Simple Moving Averages
            if len(prices) >= 20:
                indicators["sma_20"] = round(sum(prices[-20:]) / 20, 2)

            if len(prices) >= 50:
                indicators["sma_50"] = round(sum(prices[-50:]) / 50, 2)

            # RSI calculation (simplified)
            if len(prices) >= 14:
                indicators["rsi"] = TradingCalculations.calculate_rsi(prices)

            # Price change indicators
            current_price = prices[-1]
            prev_price = prices[-2] if len(prices) >= 2 else current_price

            indicators.update(
                {
                    "current_price": current_price,
                    "price_change": round(current_price - prev_price, 2),
                    "price_change_percent": (
                        round(((current_price - prev_price) / prev_price * 100), 2)
                        if prev_price != 0
                        else 0
                    ),
                }
            )

            return indicators

        except Exception as e:
            system_logger.error(f"Error calculating technical indicators: {str(e)}")
            return {"error": "Technical analysis calculation failed"}

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI

            # Calculate price changes
            changes = []
            for i in range(1, len(prices)):
                changes.append(prices[i] - prices[i - 1])

            # Separate gains and losses
            gains = [max(0, change) for change in changes]
            losses = [abs(min(0, change)) for change in changes]

            # Calculate average gains and losses
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return round(rsi, 2)
        except Exception:
            return 50.0  # Return neutral RSI on error


class RiskManagement:
    """Risk management utilities for trading"""

    @staticmethod
    def assess_portfolio_risk(
        portfolio_value: float,
        positions: Dict[str, Any],
        market_conditions: str = "neutral",
    ) -> Dict[str, Any]:
        """Assess overall portfolio risk"""
        try:
            # Validate input types
            if not isinstance(portfolio_value, (int, float)):
                raise ValueError("Invalid portfolio_value type")

            position_values, total_position_value = (
                RiskManagement._extract_position_values(positions)
            )
            num_positions = len(positions)
            largest_position = max(position_values) if position_values else 0
            concentration_ratio = (
                (largest_position / total_position_value) * 100
                if total_position_value > 0
                else 0
            )

            risk_level, risk_factors = RiskManagement._determine_risk_level(
                concentration_ratio, num_positions, market_conditions
            )

            return {
                "risk_level": risk_level,
                "risk_score": RiskManagement._calculate_risk_score(risk_level),
                "concentration_ratio": round(concentration_ratio, 2),
                "num_positions": num_positions,
                "risk_factors": risk_factors,
                "recommendations": RiskManagement._get_risk_recommendations(
                    risk_level, risk_factors
                ),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            system_logger.error(f"Error assessing portfolio risk: {str(e)}")
            return {
                "risk_level": "unknown",
                "risk_score": 0.5,
                "error": "Risk assessment failed",
                "timestamp": datetime.now().isoformat(),
            }

    @staticmethod
    def _extract_position_values(
        positions: Dict[str, Any],
    ) -> Tuple[List[float], float]:
        """Extract position values and calculate total"""
        position_values = []
        for position in positions.values():
            if isinstance(position, dict) and "value" in position:
                position_values.append(position["value"])
            elif isinstance(position, (int, float)):
                position_values.append(position)
        total_position_value = sum(position_values)
        return position_values, total_position_value

    @staticmethod
    def _determine_risk_level(
        concentration_ratio: float, num_positions: int, market_conditions: str
    ) -> Tuple[str, List[str]]:
        """Determine risk level and collect risk factors"""
        risk_level = "low"
        risk_factors = []

        if concentration_ratio > 30:
            risk_level = "high"
            risk_factors.append("High concentration in single position")
        elif concentration_ratio > 20:
            risk_level = "medium"
            risk_factors.append("Moderate concentration risk")

        if num_positions < 5:
            if risk_level == "low":
                risk_level = "medium"
            risk_factors.append("Limited diversification")

        if market_conditions == "volatile":
            if risk_level == "low":
                risk_level = "medium"
            elif risk_level == "medium":
                risk_level = "high"
            risk_factors.append("Volatile market conditions")

        return risk_level, risk_factors

    @staticmethod
    def _calculate_risk_score(risk_level: str) -> float:
        """Convert risk level to numeric score"""
        risk_scores = {"low": 0.2, "medium": 0.5, "high": 0.8, "unknown": 0.5}
        return risk_scores.get(risk_level, 0.5)

    @staticmethod
    def _get_risk_recommendations(
        risk_level: str, risk_factors: List[str]
    ) -> List[str]:
        """Get risk management recommendations"""
        recommendations = []

        if risk_level == "high":
            recommendations.extend(
                [
                    "Consider reducing position sizes",
                    "Diversify across more assets",
                    "Implement stop-loss orders",
                ]
            )
        elif risk_level == "medium":
            recommendations.extend(
                ["Monitor positions closely", "Consider additional diversification"]
            )
        else:
            recommendations.append("Continue current risk management approach")

        if "High concentration" in str(risk_factors):
            recommendations.append("Reduce concentration in largest positions")

        if "Limited diversification" in str(risk_factors):
            recommendations.append("Add positions in different sectors/asset classes")

        if "Volatile market" in str(risk_factors):
            recommendations.append(
                "Consider reducing overall position sizes during volatility"
            )

        return recommendations[:5]  # Limit to 5 recommendations
