# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-10
# @Description: Router package initialization

"""
API Routers Package

This package contains all the modular API routers for the Agentic AI system:

- customer_support: Customer Support Agent routes
- financial_trading_bot: Financial Trading Bot routes
- marketing_agent: Marketing Agent routes
- personal_virtual_assistant: Personal Virtual Assistant routes

Each router is self-contained with its own service layer and error handling.
"""

from src.que_agents.router.customer_support import router as customer_support_router
from src.que_agents.router.financial_trading_bot import router as trading_router
from src.que_agents.router.marketing import router as marketing_router
from src.que_agents.router.personal_virtual_assistant import router as pva_router

__all__ = [
    "customer_support_router",
    "trading_router",
    "marketing_router",
    "pva_router",
]
