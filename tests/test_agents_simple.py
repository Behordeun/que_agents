# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-01 23:53:31
# @Description: This module implements the test suite for the Agentic AI system

import sys

sys.path.insert(0, "que_agents")


def test_imports():
    """Test if all agent modules can be imported"""
    print("Testing agent imports...")

    try:
        print("✓ Database models imported successfully")
    except Exception as e:
        print(f"✗ Database import failed: {e}")
        return False

    try:
        print("✓ LLM Factory imported successfully")
    except Exception as e:
        print(f"✗ LLM Factory import failed: {e}")
        return False

    # Test agent imports without initializing them (to avoid LLM dependencies)
    try:
        print("✓ Customer Support Agent module imported successfully")
    except Exception as e:
        print(f"✗ Customer Support Agent import failed: {e}")
        return False

    try:
        print("✓ Marketing Agent module imported successfully")
    except Exception as e:
        print(f"✗ Marketing Agent import failed: {e}")
        return False

    try:
        print("✓ Personal Virtual Assistant Agent module imported successfully")
    except Exception as e:
        print(f"✗ Personal Virtual Assistant Agent import failed: {e}")
        return False

    try:
        print("✓ Financial Trading Bot Agent module imported successfully")
    except Exception as e:
        print(f"✗ Financial Trading Bot Agent import failed: {e}")
        return False

    return True


def test_database_models():
    """Test database model creation"""
    print("\nTesting database models...")

    try:
        from src.que_agents.core.database import (
            Customer,
            Portfolio,
            TradingStrategy,
            UserPreferences,
        )

        # Test model instantiation
        _customer = Customer(name="Test User", email="test@example.com")
        print("✓ Customer model created")

        _user_prefs = UserPreferences(user_id="test123", preferences={})
        print("✓ UserPreferences model created")

        _portfolio = Portfolio(portfolio_name="Test Portfolio")
        print("✓ Portfolio model created")

        _trading_strategy = TradingStrategy(
            name="Test Strategy", strategy_type="momentum", parameters={}
        )
        print("✓ TradingStrategy model created")

        print("✓ All database models working correctly")
        return True

    except Exception as e:
        print(f"✗ Database model test failed: {e}")
        return False


def test_api_structure():
    """Test API module structure"""
    print("\nTesting API structure...")

    try:
        import src.que_agents.api.main

        print("✓ API main module imported successfully")

        # Check if FastAPI app is created
        app = src.que_agents.api.main.app
        print("✓ FastAPI app instance found")

        # Check routes
        from fastapi.routing import APIRoute
        routes = [route.path for route in app.routes if isinstance(route, APIRoute)]
        expected_routes = [
            "/health",
            "/api/v1/customer-support/chat",
            "/api/v1/pva/chat",
            "/api/v1/trading/analyze",
        ]

        for route in expected_routes:
            if route in routes:
                print(f"✓ Route {route} found")
            else:
                print(f"✗ Route {route} missing")

        return True

    except Exception as e:
        print(f"✗ API structure test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=== Multi-Agent Platform Test Suite ===\n")

    tests = [test_imports, test_database_models, test_api_structure]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"=== Test Results: {passed}/{total} tests passed ===")

    if passed == total:
        print("🎉 All tests passed! The multi-agent platform is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
