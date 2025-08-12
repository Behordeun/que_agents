from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.que_agents.core.schemas import CustomerSupportRequest
from src.que_agents.router.customer_support import CustomerSupportService


@pytest.fixture
def agent_manager():
    return MagicMock()


@pytest.fixture
def service(agent_manager):
    return CustomerSupportService(agent_manager)


def test_get_agent_found(service, agent_manager):
    agent_manager.get_agent.return_value = "agent_obj"
    agent = service.get_agent("token")
    assert agent == "agent_obj"


def test_get_agent_not_found_logs_error(service, agent_manager):
    agent_manager.get_agent.return_value = None
    with patch("src.que_agents.router.customer_support.system_logger") as logger:
        agent = service.get_agent("token")
        assert agent is None
        logger.error.assert_called()


def test_handle_chat_request_success(service):
    mock_agent = MagicMock()
    mock_agent.handle_customer_request_enhanced.return_value = {
        "response": "Hi",
        "confidence": 0.9,
        "escalate": False,
        "suggested_actions": ["action"],
        "knowledge_sources": ["kb"],
        "sentiment": "positive",
    }
    service.get_agent = MagicMock(return_value=mock_agent)
    req = CustomerSupportRequest(customer_id=1, message="Hello")
    resp = service.handle_chat_request(req, token="token")
    assert resp.response == "Hi"
    assert resp.confidence == pytest.approx(0.9)
    assert resp.escalate is False


def test_handle_chat_request_agent_unavailable(service):
    service.get_agent = MagicMock(return_value=None)
    req = CustomerSupportRequest(customer_id=1, message="Hello")
    with pytest.raises(HTTPException) as exc:
        service.handle_chat_request(req, token="token")
    assert exc.value.status_code == 503


def test_handle_chat_request_value_error(service):
    service.get_agent = MagicMock(
        return_value=MagicMock(
            handle_customer_request_enhanced=MagicMock(side_effect=ValueError("bad id"))
        )
    )
    req = CustomerSupportRequest(customer_id=-1, message="Hello")
    resp = service.handle_chat_request(req, token="token")
    assert "apologize" in resp.response.lower()
    assert resp.confidence == pytest.approx(0.0)
    assert resp.escalate is True


def test_handle_chat_request_general_exception(service):
    service.get_agent = MagicMock(
        return_value=MagicMock(
            handle_customer_request_enhanced=MagicMock(side_effect=Exception("fail"))
        )
    )
    req = CustomerSupportRequest(customer_id=1, message="Hello")
    resp = service.handle_chat_request(req, token="token")
    assert "technical difficulties" in resp.response.lower()
    assert resp.confidence == pytest.approx(0.5)
    assert resp.escalate is True


def test_get_customer_context_data_success(service):
    mock_agent = MagicMock()
    mock_agent.get_customer_insights.return_value = {
        "customer_context": {
            "customer_id": 1,
            "name": "Test",
            "email": "t@e.com",
            "tier": "gold",
            "company": "C",
            "satisfaction_score": 4.5,
            "recent_interactions": [
                {
                    "message": "Hi",
                    "date": "2020-01-01",
                    "sentiment": "positive",
                    "satisfaction": 5,
                    "type": "chat",
                }
            ],
        },
        "interaction_stats": {"total_interactions": 2, "average_satisfaction": 4.5},
        "support_tickets": {
            "open_tickets": 1,
            "total_tickets": 2,
            "recent_tickets": [
                {
                    "id": 1,
                    "title": "T",
                    "category": "billing",
                    "priority": "high",
                    "status": "open",
                    "created_at": "2020-01-01",
                }
            ],
        },
        "feedback_insights": {
            "feedback_count": 1,
            "satisfaction_trend": {"trend_direction": "up", "latest_rating": 5},
        },
        "risk_indicators": {
            "risk_level": "medium",
            "risk_score": 0.5,
            "risk_factors": ["late payment"],
        },
        "recommendations": ["Do X"],
    }
    service.get_agent = MagicMock(return_value=mock_agent)
    data = service.get_customer_context_data(1, token="token")
    assert data["customer_id"] == 1
    assert data["customer_name"] == "Test"
    assert data["support_metrics"]["total_interactions"] == 2
    assert data["risk_assessment"]["risk_level"] == "Medium"
    assert data["recommendations"][0] == "Do X"


def test_get_customer_context_data_agent_unavailable(service):
    service.get_agent = MagicMock(return_value=None)
    with pytest.raises(HTTPException) as exc:
        service.get_customer_context_data(1, token="token")
    assert exc.value.status_code == 503


def test_get_customer_context_data_keyerror(service):
    mock_agent = MagicMock()
    mock_agent.get_customer_insights.side_effect = KeyError("fail")
    service.get_agent = MagicMock(return_value=mock_agent)
    data = service.get_customer_context_data(1, token="token")
    assert data["customer_id"] == 1
    assert "fallback" in data["data_sources"][0]


def test_get_customer_context_data_general_exception(service):
    mock_agent = MagicMock()
    mock_agent.get_customer_insights.side_effect = Exception("fail")
    service.get_agent = MagicMock(return_value=mock_agent)
    with pytest.raises(HTTPException) as exc:
        service.get_customer_context_data(1, token="token")
    assert exc.value.status_code == 500


def test_get_debug_info_success(service):
    mock_agent = MagicMock()
    mock_agent.get_customer_context.return_value = MagicMock(
        customer_id=1,
        name="Test",
        email="t@e.com",
        tier="gold",
        company="C",
        satisfaction_score=4.5,
        recent_interactions=[],
        open_tickets=[],
    )
    mock_agent.get_customer_insights.return_value = {
        "customer_context": {},
        "interaction_stats": {},
        "support_tickets": {},
        "feedback_insights": {},
        "risk_indicators": {},
        "recommendations": [],
    }
    service.get_agent = MagicMock(return_value=mock_agent)
    info = service.get_debug_info(1, token="token")
    assert info["customer_id"] == 1
    assert info["customer_context_exists"]
    assert "customer_context_data" in info


def test_get_debug_info_agent_unavailable(service):
    service.get_agent = MagicMock(return_value=None)
    info = service.get_debug_info(1, token="token")
    assert "error" in info


def test_get_debug_info_insights_exception(service):
    mock_agent = MagicMock()
    mock_agent.get_customer_context.return_value = MagicMock()
    mock_agent.get_customer_insights.side_effect = Exception("fail")
    service.get_agent = MagicMock(return_value=mock_agent)
    info = service.get_debug_info(1, token="token")
    assert info["customer_insights_success"] is False
    assert "customer_insights_error" in info


def test_get_customer_insights_data_success(service):
    mock_agent = MagicMock()
    mock_agent.get_customer_insights.return_value = {"foo": "bar"}
    service.get_agent = MagicMock(return_value=mock_agent)
    data = service.get_customer_insights_data(1, token="token")
    assert data["foo"] == "bar"


def test_get_customer_insights_data_agent_unavailable(service):
    service.get_agent = MagicMock(return_value=None)
    with pytest.raises(HTTPException) as exc:
        service.get_customer_insights_data(1, token="token")
    assert exc.value.status_code == 503


def test_get_customer_insights_data_error_in_insights(service):
    mock_agent = MagicMock()
    mock_agent.get_customer_insights.return_value = {"error": "not found"}
    service.get_agent = MagicMock(return_value=mock_agent)
    with pytest.raises(HTTPException) as exc:
        service.get_customer_insights_data(1, token="token")
    assert exc.value.status_code == 404


def test_get_customer_insights_data_general_exception(service):
    mock_agent = MagicMock()
    mock_agent.get_customer_insights.side_effect = Exception("fail")
    service.get_agent = MagicMock(return_value=mock_agent)
    with pytest.raises(HTTPException) as exc:
        service.get_customer_insights_data(1, token="token")
    assert exc.value.status_code == 500


def test_get_fallback_customer_data(service):
    data = service._get_fallback_customer_data(42)
    assert data["customer_id"] == 42
    assert data["support_tier"] == "Standard"
    assert "fallback" in data["data_sources"][0]
