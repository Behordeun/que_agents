"""
Unit tests for Personal Virtual Assistant Router to improve code coverage.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.que_agents.core.schemas import PVARequest, PVAResponse
from src.que_agents.router.personal_virtual_assistant import (
    PersonalVirtualAssistantService,
    get_pva_service,
    router,
)


@pytest.fixture
def agent_manager():
    return MagicMock()


@pytest.fixture
def service(agent_manager):
    return PersonalVirtualAssistantService(agent_manager)


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.handle_user_request.return_value = {
        "response": "Hello! How can I help you today?",
        "intent": "greeting",
        "confidence": 0.95,
        "actions_taken": ["Greeted user"],
        "suggestions": ["Ask about weather", "Set a reminder"],
        "entities": {},
        "timestamp": datetime.now().isoformat(),
    }
    agent.get_user_devices.return_value = [
        {"id": 1, "name": "Living Room Lights", "type": "light", "status": "off"}
    ]
    agent.get_user_reminders.return_value = [
        {
            "id": 1,
            "title": "Call mom",
            "time": "2024-01-01T15:00:00",
            "status": "active",
        }
    ]
    return agent


class TestPersonalVirtualAssistantService:
    """Test PersonalVirtualAssistantService functionality"""

    def test_get_agent_success(self, service, agent_manager, mock_agent):
        """Test successful agent retrieval"""
        agent_manager.get_agent.return_value = mock_agent
        agent = service.get_agent("test_token")
        assert agent == mock_agent
        agent_manager.get_agent.assert_called_once_with(
            "personal_virtual_assistant", "test_token"
        )

    def test_get_agent_not_found(self, service, agent_manager):
        """Test agent not found scenario"""
        agent_manager.get_agent.return_value = None
        with patch(
            "src.que_agents.router.personal_virtual_assistant.system_logger"
        ) as mock_logger:
            agent = service.get_agent("test_token")
            assert agent is None
            mock_logger.error.assert_called_once()

    def test_handle_chat_request_success(self, service, mock_agent):
        """Test successful chat request handling"""
        service.get_agent = MagicMock(return_value=mock_agent)
        request = PVARequest(
            user_id="test_user", message="Hello", session_id="test_session"
        )

        response = service.handle_chat_request(request, "test_token")

        assert isinstance(response, PVAResponse)
        assert response.response == "Hello! How can I help you today?"
        assert response.intent == "greeting"
        mock_agent.handle_user_request.assert_called_once()

    def test_handle_chat_request_string_response(self, service, mock_agent):
        """Test chat request with string response from agent"""
        mock_agent.handle_user_request.return_value = "Simple string response"
        service.get_agent = MagicMock(return_value=mock_agent)
        request = PVARequest(user_id="test_user", message="Hello")

        response = service.handle_chat_request(request, "test_token")

        assert isinstance(response, PVAResponse)
        assert response.response == "Simple string response"
        assert response.intent == "general_query"

    def test_handle_chat_request_agent_unavailable(self, service):
        """Test chat request with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)
        request = PVARequest(user_id="test_user", message="Hello")

        response = service.handle_chat_request(request, "test_token")

        assert isinstance(response, PVAResponse)
        assert "personal assistant" in response.response.lower()

    def test_handle_chat_request_error(self, service, mock_agent):
        """Test chat request with error"""
        mock_agent.handle_user_request.side_effect = Exception("PVA error")
        service.get_agent = MagicMock(return_value=mock_agent)
        request = PVARequest(user_id="test_user", message="Hello")

        with patch("src.que_agents.router.personal_virtual_assistant.system_logger"):
            response = service.handle_chat_request(request, "test_token")

            assert isinstance(response, PVAResponse)
            assert "technical difficulties" in response.response.lower()

    def test_generate_fallback_response_greeting(self, service):
        """Test fallback response for greeting"""
        response = service._generate_fallback_response(
            "user1", "Hello there", "test reason"
        )

        assert isinstance(response, PVAResponse)
        assert response.intent == "greeting"
        assert "personal assistant" in response.response.lower()

    def test_generate_fallback_response_weather(self, service):
        """Test fallback response for weather query"""
        response = service._generate_fallback_response(
            "user1", "What's the weather like?", ""
        )

        assert isinstance(response, PVAResponse)
        assert response.intent == "weather_inquiry"
        assert "weather" in response.response.lower()

    def test_generate_fallback_response_reminder(self, service):
        """Test fallback response for reminder request"""
        response = service._generate_fallback_response(
            "user1", "Set a reminder for tomorrow", ""
        )

        assert isinstance(response, PVAResponse)
        assert "reminder" in response.response.lower()

    def test_generate_fallback_response_smart_home(self, service):
        """Test fallback response for smart home query"""
        response = service._generate_fallback_response(
            "user1", "Turn on the lights", ""
        )

        assert isinstance(response, PVAResponse)
        assert response.intent == "smart_home_control"
        assert "smart home" in response.response.lower()

    def test_generate_fallback_response_calendar(self, service):
        """Test fallback response for calendar query"""
        response = service._generate_fallback_response(
            "user1", "What's on my calendar?", ""
        )

        assert isinstance(response, PVAResponse)
        assert response.intent == "calendar_inquiry"
        assert "calendar" in response.response.lower()

    def test_generate_fallback_response_help(self, service):
        """Test fallback response for help request"""
        response = service._generate_fallback_response("user1", "What can you do?", "")

        assert isinstance(response, PVAResponse)
        assert response.intent == "help_request"
        assert "weather" in response.response.lower()
        assert "reminders" in response.response.lower()

    def test_generate_fallback_response_generic(self, service):
        """Test fallback response for generic query"""
        response = service._generate_fallback_response("user1", "Random question", "")

        assert isinstance(response, PVAResponse)
        assert "Random question" in response.response

    def test_extract_basic_entities(self, service):
        """Test basic entity extraction"""
        entities = service._extract_basic_entities(
            "Turn on the lights tomorrow morning"
        )

        assert "time" in entities
        assert "devices" in entities
        assert "tomorrow" in entities["time"]
        assert "morning" in entities["time"]
        assert "lights" in entities["devices"]

    def test_get_user_reminders_success(self, service, mock_agent):
        """Test successful user reminders retrieval"""
        mock_agent.get_user_reminders.return_value = {
            "reminders": [{"id": 1, "title": "Test reminder"}],
            "total": 1,
        }
        service.get_agent = MagicMock(return_value=mock_agent)

        result = service.get_user_reminders("test_user", "test_token")

        assert "reminders" in result
        assert result["data_source"] == "agent_data"
        mock_agent.get_user_reminders.assert_called_once_with("test_user")

    def test_get_user_reminders_agent_unavailable(self, service):
        """Test user reminders with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)

        result = service.get_user_reminders("test_user", "test_token")

        assert "reminders" in result
        assert result["data_source"] == "fallback_data"

    def test_get_user_reminders_agent_error(self, service, mock_agent):
        """Test user reminders with agent error"""
        mock_agent.get_user_reminders.side_effect = Exception("Reminders error")
        service.get_agent = MagicMock(return_value=mock_agent)

        with patch("src.que_agents.router.personal_virtual_assistant.system_logger"):
            result = service.get_user_reminders("test_user", "test_token")

            assert result["data_source"] == "fallback_data"

    def test_generate_fallback_reminders(self, service):
        """Test fallback reminders generation"""
        result = service._generate_fallback_reminders("test_user", "test reason")

        assert "reminders" in result
        assert result["data_source"] == "fallback_data"
        assert "test reason" in result["note"]
        assert len(result["reminders"]) > 0

    def test_enhance_reminders_data(self, service):
        """Test reminders data enhancement"""
        reminders = {"reminders": [{"id": 1}], "total": 1}

        result = service._enhance_reminders_data(reminders)

        assert result["reminders"] == [{"id": 1}]
        assert result["data_source"] == "agent_data"

    def test_get_user_devices_success(self, service, mock_agent):
        """Test successful user devices retrieval"""
        mock_agent.get_user_devices.return_value = {
            "devices": [{"id": 1, "name": "Test Device"}],
            "total": 1,
        }
        service.get_agent = MagicMock(return_value=mock_agent)

        result = service.get_user_devices("test_user", "test_token")

        assert "devices" in result
        assert result["data_source"] == "agent_data"
        mock_agent.get_user_devices.assert_called_once_with("test_user")

    def test_get_user_devices_agent_unavailable(self, service):
        """Test user devices with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)

        result = service.get_user_devices("test_user", "test_token")

        assert "devices" in result
        assert result["data_source"] == "fallback_data"

    def test_get_user_devices_agent_error(self, service, mock_agent):
        """Test user devices with agent error"""
        mock_agent.get_user_devices.side_effect = Exception("Devices error")
        service.get_agent = MagicMock(return_value=mock_agent)

        with patch("src.que_agents.router.personal_virtual_assistant.system_logger"):
            result = service.get_user_devices("test_user", "test_token")

            assert result["data_source"] == "fallback_data"

    def test_generate_fallback_devices(self, service):
        """Test fallback devices generation"""
        result = service._generate_fallback_devices("test_user", "test reason")

        assert "devices" in result
        assert result["data_source"] == "fallback_data"
        assert "test reason" in result["note"]
        assert len(result["devices"]) > 0

    def test_enhance_devices_data(self, service):
        """Test devices data enhancement"""
        devices = {"devices": [{"id": 1}], "total": 1}

        result = service._enhance_devices_data(devices)

        assert result["devices"] == [{"id": 1}]
        assert result["data_source"] == "agent_data"

    def test_get_user_context_success(self, service, mock_agent):
        """Test successful user context retrieval"""
        mock_context = MagicMock()
        mock_context.__dict__ = {"user_id": "test_user", "preferences": {}}
        mock_agent.get_user_context.return_value = mock_context
        service.get_agent = MagicMock(return_value=mock_agent)

        result = service.get_user_context("test_user", "test_token")

        assert "user_id" in result
        assert result["data_source"] == "agent_context"

    def test_get_user_context_agent_unavailable(self, service):
        """Test user context with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)

        result = service.get_user_context("test_user", "test_token")

        assert result["user_id"] == "test_user"
        assert result["data_source"] == "fallback_context"

    def test_get_user_context_no_context(self, service, mock_agent):
        """Test user context when agent returns None"""
        mock_agent.get_user_context.return_value = None
        service.get_agent = MagicMock(return_value=mock_agent)

        result = service.get_user_context("test_user", "test_token")

        assert result["data_source"] == "fallback_context"

    def test_generate_fallback_user_context(self, service):
        """Test fallback user context generation"""
        result = service._generate_fallback_user_context("test_user")

        assert result["user_id"] == "test_user"
        assert "preferences" in result
        assert result["data_source"] == "fallback_context"

    def test_enhance_user_context_dict(self, service):
        """Test user context enhancement with dictionary"""
        context = {"user_id": "test_user", "preferences": {}}

        result = service._enhance_user_context(context)

        assert result["user_id"] == "test_user"
        assert result["data_source"] == "agent_context"

    def test_enhance_user_context_object(self, service):
        """Test user context enhancement with object"""
        context = MagicMock()
        context.__dict__ = {"user_id": "test_user", "preferences": {}}

        result = service._enhance_user_context(context)

        assert result["user_id"] == "test_user"
        assert result["data_source"] == "agent_context"

    def test_enhance_user_context_other(self, service):
        """Test user context enhancement with other type"""
        context = "string context"

        result = service._enhance_user_context(context)

        assert result["raw_context"] == "string context"
        assert result["data_source"] == "agent_context"

    def test_control_smart_device_success(self, service, mock_agent):
        """Test successful smart device control"""
        mock_agent.control_smart_device.return_value = {
            "success": True,
            "message": "Device controlled successfully",
        }
        service.get_agent = MagicMock(return_value=mock_agent)

        result = service.control_smart_device(
            "user1", "device1", "turn_on", {}, "test_token"
        )

        assert result["data_source"] == "agent_control"
        mock_agent.control_smart_device.assert_called_once()

    def test_control_smart_device_agent_unavailable(self, service):
        """Test smart device control with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)

        result = service.control_smart_device(
            "user1", "device1", "turn_on", {}, "test_token"
        )

        assert result["data_source"] == "fallback_control"
        assert result["status"] == "simulated"

    def test_control_smart_device_agent_error(self, service, mock_agent):
        """Test smart device control with agent error"""
        mock_agent.control_smart_device.side_effect = Exception("Control error")
        service.get_agent = MagicMock(return_value=mock_agent)

        with patch("src.que_agents.router.personal_virtual_assistant.system_logger"):
            result = service.control_smart_device(
                "user1", "device1", "turn_on", {}, "test_token"
            )

            assert result["data_source"] == "fallback_control"

    def test_generate_device_control_fallback(self, service):
        """Test device control fallback generation"""
        result = service._generate_device_control_fallback(
            "device1", "turn_on", "test reason"
        )

        assert result["device_id"] == "device1"
        assert result["action"] == "turn_on"
        assert result["status"] == "simulated"
        assert "test reason" in result["note"]

    def test_enhance_control_result(self, service):
        """Test control result enhancement"""
        result = {"success": True, "message": "Controlled"}

        enhanced = service._enhance_control_result(result)

        assert enhanced["success"] is True
        assert enhanced["data_source"] == "agent_control"

    def test_create_reminder_success(self, service, mock_agent):
        """Test successful reminder creation"""
        mock_agent.create_reminder.return_value = {
            "reminder_id": "123",
            "success": True,
        }
        service.get_agent = MagicMock(return_value=mock_agent)
        reminder_data = {"title": "Test reminder"}

        result = service.create_reminder("user1", reminder_data, "test_token")

        assert result["data_source"] == "agent_creation"
        mock_agent.create_reminder.assert_called_once()

    def test_create_reminder_agent_unavailable(self, service):
        """Test reminder creation with unavailable agent"""
        service.get_agent = MagicMock(return_value=None)
        reminder_data = {"title": "Test reminder"}

        result = service.create_reminder("user1", reminder_data, "test_token")

        assert result["data_source"] == "fallback_creation"
        assert result["success"] is True

    def test_create_reminder_agent_error(self, service, mock_agent):
        """Test reminder creation with agent error"""
        mock_agent.create_reminder.side_effect = Exception("Creation error")
        service.get_agent = MagicMock(return_value=mock_agent)
        reminder_data = {"title": "Test reminder"}

        with patch("src.que_agents.router.personal_virtual_assistant.system_logger"):
            result = service.create_reminder("user1", reminder_data, "test_token")

            assert result["data_source"] == "fallback_creation"

    def test_generate_reminder_creation_fallback(self, service):
        """Test reminder creation fallback generation"""
        reminder_data = {"title": "Test reminder", "priority": "high"}

        result = service._generate_reminder_creation_fallback(
            reminder_data, "test reason"
        )

        assert result["title"] == "Test reminder"
        assert result["priority"] == "high"
        assert result["success"] is True
        assert "test reason" in result["note"]

    def test_enhance_reminder_creation_result(self, service):
        """Test reminder creation result enhancement"""
        result = {"reminder_id": "123", "success": True}

        enhanced = service._enhance_reminder_creation_result(result)

        assert enhanced["reminder_id"] == "123"
        assert enhanced["data_source"] == "agent_creation"


class TestRouterDependencies:
    """Test router dependencies and endpoints"""

    def test_get_pva_service(self):
        """Test service dependency creation"""
        with patch(
            "src.que_agents.router.personal_virtual_assistant.agent_manager"
        ) as mock_manager:
            service = get_pva_service()
            assert isinstance(service, PersonalVirtualAssistantService)
            assert service.agent_manager == mock_manager

    def test_router_exists(self):
        """Test router is properly configured"""
        assert router is not None
        assert hasattr(router, "routes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
