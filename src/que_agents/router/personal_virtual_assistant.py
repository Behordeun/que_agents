# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-10
# @Description: Personal Virtual Assistant API routes and handlers

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.que_agents.core.schemas import PVARequest, PVAResponse
from src.que_agents.error_trace.errorlogger import system_logger
from src.que_agents.utils.agent_manager import AgentManager
from src.que_agents.utils.auth import get_token_from_state

# Constant for UTC offset replacement
UTC_OFFSET = "+00:00"


METHOD_NOT_AVAILABLE = "Method not available"
PVA_AGENT_NOT_AVAILABLE = "PVA agent not available"
TRY_AGAIN_LATER = "Try again later"
UNKNOWN_ERROR = "Unknown error"


class PersonalVirtualAssistantService:
    """Service class for Personal Virtual Assistant operations"""

    PVA_CONTEXT = "Personal Virtual Assistant"

    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        self.PVA_UNAVAILABLE = "Personal Virtual Assistant not available"

    def get_agent(self, token: str):
        """Get Personal Virtual Assistant agent"""
        agent = self.agent_manager.get_agent("personal_virtual_assistant", token)
        if agent is None:
            system_logger.error(
                "Personal Virtual Assistant agent is not available in AgentManager.",
                additional_info={"context": self.PVA_CONTEXT},
            )
        return agent

    def handle_chat_request(
        self, request: PVARequest, token: str = Depends(get_token_from_state)
    ) -> PVAResponse:
        """Handle PVA chat request with enhanced error handling"""
        try:
            agent = self.get_agent(token)
            if not agent:
                return self._generate_fallback_response(
                    request.user_id, request.message, PVA_AGENT_NOT_AVAILABLE
                )

            result = agent.handle_user_request(
                user_id=request.user_id,
                user_message=request.message,
                session_id=request.session_id or "",
            )

            # Handle case where result might be a string instead of dict
            if isinstance(result, str):
                return PVAResponse(
                    response=result,
                    intent="general_query",
                    entities={},
                    confidence=0.5,
                    actions_taken=[],
                    suggestions=[],
                    timestamp=datetime.now().isoformat(),
                )

            # Handle normal dictionary response
            return PVAResponse(
                response=result.get("response", "I'm here to help!"),
                intent=result.get("intent", "general_query"),
                entities=result.get("entities", {}),
                confidence=result.get("confidence", 0.5),
                actions_taken=result.get("actions_taken", []),
                suggestions=result.get("suggestions", []),
                timestamp=result.get("timestamp", datetime.now().isoformat()),
            )

        except Exception as e:
            system_logger.error(
                f"Error in PVA chat: {str(e)}",
                additional_info={
                    "context": "PVA Chat",
                    "user_id": request.user_id,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

            return self._generate_fallback_response(
                request.user_id, request.message, "Technical difficulties"
            )

    def _generate_fallback_response(
        self, _user_id: str, message: str, reason: str = ""
    ) -> PVAResponse:
        """Generate intelligent fallback response based on user message"""
        message_lower = message.lower()
        intent = "unknown"

        # Analyze message intent for better fallback responses
        if any(
            word in message_lower
            for word in [
                "hello",
                "hi",
                "hey",
                "good morning",
                "good afternoon",
                "good evening",
            ]
        ):
            response = f"Hello! I'm your personal assistant. I'm currently experiencing some {reason.lower() if reason else 'technical difficulties'}, but I'm here to help as best I can. How can I assist you today?"
            intent = "greeting"
            suggestions = [
                "Ask about weather",
                "Set a reminder",
                "Check calendar",
                "Smart home control",
            ]

        elif any(
            word in message_lower
            for word in ["weather", "temperature", "forecast", "rain", "sunny"]
        ):
            response = "I'd love to help you with weather information! While I'm having some technical difficulties accessing real-time weather data, I recommend checking your local weather app or website for the most current conditions."
            intent = "weather_inquiry"
            suggestions = [
                "Check weather app",
                "Ask about tomorrow's forecast",
                "Set weather reminder",
            ]

        elif any(
            word in message_lower
            for word in ["reminder", "remind", "schedule", "appointment"]
        ):
            response = "I understand you'd like to set a reminder. While my reminder system is temporarily unavailable, I suggest using your device's built-in reminder or calendar app for now."
            suggestions = ["Use device reminder", "Check calendar", TRY_AGAIN_LATER]

        elif any(
            word in message_lower
            for word in ["smart", "device", "light", "thermostat", "home"]
        ):
            response = "I can help with smart home devices! However, I'm currently having trouble connecting to your smart home systems. Please try using your device-specific apps or voice commands directly."
            intent = "smart_home_control"
            suggestions = [
                "Use device app",
                "Try voice commands",
                "Check device status",
            ]

        elif any(
            word in message_lower
            for word in ["calendar", "meeting", "event", "appointment"]
        ):
            response = "I'd be happy to help with your calendar! While I'm temporarily unable to access your calendar directly, you can check your calendar app for upcoming events and appointments."
            intent = "calendar_inquiry"
            suggestions = [
                "Open calendar app",
                "Check today's schedule",
                "View upcoming events",
            ]

        elif any(
            word in message_lower for word in ["email", "message", "send", "compose"]
        ):
            response = "I can assist with email and messaging! Currently experiencing some connectivity issues, but I recommend using your email app directly or trying again in a few minutes."
            suggestions = ["Open email app", "Draft message", TRY_AGAIN_LATER]

        elif any(
            word in message_lower
            for word in ["help", "what can you do", "capabilities", "features"]
        ):
            response = """I'm your personal virtual assistant! Even though I'm experiencing some technical difficulties right now, I can typically help you with:

• Weather updates and forecasts
• Setting reminders and managing tasks
• Calendar management and scheduling
• Smart home device control
• Email and message assistance
• General information and questions
• Daily planning and organization

Please try again in a few minutes, or feel free to ask me anything!"""
            intent = "help_request"
            suggestions = [
                "Try specific request",
                "Check system status",
                "Contact support",
            ]

        else:
            response = f"I understand you're asking about '{message[:50]}{'...' if len(message) > 50 else ''}'. While I'm experiencing some technical difficulties, I'm still here to help! Could you try rephrasing your request or ask me something else?"
            suggestions = [
                "Rephrase request",
                "Ask something else",
                TRY_AGAIN_LATER,
                "Get help",
            ]

        return PVAResponse(
            response=response,
            intent=intent,
            entities=self._extract_basic_entities(message),
            confidence=0.6,
            actions_taken=["fallback_response", "intent_analysis"],
            suggestions=suggestions,
            timestamp=datetime.now().isoformat(),
        )

    def _extract_basic_entities(self, message: str) -> Dict[str, Any]:
        """Extract basic entities from user message for fallback response"""
        entities = {}
        message_lower = message.lower()

        # Time entities
        time_words = [
            "today",
            "tomorrow",
            "tonight",
            "morning",
            "afternoon",
            "evening",
            "now",
        ]
        found_times = [word for word in time_words if word in message_lower]
        if found_times:
            entities["time"] = found_times

        # Location entities (basic)
        location_words = ["home", "office", "work", "here", "there"]
        found_locations = [word for word in location_words if word in message_lower]
        if found_locations:
            entities["location"] = found_locations

        # Device entities
        device_words = [
            "light",
            "lights",
            "thermostat",
            "temperature",
            "music",
            "tv",
            "television",
        ]
        found_devices = [word for word in device_words if word in message_lower]
        if found_devices:
            entities["devices"] = found_devices

        return entities

    def get_user_reminders(
        self, user_id: str, token: str = Depends(get_token_from_state)
    ) -> Dict[str, Any]:
        """Get user reminders with fallback data"""
        try:
            agent = self.get_agent(token)
            if not agent:
                return self._generate_fallback_reminders(
                    user_id, PVA_AGENT_NOT_AVAILABLE
                )

            if hasattr(agent, "get_user_reminders"):
                try:
                    reminders = agent.get_user_reminders(user_id)
                    return self._enhance_reminders_data(reminders)
                except Exception as e:
                    system_logger.warning(f"Agent reminders retrieval failed: {e}")
                    return self._generate_fallback_reminders(
                        user_id, METHOD_NOT_AVAILABLE
                    )

        except Exception as e:
            system_logger.error(
                f"Error getting user reminders: {str(e)}",
                additional_info={
                    "context": "User Reminders Retrieval",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return self._generate_fallback_reminders(user_id, str(e))
        return self._generate_fallback_reminders(user_id, UNKNOWN_ERROR)

    def _generate_fallback_reminders(
        self, _user_id: str, reason: str = ""
    ) -> Dict[str, Any]:
        """Generate fallback reminder data"""
        # Generate some realistic sample reminders
        sample_reminders = [
            {
                "id": "reminder_001",
                "title": "Daily standup meeting",
                "description": "Weekly team standup meeting",
                "target_time": (datetime.now() + timedelta(hours=2)).isoformat(),
                "status": "active",
                "priority": "medium",
                "category": "work",
                "recurring": True,
                "created_at": (datetime.now() - timedelta(days=7)).isoformat(),
            },
            {
                "id": "reminder_002",
                "title": "Review quarterly goals",
                "description": "Quarterly performance review preparation",
                "target_time": (datetime.now() + timedelta(days=1)).isoformat(),
                "status": "active",
                "priority": "high",
                "category": "work",
                "recurring": False,
                "created_at": (datetime.now() - timedelta(days=3)).isoformat(),
            },
            {
                "id": "reminder_003",
                "title": "Call dentist for appointment",
                "description": "Schedule routine dental checkup",
                "target_time": (datetime.now() + timedelta(hours=5)).isoformat(),
                "status": "active",
                "priority": "low",
                "category": "personal",
                "recurring": False,
                "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
            },
        ]

        return {
            "reminders": sample_reminders,
            "total": len(sample_reminders),
            "active_count": len(
                [r for r in sample_reminders if r["status"] == "active"]
            ),
            "categories": {
                "work": len([r for r in sample_reminders if r["category"] == "work"]),
                "personal": len(
                    [r for r in sample_reminders if r["category"] == "personal"]
                ),
            },
            "upcoming_today": len(
                [
                    r
                    for r in sample_reminders
                    if datetime.fromisoformat(
                        r["target_time"].replace("Z", UTC_OFFSET)
                    ).date()
                    == datetime.now().date()
                ]
            ),
            "data_source": "fallback_data",
            "note": (
                f"Using sample reminder data. {reason}"
                if reason
                else "Using sample reminder data"
            ),
            "last_updated": datetime.now().isoformat(),
        }

    def _enhance_reminders_data(self, reminders: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance reminders data with additional insights"""
        enhanced = reminders.copy()
        enhanced.update(
            {"last_updated": datetime.now().isoformat(), "data_source": "agent_data"}
        )
        return enhanced

    def get_user_devices(
        self, user_id: str, token: str = Depends(get_token_from_state)
    ) -> Dict[str, Any]:
        """Get user smart devices with fallback data"""
        try:
            agent = self.get_agent(token)
            if not agent:
                return self._generate_fallback_devices(user_id, PVA_AGENT_NOT_AVAILABLE)

            if hasattr(agent, "get_user_devices"):
                try:
                    devices = agent.get_user_devices(user_id)
                    return self._enhance_devices_data(devices)
                except Exception as e:
                    system_logger.warning(f"Agent devices retrieval failed: {e}")
                    return self._generate_fallback_devices(
                        user_id, METHOD_NOT_AVAILABLE
                    )

        except Exception as e:
            system_logger.error(
                f"Error getting user devices: {str(e)}",
                additional_info={
                    "context": "User Devices Retrieval",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return self._generate_fallback_devices(user_id, str(e))
        return self._generate_fallback_devices(user_id, UNKNOWN_ERROR)

    def _generate_fallback_devices(
        self, _user_id: str, reason: str = ""
    ) -> Dict[str, Any]:
        """Generate fallback smart home devices data"""
        sample_devices = [
            {
                "id": "device_001",
                "name": "Living Room Lights",
                "type": "lighting",
                "status": "on",
                "location": "living_room",
                "brightness": 75,
                "color": "warm_white",
                "controllable": True,
                "last_updated": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "manufacturer": "Philips Hue",
                "model": "Smart Bulb",
            },
            {
                "id": "device_002",
                "name": "Smart Thermostat",
                "type": "climate",
                "status": "auto",
                "location": "hallway",
                "current_temp": 72.5,
                "target_temp": 73.0,
                "mode": "heat",
                "controllable": True,
                "last_updated": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "manufacturer": "Nest",
                "model": "Learning Thermostat",
            },
            {
                "id": "device_003",
                "name": "Kitchen Speaker",
                "type": "audio",
                "status": "idle",
                "location": "kitchen",
                "volume": 45,
                "currently_playing": None,
                "controllable": True,
                "last_updated": (datetime.now() - timedelta(hours=2)).isoformat(),
                "manufacturer": "Amazon",
                "model": "Echo Dot",
            },
            {
                "id": "device_004",
                "name": "Front Door Lock",
                "type": "security",
                "status": "locked",
                "location": "front_door",
                "battery_level": 85,
                "controllable": True,
                "last_updated": (datetime.now() - timedelta(minutes=30)).isoformat(),
                "manufacturer": "August",
                "model": "Smart Lock Pro",
            },
            {
                "id": "device_005",
                "name": "Bedroom Fan",
                "type": "climate",
                "status": "off",
                "location": "bedroom",
                "speed": 0,
                "controllable": True,
                "last_updated": (datetime.now() - timedelta(hours=8)).isoformat(),
                "manufacturer": "Hunter",
                "model": "Smart Ceiling Fan",
            },
        ]

        # Calculate summary statistics
        total_devices = len(sample_devices)
        online_devices = len([d for d in sample_devices if d["status"] != "offline"])
        device_types = {}
        locations = {}

        for device in sample_devices:
            device_type = device["type"]
            location = device["location"]
            device_types[device_type] = device_types.get(device_type, 0) + 1
            locations[location] = locations.get(location, 0) + 1

        return {
            "devices": sample_devices,
            "total": total_devices,
            "online": online_devices,
            "offline": total_devices - online_devices,
            "device_types": device_types,
            "controllable_devices": len(
                [d for d in sample_devices if d.get("controllable", False)]
            ),
            "last_activity": max(
                [
                    datetime.fromisoformat(d["last_updated"].replace("Z", UTC_OFFSET))
                    for d in sample_devices
                ]
            ).isoformat(),
            "data_source": "fallback_data",
            "note": (
                f"Using sample device data. {reason}"
                if reason
                else "Using sample device data"
            ),
            "last_updated": datetime.now().isoformat(),
        }

    def _enhance_devices_data(self, devices: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance devices data with additional metadata"""
        enhanced = devices.copy()
        enhanced.update(
            {"last_updated": datetime.now().isoformat(), "data_source": "agent_data"}
        )
        return enhanced

    def get_user_context(
        self, user_id: str, token: str = Depends(get_token_from_state)
    ) -> Dict[str, Any]:
        """Get user context and preferences"""
        try:
            agent = self.get_agent(token)
            if not agent:
                return self._generate_fallback_user_context(user_id)

            if hasattr(agent, "get_user_context"):
                try:
                    context = agent.get_user_context(user_id)
                    if context:
                        return self._enhance_user_context(context)
                    else:
                        return self._generate_fallback_user_context(user_id)
                except Exception as e:
                    system_logger.warning(f"Agent user context retrieval failed: {e}")
                    return self._generate_fallback_user_context(user_id)
            else:
                return self._generate_fallback_user_context(user_id)

        except Exception as e:
            system_logger.error(
                f"Error getting user context: {str(e)}",
                additional_info={
                    "context": "User Context Retrieval",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return self._generate_fallback_user_context(user_id)

    def _generate_fallback_user_context(self, user_id: str) -> Dict[str, Any]:
        """Generate fallback user context data"""
        return {
            "user_id": user_id,
            "name": f"User {user_id}",
            "preferences": {
                "timezone": "UTC",
                "language": "en-US",
                "units": "imperial",
                "notification_preference": "push",
                "theme": "auto",
                "voice_response": True,
            },
            "interaction_history": {
                "total_conversations": 47,
                "last_interaction": (datetime.now() - timedelta(hours=3)).isoformat(),
                "most_used_features": ["weather", "reminders", "smart_home"],
                "avg_session_length": "4.5 minutes",
            },
            "smart_home_setup": {
                "total_devices": 5,
                "connected_devices": 4,
                "favorite_scenes": ["Good Morning", "Movie Night", "Sleep"],
                "most_controlled": "lighting",
            },
            "calendar_integration": {
                "connected": True,
                "default_calendar": "primary",
                "reminder_settings": "15_minutes_before",
            },
            "assistant_settings": {
                "proactive_suggestions": True,
                "learning_mode": True,
                "privacy_mode": False,
                "data_retention": "1_year",
            },
            "data_source": "fallback_context",
            "last_updated": datetime.now().isoformat(),
        }

    def _enhance_user_context(self, context: Any) -> Dict[str, Any]:
        """Enhance user context with additional metadata"""
        if hasattr(context, "__dict__"):
            enhanced = context.__dict__.copy()
        elif isinstance(context, dict):
            enhanced = context.copy()
        else:
            enhanced = {"raw_context": str(context)}

        enhanced.update(
            {"last_updated": datetime.now().isoformat(), "data_source": "agent_context"}
        )
        return enhanced

    def control_smart_device(
        self,
        user_id: str,
        device_id: str,
        action: str,
        parameters: Optional[Dict[str, Any]] = None,
        token: str = Depends(get_token_from_state),
    ) -> Dict[str, Any]:
        """Control smart home devices"""
        try:
            agent = self.get_agent(token)
            if not agent:
                return self._generate_device_control_fallback(
                    device_id, action, PVA_AGENT_NOT_AVAILABLE
                )
            if hasattr(agent, "control_smart_device"):
                try:
                    result = agent.control_smart_device(
                        user_id, device_id, action, parameters or {}
                    )
                    return self._enhance_control_result(result)
                except Exception as e:
                    system_logger.warning(f"Agent device control failed: {e}")
                    return self._generate_device_control_fallback(
                        device_id, action, METHOD_NOT_AVAILABLE
                    )
            else:
                return self._generate_device_control_fallback(
                    device_id, action, METHOD_NOT_AVAILABLE
                )
        except Exception as e:
            system_logger.error(
                f"Error controlling smart device: {str(e)}",
                additional_info={
                    "context": "Device Control",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return self._generate_device_control_fallback(device_id, action, str(e))

    def _generate_device_control_fallback(
        self, device_id: str, action: str, reason: str = ""
    ) -> Dict[str, Any]:
        """Generate fallback device control response"""
        action_responses = {
            "turn_on": f"Device {device_id} would be turned on",
            "turn_off": f"Device {device_id} would be turned off",
            "set_brightness": f"Device {device_id} brightness would be adjusted",
            "set_temperature": f"Device {device_id} temperature would be set",
            "play_music": f"Device {device_id} would start playing music",
            "stop_music": f"Device {device_id} would stop playing music",
            "lock": f"Device {device_id} would be locked",
            "unlock": f"Device {device_id} would be unlocked",
        }

        return {
            "device_id": device_id,
            "action": action,
            "status": "simulated",
            "message": action_responses.get(
                action, f"Device {device_id} action '{action}' would be executed"
            ),
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "data_source": "fallback_control",
            "note": (
                f"Device control simulation. {reason}"
                if reason
                else "Device control simulation"
            ),
        }

    def _enhance_control_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance device control result"""
        enhanced = result.copy()
        enhanced.update(
            {"timestamp": datetime.now().isoformat(), "data_source": "agent_control"}
        )
        return enhanced

    def create_reminder(
        self,
        user_id: str,
        reminder_data: Dict[str, Any],
        token: str = Depends(get_token_from_state),
    ) -> Dict[str, Any]:
        """Create a new reminder"""
        try:
            agent = self.get_agent(token)
            if not agent:
                return self._generate_reminder_creation_fallback(
                    reminder_data, PVA_AGENT_NOT_AVAILABLE
                )
            if not agent:
                return self._generate_reminder_creation_fallback(
                    reminder_data, "PVA agent not available"
                )

            if hasattr(agent, "create_reminder"):
                try:
                    result = agent.create_reminder(user_id, reminder_data)
                    return self._enhance_reminder_creation_result(result)
                except Exception as e:
                    system_logger.warning(f"Agent reminder creation failed: {e}")
                return self._generate_reminder_creation_fallback(
                    reminder_data, METHOD_NOT_AVAILABLE
                )

        except Exception as e:
            system_logger.error(
                f"Error creating reminder: {str(e)}",
                additional_info={
                    "context": "Reminder Creation",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return self._generate_reminder_creation_fallback(reminder_data, str(e))
        return self._generate_reminder_creation_fallback(reminder_data, UNKNOWN_ERROR)

    def _generate_reminder_creation_fallback(
        self, reminder_data: Dict[str, Any], reason: str = ""
    ) -> Dict[str, Any]:
        """Generate fallback reminder creation response"""
        reminder_id = f"fallback_reminder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return {
            "reminder_id": reminder_id,
            "title": reminder_data.get("title", "New Reminder"),
            "description": reminder_data.get("description", ""),
            "target_time": reminder_data.get(
                "target_time", (datetime.now() + timedelta(hours=1)).isoformat()
            ),
            "status": "created",
            "priority": reminder_data.get("priority", "medium"),
            "category": reminder_data.get("category", "general"),
            "recurring": reminder_data.get("recurring", False),
            "success": True,
            "message": "Reminder created successfully (simulated)",
            "created_at": datetime.now().isoformat(),
            "data_source": "fallback_creation",
            "note": (
                f"Reminder creation simulation. {reason}"
                if reason
                else "Reminder creation simulation"
            ),
        }

    def _enhance_reminder_creation_result(
        self, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance reminder creation result"""
        enhanced = result.copy()
        enhanced.update(
            {"created_at": datetime.now().isoformat(), "data_source": "agent_creation"}
        )
        return enhanced


# Create router
router = APIRouter(tags=["Personal Virtual Assistant"])


# Dependency to get PVA service
def get_pva_service() -> PersonalVirtualAssistantService:
    """Get PVA service instance"""
    from src.que_agents.api.main import agent_manager

    return PersonalVirtualAssistantService(agent_manager)


# Personal Virtual Assistant endpoints
@router.post("/chat", response_model=PVAResponse)
async def pva_chat(
    request: PVARequest,
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Personal Virtual Assistant Chat**

    Engage in natural conversation with your AI assistant for task management, smart home control, and general assistance.

    **Request Body:**
    ```json
    {
        "user_id": "user_12345",
        "message": "Turn on the living room lights and set them to 75% brightness",
        "session_id": "session_abc123"
    }
    ```

    **Response:**
    ```json
    {
        "response": "I've turned on the living room lights and set them to 75% brightness. Is there anything else you'd like me to adjust?",
        "intent": "smart_home_control",
        "entities": {
            "device": "living room lights",
            "action": "turn_on",
            "brightness": 75
        },
        "confidence": 0.95,
        "actions_taken": ["device_control", "brightness_adjustment"],
        "suggestions": [
            "Set a lighting schedule",
            "Adjust thermostat",
            "Check other smart devices"
        ],
        "timestamp": "2024-01-15T10:30:00Z"
    }
    ```

    **Supported Intents:**
    - 🏠 **Smart Home**: Device control, automation, scenes
    - 📅 **Calendar**: Scheduling, reminders, appointments
    - ☁️ **Weather**: Current conditions, forecasts, alerts
    - 📝 **Tasks**: To-do management, productivity
    - 📧 **Communication**: Email, messages, calls
    - 📰 **Information**: General queries, research

    **Features:**
    - 🤖 Natural language understanding
    - 🎯 Context-aware responses
    - 🔄 Multi-turn conversations
    - ⚡ Real-time action execution
    - 💡 Proactive suggestions
    - 📊 Confidence scoring
    - 🔍 Entity extraction

    **Smart Home Commands:**
    - "Turn on/off [device]"
    - "Set [device] to [value]"
    - "What's the temperature?"
    - "Lock the doors"
    - "Play music in [room]"

    **Status Codes:**
    - `200`: Response generated successfully
    - `400`: Invalid request format
    - `503`: PVA service temporarily unavailable
    """
    return service.handle_chat_request(request)


@router.get("/user/{user_id}/reminders")
async def get_user_reminders(
    user_id: str,
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Get User Reminders**

    Retrieve all active reminders and scheduled tasks for a specific user.

    **Path Parameters:**
    - `user_id` (str): Unique user identifier

    **Response:**
    ```json
    {
        "reminders": [
            {
                "id": "reminder_001",
                "title": "Daily standup meeting",
                "description": "Weekly team standup meeting",
                "target_time": "2024-01-15T14:00:00Z",
                "status": "active",
                "priority": "medium",
                "category": "work",
                "recurring": true,
                "created_at": "2024-01-08T10:00:00Z"
            }
        ],
        "total": 3,
        "active_count": 2,
        "categories": {
            "work": 2,
            "personal": 1
        },
        "upcoming_today": 1,
        "last_updated": "2024-01-15T10:30:00Z"
    }
    ```

    **Reminder Properties:**
    - 📝 **Title & Description**: Clear reminder details
    - ⏰ **Target Time**: When the reminder should trigger
    - 🏷️ **Priority**: Low, medium, high urgency levels
    - 📂 **Category**: Work, personal, health, etc.
    - 🔄 **Recurring**: One-time or repeating reminders
    - 🟢 **Status**: Active, completed, snoozed

    **Features:**
    - 📅 Today's upcoming reminders
    - 📈 Category-based organization
    - 🔄 Recurring reminder support
    - ⏰ Time-based sorting
    - 🏷️ Priority-based filtering
    - 📊 Summary statistics

    **Use Cases:**
    - Reminder management interfaces
    - Daily planning dashboards
    - Productivity tracking
    - Task organization systems

    **Status Codes:**
    - `200`: Reminders retrieved successfully
    - `404`: User not found
    - `500`: Reminder service unavailable
    """
    return service.get_user_reminders(user_id)


@router.get("/user/{user_id}/devices")
async def get_user_devices(
    user_id: str,
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Get Smart Home Devices**

    Retrieve all connected smart home devices with current status and control capabilities.

    **Path Parameters:**
    - `user_id` (str): Unique user identifier

    **Response:**
    ```json
    {
        "devices": [
            {
                "id": "device_001",
                "name": "Living Room Lights",
                "type": "lighting",
                "status": "on",
                "location": "living_room",
                "brightness": 75,
                "color": "warm_white",
                "controllable": true,
                "last_updated": "2024-01-15T10:15:00Z",
                "manufacturer": "Philips Hue",
                "model": "Smart Bulb"
            },
            {
                "id": "device_002",
                "name": "Smart Thermostat",
                "type": "climate",
                "status": "auto",
                "location": "hallway",
                "current_temp": 72.5,
                "target_temp": 73.0,
                "mode": "heat",
                "controllable": true,
                "last_updated": "2024-01-15T10:25:00Z",
                "manufacturer": "Nest",
                "model": "Learning Thermostat"
            }
        ],
        "total": 5,
        "online": 4,
        "offline": 1,
        "device_types": {
            "lighting": 2,
            "climate": 1,
            "security": 1,
            "audio": 1
        },
        "controllable_devices": 4,
        "last_activity": "2024-01-15T10:25:00Z",
        "last_updated": "2024-01-15T10:30:00Z"
    }
    ```

    **Device Types:**
    - 💡 **Lighting**: Smart bulbs, switches, dimmers
    - 🌡️ **Climate**: Thermostats, fans, HVAC systems
    - 🔒 **Security**: Locks, cameras, sensors
    - 🔊 **Audio**: Speakers, sound systems
    - 📺 **Entertainment**: TVs, streaming devices
    - 🏠 **Appliances**: Smart plugs, outlets

    **Device Properties:**
    - 🟢 **Status**: Current operational state
    - 📍 **Location**: Room or area placement
    - ⚙️ **Controls**: Available control options
    - 🔋 **Connectivity**: Online/offline status
    - 🏷️ **Metadata**: Manufacturer, model info
    - ⏰ **Activity**: Last update timestamp

    **Features:**
    - Real-time device status monitoring
    - Device type categorization
    - Location-based organization
    - Controllability indicators
    - Connectivity status tracking
    - Activity timeline

    **Use Cases:**
    - Smart home dashboards
    - Device control interfaces
    - Home automation setup
    - Energy monitoring systems

    **Status Codes:**
    - `200`: Devices retrieved successfully
    - `404`: User not found
    - `500`: Smart home service unavailable
    """
    return service.get_user_devices(user_id)


@router.get("/user/{user_id}/context")
async def get_user_context(
    user_id: str,
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Get User Context & Preferences**

    Retrieve comprehensive user profile, preferences, and interaction history for personalized assistance.

    **Path Parameters:**
    - `user_id` (str): Unique user identifier

    **Response:**
    ```json
    {
        "user_id": "user_12345",
        "name": "John Smith",
        "preferences": {
            "timezone": "America/New_York",
            "language": "en-US",
            "units": "imperial",
            "notification_preference": "push",
            "theme": "auto",
            "voice_response": true
        },
        "interaction_history": {
            "total_conversations": 47,
            "last_interaction": "2024-01-15T07:30:00Z",
            "most_used_features": ["weather", "reminders", "smart_home"],
            "avg_session_length": "4.5 minutes"
        },
        "smart_home_setup": {
            "total_devices": 5,
            "connected_devices": 4,
            "favorite_scenes": ["Good Morning", "Movie Night", "Sleep"],
            "most_controlled": "lighting"
        },
        "calendar_integration": {
            "connected": true,
            "default_calendar": "primary",
            "reminder_settings": "15_minutes_before"
        },
        "assistant_settings": {
            "proactive_suggestions": true,
            "learning_mode": true,
            "privacy_mode": false,
            "data_retention": "1_year"
        },
        "last_updated": "2024-01-15T10:30:00Z"
    }
    ```

    **Context Categories:**
    - 👤 **Profile**: Basic user information and identity
    - ⚙️ **Preferences**: Language, timezone, units, themes
    - 💬 **History**: Interaction patterns and usage statistics
    - 🏠 **Smart Home**: Device setup and automation preferences
    - 📅 **Calendar**: Integration status and settings
    - 🤖 **Assistant**: AI behavior and learning preferences

    **Personalization Features:**
    - Timezone-aware scheduling
    - Language and locale preferences
    - Measurement unit preferences
    - Notification delivery preferences
    - Theme and interface customization
    - Voice interaction settings

    **Privacy Controls:**
    - Data retention policies
    - Learning mode toggle
    - Privacy mode settings
    - Interaction history management

    **Use Cases:**
    - Personalized assistant responses
    - User interface customization
    - Proactive suggestion systems
    - Privacy compliance reporting

    **Status Codes:**
    - `200`: User context retrieved successfully
    - `404`: User not found
    - `500`: User service unavailable
    """
    return service.get_user_context(user_id)


class DeviceControlRequest(BaseModel):
    action: str
    parameters: Optional[Dict[str, Any]] = None


@router.post("/user/{user_id}/device/{device_id}/control")
async def control_smart_device(
    user_id: str,
    device_id: str,
    request: DeviceControlRequest,
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Control Smart Home Device**

    Execute control commands on smart home devices with real-time feedback and status updates.

    **Path Parameters:**
    - `user_id` (str): Unique user identifier
    - `device_id` (str): Unique device identifier

    **Request Body:**
    ```json
    {
        "action": "set_brightness",
        "parameters": {
            "brightness": 75,
            "transition_time": 2
        }
    }
    ```

    **Response:**
    ```json
    {
        "device_id": "device_001",
        "action": "set_brightness",
        "status": "success",
        "message": "Living room lights brightness set to 75%",
        "success": true,
        "timestamp": "2024-01-15T10:30:00Z",
        "data_source": "agent_control"
    }
    ```

    **Supported Actions:**

    **Lighting Controls:**
    - `turn_on` / `turn_off`: Basic power control
    - `set_brightness`: Adjust brightness (0-100)
    - `set_color`: Change color (RGB/HSV values)
    - `set_temperature`: Adjust color temperature

    **Climate Controls:**
    - `set_temperature`: Target temperature adjustment
    - `set_mode`: Heat, cool, auto, off modes
    - `set_fan_speed`: Fan speed control

    **Security Controls:**
    - `lock` / `unlock`: Door lock control
    - `arm` / `disarm`: Security system control

    **Audio Controls:**
    - `play_music` / `stop_music`: Media playback
    - `set_volume`: Volume adjustment (0-100)
    - `next_track` / `previous_track`: Track control

    **Common Parameters:**
    - `brightness`: 0-100 percentage
    - `temperature`: Degrees (F/C based on user preference)
    - `volume`: 0-100 percentage
    - `duration`: Time in seconds
    - `transition_time`: Smooth transition duration

    **Features:**
    - ⚡ Real-time device control
    - 🔄 Status confirmation feedback
    - 🎯 Parameter validation
    - 🛡️ Error handling and recovery
    - 📈 Action logging and history
    - 🔒 Security and permission checks

    **Status Codes:**
    - `200`: Device controlled successfully
    - `400`: Invalid action or parameters
    - `404`: Device or user not found
    - `403`: Insufficient permissions
    - `500`: Device control service unavailable
    """
    return service.control_smart_device(
        user_id, device_id, request.action, request.parameters
    )


@router.post("/user/{user_id}/reminder")
async def create_reminder(
    user_id: str,
    reminder_data: Dict[str, Any],
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Create User Reminder**

    Create a new reminder or scheduled task with customizable timing and recurrence options.

    **Path Parameters:**
    - `user_id` (str): Unique user identifier

    **Request Body:**
    ```json
    {
        "title": "Team meeting preparation",
        "description": "Prepare slides and agenda for weekly team meeting",
        "target_time": "2024-01-16T09:00:00Z",
        "priority": "high",
        "category": "work",
        "recurring": false,
        "notification_methods": ["push", "email"]
    }
    ```

    **Response:**
    ```json
    {
        "reminder_id": "reminder_2024_001",
        "title": "Team meeting preparation",
        "description": "Prepare slides and agenda for weekly team meeting",
        "target_time": "2024-01-16T09:00:00Z",
        "status": "created",
        "priority": "high",
        "category": "work",
        "recurring": false,
        "success": true,
        "message": "Reminder created successfully",
        "created_at": "2024-01-15T10:30:00Z"
    }
    ```

    **Reminder Properties:**
    - 📝 **Title**: Brief reminder description (required)
    - 📋 **Description**: Detailed reminder notes (optional)
    - ⏰ **Target Time**: When to trigger the reminder (required)
    - 🏷️ **Priority**: low, medium, high (default: medium)
    - 📂 **Category**: work, personal, health, etc. (default: general)
    - 🔄 **Recurring**: One-time or repeating (default: false)

    **Priority Levels:**
    - `low`: Non-urgent reminders, gentle notifications
    - `medium`: Standard reminders with normal notifications
    - `high`: Important reminders with prominent notifications

    **Categories:**
    - `work`: Professional tasks and meetings
    - `personal`: Personal appointments and tasks
    - `health`: Medical appointments, medication
    - `finance`: Bills, payments, financial tasks
    - `social`: Events, birthdays, social activities

    **Features:**
    - ⏰ Flexible scheduling with timezone support
    - 🔄 Recurring reminder patterns
    - 📢 Multiple notification methods
    - 🏷️ Priority-based organization
    - 📂 Category-based filtering
    - 📱 Cross-platform synchronization

    **Status Codes:**
    - `200`: Reminder created successfully
    - `400`: Invalid reminder data or timing
    - `404`: User not found
    - `500`: Reminder service unavailable
    """
    return service.create_reminder(user_id, reminder_data)


@router.get("/user/{user_id}/calendar")
async def get_user_calendar(
    user_id: str,
    days: int = 7,
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Get User Calendar Events**

    Retrieve upcoming calendar events and appointments with detailed scheduling information.

    **Path Parameters:**
    - `user_id` (str): Unique user identifier

    **Query Parameters:**
    - `days` (int): Number of days to look ahead (default: 7, max: 30)

    **Response:**
    ```json
    {
        "events": [
            {
                "id": "event_001",
                "title": "Team Meeting",
                "start_time": "2024-01-15T14:00:00Z",
                "end_time": "2024-01-15T15:00:00Z",
                "location": "Conference Room A",
                "attendees": ["john@company.com", "sarah@company.com"],
                "description": "Weekly team sync and project updates",
                "status": "confirmed",
                "meeting_type": "in_person",
                "priority": "medium"
            },
            {
                "id": "event_002",
                "title": "Client Presentation",
                "start_time": "2024-01-16T10:00:00Z",
                "end_time": "2024-01-16T11:30:00Z",
                "location": "Virtual - Zoom",
                "attendees": ["client@company.com"],
                "description": "Q1 project proposal presentation",
                "status": "confirmed",
                "meeting_type": "virtual",
                "priority": "high"
            }
        ],
        "total": 5,
        "upcoming_today": 1,
        "data_source": "calendar_integration",
        "last_updated": "2024-01-15T10:30:00Z"
    }
    ```

    **Event Properties:**
    - 📝 **Title & Description**: Event details and notes
    - ⏰ **Timing**: Start and end times with timezone
    - 📍 **Location**: Physical or virtual meeting location
    - 👥 **Attendees**: Participant email addresses
    - 🟢 **Status**: Confirmed, tentative, cancelled
    - 📹 **Type**: In-person, virtual, hybrid meetings
    - 🏷️ **Priority**: Event importance level

    **Meeting Types:**
    - `in_person`: Physical location meetings
    - `virtual`: Online meetings (Zoom, Teams, etc.)
    - `hybrid`: Mixed in-person and virtual
    - `phone`: Audio-only conferences

    **Features:**
    - 📅 Multi-day event retrieval
    - ⏰ Today's events highlighting
    - 👥 Attendee information
    - 📍 Location and meeting details
    - 🔄 Real-time calendar synchronization
    - 🏷️ Priority-based organization

    **Integration Support:**
    - Google Calendar
    - Microsoft Outlook
    - Apple Calendar
    - Exchange Server
    - CalDAV protocols

    **Use Cases:**
    - Daily planning and scheduling
    - Meeting preparation assistance
    - Calendar conflict detection
    - Automated meeting reminders

    **Status Codes:**
    - `200`: Calendar events retrieved successfully
    - `400`: Invalid date range or parameters
    - `404`: User not found or calendar not connected
    - `500`: Calendar service unavailable
    """
    try:
        agent = service.get_agent(token)
        if agent and hasattr(agent, "get_user_calendar"):
            calendar_data = agent.get_user_calendar(user_id, days)
            return calendar_data

        # Fallback calendar data
        sample_events = [
            {
                "id": "event_001",
                "title": "Team Meeting",
                "start_time": (datetime.now() + timedelta(hours=2)).isoformat(),
                "end_time": (datetime.now() + timedelta(hours=3)).isoformat(),
                "location": "Conference Room A",
                "attendees": ["john@company.com", "sarah@company.com"],
                "description": "Weekly team sync",
                "status": "confirmed",
            },
            {
                "id": "event_002",
                "title": "Lunch with Client",
                "start_time": (datetime.now() + timedelta(days=1, hours=4)).isoformat(),
                "end_time": (datetime.now() + timedelta(days=1, hours=5)).isoformat(),
                "location": "Downtown Restaurant",
                "attendees": ["client@company.com"],
                "description": "Business lunch discussion",
                "status": "confirmed",
            },
            {
                "id": "event_003",
                "title": "Doctor Appointment",
                "start_time": (datetime.now() + timedelta(days=2, hours=3)).isoformat(),
                "end_time": (datetime.now() + timedelta(days=2, hours=4)).isoformat(),
                "location": "Medical Center",
                "attendees": [],
                "description": "Annual checkup",
                "status": "confirmed",
            },
        ]

        return {
            "events": sample_events,
            "total": len(sample_events),
            "upcoming_today": len(
                [
                    e
                    for e in sample_events
                    if datetime.fromisoformat(
                        e["start_time"].replace("Z", UTC_OFFSET)
                    ).date()
                    == datetime.now().date()
                ]
            ),
            "data_source": "fallback_calendar",
            "last_updated": datetime.now().isoformat(),
        }

    except Exception as e:
        system_logger.error(
            f"Error getting user calendar: {str(e)}",
            additional_info={
                "context": "User Calendar Retrieval",
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        return {
            "events": [],
            "error": "Calendar temporarily unavailable",
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/user/{user_id}/weather")
async def get_weather_info(
    user_id: str,
    location: Optional[str] = None,
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Get Weather Information**

    Retrieve current weather conditions and forecast data for the user's location or specified area.

    **Path Parameters:**
    - `user_id` (str): Unique user identifier

    **Query Parameters:**
    - `location` (optional): Specific location override (city, zip code, coordinates)

    **Response:**
    ```json
    {
        "location": "New York, NY",
        "current": {
            "temperature": 72,
            "condition": "Partly Cloudy",
            "humidity": 65,
            "wind_speed": 8,
            "wind_direction": "NW",
            "visibility": 10,
            "uv_index": 6,
            "feels_like": 75,
            "pressure": 30.15,
            "dew_point": 58
        },
        "forecast": [
            {
                "date": "2024-01-15",
                "high": 78,
                "low": 65,
                "condition": "Partly Cloudy",
                "precipitation_chance": 20,
                "wind_speed": 10,
                "humidity": 60
            },
            {
                "date": "2024-01-16",
                "high": 80,
                "low": 68,
                "condition": "Sunny",
                "precipitation_chance": 10,
                "wind_speed": 8,
                "humidity": 55
            }
        ],
        "alerts": [],
        "sunrise": "06:45",
        "sunset": "19:30",
        "last_updated": "2024-01-15T10:30:00Z"
    }
    ```

    **Current Conditions:**
    - 🌡️ **Temperature**: Current temperature and feels-like
    - ☁️ **Condition**: Weather description (sunny, cloudy, rainy, etc.)
    - 💧 **Humidity**: Relative humidity percentage
    - 💨 **Wind**: Speed and direction information
    - 👁️ **Visibility**: Visibility distance
    - ☀️ **UV Index**: Sun exposure risk level
    - 🌡️ **Pressure**: Atmospheric pressure

    **Forecast Data:**
    - 📅 **Multi-day**: 3-7 day weather outlook
    - 🌡️ **Temperature Range**: Daily high and low temperatures
    - ☔ **Precipitation**: Chance of rain/snow
    - 💨 **Wind Conditions**: Expected wind patterns
    - 💧 **Humidity Levels**: Daily humidity ranges

    **Weather Alerts:**
    - ⚠️ Severe weather warnings
    - 🌪️ Storm and tornado watches
    - 🌨️ Temperature extremes
    - 💧 Flood and precipitation alerts

    **Features:**
    - 📍 Location-based weather data
    - ⏰ Real-time condition updates
    - 📅 Extended forecast periods
    - ⚠️ Weather alert notifications
    - 🌅 Sunrise and sunset times
    - 🌡️ Temperature unit preferences

    **Location Support:**
    - User's default location
    - City names ("New York, NY")
    - ZIP/postal codes
    - GPS coordinates
    - Airport codes

    **Use Cases:**
    - Daily weather briefings
    - Travel planning assistance
    - Outdoor activity recommendations
    - Clothing and preparation suggestions

    **Status Codes:**
    - `200`: Weather data retrieved successfully
    - `400`: Invalid location format
    - `404`: Location not found
    - `500`: Weather service unavailable
    """
    try:
        agent = service.get_agent(token)
        if agent and hasattr(agent, "get_weather_info"):
            weather_data = agent.get_weather_info(user_id, location)
            return weather_data

        # Fallback weather data
        return {
            "location": location or "Current Location",
            "current": {
                "temperature": 72,
                "condition": "Partly Cloudy",
                "humidity": 65,
                "wind_speed": 8,
                "wind_direction": "NW",
                "visibility": 10,
                "uv_index": 6,
                "feels_like": 75,
            },
            "forecast": [
                {
                    "date": datetime.now().date().isoformat(),
                    "high": 78,
                    "low": 65,
                    "condition": "Partly Cloudy",
                    "precipitation_chance": 20,
                },
                {
                    "date": (datetime.now() + timedelta(days=1)).date().isoformat(),
                    "high": 80,
                    "low": 68,
                    "condition": "Sunny",
                    "precipitation_chance": 10,
                },
                {
                    "date": (datetime.now() + timedelta(days=2)).date().isoformat(),
                    "high": 75,
                    "low": 62,
                    "condition": "Light Rain",
                    "precipitation_chance": 70,
                },
            ],
            "alerts": [],
            "data_source": "fallback_weather",
            "last_updated": datetime.now().isoformat(),
        }

    except Exception as e:
        system_logger.error(
            f"Error getting weather info: {str(e)}",
            additional_info={
                "context": "Weather Information Retrieval",
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        return {
            "error": "Weather service temporarily unavailable",
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/user/{user_id}/tasks")
async def get_user_tasks(
    user_id: str,
    status: Optional[str] = None,
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Get User Tasks & To-Do Items**

    Retrieve user's task list with progress tracking, priority management, and status filtering.

    **Path Parameters:**
    - `user_id` (str): Unique user identifier

    **Query Parameters:**
    - `status` (optional): Filter by task status (pending, in_progress, completed)

    **Response:**
    ```json
    {
        "tasks": [
            {
                "id": "task_001",
                "title": "Complete project proposal",
                "description": "Finish the Q4 project proposal document",
                "status": "in_progress",
                "priority": "high",
                "due_date": "2024-01-17T17:00:00Z",
                "category": "work",
                "progress": 75,
                "created_at": "2024-01-10T09:00:00Z",
                "estimated_duration": "4 hours",
                "tags": ["urgent", "client-work"]
            },
            {
                "id": "task_002",
                "title": "Buy groceries",
                "description": "Weekly grocery shopping",
                "status": "pending",
                "priority": "medium",
                "due_date": "2024-01-16T18:00:00Z",
                "category": "personal",
                "progress": 0,
                "created_at": "2024-01-14T10:00:00Z",
                "subtasks": [
                    "Milk and eggs",
                    "Fresh vegetables",
                    "Cleaning supplies"
                ]
            }
        ],
        "total": 8,
        "status_filter": null,
        "summary": {
            "pending": 3,
            "in_progress": 2,
            "completed": 3
        },
        "overdue_count": 1,
        "due_today": 2,
        "last_updated": "2024-01-15T10:30:00Z"
    }
    ```

    **Task Properties:**
    - 📝 **Title & Description**: Task details and notes
    - 🟢 **Status**: pending, in_progress, completed
    - 🏷️ **Priority**: low, medium, high, urgent
    - 📅 **Due Date**: Task deadline with timezone
    - 📂 **Category**: work, personal, health, etc.
    - 📈 **Progress**: Completion percentage (0-100)
    - 🏷️ **Tags**: Custom labels and keywords
    - ⏱️ **Duration**: Estimated time to complete

    **Task Statuses:**
    - `pending`: Not yet started
    - `in_progress`: Currently being worked on
    - `completed`: Finished tasks
    - `on_hold`: Temporarily paused
    - `cancelled`: Abandoned tasks

    **Priority Levels:**
    - `low`: Nice-to-have tasks
    - `medium`: Standard priority
    - `high`: Important tasks
    - `urgent`: Critical, time-sensitive tasks

    **Features:**
    - 🔍 Status-based filtering
    - 📈 Progress tracking and visualization
    - 📅 Due date management and alerts
    - 🏷️ Priority-based organization
    - 📂 Category-based grouping
    - 🔄 Subtask support and breakdown
    - 🏷️ Custom tagging system

    **Task Categories:**
    - `work`: Professional tasks and projects
    - `personal`: Personal errands and activities
    - `health`: Medical appointments, fitness goals
    - `finance`: Bills, budgeting, financial tasks
    - `learning`: Education, skill development
    - `home`: Household chores and maintenance

    **Use Cases:**
    - Personal productivity management
    - Project task tracking
    - Daily planning and organization
    - Goal setting and achievement

    **Status Codes:**
    - `200`: Tasks retrieved successfully
    - `400`: Invalid status filter
    - `404`: User not found
    - `500`: Task service unavailable
    """
    try:
        agent = service.get_agent(token)
        if agent and hasattr(agent, "get_user_tasks"):
            tasks_data = agent.get_user_tasks(user_id, status)
            return tasks_data

        # Fallback tasks data
        sample_tasks = [
            {
                "id": "task_001",
                "title": "Complete project proposal",
                "description": "Finish the Q4 project proposal document",
                "status": "in_progress",
                "priority": "high",
                "due_date": (datetime.now() + timedelta(days=2)).isoformat(),
                "category": "work",
                "progress": 75,
                "created_at": (datetime.now() - timedelta(days=5)).isoformat(),
            },
            {
                "id": "task_002",
                "title": "Buy groceries",
                "description": "Weekly grocery shopping",
                "status": "pending",
                "priority": "medium",
                "due_date": (datetime.now() + timedelta(days=1)).isoformat(),
                "category": "personal",
                "progress": 0,
                "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
            },
            {
                "id": "task_003",
                "title": "Review team feedback",
                "description": "Review and respond to team feedback from last sprint",
                "status": "completed",
                "priority": "medium",
                "due_date": datetime.now().isoformat(),
                "category": "work",
                "progress": 100,
                "created_at": (datetime.now() - timedelta(days=7)).isoformat(),
                "completed_at": (datetime.now() - timedelta(hours=2)).isoformat(),
            },
        ]

        if status:
            sample_tasks = [task for task in sample_tasks if task["status"] == status]

        return {
            "tasks": sample_tasks,
            "total": len(sample_tasks),
            "status_filter": status,
            "summary": {
                "pending": len([t for t in sample_tasks if t["status"] == "pending"]),
                "in_progress": len(
                    [t for t in sample_tasks if t["status"] == "in_progress"]
                ),
                "completed": len(
                    [t for t in sample_tasks if t["status"] == "completed"]
                ),
            },
            "data_source": "fallback_tasks",
            "last_updated": datetime.now().isoformat(),
        }

    except Exception as e:
        system_logger.error(
            f"Error getting user tasks: {str(e)}",
            additional_info={
                "context": "User Tasks Retrieval",
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        return {
            "tasks": [],
            "error": "Tasks service temporarily unavailable",
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/capabilities")
async def get_pva_capabilities(
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """
    **Get PVA Capabilities & Features**

    Retrieve comprehensive information about Personal Virtual Assistant capabilities, supported features, and integrations.

    **Response:**
    ```json
    {
        "core_features": [
            "Natural language conversation",
            "Smart home device control",
            "Calendar and scheduling management",
            "Reminder and task management",
            "Weather information",
            "Email and message assistance",
            "General information queries"
        ],
        "smart_home_support": [
            "Lighting control",
            "Climate control",
            "Audio/media devices",
            "Security systems",
            "Appliance control"
        ],
        "integrations": [
            "Google Calendar",
            "Microsoft Outlook",
            "Weather services",
            "Smart home platforms",
            "Task management apps"
        ],
        "languages": ["English (US)", "English (UK)"],
        "availability": "24/7",
        "response_modes": ["Text", "Voice (when available)"],
        "supported_platforms": [
            "Web browsers",
            "Mobile apps",
            "Smart speakers",
            "Desktop applications"
        ],
        "ai_capabilities": {
            "natural_language_understanding": true,
            "context_awareness": true,
            "learning_adaptation": true,
            "multi_turn_conversations": true,
            "intent_recognition": true,
            "entity_extraction": true
        },
        "version": "2.0.0",
        "last_updated": "2024-01-15T10:30:00Z"
    }
    ```

    **Core Capabilities:**
    - 🤖 **Conversational AI**: Natural language understanding and generation
    - 🏠 **Smart Home**: Device control and automation
    - 📅 **Scheduling**: Calendar and appointment management
    - ⏰ **Reminders**: Task and reminder creation
    - ☁️ **Weather**: Current conditions and forecasts
    - 📧 **Communication**: Email and messaging assistance
    - 📰 **Information**: General knowledge and research

    **Smart Home Features:**
    - 💡 Lighting systems (Philips Hue, LIFX, etc.)
    - 🌡️ Climate control (Nest, Ecobee, etc.)
    - 🔊 Audio devices (Sonos, Alexa, etc.)
    - 🔒 Security systems (locks, cameras, alarms)
    - 🏠 Appliances (smart plugs, switches)

    **Integration Ecosystem:**
    - 📅 Calendar platforms (Google, Outlook, Apple)
    - ☁️ Weather services (OpenWeather, AccuWeather)
    - 🏠 Smart home hubs (SmartThings, Hubitat)
    - 📝 Task managers (Todoist, Any.do)
    - 📧 Email services (Gmail, Outlook)

    **AI Features:**
    - Context-aware conversations
    - Intent recognition and classification
    - Entity extraction from natural language
    - Multi-turn dialogue management
    - Personalized learning and adaptation
    - Proactive suggestions and recommendations

    **Platform Support:**
    - Web-based interfaces
    - Mobile applications (iOS, Android)
    - Smart speakers and displays
    - Desktop applications
    - API integrations

    **Use Cases:**
    - Feature discovery and exploration
    - Integration planning and setup
    - Capability assessment for developers
    - User onboarding and education

    **Status Codes:**
    - `200`: Capabilities retrieved successfully
    - `500`: Capabilities service temporarily unavailable
    """
    try:
        agent = service.get_agent(token)
        if agent and hasattr(agent, "get_capabilities"):
            return agent.get_capabilities()

        # Fallback capabilities
        return {
            "core_features": [
                "Natural language conversation",
                "Smart home device control",
                "Calendar and scheduling management",
                "Reminder and task management",
                "Weather information",
                "Email and message assistance",
                "General information queries",
            ],
            "smart_home_support": [
                "Lighting control",
                "Climate control",
                "Audio/media devices",
                "Security systems",
                "Appliance control",
            ],
            "integrations": [
                "Calendar apps",
                "Email services",
                "Weather services",
                "Smart home platforms",
                "Task management apps",
            ],
            "languages": ["English (US)", "English (UK)"],
            "availability": "24/7",
            "response_modes": ["Text", "Voice (when available)"],
            "data_source": "fallback_capabilities",
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
        }

    except Exception as e:
        system_logger.error(
            f"Error getting PVA capabilities: {str(e)}",
            additional_info={
                "context": "PVA Capabilities Retrieval",
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        return {
            "error": "Capabilities service temporarily unavailable",
            "timestamp": datetime.now().isoformat(),
        }
