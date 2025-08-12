# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-10
# @Description: Personal Virtual Assistant API routes and handlers

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Body
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
    """Handle Personal Virtual Assistant chat request with enhanced fallback"""
    return service.handle_chat_request(request)


@router.get("/user/{user_id}/reminders")
async def get_user_reminders(
    user_id: str,
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """Get user reminders with comprehensive fallback data"""
    return service.get_user_reminders(user_id)


@router.get("/user/{user_id}/devices")
async def get_user_devices(
    user_id: str,
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """Get user smart devices with detailed fallback data"""
    return service.get_user_devices(user_id)


@router.get("/user/{user_id}/context")
async def get_user_context(
    user_id: str,
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """Get user context and preferences"""
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
    """Control smart home devices"""
    return service.control_smart_device(user_id, device_id, request.action, request.parameters)


@router.post("/user/{user_id}/reminder")
async def create_reminder(
    user_id: str,
    reminder_data: Dict[str, Any],
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """Create a new reminder for the user"""
    return service.create_reminder(user_id, reminder_data)


@router.get("/user/{user_id}/calendar")
async def get_user_calendar(
    user_id: str,
    days: int = 7,
    service: PersonalVirtualAssistantService = Depends(get_pva_service),
    token: str = Depends(get_token_from_state),
):
    """Get user calendar events"""
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
    """Get weather information for user"""
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
    """Get user tasks and to-do items"""
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
    """Get PVA capabilities and features"""
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
