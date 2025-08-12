# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-05 12:00:00
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-05 12:00:00
# @Description: This module implements a Personal Virtual Assistant agent using LangChain and SQLAlchemy

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yaml
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.que_agents.core.database import (
    PVAInteraction,
    Reminder,
    SmartDevice,
    UserPreferences,
    get_session,
)
from src.que_agents.core.llm_factory import LLMFactory
from src.que_agents.core.pva_schemas import PVAAgentResponse
from src.que_agents.core.schemas import IntentResult, UserContext
from src.que_agents.error_trace.errorlogger import system_logger
from src.que_agents.knowledge_base.kb_manager import (
    search_agent_knowledge_base,
    search_knowledge_base,
)

system_logger.info("Personal Virtual Assistant Agent initialized...")

TURN_ON = "turn on"
TURN_OFF = "turn off"
REMIND_ME_TO = "remind me to"
SET_A_REMINDER = "set a reminder"
DEFAULT_LOCATION = "New York"

# Load agent configuration
with open("./configs/agent_config.yaml", "r") as f:
    agent_config = yaml.safe_load(f)


class PersonalVirtualAssistantAgent:
    """Personal Virtual Assistant Agent using LangChain"""

    def __init__(self):
        # Try different config key names
        config_key = "personal_virtual_assistant"
        if config_key not in agent_config:
            config_key = "personal_virtual_assistant"

        config = agent_config[config_key]
        self.llm = LLMFactory.get_llm(
            agent_type="personal_virtual_assistant",
            model_name=config["model_name"],
            temperature=config["temperature"],
            max_tokens=config.get("max_tokens", 600),
        )

        # Memory for conversation history
        self._session_histories: Dict[str, ChatMessageHistory] = {}

        # Supported intents
        self.supported_intents = [
            "weather",
            "set_reminder",
            "list_reminders",
            "device_control",
            "general_query",
            "recommendation",
            "news",
            "time_date",
            "cancel_reminder",
            "greeting",
            "goodbye",
            "smart_home_help",
            "productivity_tips",
        ]

        # Initialize prompt templates
        self.intent_prompt = self._create_intent_prompt()
        self.response_prompt = self._create_response_prompt()
        self.entity_prompt = self._create_entity_prompt()

        # Create chains
        self.intent_chain = self._create_intent_chain()
        self.entity_chain = self._create_entity_chain()

        # Base response chain
        base_response_chain = self._create_response_chain_base()

        # Wrap with RunnableWithMessageHistory to manage chat history per session
        self.response_chain = RunnableWithMessageHistory(
            base_response_chain,
            get_session_history=self._get_session_history,  # called with config
            history_messages_key="history",  # must match MessagesPlaceholder key
            input_messages_key="user_message",  # which input field is the latest human message
        )

    def _get_session_history(self, config: Dict[str, Any]) -> ChatMessageHistory:
        """Retrieve or create ChatMessageHistory for a given session_id.

        Expects config={"configurable": {"session_id": "<id>"}} as per LangChain convention.
        """
        session_id = (
            config.get("configurable", {}).get("session_id") or "default_session"
        )
        history = self._session_histories.get(session_id)
        if history is None:
            history = ChatMessageHistory()
            self._session_histories[session_id] = history
        return history

    def get_assistant_knowledge(self, query: str) -> List[Dict]:
        """Get personal assistant knowledge from knowledge base"""
        try:
            return search_agent_knowledge_base(
                "personal_virtual_assistant", query, limit=3
            )
        except Exception as e:
            system_logger.error(
                f"Error searching assistant knowledge: {e}",
                additional_info={"query": query},
                exc_info=True,
            )
            return []

    def get_enhanced_context(self, user_message: str, intent: str) -> str:
        """Get enhanced context from knowledge base"""
        try:
            # Search for relevant knowledge based on intent and message
            knowledge_results = self.get_assistant_knowledge(f"{intent} {user_message}")

            if knowledge_results:
                context = "Relevant Knowledge:\n"
                for kb_item in knowledge_results:
                    context += f"- {kb_item['title']}: {kb_item['content'][:150]}...\n"
                return context

            return ""
        except Exception as e:
            system_logger.error(
                f"Error getting enhanced context: {e}",
                additional_info={"user_message": user_message, "intent": intent},
                exc_info=True,
            )
            return ""

    def _create_intent_prompt(self) -> ChatPromptTemplate:
        system_message = f"""You are an intent recognition expert for a Personal Virtual Assistant. Your role is to identify the user's intent from their message.

SUPPORTED INTENTS:
{', '.join(self.supported_intents)}

INTENT DEFINITIONS:
- weather: User wants weather information
- set_reminder: User wants to set a reminder or schedule something
- list_reminders: User wants to see their reminders
- device_control: User wants to control smart devices (lights, thermostat, etc.)
- general_query: General questions or information requests
- recommendation: User wants recommendations (restaurants, movies, etc.)
- news: User wants news or current events
- time_date: User wants current time or date
- cancel_reminder: User wants to cancel or delete a reminder
- greeting: User is greeting or starting conversation
- goodbye: User is ending conversation
- smart_home_help: User needs help with smart home setup or troubleshooting
- productivity_tips: User wants productivity advice or tips

User message: {{user_message}}

Respond with ONLY the intent name from the supported intents list. If uncertain, use 'general_query'."""
        return ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", "{user_message}")]
        )

    def _create_entity_prompt(self) -> ChatPromptTemplate:
        system_message = """You are an entity extraction expert. Extract relevant entities from the user message based on the identified intent.

ENTITY TYPES TO EXTRACT:
- location: City, state, country for weather queries
- datetime: Date and time information for reminders
- device_name: Name of smart device to control
- device_action: Action to perform on device (turn on/off, set temperature, etc.)
- reminder_title: Title or description of reminder
- query_topic: Main topic of general queries
- recommendation_type: Type of recommendation needed (restaurant, movie, etc.)
- temperature: Temperature value for thermostat control
- brightness: Brightness level for lights
- color: Color for smart lights

User message: {user_message}
Intent: {intent}

Extract entities as a JSON object. Only include entities that are clearly present in the message. Use null for missing entities.

Example format:
{{"location": "New York", "datetime": "tomorrow 3pm", "device_name": "living room lights", "device_action": "turn on"}}"""
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "Extract entities from: {user_message}"),
            ]
        )

    def _create_response_prompt(self) -> ChatPromptTemplate:
        system_message = """You are a helpful and friendly Personal Virtual Assistant. Your role is to:

1. Provide helpful responses to user requests
2. Be conversational and personable
3. Offer relevant suggestions when appropriate
4. Acknowledge when you've taken actions (like setting reminders)
5. Be honest about limitations
6. Use knowledge base information when available

RESPONSE GUIDELINES:
- Keep responses concise but friendly
- Use natural, conversational language
- Acknowledge the user's request
- Provide relevant information or confirm actions taken
- Offer helpful suggestions when appropriate
- If you can't fulfill a request, explain why and suggest alternatives
- Use any enhanced context to provide better responses

User Context: {user_context}
Intent: {intent}
Extracted Entities: {entities}
Actions Taken: {actions_taken}
Additional Information: {additional_info}
Enhanced Context: {enhanced_context}

User message: {user_message}

Provide a helpful and friendly response that addresses the user's request."""
        # Use variable_name="history" for new LangChain memory API
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{user_message}"),
            ]
        )

    def _create_intent_chain(self):
        """Create intent recognition chain"""
        return self.intent_prompt | self.llm | StrOutputParser()

    def _create_entity_chain(self):
        """Create entity extraction chain"""
        return self.entity_prompt | self.llm | StrOutputParser()

    def _create_response_chain_base(self):
        """Create base response generation chain"""
        return self.response_prompt | self.llm | StrOutputParser()

    def _create_response_chain(self):
        """Create response generation chain"""
        return self.response_prompt | self.llm | StrOutputParser()

    def get_user_context(self, user_id: str) -> Optional[UserContext]:
        """Retrieve user context from database"""
        session = get_session()
        try:
            user_prefs = (
                session.query(UserPreferences)
                .filter(UserPreferences.user_id == user_id)
                .first()
            )

            if not user_prefs:
                user_prefs = self._create_default_user_preferences(session, user_id)

            active_reminders = (
                session.query(Reminder)
                .filter(Reminder.user_id == user_id, Reminder.status == "active")
                .all()
            )

            smart_devices = (
                session.query(SmartDevice).filter(SmartDevice.user_id == user_id).all()
            )

            if not smart_devices:
                smart_devices = self._create_default_smart_devices(session, user_id)

            preferences = self._extract_json_field(user_prefs, "preferences")
            learned_behaviors = self._extract_json_field(
                user_prefs, "learned_behaviors"
            )

            return UserContext(
                user_id=user_id,
                preferences=preferences,
                learned_behaviors=learned_behaviors,
                active_reminders=[
                    {
                        "id": r.id,
                        "title": r.title,
                        "description": r.description,
                        "reminder_time": (
                            r.reminder_time.isoformat()
                            if r.reminder_time is not None
                            else None
                        ),
                        "is_recurring": r.is_recurring,
                        "recurrence_pattern": r.recurrence_pattern,
                    }
                    for r in active_reminders
                ],
                smart_devices=[
                    {
                        "id": d.id,
                        "name": d.device_name,
                        "type": d.device_type,
                        "location": d.location,
                        "state": d.current_state,
                        "capabilities": d.capabilities,
                        "is_online": d.is_online,
                    }
                    for d in smart_devices
                ],
            )
        finally:
            session.close()

    def _create_default_user_preferences(
        self, session, user_id: str
    ) -> UserPreferences:
        """Create and persist default user preferences"""
        user_prefs = UserPreferences(
            user_id=user_id,
            preferences={
                "location": DEFAULT_LOCATION,
                "timezone": "UTC",
                "temperature_unit": "fahrenheit",
                "preferred_language": "english",
            },
            learned_behaviors={},
        )
        session.add(user_prefs)
        session.commit()
        return user_prefs

    def _create_default_smart_devices(self, session, user_id: str) -> List[SmartDevice]:
        """Create and persist default smart devices for the user"""
        default_devices = [
            SmartDevice(
                user_id=user_id,
                device_name="Living Room Lights",
                device_type="light",
                location="living room",
                current_state={"power": "off", "brightness": 50},
                capabilities=["power", "brightness", "color"],
                is_online=True,
            ),
            SmartDevice(
                user_id=user_id,
                device_name="Thermostat",
                device_type="thermostat",
                location="main",
                current_state={"temperature": 72, "mode": "auto"},
                capabilities=["temperature", "mode"],
                is_online=True,
            ),
            SmartDevice(
                user_id=user_id,
                device_name="Bedroom Lights",
                device_type="light",
                location="bedroom",
                current_state={"power": "off", "brightness": 30},
                capabilities=["power", "brightness"],
                is_online=True,
            ),
        ]
        for device in default_devices:
            session.add(device)
        session.commit()
        return default_devices

    def _extract_json_field(self, obj, field: str) -> dict:
        """Safely extract a JSON/dict field from an object"""
        value = getattr(obj, field, None)
        if value is not None:
            if isinstance(value, dict):
                return value
            else:
                try:
                    return json.loads(value)
                except Exception:
                    return {}
        return {}

    def recognize_intent(self, user_message: str) -> IntentResult:
        """Recognize user intent from message"""
        try:
            # First try rule-based recognition for common patterns
            rule_based_intent = self._rule_based_intent_recognition(user_message)
            if rule_based_intent:
                confidence = self._calculate_intent_confidence(
                    user_message, rule_based_intent
                )
                return IntentResult(
                    intent=rule_based_intent, confidence=confidence, entities={}
                )

            # Fallback to LLM-based recognition
            intent = (
                self.intent_chain.invoke({"user_message": user_message}).strip().lower()
            )

            # Validate intent
            if intent not in self.supported_intents:
                intent = "general_query"

            # Enhanced confidence scoring
            confidence = self._calculate_intent_confidence(user_message, intent)

            return IntentResult(intent=intent, confidence=confidence, entities={})
        except Exception as e:
            system_logger.error(
                f"Error recognizing intent: {e}",
                additional_info={"user_message": user_message},
                exc_info=True,
            )
            return IntentResult(intent="general_query", confidence=0.5, entities={})

    def _calculate_intent_confidence(self, user_message: str, intent: str) -> float:
        """Calculate confidence score for intent recognition"""
        message_lower = user_message.lower()

        # Define intent keywords
        intent_keywords = {
            "weather": [
                "weather",
                "temperature",
                "rain",
                "sunny",
                "cloudy",
                "forecast",
            ],
            "set_reminder": [
                "remind",
                "reminder",
                "schedule",
                "appointment",
                "meeting",
                SET_A_REMINDER,
                "call",
                "tomorrow",
                "at",
            ],
            "list_reminders": [
                "reminders",
                "what's scheduled",
                "appointments",
                "my schedule",
            ],
            "device_control": [
                TURN_ON,
                TURN_OFF,
                "lights",
                "thermostat",
                "dim",
                "brighten",
            ],
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
            "goodbye": ["bye", "goodbye", "see you", "talk later"],
            "time_date": ["time", "date", "what time", "what day"],
            "recommendation": ["recommend", "suggest", "what's good", "where should"],
        }

        keywords = intent_keywords.get(intent, [])
        matches = sum(1 for keyword in keywords if keyword in message_lower)

        # Special handling for set_reminder intent
        if intent == "set_reminder":
            # Check for common reminder patterns
            reminder_patterns = [
                "remind me to",
                "set a reminder",
                "tomorrow at",
                "call",
                "meeting",
                "appointment",
            ]
            pattern_matches = sum(
                1 for pattern in reminder_patterns if pattern in message_lower
            )
            if pattern_matches > 0:
                return min(0.9, 0.7 + (pattern_matches * 0.1))

        if matches > 0:
            return min(0.9, 0.6 + (matches * 0.1))
        else:
            return 0.7  # Increased default confidence

    def extract_entities(self, user_message: str, intent: str) -> Dict[str, Any]:
        """Extract entities from user message"""
        try:
            entities_str = self.entity_chain.invoke(
                {"user_message": user_message, "intent": intent}
            )

            # Try to parse as JSON
            try:
                entities = json.loads(entities_str)
                return entities if isinstance(entities, dict) else {}
            except json.JSONDecodeError:
                system_logger.warning(
                    f"Failed to parse entities as JSON: {entities_str}"
                )
                # Fallback to regex extraction
                return self._extract_entities_fallback(user_message, intent)

        except Exception as e:
            system_logger.error(
                f"Error extracting entities: {e}",
                additional_info={"user_message": user_message},
                exc_info=True,
            )
            return self._extract_entities_fallback(user_message, intent)

    def _extract_entities_fallback(
        self, user_message: str, intent: str
    ) -> Dict[str, Any]:
        """Fallback entity extraction using regex"""
        if intent == "weather":
            return self._extract_weather_entities(user_message)
        elif intent == "set_reminder":
            return self._extract_set_reminder_entities(user_message)
        elif intent == "device_control":
            return self._extract_device_control_entities(user_message)
        return {}

    def _extract_weather_entities(self, user_message: str) -> Dict[str, Any]:
        """Extract weather-related entities"""
        entities = {}
        location_patterns = [
            r"weather in ([A-Za-z\s]+)",
            r"weather for ([A-Za-z\s]+)",
            r"in ([A-Za-z\s,]+)",
            r"at ([A-Za-z\s,]+)",
        ]
        for pattern in location_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                entities["location"] = match.group(1).strip()
                break
        return entities

    def _extract_set_reminder_entities(self, user_message: str) -> Dict[str, Any]:
        """Extract set reminder-related entities"""
        entities = {}
        entities["reminder_title"] = (
            user_message.replace(REMIND_ME_TO, "")
            .replace(SET_A_REMINDER, "")
            .strip()
        )
        time_patterns = [
            r"at (\d{1,2}:\d{2}(?:\s*[ap]m)?)",
            r"(\d{1,2}:\d{2}(?:\s*[ap]m)?)",
            r"(tomorrow|today|next week|next month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"in (\d+) (minutes?|hours?|days?)",
        ]
        for pattern in time_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                entities["datetime"] = match.group(1)
                break
        return entities

    def _extract_device_control_entities(self, user_message: str) -> Dict[str, Any]:
        """Extract device control-related entities"""
        entities = {}
        device_patterns = [
            r"turn (on|off) (?:the )?([A-Za-z\s]+?)(?:\s|$)",
            r"(turn on|turn off|dim|brighten|set) (?:the )?([A-Za-z\s]+?)(?:\s|$)",
            r"set (?:the )?([A-Za-z\s]+) to (\d+)",
            r"(dim|brighten) (?:the )?([A-Za-z\s]+)",
        ]
        for pattern in device_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                if "set" in pattern and len(match.groups()) >= 2:
                    entities["device_name"] = match.group(1)
                    entities["device_action"] = "set"
                    if len(match.groups()) >= 3:
                        entities["value"] = match.group(2)
                else:
                    entities["device_action"] = match.group(1)
                    entities["device_name"] = match.group(2)
                break
        temp_match = re.search(r"(\d+)\s*degrees?", user_message, re.IGNORECASE)
        if temp_match:
            entities["temperature"] = int(temp_match.group(1))
        return entities

    def handle_weather_request(
        self, _entities: Dict[str, Any], user_context: UserContext
    ) -> tuple[str, List[str]]:
        """Handle weather request with knowledge base enhancement"""
        # Extract location from entities
        location = _entities.get("location")
        # Safely get location with fallback
        if (
            not location
            and hasattr(user_context, "preferences")
            and isinstance(user_context.preferences, dict)
        ):
            location = user_context.preferences.get("location", DEFAULT_LOCATION)
        elif not location:
            location = DEFAULT_LOCATION

        # Get weather knowledge from knowledge base
        weather_knowledge = self.get_assistant_knowledge(
            f"weather information {location}"
        )

        # Simulate weather API call
        weather_info = self._get_weather_simulation(location)

        # Enhance with knowledge base information
        if weather_knowledge:
            weather_tips = weather_knowledge[0]["content"][:200]
            weather_info += f"\n\nTip: {weather_tips}..."

        actions_taken = [f"Retrieved weather for {location}"]
        return weather_info, actions_taken

    def _get_weather_simulation(self, location: str) -> str:
        """Simulate weather API response"""
        import random

        conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "clear", "overcast"]
        temp = random.randint(60, 85)
        condition = random.choice(conditions)
        humidity = random.randint(40, 80)
        wind_speed = random.randint(5, 15)

        return f"The weather in {location} is currently {condition} with a temperature of {temp}°F. Humidity is {humidity}% with winds at {wind_speed} mph."

    def handle_set_reminder(
        self, entities: Dict[str, Any], user_context: UserContext
    ) -> tuple[str, List[str]]:
        """Handle setting a reminder with enhanced parsing"""
        session = get_session()
        try:
            title = entities.get("reminder_title", "Reminder").strip()
            datetime_str = entities.get("datetime", "")

            # Clean up the title
            if title.lower().startswith("to "):
                title = title[3:]

            # Parse datetime
            reminder_time = self._parse_datetime(datetime_str)

            if not reminder_time:
                # Get productivity knowledge for reminder tips
                reminder_knowledge = self.get_assistant_knowledge(
                    "reminder time scheduling"
                )
                tip = ""
                if reminder_knowledge:
                    tip = f" Tip: {reminder_knowledge[0]['content'][:100]}..."

                return (
                    f"I couldn't understand the time for your reminder. Please specify a time like 'tomorrow at 3pm' or '2pm today'.{tip}",
                    [],
                )

            # Create reminder
            reminder = Reminder(
                user_id=user_context.user_id,
                title=title,
                description="",
                reminder_time=reminder_time,
                is_recurring=False,
                status="active",
            )

            session.add(reminder)
            session.commit()

            actions_taken = [
                f"Set reminder: {title} for {reminder_time.strftime('%Y-%m-%d %H:%M')}"
            ]
            return (
                f"✓ I've set a reminder for '{title}' on {reminder_time.strftime('%B %d at %I:%M %p')}.",
                actions_taken,
            )

        except Exception as e:
            session.rollback()
            system_logger.error(
                f"Error setting reminder: {e}",
                additional_info={"entities": entities},
                exc_info=True,
            )
            return "I had trouble setting your reminder. Please try again.", []
        finally:
            session.close()

    def _parse_datetime(self, datetime_str: str) -> Optional[datetime]:
        """Enhanced datetime parsing with reduced cognitive complexity"""
        if not datetime_str:
            return None

        now = datetime.now()
        datetime_str = datetime_str.lower().strip()

        base_date = self._parse_relative_day(datetime_str, now)
        if base_date is None:
            base_date = self._parse_weekday(datetime_str, now)
        if base_date is None:
            base_date = now

        time_result = self._extract_time(datetime_str, base_date)
        if time_result:
            return time_result

        relative_result = self._extract_relative_time(datetime_str, now)
        if relative_result:
            return relative_result

        return None

    def _parse_relative_day(
        self, datetime_str: str, now: datetime
    ) -> Optional[datetime]:
        if "tomorrow" in datetime_str:
            return now + timedelta(days=1)
        if "today" in datetime_str:
            return now
        if "next week" in datetime_str:
            return now + timedelta(days=7)
        if "next month" in datetime_str:
            return now + timedelta(days=30)
        return None

    def _parse_weekday(self, datetime_str: str, now: datetime) -> Optional[datetime]:
        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        for day_name, day_num in weekdays.items():
            if day_name in datetime_str:
                days_ahead = day_num - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                return now + timedelta(days=days_ahead)
        return None

    def _extract_time(
        self, datetime_str: str, base_date: datetime
    ) -> Optional[datetime]:
        time_match = re.search(
            r"(\d{1,2}):?(\d{2})?\s*(am|pm)?", datetime_str, re.IGNORECASE
        )
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            am_pm = time_match.group(3)
            if am_pm:
                if am_pm.lower() == "pm" and hour != 12:
                    hour += 12
                elif am_pm.lower() == "am" and hour == 12:
                    hour = 0
            return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return None

    def _extract_relative_time(
        self, datetime_str: str, now: datetime
    ) -> Optional[datetime]:
        relative_match = re.search(r"in (\d+) (minutes?|hours?)", datetime_str)
        if relative_match:
            amount = int(relative_match.group(1))
            unit = relative_match.group(2)
            if "minute" in unit:
                return now + timedelta(minutes=amount)
            elif "hour" in unit:
                return now + timedelta(hours=amount)
        return None

    def handle_list_reminders(self, user_context: UserContext) -> tuple[str, List[str]]:
        """Handle listing reminders with enhanced formatting"""
        if not user_context.active_reminders:
            productivity_tips = self.get_assistant_knowledge(
                "productivity reminder tips"
            )
            tip = ""
            if productivity_tips:
                tip = f"\n\n💡 Tip: {productivity_tips[0]['content'][:150]}..."

            return f"You don't have any active reminders.{tip}", []

        reminder_list = []
        for i, reminder in enumerate(user_context.active_reminders, 1):
            time_str = reminder["reminder_time"]
            if time_str:
                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                time_formatted = dt.strftime("%B %d at %I:%M %p")
                reminder_list.append(f"{i}. {reminder['title']} - {time_formatted}")
            else:
                reminder_list.append(f"{i}. {reminder['title']}")

        response = "📅 Here are your active reminders:\n" + "\n".join(reminder_list)
        actions_taken = ["Retrieved active reminders"]
        return response, actions_taken

    def handle_device_control(
        self, entities: Dict[str, Any], user_context: UserContext
    ) -> tuple[str, List[str]]:
        """Handle smart device control with enhanced capabilities"""
        device_name = entities.get("device_name", "").lower().strip()
        device_action = entities.get("device_action", "").lower().strip()
        temperature = entities.get("temperature")
        value = entities.get("value")

        if not device_name or not device_action:
            # Get smart home help from knowledge base
            smart_home_help = self.get_assistant_knowledge("smart home device control")
            help_text = ""
            if smart_home_help:
                help_text = f"\n\n💡 {smart_home_help[0]['content'][:150]}..."

            available_devices = [d["name"] for d in user_context.smart_devices]
            return (
                f"I need to know which device and what action you want me to perform. Available devices: {', '.join(available_devices)}.{help_text}",
                [],
            )

        # Find matching device (improved matching)
        matching_device = None
        for device in user_context.smart_devices:
            device_words = device["name"].lower().split()
            name_words = device_name.split()

            # Check for partial matches
            if any(word in device["name"].lower() for word in name_words) or any(
                word in device_name for word in device_words
            ):
                matching_device = device
                break

        if not matching_device:
            available_devices = [d["name"] for d in user_context.smart_devices]
            return (
                f"I couldn't find a device matching '{device_name}'. Your available devices are: {', '.join(available_devices)}",
                [],
            )

        # Execute device control with enhanced actions
        success, result_message = self._control_device_enhanced(
            matching_device, device_action, temperature, value
        )

        if success:
            actions_taken = [
                f"Controlled device: {matching_device['name']} - {device_action}"
            ]
            return f"✓ {result_message}", actions_taken
        else:
            return (
                f"❌ I couldn't {device_action} the {matching_device['name']}. {result_message}",
                [],
            )

    def _control_device_enhanced(
        self,
        device: Dict[str, Any],
        action: str,
        temperature: Optional[int] = None,
        value: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Enhanced device control with more actions and feedback"""
        if not device["is_online"]:
            return (
                False,
                "The device appears to be offline. Please check your connection.",
            )

        device_type = device["type"]
        device_name = device["name"]
        capabilities = device.get("capabilities", [])

        try:
            if action in [TURN_ON, "on"]:
                return True, f"I've turned on the {device_name}."

            elif action in [TURN_OFF, "off"]:
                return True, f"I've turned off the {device_name}."

            elif action == "dim" and "brightness" in capabilities:
                return True, f"I've dimmed the {device_name}."

            elif action == "brighten" and "brightness" in capabilities:
                return True, f"I've brightened the {device_name}."

            elif (
                action == "set"
                and temperature is not None
                and device_type == "thermostat"
            ):
                return True, f"I've set the {device_name} to {temperature}°F."

            elif action == "set" and value is not None:
                return True, f"I've set the {device_name} to {value}."

            else:
                available_actions = self._get_device_actions(device_type, capabilities)
                return (
                    False,
                    f"I don't understand that action. Available actions for {device_name}: {', '.join(available_actions)}",
                )

        except Exception as e:
            system_logger.error(
                f"Error controlling device: {e}",
                additional_info={
                    "device": device,
                    "action": action,
                    "temperature": temperature,
                    "value": value,
                },
                exc_info=True,
            )
            return False, f"There was an error controlling the device: {str(e)}"

    def _get_device_actions(
        self, device_type: str, capabilities: List[str]
    ) -> List[str]:
        """Get available actions for a device type"""
        actions = [TURN_ON, TURN_OFF]

        if "brightness" in capabilities:
            actions.extend(["dim", "brighten", "set brightness"])

        if "color" in capabilities:
            actions.extend(["set color"])

        if device_type == "thermostat":
            actions.extend(
                ["set temperature", "increase temperature", "decrease temperature"]
            )

        return actions

    def handle_enhanced_general_query(
        self, user_message: str, _entities: Dict[str, Any]
    ) -> tuple[str, List[str]]:
        """Enhanced general query handling with knowledge base"""
        try:
            # Search agent-specific knowledge first
            pva_results = self.get_assistant_knowledge(user_message)

            if pva_results:
                result = pva_results[0]
                response = f"Based on my knowledge: {result['content'][:300]}..."
                actions_taken = ["Searched personal assistant knowledge base"]
                return response, actions_taken

            # Fall back to general knowledge base
            kb_results = search_knowledge_base(user_message, category=None, limit=3)

            if kb_results:
                result = kb_results[0]
                response = f"Here's what I found: {result['content'][:300]}..."
                actions_taken = ["Searched general knowledge base"]
                return response, actions_taken
            else:
                return (
                    "I don't have specific information about that, but I'm here to help with reminders, weather, device control, and more!",
                    [],
                )

        except Exception as e:
            system_logger.error(
                f"Error handling enhanced general query: {e}",
                additional_info={"user_message": user_message, "entities": _entities},
                exc_info=True,
            )
            return (
                "I'm having trouble finding information about that right now. Is there something else I can help you with?",
                [],
            )

    def handle_smart_home_help(
        self, _entities: Dict[str, Any], _user_context: UserContext
    ) -> tuple[str, List[str]]:
        """Handle smart home help requests"""
        smart_home_knowledge = self.get_assistant_knowledge(
            "smart home setup troubleshooting"
        )

        if smart_home_knowledge:
            response = (
                f"🏠 Smart Home Help:\n{smart_home_knowledge[0]['content'][:400]}..."
            )
        else:
            response = """🏠 Smart Home Help:

I can help you control your smart devices! Here's what I can do:
• Turn devices on/off
• Adjust brightness for lights
• Set temperature for thermostats
• Control multiple devices

Try saying things like:
• "Turn on the living room lights"
• "Set the thermostat to 72 degrees"
• "Dim the bedroom lights"
"""

        actions_taken = ["Provided smart home help"]
        return response, actions_taken

    def handle_productivity_tips(
        self, _entities: Dict[str, Any], _user_context: UserContext
    ) -> tuple[str, List[str]]:
        """Handle productivity tips requests"""
        productivity_knowledge = self.get_assistant_knowledge(
            "productivity tips personal management"
        )

        if productivity_knowledge:
            response = f"💡 Productivity Tips:\n{productivity_knowledge[0]['content'][:400]}..."
        else:
            response = """💡 Productivity Tips:

Here are some ways I can help boost your productivity:
• Set reminders for important tasks
• Schedule recurring reminders for habits
• Use voice commands for quick device control
• Set location-based reminders (coming soon)

Try setting a reminder: "Remind me to take a break in 1 hour"
"""

        actions_taken = ["Provided productivity tips"]
        return response, actions_taken

    def handle_recommendation(
        self, entities: Dict[str, Any], user_context: UserContext
    ) -> tuple[str, List[str]]:
        """Handle recommendation requests with knowledge enhancement"""
        rec_type = entities.get("recommendation_type", "general")
        # Safely get location with fallback
        location = "your area"
        if hasattr(user_context, "preferences") and isinstance(
            user_context.preferences, dict
        ):
            location = user_context.preferences.get("location", "your area")

        # Get recommendation knowledge from knowledge base
        rec_knowledge = self.get_assistant_knowledge(
            f"{rec_type} recommendations {location}"
        )

        if rec_knowledge:
            response = f"Based on my knowledge: {rec_knowledge[0]['content'][:300]}..."
        else:
            # Fallback recommendations
            recommendations = {
                "restaurant": f"Based on your location in {location}, I'd recommend trying the local Italian place downtown or the new sushi restaurant on Main Street.",
                "movie": "For movies, I'd suggest checking out the latest action thriller or the new comedy that just came out.",
                "book": "For books, I'd recommend checking bestseller lists or asking for genre-specific recommendations.",
                "general": "I'd be happy to help with recommendations! What type of recommendation are you looking for - restaurants, movies, books, or something else?",
            }
            response = recommendations.get(rec_type, recommendations["general"])

        actions_taken = [f"Provided {rec_type} recommendation"]
        return response, actions_taken

    def handle_cancel_reminder(
        self, _entities: Dict[str, Any], user_context: UserContext
    ) -> tuple[str, List[str]]:
        """Handle reminder cancellation"""
        session = get_session()
        try:
            if not user_context.active_reminders:
                return "You don't have any active reminders to cancel.", []

            # Simple implementation - cancel the most recent reminder
            # In a more sophisticated version, we'd parse which specific reminder
            latest_reminder = user_context.active_reminders[0]

            reminder = (
                session.query(Reminder)
                .filter(Reminder.id == latest_reminder["id"])
                .first()
            )

            if reminder:
                setattr(reminder, "status", "cancelled")
                session.commit()

                actions_taken = [f"Cancelled reminder: {reminder.title}"]
                return (
                    f"✓ I've cancelled the reminder '{reminder.title}'.",
                    actions_taken,
                )
            else:
                return "I couldn't find that reminder to cancel.", []

        except Exception as e:
            session.rollback()
            system_logger.error(
                f"Error cancelling reminder: {e}",
                additional_info={"entities": _entities, "user_context": user_context},
                exc_info=True,
            )
            return "I had trouble cancelling the reminder. Please try again.", []
        finally:
            session.close()

    def handle_time_date(self) -> tuple[str, List[str]]:
        """Handle time and date requests"""
        now = datetime.now()
        response = f"The current time is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}."
        actions_taken = ["Provided current time and date"]
        return response, actions_taken

    def process_user_request(
        self, user_id: str, user_message: str, session_id: Optional[str] = None
    ) -> PVAAgentResponse:
        """Process a user request and generate a response (refactored for lower cognitive complexity)"""
        start_time = datetime.now()

        user_context = self.get_user_context(user_id)
        if not self._is_valid_user_context(user_context, user_id, session_id):
            return self._error_response(
                "I'm having trouble accessing your information. Please try again.",
                session_id,
                user_context_used=False,
            )

        intent_result = self.recognize_intent(user_message)
        entities = self.extract_entities(user_message, intent_result.intent)
        enhanced_context = self.get_enhanced_context(user_message, intent_result.intent)
        knowledge_base_used = bool(enhanced_context)

        try:
            additional_info, actions_taken = self._get_intent_result_tuple(
                intent_result, user_message, entities, user_context
            )
            context_str = self._build_context_str(user_context)
            response = self._generate_response(
                intent_result, entities, actions_taken, additional_info,
                enhanced_context, user_message, context_str, session_id
            )
            self._update_session_history(session_id, user_message, response)
            suggestions = self._generate_suggestions(intent_result.intent, user_context if user_context is not None else UserContext(
                user_id="unknown",
                preferences={},
                learned_behaviors={},
                active_reminders=[],
                smart_devices=[],
            ))
            pva_response = PVAAgentResponse(
                message=response,
                confidence=intent_result.confidence,
                intent=intent_result.intent,
                entities=entities,
                actions_taken=actions_taken,
                suggestions=suggestions,
                session_id=session_id,
                user_context_used=True,
                knowledge_base_used=knowledge_base_used,
            )
            self._track_interactions(intent_result.intent, actions_taken, pva_response)
            return pva_response

        except Exception as e:
            return self._handle_processing_exception(
                e, start_time, user_id, user_message, intent_result, entities, session_id, knowledge_base_used
            )

    def _is_valid_user_context(self, user_context, user_id, session_id):
        if not user_context:
            system_logger.warning(
                f"User context not found for user_id: {user_id}",
                additional_info={"user_id": user_id, "session_id": session_id},
            )
            return False
        if not hasattr(user_context, "preferences"):
            system_logger.error(
                f"User context missing expected attributes: {type(user_context)}",
                additional_info={
                    "user_id": user_id,
                    "context_type": str(type(user_context)),
                },
            )
            return False
        return True

    def _error_response(self, message, session_id, user_context_used):
        return PVAAgentResponse(
            message=message,
            confidence=0.0,
            intent="error",
            entities={},
            actions_taken=[],
            suggestions=[],
            session_id=session_id,
            user_context_used=user_context_used,
        )

    def _get_intent_result_tuple(self, intent_result, user_message, entities, user_context):
        intent_result_tuple = self._handle_intent(
            intent_result.intent, user_message, entities, user_context
        )
        if isinstance(intent_result_tuple, tuple) and len(intent_result_tuple) == 2:
            additional_info, actions_taken = intent_result_tuple
            if not isinstance(additional_info, str):
                additional_info = str(additional_info)
            if not isinstance(actions_taken, list):
                actions_taken = [str(actions_taken)] if actions_taken else []
        else:
            system_logger.error(
                f"_handle_intent returned unexpected type: {type(intent_result_tuple)}, value: {intent_result_tuple}"
            )
            additional_info = "Error processing intent"
            actions_taken = []
        return additional_info, actions_taken

    def _build_context_str(self, user_context):
        user_id = "unknown"
        try:
            user_id = getattr(user_context, "user_id", "unknown")
            preferences = getattr(user_context, "preferences", {})
            active_reminders = getattr(user_context, "active_reminders", [])
            smart_devices = getattr(user_context, "smart_devices", [])
            context_str = f"""
User ID: {user_id}
Preferences: {json.dumps(preferences) if isinstance(preferences, dict) else str(preferences)}
Active Reminders: {len(active_reminders) if isinstance(active_reminders, list) else 0}
Smart Devices: {len(smart_devices) if isinstance(smart_devices, list) else 0} ({', '.join([d.get('name', 'unknown') if isinstance(d, dict) else str(d) for d in smart_devices[:3]])})
"""
        except Exception as context_error:
            system_logger.error(f"Error creating context string: {context_error}")
            context_str = f"User ID: {user_id}\nContext creation failed"
        return context_str

    def _generate_response(
        self, intent_result, entities, actions_taken, additional_info,
        enhanced_context, user_message, context_str, session_id
    ):
        try:
            chain_input = {
                "user_context": str(context_str),
                "intent": str(intent_result.intent),
                "entities": (
                    json.dumps(entities)
                    if isinstance(entities, dict)
                    else str(entities)
                ),
                "actions_taken": (
                    ", ".join(actions_taken)
                    if isinstance(actions_taken, list)
                    else str(actions_taken)
                ),
                "additional_info": str(additional_info) if additional_info else "",
                "enhanced_context": (
                    str(enhanced_context) if enhanced_context else ""
                ),
                "user_message": str(user_message),
            }
            for key, value in chain_input.items():
                if not isinstance(value, str):
                    chain_input[key] = str(value)
            response = self.response_chain.invoke(
                chain_input,
                config={
                    "configurable": {"session_id": session_id or "default_session"}
                },
            )
            if not isinstance(response, str):
                response = str(response)
        except Exception as chain_error:
            system_logger.error(f"Response chain error: {chain_error}")
            # Fallback to simple response based on intent
            if intent_result.intent == "greeting":
                response = (
                    "Hello! I'm your personal assistant. How can I help you today?"
                )
            elif intent_result.intent == "set_reminder":
                response = (
                    f"I'd be happy to help you set a reminder. {additional_info}"
                )
            else:
                response = f"I understand you're asking about {intent_result.intent}. {additional_info}"
        return response

    def _update_session_history(self, session_id, user_message, response):
        session_id_key = session_id or "default_session"
        history = self._session_histories.get(session_id_key)
        if history is not None:
            history.add_user_message(user_message)
            history.add_ai_message(response)

    def _handle_processing_exception(
        self, e, start_time, user_id, user_message, intent_result, entities, session_id, knowledge_base_used
    ):
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        system_logger.error(
            f"Error processing request: {e}",
            additional_info={
                "user_id": user_id,
                "user_message": user_message,
                "intent": getattr(intent_result, 'intent', 'unknown'),
                "entities": entities,
                "actions_taken": [],
                "session_id": session_id,
                "processing_time_ms": processing_time,
            },
            exc_info=True,
        )
        return PVAAgentResponse(
            message="I'm sorry, I encountered an error while processing your request. Please try again.",
            confidence=0.0,
            intent=getattr(intent_result, 'intent', 'error'),
            entities=entities,
            actions_taken=[],
            suggestions=[],
            session_id=session_id,
            user_context_used=True,
            knowledge_base_used=knowledge_base_used,
        )

    def _handle_intent(
        self,
        intent: str,
        user_message: str,
        entities: Dict[str, Any],
        user_context: UserContext,
    ) -> tuple[str, List[str]]:
        """Handle the request based on intent (extracted from process_user_request)"""
        if intent == "weather":
            return self.handle_weather_request(entities, user_context)
        if intent == "set_reminder":
            return self.handle_set_reminder(entities, user_context)
        if intent == "list_reminders":
            return self.handle_list_reminders(user_context)
        if intent == "cancel_reminder":
            return self.handle_cancel_reminder(entities, user_context)
        if intent == "device_control":
            return self.handle_device_control(entities, user_context)
        if intent == "smart_home_help":
            return self.handle_smart_home_help(entities, user_context)
        if intent == "productivity_tips":
            return self.handle_productivity_tips(entities, user_context)
        if intent == "recommendation":
            return self.handle_recommendation(entities, user_context)
        if intent == "time_date":
            return self.handle_time_date()
        if intent == "general_query":
            return self.handle_enhanced_general_query(user_message, entities)
        if intent == "greeting":
            return (
                "Hello! I'm your personal assistant. I can help you with weather, reminders, device control, and more. What can I do for you today?",
                ["Greeted user"],
            )
        if intent == "goodbye":
            return (
                "Goodbye! Feel free to ask me anything anytime. Have a great day!",
                ["Said goodbye"],
            )
        return (
            "I'm here to help! I can assist with weather, reminders, smart devices, productivity tips, and general questions. What would you like to do?",
            ["Provided general help"],
        )

    def _track_interactions(
        self, intent: str, actions_taken: List[str], pva_response: PVAAgentResponse
    ):
        """Track device and reminder interactions for analytics/logging"""
        if intent == "device_control" and actions_taken:
            self._track_device_interactions(actions_taken, pva_response)
        if intent in ["set_reminder", "cancel_reminder"] and actions_taken:
            self._track_reminder_interactions(actions_taken, intent, pva_response)

    def _track_device_interactions(
        self, actions_taken: List[str], pva_response: PVAAgentResponse
    ):
        """Helper to track device interactions"""
        for action in actions_taken:
            if "Controlled device:" in action:
                device_info = action.replace("Controlled device: ", "")
                parts = device_info.split(" - ")
                if len(parts) == 2:
                    pva_response.add_device_interaction(parts[0], parts[1])

    def _track_reminder_interactions(
        self, actions_taken: List[str], intent: str, pva_response: PVAAgentResponse
    ):
        """Helper to track reminder interactions"""
        for action in actions_taken:
            if "reminder:" in action.lower():
                reminder_info = action.split(": ", 1)[1] if ": " in action else action
                pva_response.add_reminder_interaction(reminder_info, intent)

    def _generate_suggestions(
        self, intent: str, user_context: UserContext
    ) -> List[str]:
        """Generate helpful suggestions based on intent and context"""
        suggestions = []

        if intent == "weather":
            suggestions.extend(
                [
                    "Would you like me to set a reminder if rain is expected?",
                    "I can also check weather for other locations",
                ]
            )
        elif intent == "set_reminder":
            suggestions.extend(
                [
                    "I can set recurring reminders too",
                    "Try: 'Remind me to exercise every Monday at 7am'",
                ]
            )
        elif intent == "device_control":
            if user_context.smart_devices:
                suggestions.extend(
                    [
                        "You can control multiple devices at once",
                        "Ask me about smart home automation tips",
                    ]
                )
        elif intent == "general_query":
            suggestions.extend(
                [
                    "I can help you set reminders",
                    "Ask me about the weather",
                    "I can control your smart devices",
                    "Need productivity tips?",
                ]
            )
        elif intent == "greeting":
            suggestions.extend(
                [
                    "Ask me about the weather",
                    "Set a reminder for something important",
                    "Control your smart devices",
                    "Get productivity tips",
                ]
            )
        elif intent == "smart_home_help":
            suggestions.extend(
                [
                    "Try: 'Turn on the living room lights'",
                    "Say: 'Set the thermostat to 72 degrees'",
                ]
            )

        return suggestions[:3]  # Limit to 3 suggestions

    def _rule_based_intent_recognition(self, user_message: str) -> Optional[str]:
        """Rule-based intent recognition for high-confidence patterns"""
        message_lower = user_message.lower()
        if any(
            pattern in message_lower
            for pattern in [
                REMIND_ME_TO,
                SET_A_REMINDER,
                "reminder for me to",
                "call john tomorrow",
                "tomorrow at",
                "remind me",
            ]
        ):
            return "set_reminder"

        # Other high-confidence patterns
        if any(
            pattern in message_lower
            for pattern in ["turn on", "turn off", "dim", "brighten"]
        ):
            return "device_control"

        if any(
            pattern in message_lower
            for pattern in ["weather", "temperature", "forecast"]
        ):
            return "weather"

        if any(
            pattern in message_lower
            for pattern in ["hello", "hi", "hey", "good morning"]
        ):
            return "greeting"

        return None

    def log_interaction(
        self,
        user_id: str,
        user_message: str,
        response: PVAAgentResponse,
        session_id: Optional[str] = None,
    ):
        """Log the interaction to the database"""
        session = get_session()
        try:
            interaction = PVAInteraction(
                user_id=user_id,
                intent=response.intent,
                user_message=user_message,
                agent_response=response.message,
                entities_extracted=response.entities,
                confidence_score=response.confidence,
                session_id=session_id
                or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            session.add(interaction)
            session.commit()
        except Exception as e:
            system_logger.error(
                f"Error logging interaction: {e}",
                additional_info={
                    "user_id": user_id,
                    "user_message": user_message,
                    "response": response.to_dict(),
                    "session_id": session_id,
                },
                exc_info=True,
            )
            session.rollback()
        finally:
            session.close()

    def handle_user_request(
        self, user_id: str, user_message: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main method to handle a user request"""
        # Process the message
        response = self.process_user_request(user_id, user_message, session_id)

        # Log the interaction
        self.log_interaction(user_id, user_message, response, session_id)

        # Return structured response as dictionary
        return response.to_dict()

    def get_user_devices(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's smart devices"""
        user_context = self.get_user_context(user_id)
        return user_context.smart_devices if user_context else []

    def get_user_reminders(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's active reminders"""
        user_context = self.get_user_context(user_id)
        return user_context.active_reminders if user_context else []


def test_personal_virtual_assistant_agent():
    """Test the Personal Virtual Assistant agent with comprehensive scenarios"""
    agent = PersonalVirtualAssistantAgent()

    # Test scenarios with enhanced functionality
    test_cases = [
        {
            "user_id": "user123",
            "message": "Hello! How are you?",
            "description": "Greeting test",
        },
        {
            "user_id": "user123",
            "message": "What's the weather like in San Francisco?",
            "description": "Weather request with location",
        },
        {
            "user_id": "user123",
            "message": "Set a reminder to call mom tomorrow at 3pm",
            "description": "Reminder setting with specific time",
        },
        {
            "user_id": "user123",
            "message": "Turn on the living room lights",
            "description": "Smart device control",
        },
        {
            "user_id": "user123",
            "message": "Set the thermostat to 72 degrees",
            "description": "Thermostat control with temperature",
        },
        {
            "user_id": "user123",
            "message": "What are my reminders?",
            "description": "List reminders",
        },
        {
            "user_id": "user123",
            "message": "Can you recommend a good restaurant?",
            "description": "Recommendation request",
        },
        {
            "user_id": "user123",
            "message": "What time is it?",
            "description": "Time request",
        },
        {
            "user_id": "user123",
            "message": "Help me with smart home setup",
            "description": "Smart home help",
        },
        {
            "user_id": "user123",
            "message": "Give me some productivity tips",
            "description": "Productivity advice",
        },
        {
            "user_id": "user123",
            "message": "How do I improve my morning routine?",
            "description": "General knowledge query",
        },
        {
            "user_id": "user123",
            "message": "Goodbye for now",
            "description": "Goodbye message",
        },
    ]

    print("=== Personal Virtual Assistant Agent Test ===\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['description']}")
        print(f"User ID: {test_case['user_id']}")
        print(f"Message: {test_case['message']}")

        result = agent.handle_user_request(test_case["user_id"], test_case["message"])

        print(f"Response: {result['response']}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Actions Taken: {', '.join(result['actions_taken'])}")
        print(f"Suggestions: {', '.join(result['suggestions'])}")
        print(f"Knowledge Base Used: {result['knowledge_base_used']}")
        print(f"Device Interactions: {result['device_interactions']}")
        print(f"Reminder Interactions: {result['reminder_interactions']}")
        print("-" * 80)

    # Test knowledge base integration
    print("\n=== Knowledge Base Integration Test ===")
    knowledge = agent.get_assistant_knowledge("smart home automation")
    print(f"Knowledge base results: {len(knowledge)} items found")
    if knowledge:
        print(f"First result: {knowledge[0]['title']}")

    # Test device and reminder retrieval
    print("\n=== Device and Reminder Status ===")
    devices = agent.get_user_devices("user123")
    reminders = agent.get_user_reminders("user123")
    print(f"User devices: {len(devices)}")
    print(f"User reminders: {len(reminders)}")


if __name__ == "__main__":
    test_personal_virtual_assistant_agent()
