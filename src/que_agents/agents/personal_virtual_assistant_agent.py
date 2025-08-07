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
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from src.que_agents.core.database import (
    PVAInteraction,
    Reminder,
    SmartDevice,
    UserPreferences,
    get_session,
)
from src.que_agents.core.llm_factory import LLMFactory
from src.que_agents.core.schemas import AgentResponse, IntentResult, UserContext
from src.que_agents.error_trace.errorlogger import system_logger
from src.que_agents.knowledge_base.kb_manager import (
    search_agent_knowledge_base,
    search_knowledge_base,
)

system_logger.info("Personal Virtual Assistant Agent initialized...")


# Load agent configuration
with open("configs/agent_config.yaml", "r") as f:
    agent_config = yaml.safe_load(f)


class PersonalVirtualAssistantAgent:
    """Personal Virtual Assistant Agent using LangChain"""

    def __init__(self):
        config = agent_config["personal_virtual_assistant_agent"]
        self.llm = LLMFactory.get_llm(
            agent_type="personal_virtual_assistant",
            model_name=config["model_name"],
            temperature=config["temperature"],
            max_tokens=600,
        )

        # Memory for conversation history
        self.memory = ConversationBufferWindowMemory(
            k=15,  # Keep last 15 exchanges
            return_messages=True,
            memory_key="chat_history",
        )

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
        self.response_chain = self._create_response_chain()
        self.entity_chain = self._create_entity_chain()

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
        """Create prompt template for intent recognition"""
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
        """Create prompt template for entity extraction"""
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
        """Create prompt template for response generation"""
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

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{user_message}"),
            ]
        )

    def _create_intent_chain(self):
        """Create intent recognition chain"""
        return self.intent_prompt | self.llm | StrOutputParser()

    def _create_entity_chain(self):
        """Create entity extraction chain"""
        return self.entity_prompt | self.llm | StrOutputParser()

    def _create_response_chain(self):
        """Create response generation chain"""
        return (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.chat_memory.messages
            )
            | self.response_prompt
            | self.llm
            | StrOutputParser()
        )

    def get_user_context(self, user_id: str) -> Optional[UserContext]:
        """Retrieve user context from database"""
        session = get_session()
        try:
            # Get user preferences
            user_prefs = (
                session.query(UserPreferences)
                .filter(UserPreferences.user_id == user_id)
                .first()
            )

            if not user_prefs:
                # Create default user preferences
                user_prefs = UserPreferences(
                    user_id=user_id,
                    preferences={
                        "location": "New York",
                        "timezone": "UTC",
                        "temperature_unit": "fahrenheit",
                        "preferred_language": "english",
                    },
                    learned_behaviors={},
                )
                session.add(user_prefs)
                session.commit()

            # Get active reminders
            active_reminders = (
                session.query(Reminder)
                .filter(Reminder.user_id == user_id, Reminder.status == "active")
                .all()
            )

            # Get smart devices
            smart_devices = (
                session.query(SmartDevice).filter(SmartDevice.user_id == user_id).all()
            )

            # If no devices exist, create some default ones for demo
            if not smart_devices:
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
                smart_devices = default_devices

            # Extract preferences safely
            preferences_value = getattr(user_prefs, "preferences", None)
            if preferences_value is not None:
                if isinstance(preferences_value, dict):
                    preferences = preferences_value
                else:
                    try:
                        preferences = json.loads(preferences_value)
                    except Exception:
                        preferences = {}
            else:
                preferences = {}

            # Extract learned_behaviors safely
            learned_behaviors_value = getattr(user_prefs, "learned_behaviors", None)
            if learned_behaviors_value is not None:
                if isinstance(learned_behaviors_value, dict):
                    learned_behaviors = learned_behaviors_value
                else:
                    try:
                        learned_behaviors = json.loads(learned_behaviors_value)
                    except Exception:
                        learned_behaviors = {}
            else:
                learned_behaviors = {}

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

    def recognize_intent(self, user_message: str) -> IntentResult:
        """Recognize user intent from message"""
        try:
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
            ],
            "list_reminders": [
                "reminders",
                "what's scheduled",
                "appointments",
                "my schedule",
            ],
            "device_control": [
                "turn on",
                "turn off",
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

        if matches > 0:
            return min(0.9, 0.6 + (matches * 0.1))
        else:
            return 0.6  # Default confidence

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
        entities = {}

        if intent == "weather":
            # Extract location
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

        elif intent == "set_reminder":
            # Extract reminder title and time
            entities["reminder_title"] = (
                user_message.replace("remind me to", "")
                .replace("set a reminder", "")
                .strip()
            )

            # Enhanced time extraction
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

        elif intent == "device_control":
            # Enhanced device and action extraction
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

            # Extract temperature for thermostat
            temp_match = re.search(r"(\d+)\s*degrees?", user_message, re.IGNORECASE)
            if temp_match:
                entities["temperature"] = int(temp_match.group(1))

        return entities

    def handle_weather_request(
        self, entities: Dict[str, Any], user_context: UserContext
    ) -> tuple[str, List[str]]:
        """Handle weather request with knowledge base enhancement"""
        location = entities.get("location") or user_context.preferences.get(
            "location", "New York"
        )

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
        """Enhanced datetime parsing"""
        if not datetime_str:
            return None

        now = datetime.now()
        datetime_str = datetime_str.lower().strip()

        # Handle relative days
        if "tomorrow" in datetime_str:
            base_date = now + timedelta(days=1)
        elif "today" in datetime_str:
            base_date = now
        elif "next week" in datetime_str:
            base_date = now + timedelta(days=7)
        elif "next month" in datetime_str:
            base_date = now + timedelta(days=30)
        else:
            # Handle specific days of week
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
                    if days_ahead <= 0:  # Target day already happened this week
                        days_ahead += 7
                    base_date = now + timedelta(days=days_ahead)
                    break
            else:
                base_date = now

        # Extract time
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

        # Handle relative time (in X minutes/hours)
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
            if action in ["turn on", "on"]:
                return True, f"I've turned on the {device_name}."

            elif action in ["turn off", "off"]:
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
        actions = ["turn on", "turn off"]

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
                reminder.status = "cancelled"
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
    ) -> AgentResponse:
        """Process a user request and generate a response"""
        # Get user context
        user_context = self.get_user_context(user_id)
        if not user_context:
            system_logger.warning(
                f"User context not found for user_id: {user_id}",
                additional_info={"user_id": user_id, "session_id": session_id},
            )
            return AgentResponse(
                message="I'm having trouble accessing your information. Please try again.",
                intent="error",
                entities={},
                confidence=0.0,
                actions_taken=[],
                suggestions=[],
            )

        # Recognize intent
        intent_result = self.recognize_intent(user_message)

        # Extract entities
        entities = self.extract_entities(user_message, intent_result.intent)

        # Get enhanced context from knowledge base
        enhanced_context = self.get_enhanced_context(user_message, intent_result.intent)

        # Handle the request based on intent
        actions_taken = []
        additional_info = ""

        try:
            if intent_result.intent == "weather":
                additional_info, actions_taken = self.handle_weather_request(
                    entities, user_context
                )
            elif intent_result.intent == "set_reminder":
                additional_info, actions_taken = self.handle_set_reminder(
                    entities, user_context
                )
            elif intent_result.intent == "list_reminders":
                additional_info, actions_taken = self.handle_list_reminders(
                    user_context
                )
            elif intent_result.intent == "cancel_reminder":
                additional_info, actions_taken = self.handle_cancel_reminder(
                    entities, user_context
                )
            elif intent_result.intent == "device_control":
                additional_info, actions_taken = self.handle_device_control(
                    entities, user_context
                )
            elif intent_result.intent == "smart_home_help":
                additional_info, actions_taken = self.handle_smart_home_help(
                    entities, user_context
                )
            elif intent_result.intent == "productivity_tips":
                additional_info, actions_taken = self.handle_productivity_tips(
                    entities, user_context
                )
            elif intent_result.intent == "recommendation":
                additional_info, actions_taken = self.handle_recommendation(
                    entities, user_context
                )
            elif intent_result.intent == "time_date":
                additional_info, actions_taken = self.handle_time_date()
            elif intent_result.intent == "general_query":
                additional_info, actions_taken = self.handle_enhanced_general_query(
                    user_message, entities
                )
            elif intent_result.intent == "greeting":
                additional_info = f"Hello! I'm your personal assistant. I can help you with weather, reminders, device control, and more. What can I do for you today?"
                actions_taken = ["Greeted user"]
            elif intent_result.intent == "goodbye":
                additional_info = (
                    "Goodbye! Feel free to ask me anything anytime. Have a great day!"
                )
                actions_taken = ["Said goodbye"]
            else:
                additional_info = "I'm here to help! I can assist with weather, reminders, smart devices, productivity tips, and general questions. What would you like to do?"
                actions_taken = ["Provided general help"]

            # Generate response using LLM
            context_str = f"""
User ID: {user_context.user_id}
Preferences: {json.dumps(user_context.preferences)}
Active Reminders: {len(user_context.active_reminders)}
Smart Devices: {len(user_context.smart_devices)} ({', '.join([d['name'] for d in user_context.smart_devices])})
"""

            response = self.response_chain.invoke(
                {
                    "user_context": context_str,
                    "intent": intent_result.intent,
                    "entities": json.dumps(entities),
                    "actions_taken": ", ".join(actions_taken),
                    "additional_info": additional_info,
                    "enhanced_context": enhanced_context,
                    "user_message": user_message,
                }
            )

            # Update memory
            self.memory.chat_memory.add_user_message(user_message)
            self.memory.chat_memory.add_ai_message(response)

            # Generate suggestions
            suggestions = self._generate_suggestions(intent_result.intent, user_context)

            return AgentResponse(
                message=response,
                intent=intent_result.intent,
                entities=entities,
                confidence=intent_result.confidence,
                actions_taken=actions_taken,
                suggestions=suggestions,
            )

        except Exception as e:
            system_logger.error(
                f"Error processing request: {e}",
                additional_info={
                    "user_id": user_id,
                    "user_message": user_message,
                    "intent": intent_result.intent,
                    "entities": entities,
                    "actions_taken": actions_taken,
                    "session_id": session_id,
                },
                exc_info=True,
            )
            return AgentResponse(
                message="I'm sorry, I encountered an error while processing your request. Please try again.",
                intent=intent_result.intent,
                entities=entities,
                confidence=0.0,
                actions_taken=[],
                suggestions=[],
            )

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

    def log_interaction(
        self,
        user_id: str,
        user_message: str,
        response: AgentResponse,
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
                    "response": response,
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

        # Return structured response
        return {
            "response": response.message,
            "intent": response.intent,
            "entities": response.entities,
            "confidence": response.confidence,
            "actions_taken": response.actions_taken,
            "suggestions": response.suggestions,
            "timestamp": datetime.now().isoformat(),
        }

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
