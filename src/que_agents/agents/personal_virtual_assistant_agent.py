# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-05 12:00:00
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-05 12:00:00
# @Description: This module implements a Personal Virtual Assistant agent using LangChain and SQLAlchemy

import json
import re
from dataclasses import dataclass
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

try:
    from src.que_agents.knowledge_base.kb_manager import search_knowledge_base
except ImportError:
    from src.que_agents.knowledge_base.kb_manager_mock import search_knowledge_base


@dataclass
class UserContext:
    """User context information"""

    user_id: str
    preferences: Dict[str, Any]
    learned_behaviors: Dict[str, Any]
    active_reminders: List[Dict]
    smart_devices: List[Dict]


@dataclass
class IntentResult:
    """Intent recognition result"""

    intent: str
    confidence: float
    entities: Dict[str, Any]


@dataclass
class AgentResponse:
    """Agent response structure"""

    message: str
    intent: str
    entities: Dict[str, Any]
    confidence: float
    actions_taken: List[str]
    suggestions: List[str]


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
        ]

        # Initialize prompt templates
        self.intent_prompt = self._create_intent_prompt()
        self.response_prompt = self._create_response_prompt()
        self.entity_prompt = self._create_entity_prompt()

        # Create chains
        self.intent_chain = self._create_intent_chain()
        self.response_chain = self._create_response_chain()
        self.entity_chain = self._create_entity_chain()

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

User message: {user_message}
Intent: {intent}

Extract entities as a JSON object. Only include entities that are clearly present in the message. Use null for missing entities.

Example format:
{{"location": "New York", "datetime": "tomorrow 3pm", "device_name": null, "device_action": null}}"""

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

RESPONSE GUIDELINES:
- Keep responses concise but friendly
- Use natural, conversational language
- Acknowledge the user's request
- Provide relevant information or confirm actions taken
- Offer helpful suggestions when appropriate
- If you can't fulfill a request, explain why and suggest alternatives

User Context: {user_context}
Intent: {intent}
Extracted Entities: {entities}
Actions Taken: {actions_taken}
Additional Information: {additional_info}

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
                    preferences={"location": "New York", "timezone": "UTC"},
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

            return UserContext(
                user_id=user_id,
                preferences=user_prefs.preferences or {},
                learned_behaviors=user_prefs.learned_behaviors or {},
                active_reminders=[
                    {
                        "id": r.id,
                        "title": r.title,
                        "description": r.description,
                        "reminder_time": (
                            r.reminder_time.isoformat() if r.reminder_time else None
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

            # Simple confidence scoring based on keywords
            confidence = 0.8  # Default confidence

            return IntentResult(intent=intent, confidence=confidence, entities={})
        except Exception as e:
            print(f"Error recognizing intent: {e}")
            return IntentResult(intent="general_query", confidence=0.5, entities={})

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
                # Fallback to simple regex extraction
                return self._extract_entities_fallback(user_message, intent)

        except Exception as e:
            print(f"Error extracting entities: {e}")
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
                r"in ([A-Za-z\s]+)",
            ]
            for pattern in location_patterns:
                match = re.search(pattern, user_message, re.IGNORECASE)
                if match:
                    entities["location"] = match.group(1).strip()
                    break

        elif intent == "set_reminder":
            # Extract reminder title and time
            entities["reminder_title"] = user_message

            # Simple time extraction
            time_patterns = [
                r"at (\d{1,2}:\d{2})",
                r"(\d{1,2}:\d{2})",
                r"(tomorrow|today|next week|next month)",
            ]
            for pattern in time_patterns:
                match = re.search(pattern, user_message, re.IGNORECASE)
                if match:
                    entities["datetime"] = match.group(1)
                    break

        elif intent == "device_control":
            # Extract device and action
            device_patterns = [
                r"turn (on|off) the ([A-Za-z\s]+)",
                r"(turn on|turn off|dim|brighten) ([A-Za-z\s]+)",
                r"set ([A-Za-z\s]+) to (\d+)",
            ]
            for pattern in device_patterns:
                match = re.search(pattern, user_message, re.IGNORECASE)
                if match:
                    if len(match.groups()) >= 2:
                        entities["device_action"] = match.group(1)
                        entities["device_name"] = match.group(2)
                    break

        return entities

    def handle_weather_request(
        self, entities: Dict[str, Any], user_context: UserContext
    ) -> tuple[str, List[str]]:
        """Handle weather request"""
        location = entities.get("location") or user_context.preferences.get(
            "location", "New York"
        )

        # Simulate weather API call (in real implementation, use actual weather API)
        weather_info = self._get_weather_simulation(location)

        actions_taken = [f"Retrieved weather for {location}"]
        return weather_info, actions_taken

    def _get_weather_simulation(self, location: str) -> str:
        """Simulate weather API response"""
        import random

        conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "clear"]
        temp = random.randint(60, 85)
        condition = random.choice(conditions)

        return f"The weather in {location} is currently {condition} with a temperature of {temp}°F."

    def handle_set_reminder(
        self, entities: Dict[str, Any], user_context: UserContext
    ) -> tuple[str, List[str]]:
        """Handle setting a reminder"""
        session = get_session()
        try:
            title = entities.get("reminder_title", "Reminder")
            datetime_str = entities.get("datetime", "")

            # Parse datetime (simplified)
            reminder_time = self._parse_datetime(datetime_str)

            if not reminder_time:
                return (
                    "I couldn't understand the time for your reminder. Please specify a time like 'tomorrow at 3pm' or '2pm today'.",
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
                f"I've set a reminder for '{title}' on {reminder_time.strftime('%B %d at %I:%M %p')}.",
                actions_taken,
            )

        except Exception as e:
            print(f"Error setting reminder: {e}")
            return "I had trouble setting your reminder. Please try again.", []
        finally:
            session.close()

    def _parse_datetime(self, datetime_str: str) -> Optional[datetime]:
        """Parse datetime string (simplified implementation)"""
        if not datetime_str:
            return None

        now = datetime.now()

        # Simple parsing
        if "tomorrow" in datetime_str.lower():
            base_date = now + timedelta(days=1)
        elif "today" in datetime_str.lower():
            base_date = now
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

            if am_pm and am_pm.lower() == "pm" and hour != 12:
                hour += 12
            elif am_pm and am_pm.lower() == "am" and hour == 12:
                hour = 0

            return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

        return None

    def handle_list_reminders(self, user_context: UserContext) -> tuple[str, List[str]]:
        """Handle listing reminders"""
        if not user_context.active_reminders:
            return "You don't have any active reminders.", []

        reminder_list = []
        for reminder in user_context.active_reminders:
            time_str = reminder["reminder_time"]
            if time_str:
                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                time_formatted = dt.strftime("%B %d at %I:%M %p")
                reminder_list.append(f"• {reminder['title']} - {time_formatted}")
            else:
                reminder_list.append(f"• {reminder['title']}")

        response = "Here are your active reminders:\n" + "\n".join(reminder_list)
        actions_taken = ["Retrieved active reminders"]
        return response, actions_taken

    def handle_device_control(
        self, entities: Dict[str, Any], user_context: UserContext
    ) -> tuple[str, List[str]]:
        """Handle smart device control"""
        device_name = entities.get("device_name", "").lower()
        device_action = entities.get("device_action", "").lower()

        if not device_name or not device_action:
            return (
                "I need to know which device and what action you want me to perform.",
                [],
            )

        # Find matching device
        matching_device = None
        for device in user_context.smart_devices:
            if device_name in device["name"].lower():
                matching_device = device
                break

        if not matching_device:
            return (
                f"I couldn't find a device named '{device_name}'. Your available devices are: {', '.join([d['name'] for d in user_context.smart_devices])}",
                [],
            )

        # Simulate device control
        success = self._control_device_simulation(matching_device, device_action)

        if success:
            actions_taken = [
                f"Controlled device: {matching_device['name']} - {device_action}"
            ]
            return f"I've {device_action} the {matching_device['name']}.", actions_taken
        else:
            return (
                f"I couldn't {device_action} the {matching_device['name']}. Please check if the device is online.",
                [],
            )

    def _control_device_simulation(self, device: Dict[str, Any], action: str) -> bool:
        """Simulate device control (in real implementation, call actual device APIs)"""
        if not device["is_online"]:
            return False

        # Simulate successful control
        return True

    def handle_general_query(
        self, user_message: str, entities: Dict[str, Any]
    ) -> tuple[str, List[str]]:
        """Handle general queries using knowledge base"""
        try:
            # Search knowledge base
            kb_results = search_knowledge_base(user_message, limit=3)

            if kb_results:
                # Use the first result
                result = kb_results[0]
                response = f"Based on what I know: {result['content'][:300]}..."
                actions_taken = ["Searched knowledge base"]
                return response, actions_taken
            else:
                return (
                    "I don't have specific information about that, but I'm here to help with reminders, weather, device control, and more!",
                    [],
                )

        except Exception as e:
            print(f"Error handling general query: {e}")
            return (
                "I'm having trouble finding information about that right now. Is there something else I can help you with?",
                [],
            )

    def handle_recommendation(
        self, entities: Dict[str, Any], user_context: UserContext
    ) -> tuple[str, List[str]]:
        """Handle recommendation requests"""
        rec_type = entities.get("recommendation_type", "general")
        location = user_context.preferences.get("location", "your area")

        # Simulate recommendations
        recommendations = {
            "restaurant": f"Based on your location in {location}, I'd recommend trying the local Italian place downtown or the new sushi restaurant on Main Street.",
            "movie": "For movies, I'd suggest checking out the latest action thriller or the new comedy that just came out.",
            "general": "I'd be happy to help with recommendations! What type of recommendation are you looking for - restaurants, movies, books, or something else?",
        }

        response = recommendations.get(rec_type, recommendations["general"])
        actions_taken = [f"Provided {rec_type} recommendation"]
        return response, actions_taken

    def process_user_request(
        self, user_id: str, user_message: str, session_id: str = None
    ) -> AgentResponse:
        """Process a user request and generate a response"""
        # Get user context
        user_context = self.get_user_context(user_id)
        if not user_context:
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
            elif intent_result.intent == "device_control":
                additional_info, actions_taken = self.handle_device_control(
                    entities, user_context
                )
            elif intent_result.intent == "recommendation":
                additional_info, actions_taken = self.handle_recommendation(
                    entities, user_context
                )
            elif intent_result.intent == "general_query":
                additional_info, actions_taken = self.handle_general_query(
                    user_message, entities
                )
            elif intent_result.intent == "greeting":
                additional_info = f"Hello! I'm your personal assistant. I can help you with weather, reminders, device control, and more."
                actions_taken = ["Greeted user"]
            elif intent_result.intent == "goodbye":
                additional_info = "Goodbye! Feel free to ask me anything anytime."
                actions_taken = ["Said goodbye"]
            else:
                additional_info = "I'm here to help! I can assist with weather, reminders, smart devices, and general questions."
                actions_taken = ["Provided general help"]

            # Generate response using LLM
            context_str = f"""
User ID: {user_context.user_id}
Preferences: {json.dumps(user_context.preferences)}
Active Reminders: {len(user_context.active_reminders)}
Smart Devices: {len(user_context.smart_devices)}
"""

            response = self.response_chain.invoke(
                {
                    "user_context": context_str,
                    "intent": intent_result.intent,
                    "entities": json.dumps(entities),
                    "actions_taken": ", ".join(actions_taken),
                    "additional_info": additional_info,
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
            print(f"Error processing request: {e}")
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
            suggestions.append(
                "Would you like me to set a reminder if rain is expected?"
            )
        elif intent == "set_reminder":
            suggestions.append("I can also set recurring reminders if needed")
        elif intent == "device_control":
            if user_context.smart_devices:
                suggestions.append(
                    "You can also ask me to control multiple devices at once"
                )
        elif intent == "general_query":
            suggestions.extend(
                [
                    "I can help you set reminders",
                    "Ask me about the weather",
                    "I can control your smart devices",
                ]
            )

        return suggestions

    def log_interaction(
        self,
        user_id: str,
        user_message: str,
        response: AgentResponse,
        session_id: str = None,
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
            print(f"Error logging interaction: {e}")
            session.rollback()
        finally:
            session.close()

    def handle_user_request(
        self, user_id: str, user_message: str, session_id: str = None
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


def test_personal_virtual_assistant_agent():
    """Test the Personal Virtual Assistant agent with sample interactions"""
    agent = PersonalVirtualAssistantAgent()

    # Test scenarios
    test_cases = [
        {
            "user_id": "user123",
            "message": "What's the weather like in San Francisco?",
        },
        {
            "user_id": "user123",
            "message": "Set a reminder to call mom tomorrow at 3pm",
        },
        {
            "user_id": "user123",
            "message": "Turn on the living room lights",
        },
        {
            "user_id": "user123",
            "message": "What are my reminders?",
        },
        {
            "user_id": "user123",
            "message": "Can you recommend a good restaurant?",
        },
    ]

    print("=== Personal Virtual Assistant Agent Test ===\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"User ID: {test_case['user_id']}")
        print(f"Message: {test_case['message']}")

        result = agent.handle_user_request(test_case["user_id"], test_case["message"])

        print(f"Response: {result['response']}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Actions Taken: {', '.join(result['actions_taken'])}")
        print(f"Suggestions: {', '.join(result['suggestions'])}")
        print("-" * 80)


if __name__ == "__main__":
    test_personal_virtual_assistant_agent()
