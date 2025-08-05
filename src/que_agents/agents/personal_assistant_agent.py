# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-01 23:53:31
# @Description: This module implements a Personal Assistant Agent that can set reminders, provide recommendations, control smart devices, and manage user preferences.

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from src.que_agents.core.database import SmartDevice, UserPreference, get_session

# Load agent configuration
with open("configs/agent_config.yaml", "r") as f:
    agent_config = yaml.safe_load(f)


@dataclass
class Reminder:
    task: str
    time: datetime
    user_id: int


@dataclass
class Recommendation:
    category: str
    item: str
    reason: str


@dataclass
class SmartDeviceStatus:
    device_id: str
    name: str
    type: str
    status: Dict[str, Any]


# --- Tools for the Personal Assistant Agent ---


@tool
def set_reminder(task: str, time_str: str, user_id: int) -> str:
    """Sets a reminder for the user. Time should be in a parseable format (e.g., 'YYYY-MM-DD HH:MM:SS')."""
    try:
        # In a real system, this would interact with a reminder service or database
        reminder_time = (
            datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            if len(time_str) == 19
            else datetime.fromisoformat(time_str)
        )
        reminder = Reminder(task=task, time=reminder_time, user_id=user_id)
        print(f"[TOOL] Setting reminder for user {user_id}: {task} at {reminder_time}")
        # Simulate saving reminder
        # For PoC, we'll just print it. In a real app, save to DB or external service.
        return f"Reminder '{task}' set successfully for {time_str}."
    except ValueError:
        return "Error: Could not parse the time. Please provide time in 'YYYY-MM-DD HH:MM:SS' format."


@tool
def get_recommendation(user_id: int, category: str) -> str:
    """Provides a recommendation to the user based on a category (e.g., 'movie', 'book', 'restaurant')."""
    # In a real system, this would query a recommendation engine or user preferences
    print(f"[TOOL] Getting recommendation for user {user_id} in category: {category}")

    # Simulate recommendations based on category
    recommendations = {
        "movie": ["Inception", "The Matrix", "Dune"],
        "book": ["Sapiens", "The Lord of the Rings", "1984"],
        "restaurant": ["Italian Bistro", "Sushi Place", "Vegan Cafe"],
        "music": ["Jazz Fusion", "Classical Symphony", "Electronic Chill"],
    }

    items = recommendations.get(category.lower(), ["a generic item"])
    chosen_item = items[user_id % len(items)]  # Simple way to pick one

    return f"I recommend '{chosen_item}' in the '{category}' category."


@tool
def control_smart_device(
    user_id: int, device_name: str, command: str, value: Optional[Any] = None
) -> str:
    """Controls a smart device (e.g., 'light', 'thermostat', 'speaker'). Commands can be 'turn_on', 'turn_off', 'set_brightness', 'set_temperature', 'play_music'."""
    print(
        f"[TOOL] User {user_id} attempting to control device '{device_name}' with command '{command}' and value '{value}'"
    )
    session = get_session()
    try:
        device = (
            session.query(SmartDevice)
            .filter_by(user_id=user_id, name=device_name)
            .first()
        )
        if not device:
            return f"Error: Device '{device_name}' not found or not registered to user {user_id}."

        # Simulate device control
        if command == "turn_on":
            device.status["power"] = "on"
            response = f"Turned on {device_name}."
        elif command == "turn_off":
            device.status["power"] = "off"
            response = f"Turned off {device_name}."
        elif command == "set_brightness" and isinstance(value, (int, float)):
            device.status["brightness"] = value
            response = f"Set {device_name} brightness to {value}."
        elif command == "set_temperature" and isinstance(value, (int, float)):
            device.status["temperature"] = value
            response = f"Set {device_name} temperature to {value} degrees."
        elif command == "play_music" and isinstance(value, str):
            device.status["playing"] = value
            response = f"Playing {value} on {device_name}."
        else:
            response = (
                f"Unsupported command '{command}' or invalid value for {device_name}."
            )

        session.add(device)
        session.commit()
        return response
    except Exception as e:
        session.rollback()
        return f"An error occurred while controlling {device_name}: {str(e)}"
    finally:
        session.close()


@tool
def get_smart_device_status(user_id: int, device_name: str) -> str:
    """Gets the current status of a smart device (e.g., 'light', 'thermostat')."""
    print(f"[TOOL] Getting status for user {user_id} device: {device_name}")
    session = get_session()
    try:
        device = (
            session.query(SmartDevice)
            .filter_by(user_id=user_id, name=device_name)
            .first()
        )
        if not device:
            return f"Error: Device '{device_name}' not found or not registered to user {user_id}."

        return f"Status of {device_name}: {json.dumps(device.status)}"
    finally:
        session.close()


@tool
def get_user_preferences(user_id: int, preference_key: Optional[str] = None) -> str:
    """Retrieves user preferences. If preference_key is provided, returns that specific preference, otherwise returns all preferences."""
    print(f"[TOOL] Getting preferences for user {user_id}, key: {preference_key}")
    session = get_session()
    try:
        user_pref = session.query(UserPreference).filter_by(user_id=user_id).first()
        if not user_pref:
            return f"No preferences found for user {user_id}."

        preferences = user_pref.preferences
        if preference_key:
            return json.dumps(
                {preference_key: preferences.get(preference_key, "Not set")}
            )
        else:
            return json.dumps(preferences)
    finally:
        session.close()


@tool
def update_user_preference(
    user_id: int, preference_key: str, preference_value: str
) -> str:
    """Updates a specific user preference. Creates it if it doesn't exist."""
    print(
        f"[TOOL] Updating preference for user {user_id}: {preference_key} = {preference_value}"
    )
    session = get_session()
    try:
        user_pref = session.query(UserPreference).filter_by(user_id=user_id).first()
        if not user_pref:
            user_pref = UserPreference(user_id=user_id, preferences=json.dumps({}))
            session.add(user_pref)

        # Always deserialize preferences from JSON string
        if isinstance(user_pref.preferences, str):
            preferences = json.loads(user_pref.preferences)
        elif user_pref.preferences is None:
            preferences = {}
        else:
            preferences = user_pref.preferences

        preferences[preference_key] = preference_value
        user_pref.preferences = preferences
        session.commit()
        return f"Preference '{preference_key}' updated to '{preference_value}' for user {user_id}."
    except Exception as e:
        session.rollback()
        return f"Error updating preference: {str(e)}"
    finally:
        session.close()


# Combine all tools for the agent
personal_assistant_tools = [
    set_reminder,
    get_recommendation,
    control_smart_device,
    get_smart_device_status,
    get_user_preferences,
    update_user_preference,
]


class PersonalAssistantAgent:
    """Personal Virtual Assistant Agent"""

    def __init__(self):
        config = agent_config["personal_assistant_agent"]
        self.llm = ChatOpenAI(
            model=config["model_name"], temperature=config["temperature"]
        )

        self.agent_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful personal virtual assistant. You can set reminders, provide recommendations, control smart devices, and manage user preferences. Use the available tools to assist the user.",
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        self.agent = create_openai_functions_agent(
            self.llm, personal_assistant_tools, self.agent_prompt
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=personal_assistant_tools, verbose=True
        )

    def process_request(
        self, _user_id: int, message: str, chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Processes a user request using the personal assistant agent."""
        if chat_history is None:
            chat_history = []

        # Convert chat history to LangChain format
        lc_chat_history = []
        for entry in chat_history:
            if entry["role"] == "user":
                lc_chat_history.append(HumanMessage(content=entry["content"]))
            elif entry["role"] == "assistant":
                lc_chat_history.append(AIMessage(content=entry["content"]))

        try:
            response = self.agent_executor.invoke(
                {"input": message, "chat_history": lc_chat_history}
            )

            # Extract relevant information from the response
            # The response structure from AgentExecutor.invoke can vary based on tool use
            # If a tool is called, the 'output' key will contain the tool's result
            # If no tool is called, the 'output' key will contain the LLM's direct response

            return {
                "response": response.get(
                    "output", "I'm sorry, I couldn't process that request."
                ),
                "tool_used": response.get(
                    "tool_name"
                ),  # This might need more robust extraction
                "tool_output": response.get(
                    "tool_output"
                ),  # This might need more robust extraction
            }
        except Exception as e:
            return {
                "response": f"An error occurred: {str(e)}",
                "tool_used": None,
                "tool_output": None,
            }


def test_personal_assistant_agent():
    """Test the personal assistant agent with sample scenarios"""
    agent = PersonalAssistantAgent()
    user_id = 1  # Sample user ID

    print("=== Personal Assistant Agent Test ===\n")

    # Scenario 1: Set a reminder
    print("1. Setting a reminder:")
    res1 = agent.process_request(user_id, "Remind me to call mom tomorrow at 10 AM.")
    print(f"Assistant: {res1['response']}")
    print("-" * 80)

    # Scenario 2: Get a movie recommendation
    print("2. Getting a movie recommendation:")
    res2 = agent.process_request(user_id, "Can you recommend a good movie?")
    print(f"Assistant: {res2['response']}")
    print("-" * 80)

    # Scenario 3: Control a smart device (turn on light)
    print("3. Controlling a smart device (turn on light):")
    # Ensure a SmartDevice entry exists for user_id=1, name='living room light'
    # You might need to manually add this to your DB for testing
    res3 = agent.process_request(user_id, "Turn on the living room light.")
    print(f"Assistant: {res3['response']}")
    print("-" * 80)

    # Scenario 4: Get smart device status
    print("4. Getting smart device status:")
    res4 = agent.process_request(user_id, "What's the status of the living room light?")
    print(f"Assistant: {res4['response']}")
    print("-" * 80)

    # Scenario 5: Update user preference
    print("5. Updating user preference:")
    res5 = agent.process_request(user_id, "My favorite music genre is classical.")
    print(f"Assistant: {res5['response']}")
    print("-" * 80)

    # Scenario 6: Get user preference
    print("6. Getting user preference:")
    res6 = agent.process_request(user_id, "What's my favorite music genre?")
    print(f"Assistant: {res6['response']}")
    print("-" * 80)

    # Scenario 7: General query (no tool needed)
    print("7. General query:")
    res7 = agent.process_request(user_id, "Tell me a fun fact about space.")
    print(f"Assistant: {res7['response']}")
    print("-" * 80)


if __name__ == "__main__":
    test_personal_assistant_agent()
