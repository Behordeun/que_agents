from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.que_agents.core.schemas import Priority


@dataclass
class PVAAgentResponse:
    """Personal Virtual Assistant specific response structure"""

    # Core response fields
    message: str
    confidence: float

    # PVA specific fields
    intent: str
    entities: Dict[str, Any] = field(default_factory=dict)
    actions_taken: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Session and context fields
    session_id: Optional[str] = None
    timestamp: Optional[str] = None

    # Common agent fields (for compatibility)
    sentiment: str = "neutral"
    escalate: bool = False
    response_type: str = "pva_response"
    priority: Priority = Priority.MEDIUM

    # Additional PVA context
    user_context_used: bool = True
    knowledge_base_used: bool = False
    device_interactions: List[str] = field(default_factory=list)
    reminder_interactions: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Set timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for API responses"""
        return {
            "response": self.message,
            "intent": self.intent,
            "entities": self.entities,
            "confidence": self.confidence,
            "actions_taken": self.actions_taken,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "sentiment": self.sentiment,
            "response_type": self.response_type,
            "user_context_used": self.user_context_used,
            "knowledge_base_used": self.knowledge_base_used,
            "device_interactions": self.device_interactions,
            "reminder_interactions": self.reminder_interactions,
        }

    def add_device_interaction(self, device_name: str, action: str):
        """Add a device interaction to the response"""
        self.device_interactions.append(f"{device_name}: {action}")

    def add_reminder_interaction(self, reminder_title: str, action: str):
        """Add a reminder interaction to the response"""
        self.reminder_interactions.append(f"{reminder_title}: {action}")

    def set_knowledge_base_used(self, used: bool = True):
        """Mark that knowledge base was used in generating this response"""
        self.knowledge_base_used = used


@dataclass
class PVAInteractionLog:
    """Structure for logging PVA interactions"""

    user_id: str
    user_message: str
    response: PVAAgentResponse
    session_id: Optional[str] = None
    processing_time_ms: Optional[float] = None
    error_occurred: bool = False
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.session_id is None:
            self.session_id = f"pva_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
