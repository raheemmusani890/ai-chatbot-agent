"""Conversation management utilities for tracking context across turns."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


class ConversationNotFoundError(KeyError):
    """Raised when a conversation is missing from storage."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(conversation_id)
        self.conversation_id = conversation_id

    def __str__(self) -> str:
        return f"Conversation '{self.conversation_id}' was not found."


@dataclass
class Message:
    """A single turn in a conversation."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, str]) -> "Message":
        return cls(
            role=payload["role"],
            content=payload["content"],
            timestamp=payload.get("timestamp", time.time()),
        )


class MemoryStore:
    """Simple JSON-backed memory store for conversations."""

    def __init__(self, path: str = ".conversation_store.json") -> None:
        self.path = path
        self._data: Dict[str, List[Dict[str, str]]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fp:
                    self._data = json.load(fp)
            except json.JSONDecodeError:
                # Reset corrupt storage and start fresh.
                self._data = {}
        else:
            self._data = {}

    def _persist(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fp:
            json.dump(self._data, fp, ensure_ascii=False, indent=2)

    def start(self, conversation_id: Optional[str] = None) -> str:
        conversation_id = conversation_id or str(uuid.uuid4())
        if conversation_id not in self._data:
            self._data[conversation_id] = []
            self._persist()
        return conversation_id

    def append(self, conversation_id: str, message: Message) -> None:
        if conversation_id not in self._data:
            raise ConversationNotFoundError(conversation_id)
        self._data[conversation_id].append(message.to_dict())
        self._persist()

    def get(self, conversation_id: str) -> List[Message]:
        if conversation_id not in self._data:
            raise ConversationNotFoundError(conversation_id)
        return [Message.from_dict(item) for item in self._data[conversation_id]]

    def clear(self, conversation_id: str) -> None:
        if conversation_id not in self._data:
            raise ConversationNotFoundError(conversation_id)
        del self._data[conversation_id]
        self._persist()


class ConversationManager:
    """Public interface for maintaining conversation context."""

    def __init__(self, store: Optional[MemoryStore] = None) -> None:
        self.store = store or MemoryStore()

    def start_conversation(self, conversation_id: Optional[str] = None) -> str:
        return self.store.start(conversation_id)

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        message = Message(role=role, content=content)
        self.store.append(conversation_id, message)

    def get_history(self, conversation_id: str) -> Sequence[Message]:
        return self.store.get(conversation_id)

    def recall_recent(self, conversation_id: str, limit: int = 5) -> Sequence[Message]:
        history = list(self.get_history(conversation_id))
        return history[-limit:]

    def ensure_conversation(self, conversation_id: Optional[str]) -> str:
        """
        Validate or create a conversation.

        Useful for callers that may have lost context; a new conversation
        identifier is created when the provided one is missing.
        """
        if conversation_id is None:
            return self.start_conversation()
        try:
            # Accessing history ensures the id exists.
            self.get_history(conversation_id)
            return conversation_id
        except ConversationNotFoundError:
            return self.start_conversation()
