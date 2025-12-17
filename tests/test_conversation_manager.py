import json
from pathlib import Path

import pytest

from ai_chatbot_agent.conversation import (
    ConversationManager,
    ConversationNotFoundError,
    MemoryStore,
)


def test_start_and_recall_conversation(tmp_path: Path) -> None:
    store_path = tmp_path / "store.json"
    manager = ConversationManager(MemoryStore(str(store_path)))

    conversation_id = manager.start_conversation()
    manager.add_message(conversation_id, "user", "Hello")
    manager.add_message(conversation_id, "assistant", "Hi there!")

    history = manager.get_history(conversation_id)
    assert [message.content for message in history] == ["Hello", "Hi there!"]

    recent = manager.recall_recent(conversation_id, limit=1)
    assert len(recent) == 1
    assert recent[0].content == "Hi there!"


def test_ensure_conversation_recovers_missing_context(tmp_path: Path) -> None:
    store_path = tmp_path / "store.json"
    manager = ConversationManager(MemoryStore(str(store_path)))
    conversation_id = manager.start_conversation()

    # Simulate a corrupted or missing conversation by deleting it from the store.
    manager.store.clear(conversation_id)

    recovered_id = manager.ensure_conversation(conversation_id)
    assert recovered_id != conversation_id
    assert recovered_id in json.loads(store_path.read_text())


def test_missing_conversation_raises(tmp_path: Path) -> None:
    manager = ConversationManager(MemoryStore(str(tmp_path / "store.json")))
    with pytest.raises(ConversationNotFoundError):
        manager.get_history("missing-id")
