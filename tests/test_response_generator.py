from pathlib import Path

from ai_chatbot_agent.conversation import ConversationManager, MemoryStore
from ai_chatbot_agent.response_generator import ResponseGenerator


def test_generate_response_stores_history(tmp_path: Path) -> None:
    manager = ConversationManager(MemoryStore(str(tmp_path / "store.json")))
    generator = ResponseGenerator(manager)
    # Force fallback for deterministic behavior in tests.
    generator._pipeline = None

    response, conversation_id = generator.generate_response(None, "Hello")
    assert conversation_id is not None
    assert "Hello" in response

    history = manager.get_history(conversation_id)
    assert len(history) == 2  # user + assistant
    assert history[0].content == "Hello"
    assert history[1].role == "assistant"


def test_generate_response_recovers_from_missing_context(tmp_path: Path) -> None:
    manager = ConversationManager(MemoryStore(str(tmp_path / "store.json")))
    generator = ResponseGenerator(manager)
    generator._pipeline = None

    conversation_id = manager.start_conversation()
    manager.add_message(conversation_id, "user", "First message")

    # Remove stored conversation to simulate loss of context.
    manager.store.clear(conversation_id)

    response, new_id = generator.generate_response(conversation_id, "New start")
    assert new_id != conversation_id
    assert "New start" in response
    history = manager.get_history(new_id)
    assert len(history) == 2
