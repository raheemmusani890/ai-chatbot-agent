"""Response generation utilities with history awareness."""

from __future__ import annotations

from typing import Optional, Sequence

from .conversation import ConversationManager, ConversationNotFoundError, Message


class ResponseGenerator:
    """
    Generate context-aware responses.

    Attempts to use a lightweight transformers pipeline when available.
    Falls back to a rules-based responder when a pre-trained model
    cannot be loaded (for example in offline environments).
    """

    def __init__(self, conversation_manager: ConversationManager) -> None:
        self.manager = conversation_manager
        self._pipeline = self._maybe_load_pipeline()

    def _maybe_load_pipeline(self):
        try:
            from transformers import pipeline

            # Use a small causal LM; allow downstream callers to swap models.
            return pipeline("text-generation", model="sshleifer/tiny-gpt2")
        except Exception:
            return None

    def _render_history(self, messages: Sequence[Message]) -> str:
        formatted = []
        for message in messages:
            formatted.append(f"{message.role.title()}: {message.content}")
        return "\n".join(formatted)

    def _fallback_response(self, history: Sequence[Message], user_input: str) -> str:
        if history:
            last_speaker = history[-1].role
            summary = "; ".join(msg.content for msg in history[-3:])
            return (
                f"I recall our recent exchange ({summary}). "
                f"You just said: '{user_input}'. I'm responding based on that context."
            )
        return (
            "I'm starting a new conversation and will remember what you share. "
            f"You just said: '{user_input}'."
        )

    def generate_response(
        self, conversation_id: Optional[str], user_input: str
    ) -> tuple[str, str]:
        """
        Produce a reply and the conversation id used.

        Returns:
            (response_text, resolved_conversation_id)
        """
        resolved_id = self.manager.ensure_conversation(conversation_id)
        try:
            history = list(self.manager.get_history(resolved_id))
        except ConversationNotFoundError:
            # Unexpected loss of context; start a new conversation and notify the caller.
            resolved_id = self.manager.start_conversation()
            history = []
        # Append the user message before generating the response so history is up to date.
        self.manager.add_message(resolved_id, "user", user_input)
        history.append(Message(role="user", content=user_input))

        if self._pipeline is not None:
            prompt = self._render_history(history)
            try:
                output = self._pipeline(prompt, max_length=len(prompt.split()) + 20)
                text = output[0]["generated_text"]
                response = text[len(prompt) :].strip() or text.strip()
            except Exception:
                response = self._fallback_response(history[:-1], user_input)
        else:
            response = self._fallback_response(history[:-1], user_input)

        self.manager.add_message(resolved_id, "assistant", response)
        return response, resolved_id
