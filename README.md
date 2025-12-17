# ai-chatbot-agent

Lightweight conversation manager and response generator that remembers context across turns.

## Features

- JSON-backed memory store with recovery when context is lost.
- Conversation manager API for appending and recalling messages.
- Response generator that uses a pre-trained `tiny-gpt2` model when available and falls back to a deterministic rules-based responder.
- Simple pytest suite.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running tests

```bash
pytest
```

### Example usage

```python
from ai_chatbot_agent import ConversationManager, ResponseGenerator

manager = ConversationManager()
generator = ResponseGenerator(manager)

response, conversation_id = generator.generate_response(None, "Hello!")
print(response)
```
