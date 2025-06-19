"""LLM initialization module."""

from typing import Optional
from openai import OpenAI

def init_chat_model(model_name: str, model_provider: str = "openai") -> OpenAI:
    """Initialize a chat model."""
    return OpenAI() 