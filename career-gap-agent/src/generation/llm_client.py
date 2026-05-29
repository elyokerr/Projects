from __future__ import annotations

import os
from typing import Any


def build_chat_model(temperature: float = 0.0) -> Any:
    """Return a LangChain chat model: Groq if GROQ_API_KEY set, else Gemini."""
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=temperature)
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)
    raise RuntimeError("No LLM key set: provide GROQ_API_KEY or GOOGLE_API_KEY.")


def simple_complete(prompt: str, model: Any | None = None) -> str:
    model = model or build_chat_model()
    return model.invoke(prompt).content
