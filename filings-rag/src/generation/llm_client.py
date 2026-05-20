"""Provider-agnostic LLM client (Groq + Gemini) with tenacity-based retry.

The provider is selected via the LLM_PROVIDER env var ("groq" or "gemini").
Switching providers requires only changing the env var — no code edits.
"""

import os

from tenacity import retry, stop_after_attempt, wait_exponential


class ProviderError(Exception):
    """Raised when LLM_PROVIDER is unrecognised."""


def build_llm():
    """Construct the LLM client for whichever provider is configured."""
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0.0,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0.0,
        )
    raise ProviderError(f"Unknown LLM_PROVIDER: {provider}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
)
def llm_invoke(llm, messages):
    """Invoke the LLM with exponential-backoff retry on transient failures."""
    return llm.invoke(messages)
