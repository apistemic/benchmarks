import os

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


def get_embedding_llms():
    """Get configured embedding models."""
    return [
        GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", google_api_key=os.environ["GEMINI_API_KEY"]
        ),
        OpenAIEmbeddings(model="text-embedding-ada-002"),
        OpenAIEmbeddings(model="text-embedding-3-small"),
        OpenAIEmbeddings(model="text-embedding-3-large"),
    ]


def get_chat_llms_by_key():
    """Get configured chat models."""
    google_api_key = os.environ["GEMINI_API_KEY"]
    return {
        "google__gemini-2.5-flash-lite": ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", google_api_key=google_api_key
        ),
        "google__gemini-2.5-flash": ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", google_api_key=google_api_key
        ),
        "google__gemini-2.5-pro": ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", google_api_key=google_api_key
        ),
        "anthropic__claude-opus-4-1": ChatAnthropic(
            model="claude-opus-4-1-20250805", timeout=30
        ),
        "anthropic__claude-sonnet-4": ChatAnthropic(
            model="claude-sonnet-4-20250514", timeout=30
        ),
        "anthropic__claude-3-5-haiku": ChatAnthropic(
            model="claude-3-5-haiku-20241022", timeout=30
        ),
        "openai__gpt-5": ChatOpenAI(model="gpt-5"),
        "openai__gpt-5-mini": ChatOpenAI(model="gpt-5-mini"),
        "openai__gpt-5-nano": ChatOpenAI(model="gpt-5-nano"),
    }
