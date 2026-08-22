"""Provider adapters. Import the specific adapter you need — vendor SDKs are lazy."""

from .openai import OpenAIAdapter

__all__ = ["OpenAIAdapter"]
