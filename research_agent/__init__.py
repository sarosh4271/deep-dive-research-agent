"""Core package for the Deep Dive research agent."""

from .agent import MAX_TOPIC_LENGTH, ResearchError, build_agent, research_topic
from .config import MAX_MODEL_ID_LENGTH, MODEL_OPTIONS, resolve_model

__all__ = [
    "MAX_MODEL_ID_LENGTH",
    "MAX_TOPIC_LENGTH",
    "MODEL_OPTIONS",
    "ResearchError",
    "build_agent",
    "research_topic",
    "resolve_model",
]
