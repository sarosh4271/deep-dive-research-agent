"""Core package for the Deep Dive research agent."""

from .agent import MAX_TOPIC_LENGTH, ResearchError, build_agent, research_topic

__all__ = ["MAX_TOPIC_LENGTH", "ResearchError", "build_agent", "research_topic"]
