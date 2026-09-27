from __future__ import annotations

import json

import pytest

from research_agent.agent import CitationSearchWrapper, research_topic, validate_topic


class FakeSearchWrapper(CitationSearchWrapper):
    def results(self, query: str, max_results: int, source: str | None = None):
        return [
            {
                "title": "Example source",
                "link": "https://example.com/research",
                "snippet": f"Evidence about {query}",
            }
        ]


class RetrySearchWrapper(CitationSearchWrapper):
    attempted_backends: list[str] = []

    def results(self, query: str, max_results: int, source: str | None = None):
        self.attempted_backends.append(self.backend)
        if self.backend == "duckduckgo":
            raise OSError("temporary provider failure")
        return [{"title": "Fallback", "link": "https://example.com", "snippet": query}]


class FailedSearchWrapper(CitationSearchWrapper):
    def results(self, query: str, max_results: int, source: str | None = None):
        raise OSError("temporary provider failure")


class FakeSynthesisChain:
    def __init__(self, report: str = "# Recovered report"):
        self.report = report
        self.inputs = None

    def invoke(self, inputs):
        self.inputs = inputs
        return self.report


class FakeExecutor:
    def __init__(self, result, tools=None):
        self.result = result
        self.tools = tools or []
        self.synthesis_chain = FakeSynthesisChain()

    def invoke(self, inputs):
        return dict(self.result)


def test_search_output_preserves_citation_metadata():
    # Bypass the dependency check because this fake never calls the ddgs package.
    wrapper = FakeSearchWrapper.model_construct(max_results=3, source="text")
    output = wrapper.run("quantum computing")
    result = json.loads(output)[0]

    assert result["title"] == "Example source"
    assert result["link"] == "https://example.com/research"
    assert "quantum computing" in result["snippet"]


def test_search_retries_providers_independently():
    wrapper = RetrySearchWrapper.model_construct(
        max_results=3,
        source="text",
        backend="duckduckgo",
        attempted_backends=[],
    )

    result = json.loads(wrapper.run("quantum computing"))[0]

    assert result["title"] == "Fallback"
    assert wrapper.attempted_backends == ["duckduckgo", "brave"]


def test_search_returns_tool_message_when_all_providers_fail():
    wrapper = FailedSearchWrapper.model_construct(
        max_results=3, source="text", backend="duckduckgo"
    )

    output = wrapper.run("quantum computing")

    assert output.startswith("No useful DuckDuckGo search results")


def test_validate_topic_normalizes_whitespace():
    assert validate_topic("  quantum   error\ncorrection ") == "quantum error correction"


@pytest.mark.parametrize("topic", ["", "   ", "\n\t"])
def test_validate_topic_rejects_empty_input(topic: str):
    with pytest.raises(ValueError, match="research topic"):
        validate_topic(topic)


def test_research_topic_preserves_a_completed_report():
    executor = FakeExecutor({"output": "# Finished", "intermediate_steps": []})

    result = research_topic(executor, "test topic")

    assert result["output"] == "# Finished"
    assert executor.synthesis_chain.inputs is None


def test_research_topic_synthesizes_when_agent_output_is_empty():
    evidence = '[{"title":"Source","link":"https://example.com","snippet":"Fact"}]'
    executor = FakeExecutor({"output": "", "intermediate_steps": [(object(), evidence)]})

    result = research_topic(executor, "test topic")

    assert result["output"] == "# Recovered report"
    assert result["used_synthesis_fallback"] is True
    assert evidence in executor.synthesis_chain.inputs["evidence"]
