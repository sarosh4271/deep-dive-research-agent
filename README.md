# Deep Dive Research Agent

A Streamlit research assistant built with LangChain. Give it a topic and the agent runs
multiple DuckDuckGo searches, compares the results, and writes a concise report with clickable
citations.

## Features

- LangChain `AgentExecutor` with a tool-calling OpenAI model
- `DuckDuckGoSearchRun` (no search API key required)
- Multiple autonomous searches per topic
- Six suggested OpenAI models plus a custom model-ID override
- A 500-700 word report with inline citations and a source list
- Automatic final synthesis if the autonomous tool loop ends without returning a report
- Streamlit UI and Markdown report downloads
- Bounded runtime, agent iterations, topic length, and reports per browser session
- Web content is treated as untrusted evidence rather than agent instructions

## Architecture

```text
Topic → Streamlit → AgentExecutor → DuckDuckGoSearchRun
                       ↑                    │
                       └── search results ──┘
                                  │
                                  ▼
                         cited Markdown report
```

The standard `DuckDuckGoSearchRun` normally flattens results into snippets and drops their URLs.
This project keeps that tool but supplies a small wrapper that returns each result's title, URL,
and snippet so the model can produce real links.

## Run locally

Python 3.10+ is recommended.

```bash
cd deep-dive-research-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
# Add your OpenAI API key to .env, then:
streamlit run app.py
```

You can leave `.env` unset and paste a key into the sidebar for the current session. Never commit
`.env` or a real API key.

The sidebar includes six tool-calling model presets. To use a snapshot or another model, enter its
OpenAI model ID in **Custom model ID**; that value overrides the dropdown. Custom models must be
available to the supplied API key and support Chat Completions with function calling.

## Test and lint

```bash
pytest
ruff check .
```

## Important limitation

DuckDuckGo returns search-result snippets, not the complete text of every linked article. The
agent cross-checks multiple results and cites their URLs, but the report is still a research
starting point—not a substitute for opening and verifying the primary sources. Search pages can
also contain inaccurate or adversarial text, so the prompt explicitly treats them as data rather
than instructions.
