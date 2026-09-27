"""Streamlit interface for the Deep Dive research agent."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from research_agent import (
    MAX_MODEL_ID_LENGTH,
    MAX_TOPIC_LENGTH,
    MODEL_OPTIONS,
    ResearchError,
    build_agent,
    research_topic,
    resolve_model,
)

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Deep Dive Research Agent", page_icon="🔎", layout="centered")
st.title("🔎 Deep Dive Research Agent")
st.caption("Give the agent a topic. It searches the web and writes a cited one-page report.")

MAX_RUNS_PER_SESSION = 5


def friendly_error(error: Exception) -> str:
    """Return a useful UI error without exposing request details."""
    if isinstance(error, AuthenticationError):
        return "OpenAI rejected the API key. Check it and try again."
    if isinstance(error, (PermissionDeniedError, NotFoundError)):
        return (
            "The selected model is unavailable to this API key or does not support this "
            "agent. Choose another model ID."
        )
    if isinstance(error, RateLimitError):
        return "The OpenAI account reached a rate, usage, or billing limit."
    if isinstance(error, BadRequestError):
        return "The model rejected the request. Try a shorter or more specific topic."
    if isinstance(error, (APITimeoutError, APIConnectionError)):
        return "The model connection timed out. Please retry in a moment."
    return "Research failed. Check the app logs and your network connection, then retry."


if "report" not in st.session_state:
    st.session_state.report = ""
if "runs" not in st.session_state:
    st.session_state.runs = 0

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Kept in this Streamlit session and never written by the app.",
    )
    configured_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini").strip()
    default_index = (
        MODEL_OPTIONS.index(configured_model) if configured_model in MODEL_OPTIONS else 0
    )
    selected_model = st.selectbox(
        "Suggested model",
        MODEL_OPTIONS,
        index=default_index,
        help="Choose one of six OpenAI models known to support tool calling.",
    )
    custom_model = st.text_input(
        "Custom model ID (optional)",
        value=configured_model if configured_model not in MODEL_OPTIONS else "",
        placeholder="Example: gpt-5.4-mini-2026-03-17",
        max_chars=MAX_MODEL_ID_LENGTH,
        help=(
            "Overrides the dropdown. The model must be available to your OpenAI API key "
            "and support Chat Completions with function calling."
        ),
    )
    try:
        model = resolve_model(selected_model, custom_model)
        model_error = None
        st.caption(f"Using `{model}`")
    except ValueError as error:
        model = selected_model
        model_error = error
    st.divider()
    st.caption(
        "Search uses DuckDuckGo without a search API key. Generating the report requires "
        "an OpenAI API key and sends the topic and search snippets to OpenAI."
    )
    st.caption(f"Session limit: {MAX_RUNS_PER_SESSION} reports.")

with st.form("research_form"):
    topic = st.text_area(
        "Research topic",
        placeholder="Example: Practical progress in quantum error correction since 2024",
        max_chars=MAX_TOPIC_LENGTH,
        height=110,
    )
    submitted = st.form_submit_button(
        "Research topic",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.runs >= MAX_RUNS_PER_SESSION,
    )

if submitted:
    if not api_key.strip():
        st.error("Enter an OpenAI API key in the sidebar.")
    else:
        try:
            if model_error:
                raise model_error
            with st.status("Researching the topic…", expanded=True) as status:
                st.write("Searching multiple angles and comparing sources")
                executor = build_agent(api_key, model)
                result = research_topic(executor, topic)
                report = str(result.get("output", "")).strip()
                if not report:
                    raise RuntimeError("The agent returned an empty report.")
                st.session_state.report = report
                st.session_state.runs += 1
                status.update(label="Report ready", state="complete", expanded=False)
        except ValueError as error:
            st.error(str(error))
        except ResearchError as error:
            logger.warning("Research could not produce a report: %s", error)
            st.error(str(error))
        except Exception as error:
            logger.exception("Research run failed")
            st.error(friendly_error(error))

if st.session_state.report:
    st.markdown(st.session_state.report)
    st.download_button(
        "Download report as Markdown",
        data=st.session_state.report,
        file_name="deep-dive-report.md",
        mime="text/markdown",
        use_container_width=True,
    )

with st.expander("How it works"):
    st.markdown(
        "The LangChain agent decides which queries to run, calls `DuckDuckGoSearchRun`, "
        "compares the returned snippets, and asks the selected model to synthesize a report. "
        "Open the cited links before relying on the report for important decisions."
    )
