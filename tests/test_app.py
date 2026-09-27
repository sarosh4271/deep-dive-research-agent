from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_renders_research_form():
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=20)

    assert not app.exception
    assert app.title[0].value == "🔎 Deep Dive Research Agent"
    assert app.text_area[0].label == "Research topic"
    assert app.button[0].label == "Research topic"
