"""
Smoke tests for the web interface (app/).

Each page is run headless with Streamlit's AppTest and must finish without
an exception. The pages read the pipeline outputs, so these tests are
skipped until the pipeline has been run (models/ and data/processed/ exist).
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
PAGES = ["views/home.py", "views/data.py", "views/features.py", "views/system_a.py",
         "views/system_b.py", "views/compare.py", "views/try_password.py"]

pipeline_done = ((ROOT / "models" / "random_forest.joblib").exists()
                 and (ROOT / "data" / "processed" / "password_features.csv").exists())
needs_pipeline = pytest.mark.skipif(not pipeline_done, reason="run the pipeline first")

st_testing = pytest.importorskip("streamlit.testing.v1")


def run_page(page):
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    if page != "views/home.py":
        at.switch_page(page).run()
    return at


@needs_pipeline
@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_errors(page):
    at = run_page(page)
    assert not at.exception, at.exception


@needs_pipeline
def test_try_password_shows_both_verdicts():
    at = run_page("views/try_password.py")
    at.text_input(key="pw").set_value("Summer2026!").run()
    html = " ".join(m.value for m in at.markdown)
    assert "Strong" in html                       # System A
    assert "Predictable (weak)" in html           # System B
    assert not at.exception
