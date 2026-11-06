"""
Web interface for the project (Part 7).

Run from the repository root:
    streamlit run app/streamlit_app.py
or with Docker:
    docker compose up app        ->  http://localhost:8501

Pages live in app/views/. This file only sets up the navigation.
"""
import streamlit as st

st.set_page_config(page_title="Beyond Password Rules", page_icon="🔐", layout="wide")

from common import CSS  # noqa: E402  (after set_page_config)

st.markdown(CSS, unsafe_allow_html=True)

PAGES = {
    "home": st.Page("views/home.py", title="Overview", icon=":material/home:", default=True),
    "data": st.Page("views/data.py", title="Data", icon=":material/database:"),
    "features": st.Page("views/features.py", title="Features", icon=":material/tune:"),
    "system_a": st.Page("views/system_a.py", title="System A · rules", icon=":material/rule:"),
    "system_b": st.Page("views/system_b.py", title="System B · ML", icon=":material/psychology:"),
    "compare": st.Page("views/compare.py", title="A vs B", icon=":material/compare_arrows:"),
    "try": st.Page("views/try_password.py", title="Try a password", icon=":material/key:"),
}
st.session_state["pages"] = PAGES
st.navigation(list(PAGES.values()), position="top").run()
