import streamlit as st

from common import (COLOR_A, COLOR_B, GOOD, BAD, ab_group_figure, comparison_groups, kpi,
                    metric_row, need, selected_model_name, table, MODEL_NAMES)

pages = st.session_state["pages"]

st.title("Beyond Password Rules")
st.markdown("##### Can a lightweight machine-learning model detect weak passwords that "
            "traditional rule-based checkers call secure?")

rows = metric_row("test")
ch = metric_row("challenge")
groups = comparison_groups()
need(rows, ch, groups, what="the overview")
(a_test, b_test), (a_ch, b_ch) = rows, ch
split = table("split_summary")
n_total = int(split.rows.sum()) if split is not None else None

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Test F1 · System A", f"{a_test.f1:.3f}", "rule-based checker", COLOR_A)
with c2:
    kpi("Test F1 · System B", f"{b_test.f1:.3f}", MODEL_NAMES[selected_model_name()], COLOR_B)
with c3:
    kpi("Challenge detected", f"{int(a_ch.TP)} → {int(b_ch.TP):,}",
        f"of {int(a_ch.n):,} rule-compliant weak passwords", GOOD)
with c4:
    kpi("Passwords analysed", f"{n_total:,}" if n_total else "–", "30 features each")

st.write("")
left, right = st.columns([2.2, 1])
with left:
    st.markdown("#### Who gets it right?")
    st.caption("Share of each group flagged as weak. Left of the line: weak passwords "
               "(higher is better). Right: strong random passwords (lower is better).")
    st.plotly_chart(ab_group_figure(groups, height=430), use_container_width=True)
with right:
    st.markdown("#### In one sentence")
    st.markdown(
        f"""
**System A** counts character types, so it calls every password that follows the rules
*Strong* — including **{int(a_ch.FN):,} of {int(a_ch.n):,}** passwords built from attack
templates like `Summer2026!` — and calls long random passwords without symbols *Weak*.

**System B** learns from 30 features (words, years, structure…) and detects
**{int(b_ch.TP):,} of {int(b_ch.n):,}** of those passwords, while flagging only
**{100 * b_test.FP / (b_test.FP + b_test.TN):.1f}%** of strong random passwords.
""")
    st.markdown("#### Explore")
    st.page_link(pages["compare"], label="System A vs System B", icon=":material/compare_arrows:")
    st.page_link(pages["try"], label="Try your own password", icon=":material/key:")
    st.page_link(pages["data"], label="The data", icon=":material/database:")
    st.page_link(pages["features"], label="The 30 features", icon=":material/tune:")
    st.page_link(pages["system_a"], label="System A in detail", icon=":material/rule:")
    st.page_link(pages["system_b"], label="System B in detail", icon=":material/psychology:")

st.divider()
b1, b2 = st.columns(2)
with b1:
    if st.button("Compare both systems  →", use_container_width=True, type="primary"):
        st.switch_page(pages["compare"])
with b2:
    if st.button("Test your own password  →", use_container_width=True):
        st.switch_page(pages["try"])
