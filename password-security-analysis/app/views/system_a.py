import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (BAD, GOOD, MEDIUM, STRATUM_NAMES, STRATUM_ORDER, STRONG, WEAK, COLOR_A,
                    comparison_groups, confusion_figure, kpi, need, style, table)

st.title("System A · rule-based checker")
st.caption("What most websites do: look only at length and character types. No training, "
           "evaluated directly on the test and challenge sets.")

m = table("rule_based_metrics")
by = table("rule_based_by_stratum")
need(m, by, what="System A")
test = m[m.subset == "test"].iloc[0]
ch = m[m.subset == "challenge"].iloc[0]

left, right = st.columns([1, 1.3])
with left:
    st.markdown("#### The rules")
    st.markdown("""
| Result | Rule |
| --- | --- |
| **Weak** | fewer than 8 characters, or at most 2 character types |
| **Medium** | 8+ characters and 3 character types |
| **Strong** | 8+ characters and all 4 types, or 12+ characters and 3 types |

Character types: lowercase, uppercase, digits, symbols.
To compare with the ML model, **Weak and Medium count as predictable (1)**, Strong as not predictable (0).
""")
with right:
    st.markdown("#### Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Test F1", f"{test.f1:.3f}", f"accuracy {test.accuracy:.3f}", COLOR_A)
    with c2:
        kpi("Weak passwords missed", f"{int(test.FN):,}", f"of {int(test.TP + test.FN):,} in test", BAD)
    with c3:
        kpi("Challenge detected", f"{int(ch.TP)} / {int(ch.n):,}", "recall 0.000", BAD)

st.write("")
tab1, tab2, tab3 = st.tabs(["% correct per group", "Weak / Medium / Strong per group",
                            "Confusion matrix & metrics"])

with tab1:
    g = comparison_groups()
    if g is not None:
        correct = g.a.where(g.truth == "weak", 1 - g.a)
        colors = [GOOD if v >= 0.9 else BAD for v in correct]
        fig = go.Figure(go.Bar(x=g.short, y=correct * 100, marker_color=colors,
                               text=[f"{v * 100:.0f}%" for v in correct], textposition="outside",
                               cliponaxis=False, customdata=list(zip(g.group, g.n)),
                               hovertemplate="%{customdata[0]}<br>%{y:.1f}% correct<br>n = %{customdata[1]:,}<extra></extra>"))
        n_weak = int((g.truth == "weak").sum())
        fig.add_vline(x=n_weak - 0.5, line_dash="dash", line_color="#5fa3e0")
        fig.add_annotation(x=(n_weak - 1) / 2, y=118, text="<b>Really WEAK</b>", showarrow=False,
                           font=dict(color="#ff9b9b", size=15))
        fig.add_annotation(x=n_weak + (len(g) - n_weak - 1) / 2, y=118, text="<b>Really STRONG</b>",
                           showarrow=False, font=dict(color="#6fe0b5", size=15))
        style(fig, height=450, legend=False)
        fig.update_layout(yaxis=dict(range=[0, 128], tickvals=[0, 25, 50, 75, 100], ticktext=["0%", "25%", "50%", "75%", "100%"],
                                     title="% classified correctly"))
        fig.update_xaxes(tickfont=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Two blind spots, one cause.** It accepts weak passwords that follow the rules "
                    "(`Summer2026!` → Strong) and rejects strong random passwords without symbols "
                    "(`qhzmtwxkpafr` → Weak, ≈ 10¹⁷ guesses). It counts character types instead of "
                    "estimating how easy a password is to guess.")

with tab2:
    t = by[by.split == "test"].set_index("stratum").reindex(STRATUM_ORDER).dropna(subset=["n"]).reset_index()
    chal = by[by.split == "challenge"]
    w = chal.n / chal.n.sum()
    t = pd.concat([t, pd.DataFrame([{"stratum": "challenge", "n": chal.n.sum(),
                                     "pct_weak": (chal.pct_weak * w).sum(),
                                     "pct_medium": (chal.pct_medium * w).sum(),
                                     "pct_strong": (chal.pct_strong * w).sum()}])])
    t["name"] = t.stratum.map(lambda s: STRATUM_NAMES.get(s, "Challenge set"))
    fig = go.Figure()
    for col, name, color in [("pct_weak", "Weak", WEAK), ("pct_medium", "Medium", MEDIUM),
                             ("pct_strong", "Strong", STRONG)]:
        fig.add_bar(y=t.name, x=t[col] * 100, name=name, orientation="h", marker_color=color,
                    text=[f"{v * 100:.0f}%" if v >= 0.06 else "" for v in t[col]], textposition="inside",
                    insidetextfont=dict(color="#0b1f36"))
    style(fig, height=470)
    fig.update_layout(barmode="stack", xaxis=dict(ticksuffix="%", range=[0, 100]))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Full table (every challenge template, seen and unseen words)"):
        st.dataframe(by, hide_index=True, use_container_width=True)

with tab3:
    left, right = st.columns([1, 1.25])
    with left:
        st.plotly_chart(confusion_figure(int(test.TP), int(test.FN), int(test.FP), int(test.TN),
                                         f"Test set · {int(test.n):,} passwords"), use_container_width=True)
        st.caption("TP weak → flagged · FN weak → called Strong (the dangerous error) · "
                   "FP strong → flagged (false alarm) · TN strong → called Strong")
    with right:
        P, R = test.precision, test.recall
        st.markdown(f"""
| Metric | Formula | Calculation | Result |
| --- | --- | --- | --- |
| Accuracy | (TP + TN) / total | ({int(test.TP):,} + {int(test.TN):,}) / {int(test.n):,} | **{test.accuracy:.3f}** |
| Precision | TP / (TP + FP) | {int(test.TP):,} / {int(test.TP + test.FP):,} | **{P:.3f}** |
| Recall | TP / (TP + FN) | {int(test.TP):,} / {int(test.TP + test.FN):,} | **{R:.3f}** |
| F1 | 2·P·R / (P + R) | 2·{P:.3f}·{R:.3f} / ({P:.3f}+{R:.3f}) | **{test.f1:.3f}** |
| Challenge recall | TP / (TP + FN) | {int(ch.TP)} / {int(ch.n):,} | **{ch.recall:.3f}** |
""")
        st.caption("Positive class = predictable (weak). Recall is the share of weak passwords caught; "
                   "precision is how often a 'weak' verdict is right.")
