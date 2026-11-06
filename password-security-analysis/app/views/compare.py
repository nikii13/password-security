import plotly.graph_objects as go
import streamlit as st

from common import (COLOR_A, COLOR_B, MODEL_NAMES, ab_group_figure, comparison_groups,
                    confusion_figure, metric_row, need, selected_model_name, style)

best = selected_model_name()
st.title("System A vs System B")
st.caption(f"Same passwords, same test set (never used for training) and same challenge set. "
           f"System A = rule-based checker · System B = {MODEL_NAMES[best]}.")

test, ch = metric_row("test"), metric_row("challenge")
groups = comparison_groups()
need(test, ch, groups, what="the comparison")
(a, b), (ac, bc) = test, ch

# ---- 1. headline numbers -------------------------------------------------------
rows = [
    ("F1 (test)", f"{a.f1:.3f}", f"{b.f1:.3f}"),
    ("Accuracy (test)", f"{a.accuracy:.3f}", f"{b.accuracy:.3f}"),
    ("Weak passwords missed (FN)", f"{int(a.FN):,}", f"{int(b.FN):,}"),
    ("False alarms on strong ones (FP)", f"{int(a.FP):,}", f"{int(b.FP):,}"),
    ("Challenge passwords detected", f"{int(ac.TP)} / {int(ac.n):,}", f"{int(bc.TP):,} / {int(bc.n):,}"),
]
html = "".join(
    f"<tr><td>{n}</td><td style='color:{COLOR_A}'>{va}</td><td style='color:{COLOR_B}'><b>{vb}</b></td></tr>"
    for n, va, vb in rows)
left, right = st.columns([1.1, 1])
with left:
    st.markdown(f"""
<style>
.cmp {{ width:100%; border-collapse:collapse; font-size:1.05rem; }}
.cmp th {{ text-align:left; color:#9fbfe0; font-weight:600; padding:8px 10px; border-bottom:2px solid #3f7bb5;
           font-size:.85rem; text-transform:uppercase; letter-spacing:.05em; }}
.cmp td {{ padding:11px 10px; border-bottom:1px solid #22507f; }}
.cmp td:nth-child(2), .cmp td:nth-child(3), .cmp th:nth-child(2), .cmp th:nth-child(3) {{ text-align:right; }}
.cmp td:nth-child(2), .cmp td:nth-child(3) {{ font-size:1.25rem; font-weight:700; white-space:nowrap; }}
.cmp th {{ white-space:nowrap; }}
</style>
<table class="cmp"><tr><th></th><th>System A · rules</th><th>System B · ML</th></tr>{html}</table>
""", unsafe_allow_html=True)
with right:
    names = ["Accuracy", "Precision", "Recall", "F1"]
    fig = go.Figure()
    for r, name, color in [(a, "System A (rules)", COLOR_A), (b, "System B (ML)", COLOR_B)]:
        vals = [r.accuracy, r.precision, r.recall, r.f1]
        fig.add_bar(x=names, y=vals, name=name, marker_color=color,
                    text=[f"{v:.3f}" for v in vals], textposition="outside", textfont=dict(size=14))
    style(fig, height=330)
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.12], title="test set"), bargap=0.3,
                      margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

# ---- 2. per group --------------------------------------------------------------
st.markdown("#### Group by group")
mode = st.radio("Show", ["% classified correctly", "% flagged as weak"], horizontal=True,
                label_visibility="collapsed")
st.plotly_chart(ab_group_figure(groups, height=440, correct=mode.startswith("% classified")),
                use_container_width=True)
st.caption("Challenge passwords pass every rule. 'New words' means the base word is "
           "absent from the feature dictionary, so the model must recognise the structure.")

# ---- 3. confusion matrices -----------------------------------------------------
st.markdown("#### Where the errors are (test set)")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(confusion_figure(int(a.TP), int(a.FN), int(a.FP), int(a.TN), "System A · rules"),
                    use_container_width=True)
with c2:
    st.plotly_chart(confusion_figure(int(b.TP), int(b.FN), int(b.FP), int(b.TN),
                                     f"System B · {MODEL_NAMES[best]}"), use_container_width=True)
fn_cut = 1 - b.FN / a.FN
fp_cut = 1 - b.FP / a.FP
st.markdown(f"System B misses **{fn_cut:.0%} fewer** weak passwords and raises **{fp_cut:.0%} fewer** "
            f"false alarms than System A.")

with st.expander("Limitations"):
    st.markdown("""
- The challenge set follows block templates (word, digits, symbol) similar to human RockYou passwords,
  so 100% detection is optimistic. Passphrases (`ilovemydogmax`) and weak passwords that *look* random were not tested.
- Labels are operational: they come from the source of each password, not from a measured guess count.
- RockYou is from 2009 and the word dictionary is small (about 150 words).
""")
