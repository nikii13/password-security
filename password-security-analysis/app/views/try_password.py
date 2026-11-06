import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (BAD, GOOD, MEDIUM, MODEL_NAMES, STRONG, WEAK, clean_passwords, feature_names,
                    load_model, need, selected_model_name, style)
from src.features.password_features import extract_features
from src.models.rule_based import classify

st.title("Try a password")
st.caption("See what a traditional checker says and what our model says. The password is processed in "
           "memory only: it is not saved or sent anywhere. Still, don't type a password you really use.")

best = selected_model_name()
model = load_model(best)
lr = load_model("logistic_regression")
need(model, what="the password checker (the trained model)")
features = feature_names()

RULE_LABELS = {"min_length_8": "8+ characters", "long_length_12": "12+ characters",
               "has_lowercase": "lowercase", "has_uppercase": "uppercase",
               "has_digit": "digit", "has_symbol": "symbol"}
EXAMPLES = ["Summer2026!", "P4ssw0rd7!", "Maria1990#", "qhzmtwxkpafr", "k8m2xq7wz4b", "xK9#mQ2vLp8$"]

if "pw" not in st.session_state:
    st.session_state.pw = "Summer2026!"


def set_example(p):
    st.session_state.pw = p


st.markdown("**Examples:**")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex, on_click=set_example, args=(ex,), use_container_width=True)

pw = st.text_input("Password to check", key="pw", type="password",
                   help="Click the eye icon to show it.")
if not pw:
    st.stop()

a = classify(pw)
f = extract_features(pw)
x = pd.DataFrame([f])[features]
p_best = float(model.predict_proba(x)[0, 1])
weak_b = p_best >= 0.5

# ---- verdicts ---------------------------------------------------------------------
left, right = st.columns(2)
with left:
    color = {"Weak": WEAK, "Medium": MEDIUM, "Strong": STRONG}[a["strength"]]
    pills = "".join(f"<span class='pill ok'>✔ {RULE_LABELS[r]}</span>" for r in a["passed_rules"])
    pills += "".join(f"<span class='pill no'>✘ {RULE_LABELS[r]}</span>" for r in a["failed_rules"])
    st.markdown(f"""
<div class="verdict" style="background:#143a63;border:2px solid {color}">
  <div class="vt">System A · rule-based checker says</div>
  <div class="vv" style="color:{color}">{a['strength']}</div>
  <div>{a['char_classes']} character types · {len(pw)} characters</div>
  <div style="margin-top:10px">{pills}</div>
</div>""", unsafe_allow_html=True)
with right:
    color = BAD if weak_b else GOOD
    verdict = "Predictable (weak)" if weak_b else "Not predictable"
    st.markdown(f"""
<div class="verdict" style="background:#143a63;border:2px solid {color}">
  <div class="vt">System B · {MODEL_NAMES[best]} says</div>
  <div class="vv" style="color:{color}">{verdict}</div>
  <div>Probability of being predictable: <b>{p_best:.0%}</b></div>
  <div style="margin-top:10px;background:#0d2b4e;border-radius:6px;height:14px;overflow:hidden">
    <div style="width:{p_best * 100:.1f}%;height:100%;background:{color}"></div></div>
</div>""", unsafe_allow_html=True)

a_weak = a["prediction"] == 1
if a_weak != weak_b:
    if weak_b:
        st.error("**The two systems disagree.** The rules accept this password, but the model recognises a "
                 "human pattern that attackers try early. This is exactly System A's blind spot.")
    else:
        st.info("**The two systems disagree.** The rules reject this password only because it lacks some "
                "character types, but it shows no human pattern. If it is random and long enough, "
                "it is hard to guess.")

leaked = clean_passwords()
if leaked is not None and pw in leaked:
    src = leaked[pw]
    msg = {"rockyou": "This exact password is in our sample of the **RockYou leak**: attackers have it in their lists.",
           "challenge": "This exact password is in our **challenge set** (built from attack templates).",
           "random": "This exact password is one of our **random** passwords."}[src]
    st.warning(msg)

# ---- why ---------------------------------------------------------------------------
st.markdown("#### Why?")
left, right = st.columns([1, 1.2])
with left:
    signals = [
        ("contains_common_word", "Contains a common word"),
        ("leet_used", "Uses l33t substitutions (4→a, 0→o…)"),
        ("contains_year", "Contains a year (1950–2039)"),
        ("keyboard_walk", "Keyboard run (qwer, asdf…)"),
        ("has_sequence", "Sequence (abc, 123…)"),
        ("word_digits_symbol", "Template: word + digits + symbol"),
        ("letters_then_digits", "Template: letters then digits"),
        ("starts_with_upper", "Only the first letter is capital"),
    ]
    found = [label for key, label in signals if f.get(key)]
    st.markdown("**Human patterns found**")
    if found:
        st.markdown("".join(f"<span class='pill no'>{s}</span>" for s in found), unsafe_allow_html=True)
    else:
        st.markdown("<span class='pill ok'>none of the typical patterns</span>", unsafe_allow_html=True)
    pattern = "".join("U" if c.isupper() else "L" if c.islower() else "D" if c.isdigit() else "S" for c in pw)
    st.markdown(f"**Character types:** `{pattern}`  →  **{f['class_transitions']}** changes of type")
    st.caption("People type in blocks (word, then numbers, then a symbol), so they change type few times. "
               "Random passwords change almost at every character.")
    pool = (26 * f["has_lower"] + 26 * f["has_upper"] + 10 * f["has_digits"] + 33 * f["has_symbols"]) or 1
    log_guesses = len(pw) * math.log10(pool)
    st.markdown(f"**Brute-force search space:** about 10^{log_guesses:.1f} guesses "
                f"({'above' if log_guesses >= 14 else 'below'} the 10¹⁴ offline threshold)")
    st.caption("Only meaningful if the password were random. A predictable password is guessed long "
               "before that, with dictionaries and templates.")

with right:
    if lr is not None:
        scaler, logreg = lr[0], lr[-1]
        z = (x.values[0] - scaler.mean_) / scaler.scale_
        contrib = pd.Series(z * logreg.coef_[0], index=features)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:10])[::-1]
        fig = go.Figure(go.Bar(y=top.index, x=top.values, orientation="h",
                               marker_color=[BAD if v > 0 else GOOD for v in top.values],
                               customdata=[f[k] for k in top.index],
                               hovertemplate="%{y} = %{customdata}<br>contribution %{x:.2f}<extra></extra>"))
        style(fig, height=380, legend=False)
        p_lr = float(lr.predict_proba(x)[0, 1])
        fig.update_layout(title=dict(text=f"Features that push the decision (Logistic Regression · {p_lr:.0%} weak)",
                                     font=dict(size=14)),
                          xaxis_title="← towards strong · towards weak →")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The Logistic Regression is used here because each feature's push can be read directly. "
                   "The Random Forest verdict above combines the same features in a non-linear way.")

with st.expander("All 30 feature values"):
    st.dataframe(pd.DataFrame({"value": f}).T[features].T, use_container_width=True, height=400)
