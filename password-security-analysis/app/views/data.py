import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (PROCESSED, SOURCE_COLORS, STRATUM_NAMES, features_df, kpi, need, style,
                    table)

st.title("The data")
st.caption("Three sources merged into one dataset. Labels come from the source, never from System A.")

split = table("split_summary")
strata = table("rockyou_strata")
comps = table("random_compositions")
clean_log = table("cleaning_log")
need(split, strata, comps, what="the data page")

by_source = split.groupby("source").rows.sum()
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("RockYou", f"{by_source.get('rockyou', 0):,}", "real leaked passwords · label 1", SOURCE_COLORS["rockyou"])
with c2:
    kpi("Random", f"{by_source.get('random', 0):,}", "secure generator · ≥ 10¹⁴ guesses · label 0",
        SOURCE_COLORS["random"])
with c3:
    kpi("Challenge", f"{by_source.get('challenge', 0):,}", "pass every rule but predictable · label 1",
        SOURCE_COLORS["challenge"])
with c4:
    kpi("Total", f"{int(split.rows.sum()):,}", "after cleaning and removing duplicates")

st.write("")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Train / test / challenge", "RockYou sample", "Random sample",
                                        "Challenge set", "Example rows"])

with tab1:
    left, right = st.columns([1.4, 1])
    with left:
        fig = go.Figure()
        for src in ["rockyou", "random", "challenge"]:
            part = split[split.source == src].set_index("split").reindex(["train", "test", "challenge"]).fillna(0)
            fig.add_bar(y=["Train (80%)", "Test (20%)", "Challenge"], x=part.rows, name=src.capitalize(),
                        orientation="h", marker_color=SOURCE_COLORS[src],
                        text=[f"{int(v):,}" if v else "" for v in part.rows],
                        textposition="outside" if src == "challenge" else "inside")
        style(fig, height=330)
        fig.update_layout(barmode="stack", xaxis_title="passwords")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("""
**Train (80%)** — the model learns here. Split into 5 folds for cross-validation,
used to choose between Logistic Regression and Random Forest.

**Test (20%)** — the final exam, used **once**, by both systems.

**Challenge** — never used for training or model choice. It only answers the research question.

Variants of the same password (`Maria2026!`, `maria2027#`) form a *family* and always
stay on the same side of the split, so the test score is not inflated (data leakage).
""")
    st.dataframe(split.rename(columns={"rows": "passwords"}), hide_index=True, use_container_width=True)

with tab2:
    st.markdown("`rockyou.txt` is sorted from most to least used password, so the line number "
                "tells how common each one is. We sample by frequency band.")
    s = strata.copy()
    s["name"] = s.stratum.map(STRATUM_NAMES)
    left, right = st.columns([1.3, 1])
    with left:
        fig = go.Figure(go.Bar(x=s.name, y=s.sampled, marker_color=SOURCE_COLORS["rockyou"],
                               text=[f"{v:,}" for v in s.sampled], textposition="outside"))
        style(fig, height=340, legend=False)
        fig.update_layout(yaxis_title="passwords sampled")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.dataframe(s[["name", "rank_range", "pool_size", "sampled"]].rename(
            columns={"name": "stratum", "rank_range": "line numbers", "pool_size": "available"}),
            hide_index=True, use_container_width=True)
        st.caption("The *passes the rules* group is added on purpose: without it almost no human "
                   "password would have a symbol, and the model would learn \"symbol = strong\".")
    log = table("rockyou_cleaning_log")
    if log is not None:
        with st.expander("How rockyou.txt was cleaned"):
            st.dataframe(log, hide_index=True, use_container_width=True)

with tab3:
    st.markdown("Generated with Python `secrets` (cryptographically secure). Every composition has a "
                "minimum length so that brute force needs **at least 10¹⁴ guesses** "
                "(offline-attack threshold, Florêncio et al. 2014).")
    c = comps.copy()
    c["name"] = c.composition.map(STRATUM_NAMES)
    left, right = st.columns([1.3, 1])
    with left:
        fig = go.Figure(go.Bar(x=c.name, y=c.min_log10_guesses, marker_color=SOURCE_COLORS["random"],
                               text=[f"10^{v:.1f}" for v in c.min_log10_guesses], textposition="outside"))
        fig.add_hline(y=14, line_dash="dash", line_color="#ff9b9b",
                      annotation_text="threshold 10¹⁴", annotation_position="top left")
        style(fig, height=340, legend=False)
        fig.update_layout(yaxis=dict(title="log10(guesses) of the shortest password", range=[0, 17.5]))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.dataframe(c[["name", "alphabet_size", "min_length", "generated"]].rename(
            columns={"name": "composition", "alphabet_size": "chars", "min_length": "min length"}),
            hide_index=True, use_container_width=True)
        st.caption("Compositions are varied on purpose: if every random password had a symbol, "
                   "the model would just learn the rule System A already uses.")

feat = features_df()

with tab4:
    st.markdown("Passwords built from attack templates. **All of them pass every composition rule.** "
                "Half use words that are in our feature dictionary (*seen*), half use words that are not "
                "(*unseen*), to check that the model detects the structure and not just the words.")
    if feat is not None:
        ch = feat[feat.source == "challenge"].groupby(["stratum", "vocab"]).size().unstack(fill_value=0)
        ch.index = ch.index.map(STRATUM_NAMES)
        fig = go.Figure()
        for vocab, color in [("seen", "#a98bff"), ("unseen", "#d7c8ff")]:
            if vocab in ch:
                fig.add_bar(x=ch.index, y=ch[vocab], name=f"{vocab} words", marker_color=color,
                            text=ch[vocab], textposition="outside")
        style(fig, height=340)
        fig.update_layout(barmode="group", yaxis_title="passwords")
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown("A few real rows from `passwords_clean.csv`, the merged file every later step reads.")
    path = PROCESSED / "passwords_clean.csv"
    if path.exists():
        @st.cache_data
        def examples():
            df = pd.read_csv(path, keep_default_na=False, dtype=str)
            parts = [df[df.source == s].sample(4, random_state=7) for s in ["rockyou", "random", "challenge"]]
            return pd.concat(parts)
        st.dataframe(examples(), hide_index=True, use_container_width=True)
        st.caption("label 1 = predictable (weak), 0 = not predictable. `rule_compliant` only says whether "
                   "it passes the classic rules; it is NOT the label and the model never sees it.")
    else:
        st.info("Run the pipeline to see example rows.")
