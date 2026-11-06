import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (BAD, COLOR_B, FEATURE_INFO, GOOD, GROUP_COLORS, MODEL_NAMES, SOURCE_COLORS,
                    confusion_figure, features_df, kpi, ml_predictions, need, selected_model_name,
                    style, table)

st.title("System B · machine learning")
st.caption("Two models trained on the 30 features of the train split. The best one is chosen with "
           "cross-validation, then evaluated once on the test and challenge sets.")

cv = table("ml_cv_results")
m = table("ml_metrics")
imp = table("ml_feature_importance")
need(cv, m, imp, what="System B")
best = selected_model_name()
bt = m[(m.model == best) & (m.subset == "test")].iloc[0]
bc = m[(m.model == best) & (m.subset == "challenge")].iloc[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Selected model", MODEL_NAMES[best], "best mean F1 in cross-validation", COLOR_B, size="1.55rem")
with c2:
    kpi("Test F1", f"{bt.f1:.3f}", f"accuracy {bt.accuracy:.3f}", COLOR_B)
with c3:
    kpi("Weak passwords missed", f"{int(bt.FN):,}", f"of {int(bt.TP + bt.FN):,} in test", GOOD)
with c4:
    kpi("Challenge detected", f"{int(bc.TP):,} / {int(bc.n):,}", f"recall {bc.recall:.3f}", GOOD)

st.write("")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model selection", "Test & challenge", "What the model looks at",
                                        "Robustness check", "Probabilities"])

with tab1:
    st.markdown("**5-fold cross-validation inside train.** Train on 4 folds, validate on the 5th, repeat 5 "
                "times. Families never cross folds. The test set is not touched here.")
    fig = go.Figure()
    metrics = ["accuracy", "precision", "recall", "f1"]
    for _, r in cv.iterrows():
        fig.add_bar(x=[x.capitalize() if x != "f1" else "F1" for x in metrics],
                    y=[r[f"{x}_mean"] for x in metrics],
                    error_y=dict(type="data", array=[r[f"{x}_std"] for x in metrics], color="#ffffff"),
                    name=MODEL_NAMES[r.model], marker_color=COLOR_B if r.model == best else "#5fa3e0",
                    text=[f"{r[f'{x}_mean']:.3f}" for x in metrics], textposition="outside")
    style(fig, height=380)
    fig.update_layout(barmode="group", yaxis=dict(range=[0.85, 1.0], title="mean over 5 folds"))
    st.plotly_chart(fig, use_container_width=True)
    show = cv.copy()
    show["model"] = show.model.map(MODEL_NAMES)
    st.dataframe(show.round(4), hide_index=True, use_container_width=True)
    st.caption("Logistic Regression: one weight per feature (interpretable). Random Forest: 200 decision "
               "trees that can combine features. The tiny standard deviation means the result does not "
               "depend on a lucky split.")

with tab2:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(confusion_figure(int(bt.TP), int(bt.FN), int(bt.FP), int(bt.TN),
                                         f"{MODEL_NAMES[best]} · test set ({int(bt.n):,})"),
                        use_container_width=True)
    with right:
        show = m.copy()
        show["model"] = show.model.map(MODEL_NAMES)
        st.markdown("##### Both models, evaluated once")
        st.dataframe(show[["model", "subset", "accuracy", "precision", "recall", "f1", "TP", "FN", "FP", "TN"]]
                     .round(3), hide_index=True, use_container_width=True)
        st.caption("Both models detect every challenge password, including those built from words that are "
                   "NOT in the feature dictionary.")
    by = table("ml_by_stratum")
    if by is not None:
        st.markdown("##### % flagged as weak, per group")
        t = by[by.split == "test"]
        st.dataframe(t[["source", "stratum", "n", "pct_predicted_predictable", "mean_probability"]].round(3),
                     hide_index=True, use_container_width=True)

with tab3:
    st.markdown("**Permutation importance:** how much the test F1 drops when one feature is shuffled. "
                "The bigger the drop, the more the Random Forest relies on it.")
    top = imp.head(15).iloc[::-1]
    colors = [GROUP_COLORS.get(FEATURE_INFO.get(f, ("", ""))[0], "#5fa3e0") for f in top.feature]
    fig = go.Figure(go.Bar(y=top.feature, x=top.rf_permutation_importance, orientation="h", marker_color=colors,
                           text=[f"{v:.3f}" for v in top.rf_permutation_importance], textposition="outside"))
    style(fig, height=520, legend=False)
    fig.update_layout(xaxis_title="drop in F1 when shuffled")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(" ".join(f"<span class='pill' style='background:{c};color:#0b1f36'>{g}</span>"
                         for g, c in GROUP_COLORS.items()), unsafe_allow_html=True)
    st.markdown("`class_transitions` dominates: people type in blocks (`Maria` + `2026` + `!` = 3 changes), "
                "random passwords change character type almost every character. That is the structure "
                "PCFG attacks exploit (Weir et al., 2009).")
    with st.expander("Logistic Regression coefficients (sign = direction)"):
        lr = imp[["feature", "lr_coefficient"]].sort_values("lr_coefficient")
        fig = go.Figure(go.Bar(y=lr.feature, x=lr.lr_coefficient, orientation="h",
                               marker_color=[BAD if v > 0 else GOOD for v in lr.lr_coefficient]))
        style(fig, height=760, legend=False)
        fig.update_layout(xaxis_title="standardised coefficient  (→ more weak · ← more strong)")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    ab = table("ml_ablation")
    if ab is None:
        st.info("Run `python -m src.models.train_ml --ablation` to add this check.")
    else:
        st.markdown("The Random Forest is retrained **without** groups of features. If detection collapsed, "
                    "the model would depend on a single shortcut.")
        names = {"all_features": "All 30 features", "no_class_transitions": "Without class_transitions",
                 "no_structure_features": "Without the 6 structure features",
                 "no_dictionary_features": "Without the dictionary features"}
        fig = go.Figure()
        for col, name, color in [("test_f1", "Test F1", "#5fa3e0"),
                                 ("challenge_recall_unseen", "Challenge (unseen words) detected", "#a98bff"),
                                 ("rule_compliant_rockyou_recall", "RockYou that passes the rules detected",
                                  SOURCE_COLORS["rockyou"])]:
            fig.add_bar(x=ab.experiment.map(names), y=ab[col] * 100, name=name, marker_color=color,
                        text=[f"{v * 100:.0f}%" for v in ab[col]], textposition="outside")
        style(fig, height=420)
        fig.update_layout(barmode="group", yaxis=dict(range=[0, 112], ticksuffix="%"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Even without the structure features, the challenge set is still fully detected: "
                   "the model combines several independent signals.")

with tab5:
    preds, feat = ml_predictions(), features_df()
    if preds is None or feat is None:
        st.info("Run the pipeline to see the predicted probabilities.")
    else:
        p = preds[preds.model == best].set_index("row_id").join(feat[["source"]])
        fig = go.Figure()
        for src in ["rockyou", "random", "challenge"]:
            fig.add_histogram(x=p.loc[p.source == src, "probability"], name=src.capitalize(),
                              marker_color=SOURCE_COLORS[src], histnorm="percent", nbinsx=40, opacity=0.8)
        fig.add_vline(x=0.5, line_dash="dash", line_color="#ffffff",
                      annotation_text="threshold 0.5", annotation_position="top")
        style(fig, height=400)
        fig.update_layout(barmode="overlay", xaxis_title="probability of being predictable (weak)",
                          yaxis=dict(title="% of passwords in the source", ticksuffix="%"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Test and challenge passwords only. Most passwords get a very confident score; "
                   "the errors are the few on the wrong side of 0.5.")
