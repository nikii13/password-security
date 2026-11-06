import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (FEATURE_INFO, GROUP_COLORS, SOURCE_COLORS, feature_names, features_df, need,
                    style, table)
from src.features.password_features import extract_features

st.title("The 30 features")
st.caption("password_features.py turns every password into 30 numbers. Only System B uses them; "
           "System A applies its 6 rules directly to the password.")

feat = features_df()
need(feat, what="the features page")
names = feature_names()
imp = table("ml_feature_importance")

tab1, tab2, tab3 = st.tabs(["What each feature means", "Weak vs strong, feature by feature",
                            "Compute them for any password"])

with tab1:
    info = pd.DataFrame([{"feature": f, "group": FEATURE_INFO.get(f, ("", ""))[0],
                          "meaning": FEATURE_INFO.get(f, ("", ""))[1]} for f in names])
    counts = info.group.value_counts()
    cols = st.columns(len(GROUP_COLORS))
    for col, (g, color) in zip(cols, GROUP_COLORS.items()):
        col.markdown(f"<div class='kpi'><div class='kl'>{g}</div>"
                     f"<div class='kv' style='color:{color}'>{counts.get(g, 0)}</div></div>",
                     unsafe_allow_html=True)
    st.write("")
    st.markdown("The **composition** group is what a traditional checker looks at. The other groups "
                "capture what attackers exploit: common words, years, keyboard runs and the "
                "*Word + digits + symbol* structure.")
    if imp is not None:
        info = info.merge(imp[["feature", "rf_permutation_importance"]], on="feature", how="left")
        info = info.rename(columns={"rf_permutation_importance": "importance (RF)"})
    config = {}
    if "importance (RF)" in info:
        config["importance (RF)"] = st.column_config.ProgressColumn(
            "importance (RF)", help="Drop in F1 when the feature is shuffled (System B page)",
            format="%.3f", min_value=0.0, max_value=float(info["importance (RF)"].max()))
    st.dataframe(info, hide_index=True, use_container_width=True, height=1090, column_config=config)

with tab2:
    st.markdown("Pick a feature and compare its values in leaked (weak), random (strong) and challenge passwords.")
    default = names.index("class_transitions") if "class_transitions" in names else 0
    f = st.selectbox("Feature", names, index=default,
                     format_func=lambda x: f"{x}  —  {FEATURE_INFO.get(x, ('', ''))[1]}")
    data = feat[[f, "source"]]
    is_binary = set(data[f].unique()) <= {0, 1}
    fig = go.Figure()
    for src in ["rockyou", "random", "challenge"]:
        vals = data.loc[data.source == src, f]
        if is_binary or vals.nunique() <= 25:
            vc = vals.value_counts(normalize=True).sort_index()
            fig.add_bar(x=vc.index.astype(str) if is_binary else vc.index, y=vc.values * 100,
                        name=src.capitalize(), marker_color=SOURCE_COLORS[src])
        else:
            fig.add_histogram(x=vals, name=src.capitalize(), marker_color=SOURCE_COLORS[src],
                              histnorm="percent", opacity=0.75, nbinsx=40)
    style(fig, height=400)
    fig.update_layout(barmode="group" if (is_binary or data[f].nunique() <= 25) else "overlay",
                      xaxis_title=f, yaxis=dict(title="% of passwords in the source", ticksuffix="%"))
    st.plotly_chart(fig, use_container_width=True)
    means = data.groupby("source")[f].mean().reindex(["rockyou", "random", "challenge"])
    c1, c2, c3 = st.columns(3)
    for col, src in zip([c1, c2, c3], means.index):
        col.metric(f"Mean · {src}", f"{means[src]:.2f}")

    st.markdown("##### Average of every feature, per source")
    st.caption("Values scaled 0–1 per feature (0 = lowest source average, 1 = highest) so they fit on one scale.")
    m = feat.groupby("source")[names].mean().reindex(["rockyou", "random", "challenge"])
    scaled = (m - m.min()) / (m.max() - m.min()).replace(0, 1)
    heat = go.Figure(go.Heatmap(z=scaled.values, x=names, y=[s.capitalize() for s in scaled.index],
                                customdata=m.values, colorscale=[[0, "#143a63"], [1, "#ffb547"]],
                                hovertemplate="%{y} · %{x}<br>mean = %{customdata:.2f}<extra></extra>",
                                showscale=False, xgap=2, ygap=2))
    style(heat, height=260, legend=False)
    heat.update_xaxes(tickangle=-50, tickfont=dict(size=11))
    st.plotly_chart(heat, use_container_width=True)

with tab3:
    pw = st.text_input("Password", value="Summer2026!", type="default",
                       help="Computed in memory only; nothing is saved.")
    pw2 = st.text_input("Compare with (optional)", value="xK9#mQ2vLp8$")
    if pw:
        rows = {pw: extract_features(pw)}
        if pw2:
            rows[pw2] = extract_features(pw2)
        df = pd.DataFrame(rows)
        df.insert(0, "group", [FEATURE_INFO.get(i, ("", ""))[0] for i in df.index])
        st.dataframe(df, use_container_width=True, height=600)
        st.page_link(st.session_state["pages"]["try"], label="See what both systems say about it",
                     icon=":material/key:")
