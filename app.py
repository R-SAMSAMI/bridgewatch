from __future__ import annotations

import pandas as pd
import streamlit as st

from src.constants import FHWA_DOWNLOAD_URL, FHWA_YEAR, MODEL_SAMPLE_LIMIT, PROCESSED_DATA_PATH
from src.data import load_bridgewatch_data
from src.modeling import (
    build_model_bundle,
    explain_tree_prediction,
    export_tree_rules,
    get_confusion,
    get_feature_importance,
    LEAKING_FEATURES,
    get_pipeline,
    score_state_bridges,
)
from src.visuals import confusion_chart, feature_importance_chart, state_summary_chart

st.set_page_config(page_title="BridgeWatch", page_icon=":bridge_at_night:", layout="wide")

if st.query_params.get("reset") == "1":
    st.cache_data.clear()
    st.cache_resource.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.query_params.clear()

st.markdown(
    """
    <style>
    :root {
        --bg: #f7f4ec;
        --panel: rgba(255,255,255,0.92);
        --panel-strong: rgba(255,255,255,0.98);
        --ink: #172033;
        --muted: #586174;
        --accent: #14532d;
        --accent-soft: #dcfce7;
        --rust: #b45309;
        --line: rgba(23, 32, 51, 0.10);
        --shadow: 0 16px 42px rgba(23, 32, 51, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(180, 83, 9, 0.10), transparent 22%),
            radial-gradient(circle at top right, rgba(20, 83, 45, 0.12), transparent 24%),
            linear-gradient(180deg, #faf8f2 0%, var(--bg) 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, rgba(255,255,255,0.97), rgba(220,252,231,0.72));
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 1.8rem;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--rust);
        margin-bottom: 0.55rem;
    }

    .hero-title {
        font-size: 3rem;
        line-height: 1;
        margin-bottom: 0.75rem;
        font-weight: 700;
    }

    .hero-copy {
        color: var(--muted);
        max-width: 900px;
        font-size: 1.03rem;
        margin-bottom: 0.9rem;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }

    .chip {
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid rgba(20, 83, 45, 0.14);
        padding: 0.42rem 0.78rem;
        font-weight: 700;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1rem 1rem 0.65rem 1rem;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .callout {
        background: rgba(255,255,255,0.75);
        border: 1px solid var(--line);
        border-left: 6px solid var(--rust);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        color: var(--muted);
        margin-bottom: 1rem;
    }

    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.84);
        border-right: 1px solid var(--line);
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.70);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.8rem;
    }

    .path-step {
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.85rem;
        margin-bottom: 0.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data(force_refresh: bool = False, cache_buster: int = 0) -> pd.DataFrame:
    _ = cache_buster
    return load_bridgewatch_data(force_refresh=force_refresh)


@st.cache_resource(show_spinner=False)
def get_models(data_signature: tuple[int, int], frame: pd.DataFrame):
    return build_model_bundle(frame)


st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Explainable Bridge Inspection Prioritization</div>
        <div class="hero-title">BridgeWatch</div>
        <div class="hero-copy">
            BridgeWatch uses the Federal Highway Administration's National Bridge Inventory to triage bridges
            into <strong>Priority Review</strong> and <strong>Routine Review</strong>, then compares an
            interpretable decision tree with a stronger random forest benchmark.
        </div>
        <div class="chip-row">
            <div class="chip">National FHWA data</div>
            <div class="chip">State-by-state exploration</div>
            <div class="chip">Decision tree explainability</div>
            <div class="chip">Random forest benchmark</div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dashboard Controls")
    refresh = st.button("Refresh FHWA Cache")

try:
    cache_buster = int(PROCESSED_DATA_PATH.stat().st_mtime_ns) if PROCESSED_DATA_PATH.exists() else 0
    data = get_data(force_refresh=refresh, cache_buster=cache_buster)
except Exception as exc:
    st.error(
        "BridgeWatch could not load the FHWA bridge dataset automatically. "
        "Run the app with internet access so it can download the official NBI zip once, then it will cache locally."
    )
    st.caption(f"Dataset source: {FHWA_DOWNLOAD_URL}")
    st.exception(exc)
    st.stop()
    raise SystemExit(1)

required_columns = {"priority_review", "state_name", "priority_label"}
missing_columns = required_columns.difference(data.columns)
if missing_columns:
    st.error(
        "BridgeWatch loaded an outdated processed cache file that does not match the current schema."
    )
    st.code(
        "Remove-Item .\\data\\processed\\bridgewatch_2025_processed.csv.gz -Force\n"
        "& ..\\.venv\\Scripts\\python.exe -m streamlit cache clear\n"
        "& ..\\.venv\\Scripts\\python.exe -m streamlit run app.py",
        language="powershell",
    )
    st.caption(f"Missing columns: {', '.join(sorted(missing_columns))}")
    st.stop()

model_bundle = get_models((len(data), int(data["priority_review"].sum())), data)

with st.sidebar:
    state_options = ["All States", *sorted(data["state_name"].unique())]
    if "selected_state" in st.session_state and st.session_state["selected_state"] not in state_options:
        st.session_state["selected_state"] = "All States"
    selected_state = st.selectbox("State view", state_options, key="selected_state")
    selected_model_name = st.radio("Model", ["Decision Tree", "Random Forest"], horizontal=False)
    probability_threshold = st.slider("Priority threshold", min_value=0.30, max_value=0.80, value=0.50, step=0.05)
    st.markdown(
        f"""
        <div class="callout">
            Data year: <strong>{FHWA_YEAR}</strong><br>
            Training sample cap: <strong>{MODEL_SAMPLE_LIMIT:,}</strong> bridges for fast interactivity.
        </div>
        """,
        unsafe_allow_html=True,
    )

selected_frame = data if selected_state == "All States" else data[data["state_name"] == selected_state].copy()
selected_model = get_pipeline(model_bundle, selected_model_name)
scored_state = score_state_bridges(selected_model, selected_frame)
scored_state["predicted_priority"] = scored_state["priority_probability"].ge(probability_threshold).map(
    {True: "Priority Review", False: "Routine Review"}
)

top_metrics = st.columns(4)
with top_metrics[0]:
    st.metric("Bridges in view", f"{len(selected_frame):,}")
with top_metrics[1]:
    st.metric("Actual priority share", f"{selected_frame['priority_review'].mean():.1%}")
with top_metrics[2]:
    st.metric("Predicted priority share", f"{(scored_state['predicted_priority'] == 'Priority Review').mean():.1%}")
with top_metrics[3]:
    st.metric("States covered", f"{data['state_name'].nunique()}")

intro_left, intro_right = st.columns([1.15, 1])
with intro_left:
    st.markdown(
        """
        <div class="panel">
            <strong>Target label logic.</strong> A bridge is marked <em>Priority Review</em> when it is already poor,
            has low condition under heavier traffic, or shows weak structural evaluation. That keeps the target grounded
            in inspection reality while still giving the model a meaningful triage problem to learn.
            <br><br>
            <strong>Why four ratings are excluded from the model.</strong> That rule is built from the condition
            ratings and the structural evaluation, so feeding those back in as features leaks the label &mdash; the
            model would simply re-derive its own target. With them included the decision tree scored
            <strong>0.995 accuracy</strong>, which measured nothing.
            <code>structural_evaluation &le; 4</code> alone recovers 67.5% of positives.
            They are excluded here, so the models predict priority from age, traffic, geometry, load rating, design
            and location instead. The honest numbers are in the Model Lab tab.
        </div>
        """,
        unsafe_allow_html=True,
    )
with intro_right:
    st.plotly_chart(state_summary_chart(data), width="stretch")

overview_tab, models_tab, explorer_tab = st.tabs(["State Overview", "Model Lab", "Bridge Explorer"])

with overview_tab:
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### Highest-risk bridges in view")
        display_cols = [
            "state_name",
            "structure_number",
            "condition_bucket",
            "bridge_age",
            "average_daily_traffic",
            "operating_rating",
            "priority_probability",
            "predicted_priority",
        ]
        st.dataframe(scored_state[display_cols].head(20), width="stretch", hide_index=True)
    with right:
        st.markdown("### Condition mix")
        condition_summary = (
            selected_frame["condition_bucket"]
            .value_counts(normalize=True)
            .rename_axis("condition")
            .reset_index(name="share")
        )
        st.bar_chart(condition_summary.set_index("condition"))

with models_tab:
    st.markdown("### Holdout model comparison")
    st.info(
        "**Leakage audit.** These scores exclude "
        f"`{'`, `'.join(LEAKING_FEATURES)}` from the feature set. "
        "The `priority_review` label is constructed from those ratings, so including them let the "
        "decision tree reach 0.995 accuracy by re-deriving its own target. The numbers below are what "
        "the models achieve predicting priority from age, traffic, geometry, load rating, design and "
        "location — no label inputs.",
        icon=":material/policy:",
    )
    st.dataframe(
        model_bundle.metrics.assign(
            accuracy=lambda df: df["accuracy"].round(3),
            precision=lambda df: df["precision"].round(3),
            recall=lambda df: df["recall"].round(3),
            f1=lambda df: df["f1"].round(3),
        ),
        width="stretch",
        hide_index=True,
    )

    cm_left, cm_right = st.columns(2)
    with cm_left:
        st.plotly_chart(
            confusion_chart(get_confusion(model_bundle, "Decision Tree"), "Decision Tree"),
            width="stretch",
        )
    with cm_right:
        st.plotly_chart(
            confusion_chart(get_confusion(model_bundle, "Random Forest"), "Random Forest"),
            width="stretch",
        )

    feature_frame = get_feature_importance(selected_model)
    chart_col, rules_col = st.columns([1.05, 1])
    with chart_col:
        st.markdown(f"### {selected_model_name} feature importance")
        st.plotly_chart(feature_importance_chart(feature_frame), width="stretch")
    with rules_col:
        if selected_model_name == "Decision Tree":
            st.markdown("### Tree rules")
            st.code(export_tree_rules(selected_model), language="text")
        else:
            st.markdown(
                """
                ### Why the forest helps
                A random forest averages many decision trees, which usually improves stability and accuracy.
                The tradeoff is that it is harder to explain one exact rule path, so BridgeWatch keeps the single
                decision tree around as the interpretable companion model.
                """
            )

with explorer_tab:
    st.markdown(
        """
        ### Bridge-level explanation
        Pick one bridge from the current state view. The selected model returns its priority probability,
        and the decision tree can also show the split-by-split path used to reach a prediction.
        """
    )

    explorer_options = scored_state["bridge_id"].tolist()[:500]
    selected_bridge_id = st.selectbox("Bridge example", explorer_options)
    bridge_row = scored_state.loc[scored_state["bridge_id"] == selected_bridge_id].iloc[0]

    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("Bridge age", f"{int(bridge_row['bridge_age'])} years")
    with summary_cols[1]:
        st.metric("Avg daily traffic", f"{int(bridge_row['average_daily_traffic']):,}")
    with summary_cols[2]:
        st.metric("Operating rating", f"{bridge_row['operating_rating']:.1f}")
    with summary_cols[3]:
        st.metric("Priority probability", f"{bridge_row['priority_probability']:.1%}")

    show_cols = [
        "state_name",
        "structure_number",
        "condition_bucket",
        "bridge_age",
        "lanes_on_structure",
        "average_daily_traffic",
        "average_daily_truck_traffic",
        "number_of_spans_in_main_unit",
        "structure_length",
        "deck",
        "superstructure",
        "substructure",
        "structural_evaluation",
        "predicted_priority",
    ]
    st.dataframe(pd.DataFrame([bridge_row[show_cols]]), width="stretch", hide_index=True)

    if selected_model_name == "Decision Tree":
        st.markdown("### Decision tree path")
        for step in explain_tree_prediction(model_bundle.decision_tree, bridge_row):
            st.markdown(f'<div class="path-step">{step}</div>', unsafe_allow_html=True)
    else:
        st.markdown("### Random forest note")
        st.markdown(
            """
            <div class="path-step">
                Random forests combine many trees, so they do not have one single human-readable path the way a
                decision tree does. Use the probability score and feature importance chart for the forest, and use
                the decision tree view when you want a step-by-step explanation for class.
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption(
    f"Bridge data source: FHWA National Bridge Inventory {FHWA_YEAR}. "
    f"First run downloads the official file from {FHWA_DOWNLOAD_URL} and caches a processed subset locally."
)
