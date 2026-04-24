from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def feature_importance_chart(frame: pd.DataFrame) -> go.Figure:
    top = frame.head(10).sort_values("importance", ascending=True)
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#d9f99d", "#15803d"],
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Importance",
        yaxis_title="",
        height=360,
    )
    return fig


def confusion_chart(matrix, title: str) -> go.Figure:
    labels = ["Routine Review", "Priority Review"]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            text=matrix,
            texttemplate="%{text}",
            colorscale=[[0.0, "#ecfccb"], [0.5, "#65a30d"], [1.0, "#14532d"]],
            hovertemplate="Actual=%{y}<br>Predicted=%{x}<br>Count=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=340,
    )
    return fig


def state_summary_chart(frame: pd.DataFrame) -> go.Figure:
    summary = (
        frame.groupby("state_name", as_index=False)["priority_review"]
        .mean()
        .rename(columns={"priority_review": "priority_rate"})
        .sort_values("priority_rate", ascending=False)
        .head(12)
    )
    fig = px.bar(
        summary.sort_values("priority_rate"),
        x="priority_rate",
        y="state_name",
        orientation="h",
        color="priority_rate",
        color_continuous_scale=["#fde68a", "#b45309"],
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Share labeled Priority Review",
        yaxis_title="",
        height=380,
    )
    return fig
