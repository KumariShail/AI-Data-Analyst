import pandas as pd
import plotly.express as px


# =========================================================
# COLOR PALETTES
# =========================================================

COLOR_PALETTE = [
    "#6366F1",  # Indigo
    "#EC4899",  # Pink
    "#14B8A6",  # Teal
    "#F59E0B",  # Amber
    "#8B5CF6",  # Violet
    "#EF4444",  # Red
    "#06B6D4",  # Cyan
    "#84CC16",  # Lime
    "#F97316",  # Orange
    "#A855F7",  # Purple
    "#10B981",  # Emerald
    "#E11D48",  # Rose
]


def create_chart(df, chart):

    chart_type = chart.get("type")
    x = chart.get("x")
    y = chart.get("y")
    operation = chart.get("operation")
    title = chart.get(
        "title",
        "Dashboard Chart"
    )

    if x not in df.columns:
        raise ValueError(
            f"Column '{x}' does not exist."
        )

    # =====================================================
    # HISTOGRAM
    # =====================================================

    if chart_type == "histogram":

        data = df[[x]].dropna()

        if data.empty:
            raise ValueError(
                "No usable values for histogram."
            )

        if not pd.api.types.is_numeric_dtype(
            data[x]
        ):
            raise ValueError(
                "Histogram requires numeric data."
            )

        fig = px.histogram(
            data,
            x=x,
            nbins=25,
            title=title,
            color_discrete_sequence=COLOR_PALETTE
        )

        fig.update_traces(
            marker_line_width=0.5,
            marker_line_color="rgba(255,255,255,0.35)"
        )

    # =====================================================
    # SCATTER
    # =====================================================

    elif chart_type == "scatter":

        if y not in df.columns:
            raise ValueError(
                f"Column '{y}' does not exist."
            )

        data = df[[x, y]].dropna()

        if data.empty:
            raise ValueError(
                "No usable values for scatter chart."
            )

        fig = px.scatter(
            data,
            x=x,
            y=y,
            title=title,
            opacity=0.70,
            color_discrete_sequence=COLOR_PALETTE
        )

        fig.update_traces(
            marker=dict(
                size=8,
                line=dict(
                    width=0.5,
                    color="rgba(255,255,255,0.45)"
                )
            )
        )

    # =====================================================
    # DATE / AGGREGATED CHARTS
    # =====================================================

    else:

        data = df.copy()

        converted = pd.to_datetime(
            data[x],
            errors="coerce"
        )

        is_date = (
            converted.notna().mean() > 0.8
            and not pd.api.types.is_numeric_dtype(
                data[x]
            )
        )

        if is_date:

            data["__date"] = converted

            data = data.dropna(
                subset=["__date"]
            )

            group_column = "__date"

        else:

            group_column = x

        # =================================================
        # AGGREGATION
        # =================================================

        if operation == "count":

            chart_data = (
                data
                .groupby(
                    group_column,
                    dropna=False
                )
                .size()
                .reset_index(
                    name="Count"
                )
            )

            value_column = "Count"

        elif operation in {"sum", "mean"}:

            if y not in data.columns:
                raise ValueError(
                    f"Metric '{y}' does not exist."
                )

            if not pd.api.types.is_numeric_dtype(
                data[y]
            ):
                raise ValueError(
                    f"'{y}' is not numeric."
                )

            grouped = (
                data
                .groupby(
                    group_column,
                    dropna=False
                )[y]
            )

            if operation == "sum":

                chart_data = (
                    grouped
                    .sum()
                    .reset_index()
                )

            else:

                chart_data = (
                    grouped
                    .mean()
                    .reset_index()
                )

            value_column = y

        else:

            raise ValueError(
                "Invalid chart operation."
            )

        chart_data = chart_data.dropna(
            subset=[group_column]
        )

        # =================================================
        # BAR
        # =================================================

        if chart_type == "bar":

            if len(chart_data) > 12:

                chart_data = (
                    chart_data
                    .nlargest(
                        12,
                        value_column
                    )
                )

            fig = px.bar(
                chart_data,
                x=group_column,
                y=value_column,
                title=title,
                color=group_column,
                color_discrete_sequence=COLOR_PALETTE
            )

            fig.update_traces(
                marker_line_width=0.5,
                marker_line_color="rgba(255,255,255,0.35)"
            )

        # =================================================
        # LINE
        # =================================================

        elif chart_type == "line":

            chart_data = (
                chart_data
                .sort_values(group_column)
            )

            fig = px.line(
                chart_data,
                x=group_column,
                y=value_column,
                markers=True,
                title=title,
                color_discrete_sequence=COLOR_PALETTE
            )

            fig.update_traces(
                line=dict(
                    width=3
                ),
                marker=dict(
                    size=8
                )
            )

        # =================================================
        # PIE / DONUT
        # =================================================

        elif chart_type == "pie":

            chart_data = (
                chart_data
                .nlargest(
                    8,
                    value_column
                )
            )

            fig = px.pie(
                chart_data,
                names=group_column,
                values=value_column,
                title=title,
                hole=0.45,
                color_discrete_sequence=COLOR_PALETTE
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Value: %{value}<br>"
                    "Share: %{percent}"
                    "<extra></extra>"
                ),
                marker=dict(
                    line=dict(
                        color="rgba(255,255,255,0.85)",
                        width=2
                    )
                )
            )

        else:

            raise ValueError(
                f"Unsupported chart type: {chart_type}"
            )

    # =====================================================
    # COMMON STYLE
    # =====================================================

    fig.update_layout(
        height=420,

        margin=dict(
            l=20,
            r=20,
            t=70,
            b=40
        ),

        title={
            "x": 0.02,
            "xanchor": "left"
        },

        hoverlabel={
            "namelength": -1
        },

        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 1
        },

        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    fig.update_xaxes(
        showgrid=False
    )

    fig.update_yaxes(
        gridcolor="rgba(128,128,128,0.15)"
    )

    return fig