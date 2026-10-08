import streamlit as st
import pandas as pd

from data_loader import load_dataset
from analyzer import analyze_dataset
from ai_planner import generate_dashboard_plan
from dashboard_generator import create_chart
from ai_insights import generate_insights
from ai_chat import ask_dataset


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .small-muted {
        color: #6b7280;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "dashboard" not in st.session_state:
    st.session_state.dashboard = None

if "insights" not in st.session_state:
    st.session_state.insights = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "analysis" not in st.session_state:
    st.session_state.analysis = None


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.title("📊 AI Data Analyst")

    st.markdown(
        """
        Upload a CSV or Excel dataset and let AI help you
        understand, visualize, and analyze it.
        """
    )

    st.divider()

    st.info(
        "AI Engine\n\n"
        "Groq — AI reasoning layer\n\n"
        "Python performs dataset calculations and validation, "
        "while AI handles dashboard planning, insights, and "
        "natural-language analysis."
    )

    st.divider()

    st.caption(
        "General-purpose AI-powered data analysis tool"
    )


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">AI Data Analyst</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload your dataset and turn raw data into dashboards, '
    'insights, and answers.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"],
    help="Supported formats: CSV and XLSX"
)


if uploaded_file is None:

    st.info(
        "👆 Upload a CSV or XLSX file to start analyzing your data."
    )

    st.markdown(
        """
        ### What you can do

        - 📋 Automatically profile your dataset
        - 📊 Generate an AI-powered dashboard
        - 💡 Discover important insights
        - 💬 Ask questions about your dataset
        - 🔎 Explore numerical and categorical information
        """
    )

    st.stop()


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

try:

    # Reload only when a new file is uploaded
    if st.session_state.current_file != uploaded_file.name:

        df = load_dataset(uploaded_file)

        st.session_state.current_file = uploaded_file.name
        st.session_state.dashboard = None
        st.session_state.insights = None

        st.session_state.analysis = analyze_dataset(df)

    else:

        # Re-load the file for the current session
        df = load_dataset(uploaded_file)

        if st.session_state.analysis is None:
            st.session_state.analysis = analyze_dataset(df)

except Exception as e:

    st.error(f"Could not load the dataset: {e}")
    st.stop()


analysis = st.session_state.analysis


# ---------------------------------------------------------
# DATASET HEADER
# ---------------------------------------------------------

st.markdown(
    f"### 📁 {uploaded_file.name}"
)

st.caption(
    f"{analysis['rows']:,} rows × {analysis['columns']:,} columns"
)


# ---------------------------------------------------------
# OVERVIEW METRICS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Dataset Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Rows",
        f"{analysis['rows']:,}"
    )

with col2:
    st.metric(
        "Columns",
        f"{analysis['columns']:,}"
    )

with col3:
    st.metric(
        "Numeric Columns",
        len(analysis["numeric_columns"])
    )

with col4:
    st.metric(
        "Categorical Columns",
        len(analysis["categorical_columns"])
    )


# ---------------------------------------------------------
# DATA QUALITY
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Data Quality</div>',
    unsafe_allow_html=True
)

missing_values = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

quality_col1, quality_col2, quality_col3 = st.columns(3)

with quality_col1:

    st.metric(
        "Missing Values",
        f"{missing_values:,}"
    )

with quality_col2:

    st.metric(
        "Duplicate Rows",
        f"{duplicate_rows:,}"
    )

with quality_col3:

    total_cells = df.shape[0] * df.shape[1]

    if total_cells > 0:
        completeness = (
            1 - (missing_values / total_cells)
        ) * 100
    else:
        completeness = 100

    st.metric(
        "Completeness",
        f"{completeness:.1f}%"
    )


# ---------------------------------------------------------
# COLUMN TYPES
# ---------------------------------------------------------

with st.expander("🔎 View detected column types"):

    type_col1, type_col2 = st.columns(2)

    with type_col1:

        st.markdown("**Numeric Columns**")

        if analysis["numeric_columns"]:
            st.write(
                ", ".join(
                    analysis["numeric_columns"]
                )
            )
        else:
            st.write("None detected")

        st.markdown("**Categorical Columns**")

        if analysis["categorical_columns"]:
            st.write(
                ", ".join(
                    analysis["categorical_columns"]
                )
            )
        else:
            st.write("None detected")

    with type_col2:

        st.markdown("**Date Columns**")

        if analysis["date_columns"]:
            st.write(
                ", ".join(
                    analysis["date_columns"]
                )
            )
        else:
            st.write("None detected")

        st.markdown("**Identifier Columns**")

        if analysis["identifier_columns"]:
            st.write(
                ", ".join(
                    analysis["identifier_columns"]
                )
            )
        else:
            st.write("None detected")


# ---------------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Data Preview</div>',
    unsafe_allow_html=True
)

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ---------------------------------------------------------
# NUMERICAL SUMMARY
# ---------------------------------------------------------

if analysis["numeric_columns"]:

    with st.expander("📈 Numerical Summary"):

        st.dataframe(
            df[analysis["numeric_columns"]].describe().T,
            use_container_width=True
        )


# ---------------------------------------------------------
# AI DASHBOARD
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">AI Dashboard</div>',
    unsafe_allow_html=True
)

st.write(
    "Let AI choose useful visualizations based on the "
    "structure and content of your dataset."
)


if st.button(
    "✨ Generate AI Dashboard",
    use_container_width=True
):

    with st.spinner(
        "AI is analyzing your dataset and planning the dashboard..."
    ):

        try:

            plan = generate_dashboard_plan(
                analysis
            )

            st.session_state.dashboard = plan

        except Exception as e:

            st.error(
                f"Dashboard generation failed: {e}"
            )


# ---------------------------------------------------------
# DISPLAY DASHBOARD
# ---------------------------------------------------------

if st.session_state.dashboard:

    dashboard_plan = st.session_state.dashboard

    charts = dashboard_plan.get(
        "charts",
        []
    )

    if not charts:

        st.warning(
            "The AI did not generate any valid charts."
        )

    else:

        chart_columns = st.columns(2)

        for index, chart_spec in enumerate(charts):

            with chart_columns[index % 2]:

                try:

                    figure = create_chart(
                        df,
                        chart_spec
                    )

                    if figure is not None:

                        st.plotly_chart(
                            figure,
                            use_container_width=True
                        )

                        title = chart_spec.get(
                            "title"
                        )

                        if title:
                            st.caption(title)

                except Exception as e:

                    st.warning(
                        f"Could not create chart "
                        f"{index + 1}: {e}"
                    )


# ---------------------------------------------------------
# AI INSIGHTS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">AI Insights</div>',
    unsafe_allow_html=True
)

st.write(
    "Generate a concise explanation of important patterns "
    "and observations in your dataset."
)


if st.button(
    "💡 Generate AI Insights",
    use_container_width=True
):

    with st.spinner(
        "AI is generating insights..."
    ):

        try:

            insights = generate_insights(
                df,
                analysis
            )

            st.session_state.insights = insights

        except Exception as e:

            st.error(
                f"Insight generation failed: {e}"
            )


# ---------------------------------------------------------
# DISPLAY INSIGHTS
# ---------------------------------------------------------

if st.session_state.insights:

    insights = st.session_state.insights

    summary = insights.get(
        "summary"
    )

    if summary:

        st.markdown("### 📝 Summary")

        st.write(summary)

    key_insights = insights.get(
        "key_insights",
        []
    )

    if key_insights:

        st.markdown("### 🔍 Key Insights")

        for insight in key_insights:

            st.markdown(
                f"- {insight}"
            )

    recommendations = insights.get(
        "recommendations",
        []
    )

    if recommendations:

        st.markdown("### 🚀 Recommendations")

        for recommendation in recommendations:

            st.markdown(
                f"- {recommendation}"
            )


# ---------------------------------------------------------
# ASK YOUR DATASET
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Ask Your Dataset</div>',
    unsafe_allow_html=True
)

st.write(
    "Ask questions about your data using natural language."
)


question = st.text_input(
    "Example: What is the average revenue?",
    placeholder="Ask something about your dataset..."
)


if st.button(
    "💬 Ask",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        with st.spinner(
            "Analyzing your question..."
        ):

            try:

                answer = ask_dataset(
                    df,
                    question
                )

                st.markdown("### 🤖 Answer")

                st.write(answer)

            except Exception as e:

                st.error(
                    f"Could not answer the question: {e}"
                )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "AI Data Analyst • AI-assisted analysis with Python-based "
    "calculation and validation"
)