import streamlit as st
import pandas as pd

from data_loader import load_dataset
from analyzer import analyze_dataset
from ai_planner import generate_dashboard_plan
from dashboard_generator import create_chart
from ai_insights import generate_insights
from ai_chat import ask_dataset


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 10px;
    }

    .insight-card {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 10px;
    }

    .small-text {
        font-size: 13px;
        opacity: 0.65;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🤖 AI Data Analyst</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a dataset and let AI discover, visualize and explain your data.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "dashboard" not in st.session_state:
    st.session_state.dashboard = None

if "insights" not in st.session_state:
    st.session_state.insights = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.info(
        "AI Engine\n\n"
        "Groq — Development Mode\n\n"
        "The application is designed so a local "
        "Snapdragon-compatible AI engine can be used "
        "for the target deployment."
    )

    st.divider()

    st.markdown("### 📌 How it works")

    st.markdown(
        """
        **1. Upload** your dataset  
        **2. Profile** the data  
        **3. AI plans** the dashboard  
        **4. Python validates** the plan  
        **5. Plotly renders** the charts  
        **6. AI explains** the findings
        """
    )


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload your dataset",
    type=["csv", "xlsx"],
    help="Upload a CSV or Excel file."
)


if uploaded_file is not None:

    # =====================================================
    # RESET RESULTS WHEN A NEW FILE IS UPLOADED
    # =====================================================

    if (
        st.session_state.current_file
        != uploaded_file.name
    ):

        st.session_state.dashboard = None
        st.session_state.insights = None
        st.session_state.current_file = uploaded_file.name


    # =====================================================
    # LOAD DATASET
    # =====================================================

    try:

        df = load_dataset(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"❌ Could not load dataset: {e}"
        )

        st.stop()


    # =====================================================
    # DATASET SUCCESS MESSAGE
    # =====================================================

    st.success(
        f"Dataset loaded successfully — "
        f"{len(df):,} rows × {len(df.columns)} columns"
    )


    # =====================================================
    # DATASET OVERVIEW
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Dataset Overview</div>',
        unsafe_allow_html=True
    )

    analysis = analyze_dataset(df)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Rows",
            f"{analysis['rows']:,}"
        )

    with c2:
        st.metric(
            "Columns",
            analysis["columns"]
        )

    with c3:
        st.metric(
            "Numeric Fields",
            len(
                analysis["numeric_columns"]
            )
        )

    with c4:
        st.metric(
            "Categorical Fields",
            len(
                analysis["categorical_columns"]
            )
        )


    # =====================================================
    # DATA QUALITY
    # =====================================================

    total_missing = int(
        df.isna().sum().sum()
    )

    total_cells = (
        df.shape[0] * df.shape[1]
    )

    missing_percentage = (
        (total_missing / total_cells) * 100
        if total_cells > 0
        else 0
    )

    q1, q2, q3 = st.columns(3)

    with q1:
        st.metric(
            "Missing Values",
            f"{total_missing:,}"
        )

    with q2:
        st.metric(
            "Missing %",
            f"{missing_percentage:.1f}%"
        )

    with q3:
        st.metric(
            "Duplicate Rows",
            f"{df.duplicated().sum():,}"
        )


    # =====================================================
    # COLUMN TYPES
    # =====================================================

    with st.expander(
        "🔍 View detected column types"
    ):

        col_a, col_b = st.columns(2)

        with col_a:

            st.markdown("**Numeric**")

            st.write(
                analysis["numeric_columns"]
                or "None detected"
            )

            st.markdown("**Categorical**")

            st.write(
                analysis["categorical_columns"]
                or "None detected"
            )

            st.markdown("**Dates**")

            st.write(
                analysis["date_columns"]
                or "None detected"
            )

        with col_b:

            st.markdown("**Identifiers**")

            st.write(
                analysis["identifier_columns"]
                or "None detected"
            )

            st.markdown("**Text**")

            st.write(
                analysis["text_columns"]
                or "None detected"
            )


    # =====================================================
    # DATA PREVIEW
    # =====================================================

    with st.expander(
        "📋 Preview dataset"
    ):

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


    # =====================================================
    # NUMERIC SUMMARY
    # =====================================================

    numeric_cols = analysis[
        "numeric_columns"
    ]

    if numeric_cols:

        summary_df = (
            df[numeric_cols]
            .describe()
            .round(2)
        )

        summary = summary_df.to_dict()

    else:

        summary_df = pd.DataFrame()
        summary = {}


    if not summary_df.empty:

        with st.expander(
            "📈 Statistical summary"
        ):

            st.dataframe(
                summary_df,
                use_container_width=True
            )


    # =====================================================
    # AI DASHBOARD
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🧠 AI-Generated Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "AI selects visualizations based on the structure "
        "and characteristics of your dataset."
    )


    if st.button(
        "✨ Generate AI Dashboard",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "🧠 AI is analyzing your dataset and designing the dashboard..."
        ):

            try:

                dashboard = generate_dashboard_plan(
                    analysis
                )

                st.session_state.dashboard = dashboard

            except Exception as e:

                st.session_state.dashboard = None

                st.error(
                    f"❌ Dashboard generation failed: {e}"
                )


    # =====================================================
    # DISPLAY DASHBOARD
    # =====================================================

    if st.session_state.dashboard:

        dashboard = (
            st.session_state.dashboard
        )

        st.success(
            "Dashboard plan generated successfully."
        )

        st.markdown(
            f"### {dashboard.get('dashboard_title', 'AI Dashboard')}"
        )

        charts = dashboard.get(
            "charts",
            []
        )

        if not charts:

            st.warning(
                "AI could not identify enough suitable "
                "visualizations for this dataset."
            )

        else:

            # ---------------------------------------------
            # TWO-COLUMN RESPONSIVE LAYOUT
            # ---------------------------------------------

            for i in range(
                0,
                len(charts),
                2
            ):

                row = st.columns(2)

                for position in range(2):

                    chart_index = i + position

                    if chart_index >= len(charts):
                        break

                    chart = charts[
                        chart_index
                    ]

                    with row[position]:

                        try:

                            fig = create_chart(
                                df,
                                chart
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                        except Exception as chart_error:

                            st.warning(
                                f"⚠️ Chart could not be rendered: "
                                f"{chart_error}"
                            )


    # =====================================================
    # AI INSIGHTS
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">💡 AI Data Insights</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "AI-generated findings based on the dataset statistics."
    )


    if st.button(
        "🔎 Generate AI Insights",
        use_container_width=True
    ):

        with st.spinner(
            "🤖 AI is finding patterns in your data..."
        ):

            try:

                insights = generate_insights(
                    analysis,
                    summary
                )

                st.session_state.insights = insights

            except Exception as e:

                st.session_state.insights = None

                st.error(
                    f"❌ Insight generation failed: {e}"
                )


    # =====================================================
    # DISPLAY INSIGHTS
    # =====================================================

    if st.session_state.insights:

        insights = (
            st.session_state.insights
        )

        # ---------------------------------------------
        # SUMMARY
        # ---------------------------------------------

        st.markdown("### 📝 Summary")

        st.info(
            insights.get(
                "summary",
                "No summary available."
            )
        )


        # ---------------------------------------------
        # KEY INSIGHTS
        # ---------------------------------------------

        st.markdown(
            "### 🔎 Key Findings"
        )

        key_insights = insights.get(
            "key_insights",
            []
        )

        for index, insight in enumerate(
            key_insights,
            start=1
        ):

            st.markdown(
                f"""
                <div class="insight-card">
                <b>💡 Finding {index}</b><br>
                {insight}
                </div>
                """,
                unsafe_allow_html=True
            )


        # ---------------------------------------------
        # RECOMMENDATIONS
        # ---------------------------------------------

        recommendations = insights.get(
            "recommendations",
            []
        )

        if recommendations:

            st.markdown(
                "### 🎯 Recommendations"
            )

            for recommendation in recommendations:

                st.markdown(
                    f"""
                    <div class="insight-card">
                    🎯 {recommendation}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # =====================================================
    # ASK YOUR DATASET
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">💬 Ask Your Dataset</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Ask questions about the uploaded dataset."
    )

    question = st.text_input(
        "Your question",
        placeholder=(
            "Example: Which numeric metric has the highest average?"
        ),
        label_visibility="collapsed"
    )


    if st.button(
        "🤖 Ask AI",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            with st.spinner(
                "🤖 AI is analyzing your question..."
            ):

                try:

                    answer = ask_dataset(
                        question,
                        analysis,
                        summary,
                        df
                    )

                    st.markdown(
                        "### 🤖 Answer"
                    )

                    st.info(
                        answer
                    )

                except Exception as e:

                    st.error(
                        f"❌ Question failed: {e}"
                    )


else:

    # =====================================================
    # EMPTY STATE
    # =====================================================

    st.markdown(
        """
        ### 👋 Welcome!

        Upload a **CSV or XLSX dataset** above to begin.

        The AI Data Analyst will:

        - 🔍 Profile your dataset
        - 🧠 Understand its structure
        - 📊 Select suitable visualizations
        - 📈 Generate an interactive dashboard
        - 💡 Find useful patterns
        - 💬 Answer questions about your data
        """
    )

    st.divider()

    st.markdown(
        "### 🚀 AI-powered workflow"
    )

    a, b, c, d = st.columns(4)

    with a:
        st.markdown(
            "**1️⃣ Upload**\n\n"
            "Provide any CSV or Excel dataset."
        )

    with b:
        st.markdown(
            "**2️⃣ Analyze**\n\n"
            "The system profiles columns and data quality."
        )

    with c:
        st.markdown(
            "**3️⃣ Visualize**\n\n"
            "AI selects appropriate charts."
        )

    with d:
        st.markdown(
            "**4️⃣ Understand**\n\n"
            "Get AI insights and dataset Q&A."
        )