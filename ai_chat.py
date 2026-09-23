import json
import re

import pandas as pd

from ai_client import ask_ai


# =========================================================
# HELPERS
# =========================================================

def _normalise(text):
    """Make text easier to compare with column names."""

    text = str(text).lower().strip()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def _find_column(question, columns):
    """
    Find the dataset column most closely matching
    words used in the user's question.
    """

    question_normalised = _normalise(question)

    # Exact column-name matches first
    exact_matches = []

    for column in columns:

        column_normalised = _normalise(column)

        if column_normalised in question_normalised:
            exact_matches.append(
                (column, len(column_normalised))
            )

    if exact_matches:

        exact_matches.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return exact_matches[0][0]

    # Match individual words
    question_words = set(
        question_normalised.split()
    )

    candidates = []

    for column in columns:

        column_words = set(
            _normalise(column).split()
        )

        overlap = len(
            question_words & column_words
        )

        if overlap > 0:

            candidates.append(
                (column, overlap)
            )

    if candidates:

        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return candidates[0][0]

    return None


def _find_numeric_columns(df):
    return df.select_dtypes(
        include="number"
    ).columns.tolist()


def _find_group_column(df, question, metric=None):
    """
    Find a categorical/grouping column mentioned
    in the question.
    """

    candidates = []

    for column in df.columns:

        if metric is not None and column == metric:
            continue

        normalised = _normalise(column)

        if normalised and normalised in _normalise(question):

            if (
                df[column].dtype == "object"
                or str(df[column].dtype).startswith("category")
                or df[column].nunique(dropna=True) <= 50
            ):

                candidates.append(column)

    if candidates:

        return max(
            candidates,
            key=lambda x: len(_normalise(x))
        )

    return None


def _format_number(value):
    """Readable numeric formatting."""

    if pd.isna(value):
        return "N/A"

    value = float(value)

    if value.is_integer():
        return f"{int(value):,}"

    return f"{value:,.2f}"


def _format_percentage(value):
    if pd.isna(value):
        return "N/A"

    return f"{float(value):.2f}%"


def _contains_any(question, words):
    question = _normalise(question)

    return any(
        word in question
        for word in words
    )


# =========================================================
# PROFIT / PERCENTAGE HELPERS
# =========================================================

def _find_profit_column(df):
    candidates = [
        column
        for column in df.columns
        if "profit" in _normalise(column)
    ]

    return candidates[0] if candidates else None


def _find_revenue_column(df):
    candidates = [
        column
        for column in df.columns
        if any(
            word in _normalise(column)
            for word in [
                "revenue",
                "sales",
                "sale",
                "income"
            ]
        )
    ]

    return candidates[0] if candidates else None


def _find_price_column(df):
    candidates = [
        column
        for column in df.columns
        if "price" in _normalise(column)
    ]

    return candidates[0] if candidates else None


def _calculate_profit_percentage(df):
    """
    Calculate profit percentage when suitable columns exist.

    Preferred formula:

        profit / revenue * 100

    If revenue does not exist but price and cost exist,
    calculate:

        (price - cost) / price * 100
    """

    profit_column = _find_profit_column(df)
    revenue_column = _find_revenue_column(df)

    if (
        profit_column is not None
        and revenue_column is not None
    ):

        revenue = pd.to_numeric(
            df[revenue_column],
            errors="coerce"
        )

        profit = pd.to_numeric(
            df[profit_column],
            errors="coerce"
        )

        percentage = (
            profit
            .div(revenue)
            .mul(100)
        )

        return (
            percentage,
            f"{profit_column} / {revenue_column} × 100"
        )

    price_column = _find_price_column()

    if price_column is not None:

        cost_candidates = [
            column
            for column in df.columns
            if "cost" in _normalise(column)
        ]

        if cost_candidates:

            cost_column = cost_candidates[0]

            price = pd.to_numeric(
                df[price_column],
                errors="coerce"
            )

            cost = pd.to_numeric(
                df[cost_column],
                errors="coerce"
            )

            percentage = (
                (price - cost)
                .div(price)
                .mul(100)
            )

            return (
                percentage,
                f"({price_column} - {cost_column}) / {price_column} × 100"
            )

    return None, None


# =========================================================
# DIRECT DATA ANALYSIS ENGINE
# =========================================================

def _answer_directly(question, df):

    question_normalised = _normalise(
        question
    )

    numeric_columns = _find_numeric_columns(
        df
    )

    # -----------------------------------------------------
    # EMPTY DATASET
    # -----------------------------------------------------

    if df.empty:

        return (
            "The uploaded dataset contains no rows, "
            "so there is nothing to calculate."
        )

    # -----------------------------------------------------
    # TOTAL / SUM
    # -----------------------------------------------------

    if _contains_any(
        question_normalised,
        [
            "total",
            "sum",
            "overall"
        ]
    ):

        metric = _find_column(
            question,
            numeric_columns
        )

        if metric:

            value = pd.to_numeric(
                df[metric],
                errors="coerce"
            ).sum()

            return (
                f"The total {metric} is "
                f"**{_format_number(value)}**."
            )

    # -----------------------------------------------------
    # AVERAGE / MEAN
    # -----------------------------------------------------

    if _contains_any(
        question_normalised,
        [
            "average",
            "mean",
            "avg"
        ]
    ):

        metric = _find_column(
            question,
            numeric_columns
        )

        if metric:

            group = _find_group_column(
                df,
                question,
                metric
            )

            # ---------------------------------------------
            # GROUPED AVERAGE
            # ---------------------------------------------

            if group:

                grouped = (
                    df.groupby(group)[metric]
                    .mean()
                    .dropna()
                    .sort_values(
                        ascending=False
                    )
                )

                if not grouped.empty:

                    top = grouped.iloc[0]

                    return (
                        f"The highest average **{metric}** "
                        f"is **{_format_number(top)}**, "
                        f"achieved by **{grouped.index[0]}**."
                    )

            # ---------------------------------------------
            # OVERALL AVERAGE
            # ---------------------------------------------

            value = pd.to_numeric(
                df[metric],
                errors="coerce"
            ).mean()

            return (
                f"The average {metric} is "
                f"**{_format_number(value)}**."
            )

    # -----------------------------------------------------
    # MEDIAN
    # -----------------------------------------------------

    if "median" in question_normalised:

        metric = _find_column(
            question,
            numeric_columns
        )

        if metric:

            value = pd.to_numeric(
                df[metric],
                errors="coerce"
            ).median()

            return (
                f"The median {metric} is "
                f"**{_format_number(value)}**."
            )

    # -----------------------------------------------------
    # HIGHEST / MAXIMUM / MAX
    # -----------------------------------------------------

    if _contains_any(
        question_normalised,
        [
            "highest",
            "maximum",
            "max",
            "largest",
            "top"
        ]
    ):

        # ---------------------------------------------
        # PROFIT PERCENTAGE
        # ---------------------------------------------

        if (
            "percentage" in question_normalised
            or "percent" in question_normalised
            or "margin" in question_normalised
        ):

            percentages, formula = (
                _calculate_profit_percentage(df)
            )

            if percentages is not None:

                valid = percentages.dropna()

                if not valid.empty:

                    index = valid.idxmax()
                    value = valid.loc[index]

                    group = _find_group_column(
                        df,
                        question
                    )

                    if group:

                        group_value = df.loc[
                            index,
                            group
                        ]

                        return (
                            f"The highest profit percentage "
                            f"is **{_format_percentage(value)}**, "
                            f"for **{group_value}**."
                        )

                    return (
                        f"The highest profit percentage is "
                        f"**{_format_percentage(value)}**."
                    )

        metric = _find_column(
            question,
            numeric_columns
        )

        if metric:

            group = _find_group_column(
                df,
                question,
                metric
            )

            # -----------------------------------------
            # HIGHEST GROUP TOTAL
            # -----------------------------------------

            if group:

                grouped = (
                    df.groupby(group)[metric]
                    .sum()
                    .dropna()
                    .sort_values(
                        ascending=False
                    )
                )

                if not grouped.empty:

                    return (
                        f"The highest total {metric} is "
                        f"**{_format_number(grouped.iloc[0])}**, "
                        f"for **{grouped.index[0]}**."
                    )

            # -----------------------------------------
            # HIGHEST RAW VALUE
            # -----------------------------------------

            series = pd.to_numeric(
                df[metric],
                errors="coerce"
            )

            index = series.idxmax()

            value = series.loc[index]

            return (
                f"The highest {metric} is "
                f"**{_format_number(value)}**."
            )

    # -----------------------------------------------------
    # LOWEST / MINIMUM / MIN
    # -----------------------------------------------------

    if _contains_any(
        question_normalised,
        [
            "lowest",
            "minimum",
            "min",
            "smallest",
            "bottom"
        ]
    ):

        metric = _find_column(
            question,
            numeric_columns
        )

        if metric:

            group = _find_group_column(
                df,
                question,
                metric
            )

            if group:

                grouped = (
                    df.groupby(group)[metric]
                    .sum()
                    .dropna()
                    .sort_values()
                )

                if not grouped.empty:

                    return (
                        f"The lowest total {metric} is "
                        f"**{_format_number(grouped.iloc[0])}**, "
                        f"for **{grouped.index[0]}**."
                    )

            series = pd.to_numeric(
                df[metric],
                errors="coerce"
            )

            index = series.idxmin()

            value = series.loc[index]

            return (
                f"The lowest {metric} is "
                f"**{_format_number(value)}**."
            )

    # -----------------------------------------------------
    # MOST COMMON / FREQUENCY
    # -----------------------------------------------------

    if _contains_any(
        question_normalised,
        [
            "most common",
            "appears most",
            "most frequent",
            "highest frequency",
            "best rating",
            "popular"
        ]
    ):

        column = _find_column(
            question,
            df.columns
        )

        if column:

            counts = (
                df[column]
                .dropna()
                .value_counts()
            )

            if not counts.empty:

                return (
                    f"**{counts.index[0]}** is the most "
                    f"frequent {column}, appearing "
                    f"**{int(counts.iloc[0]):,} times**."
                )

    # -----------------------------------------------------
    # COUNT / HOW MANY
    # -----------------------------------------------------

    if _contains_any(
        question_normalised,
        [
            "how many",
            "count",
            "number of"
        ]
    ):

        column = _find_column(
            question,
            df.columns
        )

        if column:

            count = int(
                df[column]
                .notna()
                .sum()
            )

            return (
                f"There are **{count:,} non-missing "
                f"values** in {column}."
            )

        return (
            f"The dataset contains "
            f"**{len(df):,} rows**."
        )

    # -----------------------------------------------------
    # PERCENTAGE OF CATEGORY
    # -----------------------------------------------------

    if (
        "percentage" in question_normalised
        or "percent" in question_normalised
        or "%" in question
    ):

        column = _find_column(
            question,
            df.columns
        )

        if column:

            # Find a quoted/category-like value
            # by checking actual unique values.
            values = (
                df[column]
                .dropna()
                .astype(str)
                .unique()
            )

            question_lower = question.lower()

            for value in values:

                if str(value).lower() in question_lower:

                    percentage = (
                        df[column]
                        .astype(str)
                        .str.lower()
                        .eq(str(value).lower())
                        .mean()
                        * 100
                    )

                    return (
                        f"**{_format_percentage(percentage)}** "
                        f"of the dataset belongs to "
                        f"**{value}** in {column}."
                    )

    # -----------------------------------------------------
    # TOP N
    # -----------------------------------------------------

    top_match = re.search(
        r"(?:top|best)\s+(\d+)",
        question_normalised
    )

    if top_match:

        n = int(
            top_match.group(1)
        )

        metric = _find_column(
            question,
            numeric_columns
        )

        if metric:

            group = _find_group_column(
                df,
                question,
                metric
            )

            if group:

                grouped = (
                    df.groupby(group)[metric]
                    .sum()
                    .dropna()
                    .sort_values(
                        ascending=False
                    )
                    .head(n)
                )

                if not grouped.empty:

                    lines = []

                    for rank, (name, value) in enumerate(
                        grouped.items(),
                        start=1
                    ):

                        lines.append(
                            f"{rank}. **{name}** — "
                            f"{_format_number(value)}"
                        )

                    return (
                        f"Top {n} by {metric}:\n\n"
                        + "\n".join(lines)
                    )

    # -----------------------------------------------------
    # NO DIRECT ANSWER
    # -----------------------------------------------------

    return None


# =========================================================
# AI FALLBACK
# =========================================================

def _build_small_ai_context(df):

    context = {
        "rows": len(df),
        "columns": list(df.columns)
    }

    numeric_columns = _find_numeric_columns(
        df
    )

    numeric_stats = {}

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if series.empty:
            continue

        numeric_stats[column] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "min": float(series.min()),
            "max": float(series.max()),
            "sum": float(series.sum())
        }

    context["numeric_statistics"] = (
        numeric_stats
    )

    categorical = {}

    for column in df.columns:

        if (
            df[column].dtype == "object"
            or str(df[column].dtype).startswith("category")
        ):

            unique = df[column].nunique(
                dropna=True
            )

            if unique <= 30:

                counts = (
                    df[column]
                    .dropna()
                    .value_counts()
                    .head(20)
                )

                categorical[column] = {
                    str(k): int(v)
                    for k, v in counts.items()
                }

    context["categorical_frequencies"] = (
        categorical
    )

    return context


def _ask_ai_fallback(question, df):

    context = _build_small_ai_context(
        df
    )

    prompt = f"""
You are an AI Data Analyst.

Answer the user's question using ONLY the supplied
dataset statistics.

Do not invent values.

The Python application has already read the actual
uploaded dataset and calculated the statistics below.

DATASET:
{json.dumps(context, indent=2, default=str)}

QUESTION:
{question}

If the exact answer cannot be determined from these
statistics, clearly explain what information is missing.

Keep the answer concise and direct.
"""

    return ask_ai(
        prompt,
        max_tokens=350
    )


# =========================================================
# MAIN FUNCTION
# =========================================================

def ask_dataset(
    question,
    analysis,
    summary,
    df
):

    question = str(
        question
    ).strip()

    if not question:

        return "Please enter a question."

    # -----------------------------------------------------
    # TRY PYTHON FIRST
    # -----------------------------------------------------

    direct_answer = _answer_directly(
        question,
        df
    )

    if direct_answer is not None:

        return direct_answer

    # -----------------------------------------------------
    # AI ONLY WHEN PYTHON CANNOT CONFIDENTLY ANSWER
    # -----------------------------------------------------

    try:

        return _ask_ai_fallback(
            question,
            df
        )

    except Exception as e:

        return (
            "I couldn't answer that question right now. "
            f"AI fallback error: {e}"
        )