import re
import pandas as pd

from ai_client import ask_ai


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def _normalise(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).strip().lower()
    )


def _find_column(df, keywords):

    columns = list(df.columns)

    # Exact / substring matching
    for keyword in keywords:

        keyword = _normalise(keyword)

        for column in columns:

            if keyword in _normalise(column):
                return column

    return None


def _find_numeric_columns(df):

    return list(
        df.select_dtypes(
            include="number"
        ).columns
    )


def _find_group_column(df):

    categorical = list(
        df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
    )

    if categorical:
        return categorical[0]

    return None


def _find_price_column(df):

    return _find_column(
        df,
        [
            "price",
            "selling price",
            "unit price",
            "cost",
            "amount"
        ]
    )


def _find_profit_column(df):

    return _find_column(
        df,
        [
            "profit",
            "net profit",
            "gross profit"
        ]
    )


def _find_revenue_column(df):

    return _find_column(
        df,
        [
            "revenue",
            "sales",
            "sale",
            "total sales"
        ]
    )


# ---------------------------------------------------------
# DIRECT CALCULATIONS
# ---------------------------------------------------------

def _calculate_profit_percentage(df):

    profit_column = _find_profit_column(df)

    if profit_column:

        profit = pd.to_numeric(
            df[profit_column],
            errors="coerce"
        ).dropna()

        if len(profit) > 0:

            return (
                f"The average profit percentage is not directly "
                f"available because the dataset contains a profit "
                f"column but no clearly identified revenue/cost "
                f"basis for calculating the percentage."
            )

    price_column = _find_price_column(df)

    if price_column:

        price = pd.to_numeric(
            df[price_column],
            errors="coerce"
        ).dropna()

        if len(price) > 0:

            return (
                f"The average value of '{price_column}' is "
                f"{price.mean():,.2f}."
            )

    return None


def _direct_answer(df, question):

    q = _normalise(question)

    numeric_columns = _find_numeric_columns(df)

    # -----------------------------------------------------
    # COUNT
    # -----------------------------------------------------

    if (
        "how many rows" in q
        or "number of rows" in q
        or "how many records" in q
        or "number of records" in q
    ):

        return f"The dataset contains {len(df):,} rows."


    # -----------------------------------------------------
    # TOTAL / SUM
    # -----------------------------------------------------

    if (
        "total" in q
        or "sum" in q
    ):

        target_column = None

        if "revenue" in q:
            target_column = _find_revenue_column(df)

        elif "profit" in q:
            target_column = _find_profit_column(df)

        elif "sales" in q:
            target_column = _find_revenue_column(df)

        else:

            for column in numeric_columns:

                if _normalise(column) in q:
                    target_column = column
                    break

        if target_column:

            values = pd.to_numeric(
                df[target_column],
                errors="coerce"
            ).dropna()

            if len(values) > 0:

                return (
                    f"The total {target_column} is "
                    f"{values.sum():,.2f}."
                )


    # -----------------------------------------------------
    # AVERAGE
    # -----------------------------------------------------

    if (
        "average" in q
        or "mean" in q
    ):

        target_column = None

        for column in numeric_columns:

            if _normalise(column) in q:

                target_column = column
                break

        if target_column is None:

            if "revenue" in q:
                target_column = _find_revenue_column(df)

            elif "profit" in q:
                target_column = _find_profit_column(df)

            elif "sales" in q:
                target_column = _find_revenue_column(df)

        if target_column:

            values = pd.to_numeric(
                df[target_column],
                errors="coerce"
            ).dropna()

            if len(values) > 0:

                return (
                    f"The average {target_column} is "
                    f"{values.mean():,.2f}."
                )


    # -----------------------------------------------------
    # MEDIAN
    # -----------------------------------------------------

    if "median" in q:

        target_column = None

        for column in numeric_columns:

            if _normalise(column) in q:

                target_column = column
                break

        if target_column:

            values = pd.to_numeric(
                df[target_column],
                errors="coerce"
            ).dropna()

            if len(values) > 0:

                return (
                    f"The median {target_column} is "
                    f"{values.median():,.2f}."
                )


    # -----------------------------------------------------
    # TOP N
    # -----------------------------------------------------

    top_match = re.search(
        r"(?:top|highest)\s+(\d+)",
        q
    )

    if top_match:

        n = int(
            top_match.group(1)
        )

        target_column = None

        for column in numeric_columns:

            if _normalise(column) in q:

                target_column = column
                break

        if target_column is None:

            if "revenue" in q:
                target_column = _find_revenue_column(df)

            elif "profit" in q:
                target_column = _find_profit_column(df)

            elif "sales" in q:
                target_column = _find_revenue_column(df)

        if target_column:

            group_column = _find_group_column(df)

            if group_column:

                grouped = (
                    df.groupby(group_column)[target_column]
                    .sum()
                    .sort_values(ascending=False)
                    .head(n)
                )

                if len(grouped) > 0:

                    result = []

                    for name, value in grouped.items():

                        result.append(
                            f"{name}: {value:,.2f}"
                        )

                    return (
                        f"Top {n} by {target_column}:\n\n"
                        + "\n".join(
                            f"{i + 1}. {item}"
                            for i, item in enumerate(result)
                        )
                    )


    # -----------------------------------------------------
    # HIGHEST
    # -----------------------------------------------------

    if (
        "highest" in q
        or "maximum" in q
        or "max" in q
    ):

        target_column = None

        for column in numeric_columns:

            if _normalise(column) in q:

                target_column = column
                break

        if target_column is None:

            if "revenue" in q:
                target_column = _find_revenue_column(df)

            elif "profit" in q:
                target_column = _find_profit_column(df)

            elif "sales" in q:
                target_column = _find_revenue_column(df)

        if target_column:

            values = pd.to_numeric(
                df[target_column],
                errors="coerce"
            )

            if values.notna().any():

                maximum = values.max()

                return (
                    f"The highest {target_column} is "
                    f"{maximum:,.2f}."
                )


    # -----------------------------------------------------
    # LOWEST
    # -----------------------------------------------------

    if (
        "lowest" in q
        or "minimum" in q
        or "min" in q
    ):

        target_column = None

        for column in numeric_columns:

            if _normalise(column) in q:

                target_column = column
                break

        if target_column is None:

            if "revenue" in q:
                target_column = _find_revenue_column(df)

            elif "profit" in q:
                target_column = _find_profit_column(df)

            elif "sales" in q:
                target_column = _find_revenue_column(df)

        if target_column:

            values = pd.to_numeric(
                df[target_column],
                errors="coerce"
            )

            if values.notna().any():

                minimum = values.min()

                return (
                    f"The lowest {target_column} is "
                    f"{minimum:,.2f}."
                )


    # -----------------------------------------------------
    # MOST COMMON CATEGORY
    # -----------------------------------------------------

    if (
        "most common" in q
        or "most popular" in q
        or "most frequent" in q
    ):

        group_column = _find_group_column(df)

        if group_column:

            counts = (
                df[group_column]
                .value_counts()
            )

            if len(counts) > 0:

                value = counts.index[0]
                count = counts.iloc[0]

                return (
                    f"The most common value in "
                    f"'{group_column}' is '{value}', "
                    f"appearing {count:,} times."
                )


    # -----------------------------------------------------
    # PROFIT PERCENTAGE
    # -----------------------------------------------------

    if (
        "profit percentage" in q
        or "profit margin" in q
        or "profit %" in q
    ):

        result = _calculate_profit_percentage(
            df
        )

        if result:
            return result


    # -----------------------------------------------------
    # NO DIRECT ANSWER
    # -----------------------------------------------------

    return None


# ---------------------------------------------------------
# AI FALLBACK
# ---------------------------------------------------------

def _ai_answer(df, question):

    numeric_columns = _find_numeric_columns(df)

    statistics = {}

    for column in numeric_columns[:15]:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        statistics[column] = {
            "count": int(values.count()),
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    categorical_columns = list(
        df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
    )

    categorical_summary = {}

    for column in categorical_columns[:10]:

        values = df[column].dropna()

        if len(values) == 0:
            continue

        categorical_summary[column] = (
            values.value_counts()
            .head(10)
            .to_dict()
        )

    prompt = f"""
You are an AI Data Analyst.

Answer the user's question using ONLY the dataset
information provided below.

Do not invent values.

If the exact answer cannot be determined from the
provided information, clearly say that.

Dataset shape:
Rows: {len(df)}
Columns: {len(df.columns)}

Columns:
{list(df.columns)}

Numeric statistics:
{statistics}

Categorical summaries:
{categorical_summary}

User question:
{question}

Give a concise, useful answer.
"""

    return ask_ai(
        prompt,
        max_tokens=500
    )


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------

def ask_dataset(df, question):

    direct_answer = _direct_answer(
        df,
        question
    )

    if direct_answer:

        return direct_answer

    return _ai_answer(
        df,
        question
    )