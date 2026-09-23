import pandas as pd


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _looks_like_identifier(column_name, series):
    name = str(column_name).strip().lower()

    identifier_words = [
        "id",
        "uuid",
        "guid",
        "email",
        "phone",
        "mobile",
        "zip",
        "postal",
        "pincode",
        "code",
        "key"
    ]

    if name in ["id", "uuid", "guid"]:
        return True

    if any(
        name.endswith("_" + word)
        or name.startswith(word + "_")
        or word in name.split("_")
        for word in identifier_words
    ):
        return True

    unique_ratio = (
        series.nunique(dropna=True) / len(series)
        if len(series) > 0
        else 0
    )

    # A mostly unique column with a name suggesting an identifier
    if unique_ratio > 0.95 and (
        "number" in name
        or "reference" in name
        or "account" in name
    ):
        return True

    return False


def analyze_dataset(df):

    rows = len(df)
    columns = len(df.columns)

    numeric_columns = []
    categorical_columns = []
    date_columns = []
    identifier_columns = []
    text_columns = []

    column_details = {}

    for column in df.columns:

        series = df[column]
        name = str(column).strip()
        lower_name = name.lower()

        unique_count = series.nunique(dropna=True)

        unique_ratio = (
            unique_count / rows
            if rows > 0
            else 0
        )

        missing_count = int(series.isna().sum())

        missing_ratio = (
            missing_count / rows
            if rows > 0
            else 0
        )

        # ==========================================
        # IDENTIFIER DETECTION
        # ==========================================

        if _looks_like_identifier(column, series):

            identifier_columns.append(column)

            column_details[column] = {
                "type": "identifier",
                "unique_values": unique_count,
                "unique_ratio": round(unique_ratio, 4),
                "missing_values": missing_count,
                "missing_ratio": round(missing_ratio, 4)
            }

            continue

        # ==========================================
        # NUMERIC
        # ==========================================

        if pd.api.types.is_numeric_dtype(series):

            numeric_columns.append(column)

            clean = series.dropna()

            column_details[column] = {
                "type": "numeric",
                "unique_values": unique_count,
                "unique_ratio": round(unique_ratio, 4),
                "missing_values": missing_count,
                "missing_ratio": round(missing_ratio, 4),
                "min": _safe_float(clean.min()) if not clean.empty else None,
                "max": _safe_float(clean.max()) if not clean.empty else None,
                "mean": _safe_float(clean.mean()) if not clean.empty else None,
                "median": _safe_float(clean.median()) if not clean.empty else None,
                "std": _safe_float(clean.std()) if not clean.empty else None
            }

            continue

        # ==========================================
        # DATE DETECTION
        # ==========================================

        converted = pd.to_datetime(
            series,
            errors="coerce"
        )

        date_ratio = (
            converted.notna().mean()
            if rows > 0
            else 0
        )

        looks_like_date = any(
            word in lower_name
            for word in [
                "date",
                "time",
                "timestamp",
                "year",
                "month",
                "day"
            ]
        )

        if date_ratio > 0.8 and (
            looks_like_date
            or series.dtype == "object"
        ):

            date_columns.append(column)

            column_details[column] = {
                "type": "date",
                "unique_values": unique_count,
                "unique_ratio": round(unique_ratio, 4),
                "missing_values": missing_count,
                "missing_ratio": round(missing_ratio, 4),
                "min": str(converted.min())
                if converted.notna().any()
                else None,
                "max": str(converted.max())
                if converted.notna().any()
                else None
            }

            continue

        # ==========================================
        # CATEGORICAL / TEXT
        # ==========================================

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):

            non_null = series.dropna()

            sample_values = [
                str(value)
                for value in non_null.head(5).tolist()
            ]

            # Low/moderate cardinality = categorical
            if unique_ratio <= 0.5:

                categorical_columns.append(column)

                column_details[column] = {
                    "type": "categorical",
                    "unique_values": unique_count,
                    "unique_ratio": round(unique_ratio, 4),
                    "missing_values": missing_count,
                    "missing_ratio": round(missing_ratio, 4),
                    "sample_values": sample_values
                }

            else:

                text_columns.append(column)

                column_details[column] = {
                    "type": "text",
                    "unique_values": unique_count,
                    "unique_ratio": round(unique_ratio, 4),
                    "missing_values": missing_count,
                    "missing_ratio": round(missing_ratio, 4),
                    "sample_values": sample_values
                }

            continue

        # ==========================================
        # OTHER
        # ==========================================

        column_details[column] = {
            "type": str(series.dtype),
            "unique_values": unique_count,
            "unique_ratio": round(unique_ratio, 4),
            "missing_values": missing_count,
            "missing_ratio": round(missing_ratio, 4)
        }

    return {
        "rows": rows,
        "columns": columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "identifier_columns": identifier_columns,
        "text_columns": text_columns,
        "column_details": column_details
    }