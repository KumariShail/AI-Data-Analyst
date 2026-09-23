import json
from ai_client import ask_ai


def generate_dashboard_plan(analysis):

    prompt = f"""
You are the AI planning engine of a general-purpose AI Data Analyst.

The user uploaded an unknown dataset.

Your job is to design a useful analytical dashboard specifically for THIS dataset.

DATASET PROFILE:

{json.dumps(analysis, indent=2, default=str)}

AVAILABLE VISUALIZATIONS:

- bar
- line
- pie
- scatter
- histogram

CHART RULES:

1. Only use columns that exist in the dataset profile.
2. Never invent column names.
3. Never use identifier columns.
4. Never use text columns as numerical metrics.
5. Use a bar chart for categorical comparisons or rankings.
6. Use a line chart for genuine time/date trends.
7. Use a pie chart only for a small categorical part-to-whole view.
8. Use scatter when two numeric variables have a potentially meaningful relationship.
9. Use histogram to show the distribution of a numeric variable.
10. Do not create a chart merely to increase the number of charts.
11. Avoid duplicate or nearly identical charts.
12. Prefer meaningful business/analytical questions.
13. If the dataset has no suitable date column, do not create a time-series chart.
14. If the dataset has no suitable categorical column, do not create categorical charts.
15. If the dataset has fewer than two suitable numeric columns, do not create scatter.
16. If the dataset has no useful numeric column, focus on categorical counts where possible.
17. Ignore identifiers such as IDs, emails, phone numbers and reference codes.
18. Select between 2 and 5 useful charts when enough information exists.
19. If fewer than 2 meaningful charts are possible, return only the meaningful charts.
20. Give every chart a clear human-readable title.
21. Prefer different analytical questions rather than simply different chart types.

CHART FIELDS:

type:
bar | line | pie | scatter | histogram

For BAR:
x = categorical/date column
y = numeric column OR null for a count
operation = sum | mean | count

For LINE:
x = date/time column
y = numeric column
operation = sum | mean

For PIE:
x = categorical column
y = numeric column OR null for a count
operation = sum | mean | count

For SCATTER:
x = numeric column
y = numeric column
operation = none

For HISTOGRAM:
x = numeric column
y = null
operation = none

Return ONLY the JSON object.
"""

    schema = {
        "type": "object",
        "properties": {
            "dashboard_title": {
                "type": "string"
            },
            "charts": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "bar",
                                "line",
                                "pie",
                                "scatter",
                                "histogram"
                            ]
                        },
                        "x": {
                            "type": "string"
                        },
                        "y": {
                            "type": [
                                "string",
                                "null"
                            ]
                        },
                        "operation": {
                            "type": "string",
                            "enum": [
                                "sum",
                                "mean",
                                "count",
                                "none"
                            ]
                        },
                        "title": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "type",
                        "x",
                        "y",
                        "operation",
                        "title"
                    ],
                    "additionalProperties": False
                }
            }
        },
        "required": [
            "dashboard_title",
            "charts"
        ],
        "additionalProperties": False
    }

    response = ask_ai(
        prompt,
        max_tokens=1200,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "dashboard_plan",
                "strict": True,
                "schema": schema
            }
        }
    )

    dashboard = json.loads(response)

    # ==========================================
    # PYTHON VALIDATION
    # ==========================================

    numeric = set(
        analysis["numeric_columns"]
    )

    categorical = set(
        analysis["categorical_columns"]
    )

    dates = set(
        analysis["date_columns"]
    )

    valid_columns = (
        numeric
        | categorical
        | dates
    )

    valid_charts = []

    for chart in dashboard.get("charts", []):

        chart_type = chart.get("type")
        x = chart.get("x")
        y = chart.get("y")
        operation = chart.get("operation")

        if chart_type not in {
            "bar",
            "line",
            "pie",
            "scatter",
            "histogram"
        }:
            continue

        if x not in valid_columns:
            continue

        # --------------------------------------
        # BAR
        # --------------------------------------

        if chart_type == "bar":

            if x not in categorical and x not in dates:
                continue

            if operation == "count":

                chart["y"] = None

            elif operation in {"sum", "mean"}:

                if y not in numeric:
                    continue

            else:
                continue

        # --------------------------------------
        # LINE
        # --------------------------------------

        elif chart_type == "line":

            if x not in dates:
                continue

            if y not in numeric:
                continue

            if operation not in {"sum", "mean"}:
                continue

        # --------------------------------------
        # PIE
        # --------------------------------------

        elif chart_type == "pie":

            if x not in categorical:
                continue

            unique_values = (
                analysis["column_details"]
                .get(x, {})
                .get("unique_values", 999999)
            )

            if unique_values > 8:
                continue

            if operation == "count":

                chart["y"] = None

            elif operation in {"sum", "mean"}:

                if y not in numeric:
                    continue

            else:
                continue

        # --------------------------------------
        # SCATTER
        # --------------------------------------

        elif chart_type == "scatter":

            if x not in numeric:
                continue

            if y not in numeric:
                continue

            if x == y:
                continue

            if operation != "none":
                continue

        # --------------------------------------
        # HISTOGRAM
        # --------------------------------------

        elif chart_type == "histogram":

            if x not in numeric:
                continue

            if y is not None:
                continue

            if operation != "none":
                continue

        valid_charts.append(chart)

    dashboard["charts"] = valid_charts

    return dashboard