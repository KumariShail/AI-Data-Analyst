import json

from ai_client import ask_ai


def generate_insights(analysis, summary):

    prompt = f"""
You are an AI Data Analyst.

Analyze the supplied dataset information and generate
useful, factual insights.

=========================================================
DATASET STRUCTURE
=========================================================

{json.dumps(
    analysis,
    indent=2,
    default=str
)}

=========================================================
STATISTICAL SUMMARY
=========================================================

{json.dumps(
    summary,
    indent=2,
    default=str
)}

=========================================================
STRICT OUTPUT RULES
=========================================================

Return ONLY the requested JSON structure.

IMPORTANT:

1. "summary" must contain exactly ONE concise paragraph.

2. "key_insights" must contain EXACTLY 3 items.

3. Every key insight MUST be a non-empty string.

4. Do NOT add a fourth insight.

5. "recommendations" must contain EXACTLY 2 items.

6. Every recommendation MUST be a non-empty string.

7. Do not invent facts or values.

8. Use only information supported by the supplied data.

9. Do not claim causation unless the data supports it.

10. If there are fewer than 3 meaningful insights available,
    use the available meaningful insights and do not create
    fake information.

11. Keep every insight concise.

12. Keep every recommendation concise.

The JSON must contain ONLY these three fields:

- summary
- key_insights
- recommendations
"""

    schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string"
            },
            "key_insights": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "string"
                }
            },
            "recommendations": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "string"
                }
            }
        },
        "required": [
            "summary",
            "key_insights",
            "recommendations"
        ],
        "additionalProperties": False
    }

    response = ask_ai(
        prompt,
        max_tokens=1200,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "dataset_insights",
                "strict": True,
                "schema": schema
            }
        }
    )

    result = json.loads(response)

    # =====================================================
    # CLEAN AI OUTPUT
    # =====================================================

    # Remove empty/whitespace-only insights
    key_insights = [
        str(item).strip()
        for item in result.get(
            "key_insights",
            []
        )
        if str(item).strip()
    ]

    # Remove empty/whitespace-only recommendations
    recommendations = [
        str(item).strip()
        for item in result.get(
            "recommendations",
            []
        )
        if str(item).strip()
    ]

    # Keep only the allowed number
    key_insights = key_insights[:3]
    recommendations = recommendations[:2]

    result["key_insights"] = key_insights
    result["recommendations"] = recommendations

    result["summary"] = str(
        result.get(
            "summary",
            ""
        )
    ).strip()

    return result