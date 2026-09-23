import os
import requests
from dotenv import load_dotenv

load_dotenv()


def ask_ai(prompt, max_tokens=700, response_format=None):

    mode = os.getenv("AI_MODE", "groq").lower()

    if mode == "groq":

        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from .env")

        # IMPORTANT: model is defined BEFORE client/request
        model = os.getenv("GROQ_MODEL")

        if not model:
            model = "openai/gpt-oss-20b"

        client = Groq(api_key=api_key)

        request_data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "reasoning_effort":"low",
            "max_completion_tokens": max_tokens
        }

        if response_format is not None:
            request_data["response_format"] = response_format

        response = client.chat.completions.create(
            **request_data
        )

        return response.choices[0].message.content


    elif mode == "geniex":

        url = os.getenv(
            "GENIEX_URL",
            "http://127.0.0.1:18181/v1/chat/completions"
        )

        model = os.getenv(
            "GENIEX_MODEL",
            "ai-hub-models/Qwen3-4B"
        )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens
        }

        response = requests.post(
            url,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]


    else:

        raise ValueError(
            f"Unsupported AI mode: {mode}"
        )