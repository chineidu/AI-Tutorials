import json
import os
from typing import Any, Literal

import instructor
import requests  # type: ignore
from openai import AsyncOpenAI

from settings import refresh_settings  # type: ignore

# Load Settings
SETTINGS = refresh_settings()


# Required for litellm
os.environ["OPENROUTER_API_KEY"] = SETTINGS.OPENROUTER_API_KEY.get_secret_value()
os.environ["OPENROUTER_API_BASE"] = SETTINGS.OPENROUTER_URL
os.environ["MISTRAL_API_KEY"] = SETTINGS.MISTRAL_API_KEY.get_secret_value()
os.environ["GEMINI_API_KEY"] = SETTINGS.GEMINI_API_KEY.get_secret_value()


def openai_client() -> instructor.AsyncInstructor:
    """
    Create an async OpenAI client configured with OpenRouter credentials.

    Returns
    -------
    AsyncOpenAI
        An authenticated async OpenAI client instance.
    """
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=SETTINGS.OPENROUTER_API_KEY.get_secret_value(),
            base_url=SETTINGS.OPENROUTER_URL,
        ),
        mode=instructor.Mode.JSON,
    )



def check_rate_limit() -> None:
    """
    Check the rate limit status for the OpenRouter API..

    Returns
    -------
    None
        Prints the JSON response from the API containing rate limit information.
    """

    response = requests.get(
        url="https://openrouter.ai/api/v1/auth/key",
        headers={"Authorization": f"Bearer {SETTINGS.OPENROUTER_API_KEY.get_secret_value()}"},
    )

    print(json.dumps(response.json(), indent=2))
