import json
import os
from typing import Any, Literal

import instructor
import requests  # type: ignore
from litellm import acompletion

from settings import refresh_settings  # type: ignore

# Load Settings
SETTINGS = refresh_settings()


# Required for litellm
os.environ["OPENROUTER_API_KEY"] = SETTINGS.OPENROUTER_API_KEY.get_secret_value()
os.environ["OPENROUTER_API_BASE"] = SETTINGS.OPENROUTER_URL
os.environ["MISTRAL_API_KEY"] = SETTINGS.MISTRAL_API_KEY.get_secret_value()
os.environ["GEMINI_API_KEY"] = SETTINGS.GEMINI_API_KEY.get_secret_value()


def get_aclient(
    return_type: Literal["litellm", "instructor"],
) -> Any | instructor.AsyncInstructor:
    """
    Create an async client for either litellm or instructor.

    Parameters
    ----------
    return_type : Literal["litellm", "instructor"]
        The type of client to return. Can be either "litellm" or "instructor".

    Returns
    -------
    Union[Any, instructor.AsyncInstructor]
        If return_type is "litellm", returns acompletion object.
        If return_type is "instructor", returns an AsyncInstructor instance.
    """
    if return_type == "litellm":
        print("Using litellm")
        return acompletion

    return instructor.from_litellm(acompletion, mode=instructor.Mode.JSON)


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
