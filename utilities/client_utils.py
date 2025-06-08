import json
from typing import Any, Literal

import instructor
import requests  # type: ignore
from litellm import acompletion

from settings import refresh_settings

# Load Settings
SETTINGS = refresh_settings()

# Load Prompt
# template_file: str = "ner_prompt.jinja2"
# template_env: Environment = setup_jinja_environment(searchpath="./prompts")
THINKING_MODE: bool = False

RESPONSE: list[dict[str, Any]] = [
    {"text": "TRF TO", "label": "transactionReason", "score": 0.84},
    {"text": "Access Bank", "label": "Miscellaneous", "score": 0.93},
    {"text": "PiggyVest", "label": "loanLender", "score": 0.96},
]
context: dict[str, Any] = {
    # TODO:  To be populated later
}
# SYSTEM_PROMPT = load_and_render_template(
#     env=template_env, template_file=template_file, context=context
# )


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
