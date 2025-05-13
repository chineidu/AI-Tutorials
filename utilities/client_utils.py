from typing import Any

import instructor
from openai import AsyncOpenAI

from settings import refresh_settings
from utilities.jinja_utils import (
    Environment,
    load_and_render_template,
    setup_jinja_environment,
)

# Load Settings
SETTINGS = refresh_settings()

# Load Prompt
# template_file: str = "ner_prompt.jinja2"
# template_env: Environment = setup_jinja_environment(searchpath="./prompts")
THINKING_MODE: bool = False

RESPONSE: list[dict[str, str]] = [
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


def get_client(is_remote: bool = True) -> instructor.AsyncInstructor:
    """Get the client to use for entity extraction."""
    if is_remote:
        # using remote
        remote_client: AsyncOpenAI = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=SETTINGS.OPENROUTER_API_KEY.get_secret_value(),
        )
        print("Using Remote")
        return instructor.patch(remote_client, mode=instructor.Mode.JSON)

    ollama_client: AsyncOpenAI = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    )
    print("Using Ollama")
    return instructor.from_openai(ollama_client, mode=instructor.Mode.JSON)
