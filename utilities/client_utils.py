from typing import Any, Literal

import instructor
from openai import AsyncOpenAI

from settings import refresh_settings

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


def get_client(
    is_remote: bool = True, mode: Literal["json_mode", "tool_mode"] = "json_mode"
) -> instructor.AsyncInstructor:
    """Get the client to use for entity extraction."""

    def _return_mode(mode: Literal["json_mode", "tool_mode"]) -> instructor.Mode:
        if mode == "json_mode":
            return instructor.Mode.JSON
        return instructor.Mode.TOOLS

    _mode = _return_mode(mode)
    print(f"Using mode: {_mode!r}")

    if is_remote:
        # using remote
        remote_client: AsyncOpenAI = AsyncOpenAI(
            base_url=SETTINGS.OPENROUTER_URL,
            api_key=SETTINGS.OPENROUTER_API_KEY.get_secret_value(),
        )
        print("Using Remote")
        return instructor.from_openai(remote_client, mode=_mode)

    ollama_client: AsyncOpenAI = AsyncOpenAI(
        base_url=SETTINGS.OLLAMA_URL,
        api_key=SETTINGS.OLLAMA_API_KEY.get_secret_value(),  # required, but unused
    )
    print("Using Ollama")
    return instructor.from_openai(ollama_client, mode=_mode)

