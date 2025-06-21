import json
import re
from dataclasses import dataclass
from typing import Any, Type

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, SecretStr

SYSTEM_MESSAGE: str = """
<system>
/no_think
<role>
You are a data extraction assistant. Extract information from the provided text and return ONLY a 
valid JSON object that matches this exact schema:

<schema>
{json_schema}
</schema>
</role>

<guidelines>
- Return only valid JSON - no explanations, markdown, or additional text
- Extract data precisely as it appears in the source text
- Do not include fields not present in the schema
- For missing required fields, use these defaults:
  * Numbers: 0
  * Strings: null
  * Booleans: false
  * Arrays: []
  * Objects: {{}}
- Preserve original data types and formatting where possible
- If text contains ambiguous information, choose the most likely interpretation
</guidelines>

<output>
Valid JSON object only
</output>

</system>
"""


def _clean_response_text_single_regex(text: str) -> str:
    """
    Clean response text by removing XML-like tags and backticks using regex pattern.

    Parameters
    ----------
    text : str
        Input text containing XML-like tags and backticks to be cleaned.

    Returns
    -------
    str
        Cleaned text with XML-like tags and backticks removed.
    """
    pattern: str = r"<think>.*?</think>|`+json|`+"
    cleaned_text: str = re.sub(pattern, "", text, flags=re.DOTALL)

    return cleaned_text.strip()


@dataclass
class LLMResponse:
    """Class for handling LLM API responses.

    Parameters
    ----------
    api_key : SecretStr
        The API key for authentication.
    base_url : str
        The base URL for the API endpoint.
    model : str
        The name of the LLM model to use.
    """

    api_key: SecretStr
    base_url: str
    model: str

    async def ainvoke(self, messages: list[dict[str, str]]) -> str | None:
        """Asynchronously invoke the LLM API with the given messages.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dictionaries containing role and content.

        Returns
        -------
        str | None
            The cleaned response text if successful, None if an error occurs.

        Notes
        -----
        The function attempts to create an async client and make an API call.
        If successful, it returns the cleaned response content.
        If an exception occurs, it returns None.
        """
        try:
            aclient: AsyncOpenAI = AsyncOpenAI(
                api_key=self.api_key.get_secret_value(),
                base_url=self.base_url,
                max_retries=3,
                timeout=180,  # type: ignore
            )

            raw_response = await aclient.chat.completions.create(  # type: ignore
                model=self.model,
                messages=messages,  # type: ignore
                temperature=0,
                seed=42,
            )

            return _clean_response_text_single_regex(raw_response.choices[0].message.content)  # type: ignore

        except Exception:
            return None

    async def get_structured_response(
        self, message: str, response_model: Type[BaseModel]
    ) -> tuple[Type[BaseModel], Type[BaseModel]] | tuple[dict[str, str], dict[str, str]]:
        """Get structured response from OpenAI API.

        Parameters
        ----------
            message : str
                The user message to send to the API.
            response_model : Type[BaseModel]
                The Pydantic model class to validate the response.

        Returns
        -------
        A tuple containing either:
        - (structured_output, raw_response)
        - (error_dict, error_info)
        """
        try:
            aclient: AsyncOpenAI = AsyncOpenAI(
                api_key=self.api_key.get_secret_value(),
                base_url=self.base_url,
            )

            json_schema: dict = response_model.model_json_schema()
            raw_response = await aclient.chat.completions.create(  # type: ignore
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_MESSAGE.format(json_schema=json_schema),
                    },
                    {"role": "user", "content": message},
                ],
                response_format={"type": "json_schema", "schema": json_schema, "strict": True},
                temperature=0,
                seed=42,
            )

            _value = _clean_response_text_single_regex(raw_response.choices[0].message.content)
            structured_output = response_model.model_validate(json.loads(_value))
            return (structured_output, raw_response)  # type: ignore

        except Exception as e:
            return (
                {"status": "error", "error": str(e)},
                {"status": "error", "raw_response": None},
            )  # type: ignore


def convert_to_openai_messages(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """
    Convert a list of messages to OpenAI compatible message format.

    Parameters
    ----------
    messages : list[AnyMessage]
        List of messages to be converted. Can contain SystemMessage,
        AIMessage, ToolMessage, or other message types.

    Returns
    -------
    list[dict[str, Any]]
    """
    formatted: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            formatted.append({"role": "system", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            formatted.append(
                {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                }
            )
        else:
            formatted.append({"role": "user", "content": msg.content})
    return formatted
