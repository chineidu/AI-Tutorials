import json
from dataclasses import dataclass
from typing import Type

from openai import AsyncOpenAI
from pydantic import BaseModel

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

<output_format>
Valid JSON object only
</output_format>

</system>
"""


@dataclass
class StructuredLLMResponse:
    """Structured response model for LLM outputs.

    Parameters
    ----------
    api_key : str
        The API key for authentication
    base_url : str
        The base URL for the API endpoint
    model : str
        The name of the model to use
    """

    api_key: str
    base_url: str
    model: str

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
        tuple[Type[BaseModel] | Type[BaseModel]] | tuple[dict[str, str], dict[str, str]]
        A tuple containing either:
        - (structured_output, raw_response)
        - (error_dict, error_info)
        """
        try:
            aclient: AsyncOpenAI = AsyncOpenAI(
                api_key=self.api_key,
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

            _value = (
                raw_response.choices[0]
                .message.content.replace("<think>", "")
                .replace("</think>", "")
            )
            structured_output = response_model.model_validate(json.loads(_value))
            return (structured_output, raw_response)  # type: ignore

        except Exception as e:
            return (
                {"status": "error", "error": str(e)},
                {"status": "error", "raw_response": None},
            )  # type: ignore
