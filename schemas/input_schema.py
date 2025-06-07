from enum import Enum
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict  # type: ignore
from pydantic.alias_generators import to_camel


def round_probability(value: float) -> float:
    """Round a float value to two decimal places.

    Returns:
        float: Rounded value.
    """
    if isinstance(value, float):
        return round(value, 2)
    return value


def strip_string(text: str) -> str:
    """Strip whitespace from beginning and end of a string.

    Parameters
    ----------
    text : str
        The input string to be stripped.

    Returns
    -------
    str
        The string with leading and trailing whitespace removed.
    """
    return text.strip()


class BaseSchema(BaseModel):
    """Base schema class that inherits from Pydantic BaseModel.

    This class provides common configuration for all schema classes including
    camelCase alias generation, population by field name, and attribute mapping.
    """

    model_config: ConfigDict = ConfigDict(  # type: ignore
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )


Float = Annotated[float, BeforeValidator(round_probability)]
String = Annotated[str, BeforeValidator(strip_string)]

class GeneralResponse(BaseSchema):
    content: String


class ModelEnum(str, Enum):
    DEEPSEEK_R1_1p5B_LOCAL = "deepseek-r1:1.5b"
    DEEPSEEK_R1_70B_REMOTE_FREE = "deepseek/deepseek-r1-distill-llama-70b:free"

    GEMMA_3p0_1B_LOCAL = "gemma3:1b"
    GEMMA_3p0_12B_REMOTE_FREE = "google/gemma-3-12b-it:free"
    GEMMA_3p0_12B_REMOTE = "google/gemma-3-12b-it"  # $0.05/1M tokens
    GEMMA_3p0_27B_REMOTE = "google/gemma-3-27b-it"  # $0.1/1M tokens
    GEMINI_2p5_FLASH_REMOTE = "google/gemini-2.5-flash-preview-05-20"  # $0.15/1M tokens
    GPT_4_p_1_NANO_REMOTE = "openai/gpt-4.1-nano"
    GPT_4_o_MINI_REMOTE = "openai/gpt-4o-mini"  # $0.15/1M tokens

    QWEN_2p5_3B_LOCAL = "qwen2.5:3b"
    QWEN_3p0_4B_LOCAL = "qwen3:4b-q4_K_M"
    QWEN_3p0_8B_REMOTE_FREE = "qwen/qwen3-8b:free"
    QWEN_3p0_8B_REMOTE = "qwen/qwen3-8b"  # $0.035/1M tokens
    QWEN_3p0_30B_REMOTE = "qwen/qwen3-30b-a3b"  # $0.08/1M tokens
    QWEN_2p5_VL_72B_INSTRUCT_REMOTE_FREE = "qwen/qwen2.5-vl-72b-instruct:free"  # $0.00/1M tokens
    QWEN_2p5_VL_72B_INSTRUCT_REMOTE = "qwen/qwen2.5-vl-72b-instruct"  # $0.25/1M tokens

    LLAMA_3p1_8B_REMOTE = "meta-llama/llama-3.1-8b-instruct"  # $0.02/1M tokens
    LLAMA_3p2_3B_INSTRUCT_REMOTE = "meta-llama/llama-3.2-3b-instruct"  # $0.01/1M tokens
    LLAMA_3p2_11B_VISION_REMOTE_FREE = "meta-llama/llama-3.2-11b-vision-instruct:free"
    LLAMA_4_MAVERICK_17B_REMOTE_FREE = "meta-llama/llama-4-maverick:free"
    LLAMA_3p3_70B_INSTRUCT_REMOTE = "meta-llama/llama-3.3-70b-instruct"  # 0.07/1M tokens
    LLAMA_GUARD_4_12B_MULTIMODAL_REMOTE = "meta-llama/llama-guard-4-12b"  # $0.05/1M tokens

    PHI_4p0_14B_REASONING_REMOTE = "microsoft/phi-4-reasoning-plus"  # $0.07/1M tokens
    PHI_4p0_5p6B_MULTIMODAL_REMOTE_FREE = "microsoft/phi-4-multimodal-instruct"  # $0.05/1M tokens
    PHI_3p5_128K_INSTRUCT_WF_TOOL_USE_REMOTE = (
        "microsoft/phi-3.5-mini-128k-instruct"  # $0.10/1M tokens
    )
    MISTRAL_7B_REMOTE_FREE = "mistralai/mistral-7b-instruct:free"
    MISTRAL_NEMO_12B_REMOTE_FREE = "mistralai/mistral-nemo:free"
    MISTRAL_8B_REMOTE = "mistralai/ministral-8b"  # $0.10/1M tokens
    MISTRAL_NEMO_12B_REMOTE = "mistralai/mistral-nemo"  # $0.03/1M tokens
    MISTRAL_7B_REMOTE = "mistralai/mistral-7b-instruct"  # $0.028/1M tokens
    PHI_4p0_8B_REMOTE_FREE = "microsoft/phi-4-reasoning-plus:free"
    DOLPHIN_3p0_MISTRAL_24B_REMOTE_FREE = "cognitivecomputations/dolphin3.0-mistral-24b:free"

    BASE_MODEL_LOCAL_1 = "gemma3:1b"
    BASE_MODEL_LOCAL_2 = "qwen3:4b-q4_K_M"
    BASE_REMOTE_MODEL_1_8B = "meta-llama/llama-3.1-8b-instruct"  # $0.02/1M tokens

    # Via LiteLLM
    MISTRAL_EMBED__MISTRAL_API = "mistral/mistral-embed"  # $0.00/1M tokens
    BASE_REMOTE_MODEL_8B_MISTRAL_API = "mistral/ministral-8b-latest"  # $0.00/1M tokens
