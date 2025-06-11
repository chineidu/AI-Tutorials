from typing import Annotated, Any, Type, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from pydantic import BaseModel

from src.schemas import ModelEnum  # noqa: E402
from src.settings import refresh_settings  # type: ignore
from src.studio import configuration  # type: ignore

settings = refresh_settings()


class DataState(TypedDict):
    messages: Annotated[list[Any], add_messages]


class DataStateValidator(BaseModel):
    messages: Annotated[list[BaseChatModel], add_messages]


def validate_data(
    data: dict[str, Any], state: dict[str, Any], response_model: Type[BaseModel]
) -> None:
    """Validate that input data matches the expected response model structure.

    Parameters
    ----------
    data : dict[str, Any]
        The input data dictionary to validate
    state : dict[str, Any]
        The state dictionary to update with input data
    response_model : Type[BaseModel]
        The Pydantic model class to validate against

    Returns
    -------
    None
        Function performs validation through assertions

    Raises
    ------
    AssertionError
        If data is not a dictionary, state is not a dictionary, or fails model validation
    """
    assert isinstance(data, dict), "Data must be a dictionary"
    assert isinstance(state, dict), "Data must be a dictionary"
    state.update(data)
    assert response_model(**state), "Data is not valid"


# Chatbot instruction
MODEL_SYSTEM_MESSAGE: str = """
<system>

<role>
You are a helpful assistant with memory that provides information about the user.
If you have memory for this user, use it to personalize your responses.
</role>

<memory>
{memory}
</memory>

<quality_standards>
- **ALWAYS** use the information in memory.
</quality_standards>

</system>
"""

# Create new memory from the chat history and any existing memory
CREATE_MEMORY_INSTRUCTION: str = """"
<system>

<role>
You are collecting information about the user to personalize your responses.
</role>

<current_user_info>
{memory}
</current_user_info>

<instruction>
1. Review the chat history below carefully
2. Identify new information about the user, such as:
   - Personal details (name, location)
   - Preferences (likes, dislikes)
   - Interests and hobbies
   - Past experiences
   - Goals or future plans
3. Merge any new information with existing memory
4. Format the memory as a clear, bulleted list
5. If new information conflicts with existing memory, keep the most recent version
Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.
</instruction>

Based on the chat history below, please update the user information:

<system>
"""


model_str: str = f"openai:{ModelEnum.LLAMA_3p2_3B_INSTRUCT_REMOTE.value}"
llm: BaseChatModel = init_chat_model(
    model=model_str,
    api_key=settings.OPENROUTER_API_KEY.get_secret_value(),
    base_url=settings.OPENROUTER_URL,
    temperature=0.0,
    seed=0,
)


async def call_llm(state: DataState, config: RunnableConfig, store: BaseStore) -> dict[str, Any]:
    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)

    # Get the user id
    user_id = configurable.user_id

    # Get the memory for the user
    prefix: str = "memory"
    key = "user_memory"
    namespace = (prefix, user_id)
    existing_memory = await store.get(namespace, key)

    if existing_memory:
        existing_memory_content = existing_memory.value.get(prefix)
    else:
        existing_memory_content = "No existing memory found"

    system_message: str = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)
    # Respond using memory + chat history
    response = await llm.ainvoke([SystemMessage(content=system_message)] + state["messages"])

    # Validate
    output = {"messages": [response]}
    validate_data(data=output, state=state, response_model=DataStateValidator)  # type: ignore

    return output


async def write_memory(state: DataState, config: RunnableConfig, store: BaseStore) -> None:
    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    prefix: str = "memory"
    key = "user_memory"
    namespace = (prefix, user_id)
    existing_memory = await store.get(namespace, key)

    if existing_memory:
        existing_memory_content = existing_memory.value.get(prefix)
    else:
        existing_memory_content = "No existing memory found"

    system_message: str = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
    # Respond using memory + chat history
    new_memory = await llm.ainvoke([SystemMessage(content=system_message)] + state["messages"])

    # Update existing memory
    await store.put(namespace, key, {prefix: new_memory.content})


# Graph
graph_builder = StateGraph(DataState, config_schema=configuration.Configuration)

# Add nodes
graph_builder.add_node("call_llm", call_llm)
graph_builder.add_node("write_memory", write_memory)

# Add edges
graph_builder.add_edge(START, "call_llm")
graph_builder.add_edge("call_llm", "write_memory")
graph_builder.add_edge("write_memory", END)

# Compile
graph = graph_builder.compile()


# from langchain_core.messages import SystemMessage
# from langchain.chat_models import init_chat_model
# from langchain_core.runnables.config import RunnableConfig
# from langgraph.graph import END, START, MessagesState, StateGraph
# from langgraph.store.base import BaseStore

# # Initialize the LLM
# model_str: str = "openai:meta-llama/llama-3.2-3b-instruct"
# llm = init_chat_model(
#     model=model_str,
#     api_key=settings.OPENROUTER_API_KEY.get_secret_value(),
#     base_url=settings.OPENROUTER_URL,
#     temperature=0.0,
#     seed=0,
# )

# # Chatbot instruction
# MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user.
# If you have memory for this user, use it to personalize your responses.
# Here is the memory (it may be empty): {memory}"""

# # Create new memory from the chat history and any existing memory
# CREATE_MEMORY_INSTRUCTION = """"You are collecting information about the user to personalize your responses.

# CURRENT USER INFORMATION:
# {memory}

# INSTRUCTIONS:
# 1. Review the chat history below carefully
# 2. Identify new information about the user, such as:
#    - Personal details (name, location)
#    - Preferences (likes, dislikes)
#    - Interests and hobbies
#    - Past experiences
#    - Goals or future plans
# 3. Merge any new information with existing memory
# 4. Format the memory as a clear, bulleted list
# 5. If new information conflicts with existing memory, keep the most recent version

# Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.

# Based on the chat history below, please update the user information:"""


# def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
#     """Load memory from the store and use it to personalize the chatbot's response."""

#     # Get configuration
#     configurable = configuration.Configuration.from_runnable_config(config)

#     # Get the user ID from the config
#     user_id = configurable.user_id

#     # Retrieve memory from the store
#     namespace = ("memory", user_id)
#     key = "user_memory"
#     existing_memory = store.get(namespace, key)

#     # Extract the memory
#     if existing_memory:
#         # Value is a dictionary with a memory key
#         existing_memory_content = existing_memory.value.get("memory")
#     else:
#         existing_memory_content = "No existing memory found."

#     # Format the memory in the system prompt
#     system_msg = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)

#     # Respond using memory as well as the chat history
#     response = llm.invoke([SystemMessage(content=system_msg)] + state["messages"])  # type: ignore

#     return {"messages": response}


# def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
#     """Reflect on the chat history and save a memory to the store."""

#     # Get configuration
#     configurable = configuration.Configuration.from_runnable_config(config)

#     # Get the user ID from the config
#     user_id = configurable.user_id

#     # Retrieve existing memory from the store
#     namespace = ("memory", user_id)
#     existing_memory = store.get(namespace, "user_memory")

#     # Extract the memory
#     if existing_memory:
#         # Value is a dictionary with a memory key
#         existing_memory_content = existing_memory.value.get("memory")
#     else:
#         existing_memory_content = "No existing memory found."

#     # Format the memory in the system prompt
#     system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
#     new_memory = llm.invoke([SystemMessage(content=system_msg)] + state["messages"])  # type: ignore

#     # Overwrite the existing memory in the store
#     key = "user_memory"
#     store.put(namespace, key, {"memory": new_memory.content})


# # Define the graph
# builder = StateGraph(MessagesState, config_schema=configuration.Configuration)
# builder.add_node("call_model", call_model)
# builder.add_node("write_memory", write_memory)
# builder.add_edge(START, "call_model")
# builder.add_edge("call_model", "write_memory")
# builder.add_edge("write_memory", END)
# graph = builder.compile()
