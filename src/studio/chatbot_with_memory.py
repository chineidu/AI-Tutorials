from typing import Annotated, Any, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from pydantic import BaseModel

from src.schemas import ModelEnum  # noqa: E402
from src.settings import refresh_settings  # type: ignore
from src.studio import configuration  # type: ignore

settings = refresh_settings()


class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class MessageStateValidator(BaseModel):
    messages: Annotated[list[BaseChatModel], add_messages]


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


async def call_llm(state: MessageState, config: RunnableConfig, store: BaseStore) -> dict[str, Any]:
    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)

    # Get the user id
    user_id = configurable.user_id

    # Get the memory for the user
    prefix: str = "memory"
    key = "user_memory"
    namespace = (prefix, user_id)
    existing_memory = await store.aget(namespace, key)  # type: ignore

    if existing_memory:
        existing_memory_content = existing_memory.value.get(prefix)
    else:
        existing_memory_content = "No existing memory found"

    system_message: str = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)
    # Respond using memory + chat history
    response = await llm.ainvoke([SystemMessage(content=system_message)] + state["messages"])  # type: ignore

    return {"messages": [response]}


async def write_memory(state: MessageState, config: RunnableConfig, store: BaseStore) -> None:
    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    prefix: str = "memory"
    key = "user_memory"
    namespace = (prefix, user_id)
    existing_memory = await store.aget(namespace, key)  # type: ignore

    if existing_memory:
        existing_memory_content = existing_memory.value.get(prefix)
    else:
        existing_memory_content = "No existing memory found"

    system_message: str = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
    # Respond using memory + chat history
    new_memory = await llm.ainvoke([SystemMessage(content=system_message)] + state["messages"])  # type: ignore

    # Update existing memory
    await store.aput(namespace, key, {prefix: new_memory.content})  # type: ignore


# Graph
graph_builder = StateGraph(MessageState, config_schema=configuration.Configuration)

# Add nodes
graph_builder.add_node("call_llm", call_llm)
graph_builder.add_node("write_memory", write_memory)

# Add edges
graph_builder.add_edge(START, "call_llm")
graph_builder.add_edge("call_llm", "write_memory")
graph_builder.add_edge("write_memory", END)

# Compile
graph = graph_builder.compile()
