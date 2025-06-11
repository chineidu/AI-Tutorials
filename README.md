# AI Tutorials

## Table of Content

- [AI Tutorials](#ai-tutorials)
  - [Table of Content](#table-of-content)
  - [Setup](#setup)
    - [Install Dependencies](#install-dependencies)
  - [LangGraph Studio](#langgraph-studio)

## Setup

### Install Dependencies

```sh
# Via uv
uv sync

# Via pip: Required for LangGraph Studio
python -m pip install -e .
```

## LangGraph Studio

- Run the command

```sh
langgraph dev --tunnel --config src/studio/langgraph.json
```

```txt
langgraph dev --tunnel \  # Exposes the dev server to the internet via a public URL
  --config src/studio/langgraph.json  # Specifies the config file for LangChain Studio
```

- Add `LANGSMITH_API_KEY and LANGSMITH_TRACING`

```sh
LANGSMITH_API_KEY="your-api-key"
LANGSMITH_TRACING=true
```
