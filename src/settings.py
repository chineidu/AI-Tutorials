from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseSettingsConfig(BaseSettings):
    """Base configuration class for settings.

    This class extends BaseSettings to provide common configuration options
    for environment variable loading and processing.

    Attributes
    ----------
    model_config : SettingsConfigDict
        Configuration dictionary for the settings model specifying env file location,
        encoding and other processing options.
    """

    model_config = SettingsConfigDict(
        env_file=str(Path(".env").absolute()),
        env_file_encoding="utf-8",
        from_attributes=True,
        populate_by_name=True,
    )


class Settings(BaseSettingsConfig):
    """Application settings class containing credentials."""

    # OLLAMA
    OLLAMA_API_KEY: SecretStr
    OLLAMA_URL: str

    # GROQ
    GROQ_API_KEY: SecretStr
    # GROQ_BASE_URL: str

    # LANGFUSE
    LANGFUSE_SECRET_KEY: SecretStr
    LANGFUSE_PUBLIC_KEY: SecretStr
    LANGFUSE_HOST: str

    # TAVILY
    TAVILY_API_KEY: SecretStr

    # OPENROUTER
    OPENROUTER_API_KEY: SecretStr
    OPENROUTER_URL: str

    # MISTRAL AI
    MISTRAL_API_KEY: SecretStr

    # GEMINI
    GEMINI_API_KEY: SecretStr

    # LANGSMITH
    LANGSMITH_API_KEY: SecretStr
    LANGSMITH_TRACING: bool = True
    LANGSMITH_PROJECT: str


def refresh_settings() -> Settings:
    """Refresh environment variables and return new Settings instance.

    This function reloads environment variables from .env file and creates
    a new Settings instance with the updated values.

    Returns
    -------
    Settings
        A new Settings instance with refreshed environment variables
    """
    load_dotenv(override=True)
    return Settings()  # type: ignore
