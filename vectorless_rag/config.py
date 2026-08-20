from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    pageindex_api_key: str
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash"
    processing_timeout_seconds: int = 300
    tree_summary_chars: int = 150

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
