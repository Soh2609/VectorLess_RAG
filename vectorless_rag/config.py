from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    pageindex_api_key: str
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash"
    
    # Groq Settings
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    
    # App Settings
    max_upload_size_mb: int = 5
    llm_provider: str = "auto" # "groq", "gemini", or "auto"
    
    processing_timeout_seconds: int = 300
    tree_summary_chars: int = 150

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
