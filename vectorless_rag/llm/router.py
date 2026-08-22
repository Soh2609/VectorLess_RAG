import logging
from typing import Generator
from vectorless_rag.config import settings
from vectorless_rag.llm import gemini
from vectorless_rag.llm import groq_client

logger = logging.getLogger(__name__)

def _can_use_groq() -> bool:
    return bool(settings.groq_api_key) and settings.llm_provider in ["groq", "auto"]

def _can_use_gemini() -> bool:
    return bool(settings.gemini_api_key) and settings.llm_provider in ["gemini", "auto"]

def generate_json(prompt: str) -> dict:
    if _can_use_groq():
        try:
            return groq_client.generate_json(prompt)
        except Exception as e:
            if settings.llm_provider == "auto" and _can_use_gemini():
                logger.warning(f"Groq failed, falling back to Gemini: {e}")
                return gemini.generate_json(prompt)
            raise e
            
    if _can_use_gemini():
        return gemini.generate_json(prompt)
        
    raise ValueError("No LLM provider available or configured.")

def generate_text_stream(prompt: str) -> Generator[str, None, None]:
    if _can_use_groq():
        try:
            # We need to test if the connection works before yielding
            # But with streaming, the first chunk might throw.
            # We use a wrapper generator to catch the error on the first chunk.
            stream = groq_client.generate_text_stream(prompt)
            first_chunk = next(stream)
            
            def wrapped_stream():
                yield first_chunk
                yield from stream
                
            return wrapped_stream()
        except Exception as e:
            if settings.llm_provider == "auto" and _can_use_gemini():
                logger.warning(f"Groq failed, falling back to Gemini: {e}")
                return gemini.generate_text_stream(prompt)
            raise e

    if _can_use_gemini():
        return gemini.generate_text_stream(prompt)
        
    raise ValueError("No LLM provider available or configured.")

def get_active_provider_info() -> dict:
    if _can_use_groq():
        return {"name": "Groq", "model": settings.groq_model, "status": "primary"}
    if _can_use_gemini():
        return {"name": "Gemini", "model": settings.gemini_model, "status": "fallback"}
    return {"name": "None", "model": "None", "status": "error"}
