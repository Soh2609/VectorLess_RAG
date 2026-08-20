import json
from typing import Generator
import google.generativeai as genai
from vectorless_rag.config import settings

# Configure Gemini with the API key from settings
genai.configure(api_key=settings.gemini_api_key)

def generate_json(prompt: str) -> dict:
    """
    Calls Gemini to generate a JSON response based on the prompt.
    Returns the parsed JSON dictionary.
    """
    model = genai.GenerativeModel(settings.gemini_model)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
        ),
    )
    
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        # Fallback handling if response text is wrapped in markdown code blocks
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())

def generate_text_stream(prompt: str) -> Generator[str, None, None]:
    """
    Calls Gemini to generate a text response and yields it token by token.
    """
    model = genai.GenerativeModel(settings.gemini_model)
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text
