import json
from typing import Generator
from groq import Groq
from vectorless_rag.config import settings

def _get_client():
    return Groq(api_key=settings.groq_api_key)

def generate_json(prompt: str) -> dict:
    """
    Calls Groq to generate a JSON response based on the prompt.
    Returns the parsed JSON dictionary.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    text = response.choices[0].message.content
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Fallback handling
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())

def generate_text_stream(prompt: str) -> Generator[str, None, None]:
    """
    Calls Groq to generate a text response and yields it token by token.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
