from typing import Generator
from vectorless_rag.llm.gemini import generate_text_stream

def build_context(nodes: list[dict]) -> str:
    """
    Builds a context string from retrieved nodes, including section titles and page numbers.
    """
    context_parts = []
    for node in nodes:
        # Fall back to 'summary' if 'text' is missing
        content = node.get("text", node.get("summary", "Content not available."))
        context_parts.append(
            f"[Section: '{node['title']}' | Page {node.get('page_index', '?')}]\n"
            f"{content}"
        )
    return "\n\n---\n\n".join(context_parts)

def generate_answer(query: str, nodes: list[dict]) -> Generator[str, None, None]:
    """
    Takes retrieved nodes as context and yields a grounded answer token by token.
    Instructs the LLM to cite section titles and page numbers.
    """
    if not nodes:
        yield "⚠️ No relevant sections found in the document."
        return
        
    context = build_context(nodes)
    
    prompt = f"""You are an expert document analyst.
Answer the question using ONLY the provided context.
For every claim you make, cite the section title and page number in parentheses.
Be concise and precise.

Question: {query}

Context:
{context}

Answer:"""

    yield from generate_text_stream(prompt)
