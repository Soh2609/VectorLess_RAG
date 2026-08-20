import json
from vectorless_rag.config import settings
from vectorless_rag.llm.gemini import generate_json

def compress_tree(nodes: list[dict]) -> list[dict]:
    """
    Compresses the tree to save tokens by only sending titles and short summaries to the LLM.
    """
    out = []
    for n in nodes:
        entry = {
            "node_id": n["node_id"],
            "title": n["title"],
            "page": n.get("page_index", "?"),
            "summary": n.get("summary", n.get("text", "")[:settings.tree_summary_chars])
        }
        if n.get("nodes"):
            entry["children"] = compress_tree(n["nodes"])
        out.append(entry)
    return out

def tree_search(query: str, tree: list[dict], expert_rules: str = None) -> dict:
    """
    Core PageIndex retrieval:
    Sends the query + document tree to an LLM.
    LLM reasons over the structure and returns relevant node_ids.
    
    Returns: dict with 'thinking' (reasoning) and 'node_list' (node IDs)
    """
    compressed_tree = compress_tree(tree)
    
    prompt = f"""You are a document analyst and domain expert.
    Your task: identify which node IDs most likely contain the answer to the query.
    Think step-by-step about which sections are relevant based on the document's structure and node summaries.
    If a section is relevant, try to select the specific child node IDs that contain the details, rather than just the root or parent node.
    If you are unsure which specific child node has the answer, include the parent node AND its likely child nodes.

    Query: {query}

    Document Tree:
    {json.dumps(compressed_tree, indent=2)}
    """
    if expert_rules:
            prompt += f"""
    Expert Routing Rules (follow these carefully):
    {expert_rules}
    """

    prompt += """
    Reply ONLY in this exact JSON format:
    {
    "thinking": "<your step-by-step reasoning>",
    "node_list": ["node_id1", "node_id2"]
    }"""

    return generate_json(prompt)

def find_nodes_by_ids(tree: list[dict], target_ids: list[str]) -> list[dict]:
    """
    Recursively walks the tree and collects nodes matching target_ids.
    """
    found = []
    for node in tree:
        if node["node_id"] in target_ids:
            found.append(node)
        if node.get("nodes"):
            found.extend(find_nodes_by_ids(node["nodes"], target_ids))
    return found
