from pageindex import PageIndexClient
from vectorless_rag.config import settings

pi_client = PageIndexClient(api_key=settings.pageindex_api_key)

def fetch_tree(doc_id: str) -> list[dict]:
    """
    Fetches the document tree structure from PageIndex.
    """
    tree_result = pi_client.get_tree(doc_id, node_summary=True)
    return tree_result.get("result", [])

def count_nodes(tree: list[dict]) -> int:
    """
    Recursively counts the total number of nodes in the tree.
    
    """
    total = len(tree)
    for node in tree:
        if node.get("nodes"):
            total += count_nodes(node["nodes"])
    return total

def format_tree_text(nodes: list[dict], indent: int = 0) -> str:
    """
    Returns a formatted string representation of the tree structure.
    """
    result = []
    for node in nodes:
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        page = node.get("page_index", "?")
        result.append(f"{prefix}[{node['node_id']}] {node['title']}  (p.{page})")
        if node.get("nodes"):
            result.append(format_tree_text(node["nodes"], indent + 1))
    return "\n".join(result)
