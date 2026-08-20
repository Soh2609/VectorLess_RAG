from dataclasses import dataclass
from typing import Generator
from vectorless_rag.retrieval import tree_search, find_nodes_by_ids
from vectorless_rag.generation import generate_answer

@dataclass
class RAGResult:
    node_ids: list[str]
    sections: list[str]
    reasoning: str
    answer_generator: Generator[str, None, None]

def run_rag(query: str, tree: list[dict], expert_rules: str = None) -> RAGResult:
    """
    Full end-to-end PageIndex RAG pipeline:
    
    Step 1: LLM Tree Search  -> finds relevant node_ids
    Step 2: Node Retrieval   -> fetches section content
    Step 3: Answer Generation -> produces cited answer stream
    """
    # Step 1: Tree Search
    search_result = tree_search(query, tree, expert_rules)
    node_ids = search_result.get("node_list", [])
    reasoning = search_result.get("thinking", "")
    
    # Step 2: Retrieve nodes
    nodes = find_nodes_by_ids(tree, node_ids)
    sections = [n["title"] for n in nodes]
    
    # Step 3: Generate answer
    answer_gen = generate_answer(query, nodes)
    
    return RAGResult(
        node_ids=node_ids,
        sections=sections,
        reasoning=reasoning,
        answer_generator=answer_gen
    )
