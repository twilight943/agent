from src.recsys.semantic_retrieval import SemanticRetriever


def semantic_search_tool(
    retriever: SemanticRetriever, query: str, top_k: int = 200
) -> list[dict[str, float | int]]:
    return retriever.semantic_retrieve(query, top_k)
