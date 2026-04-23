from sentence_transformers import CrossEncoder

reranker = None

def get_reranker():
    global reranker
    if reranker is None:
        reranker = CrossEncoder("BAAI/bge-reranker-base")
    return reranker


def rerank(query, docs):
    """
    docs: list of langchain Document objects
    returns: docs sorted by reranker score (descending)
    """
    if not docs:
        return []

    model = get_reranker()   # ✅ load only when needed

    pairs = [[query, d.page_content] for d in docs]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in ranked]