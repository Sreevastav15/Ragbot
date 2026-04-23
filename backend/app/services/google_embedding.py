import google.generativeai as genai
from langchain_core.embeddings import Embeddings
import os

class GoogleTextEmbedding(Embeddings):
    def __init__(self, model_name="gemini-embedding-001"):
        self.model_name = model_name
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    def _extract_embedding(self, result):
        """Extract embedding safely regardless of Google response format."""
        emb = result["embedding"]

        # Case 1: embedding is raw list
        if isinstance(emb, list):
            return emb

        # Case 2: embedding is dict with "values"
        if isinstance(emb, dict) and "values" in emb:
            return emb["values"]

        raise ValueError("Unrecognized embedding format from Google API.")

    def embed_documents(self, texts):
        embeddings = []
        for t in texts:
            result = genai.embed_content(
                model=self.model_name,
                content=t,
                task_type="retrieval_document"
            )
            embeddings.append(self._extract_embedding(result))
        return embeddings

    def embed_query(self, query):
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query"
        )
        return self._extract_embedding(result)
