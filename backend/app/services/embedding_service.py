from langchain_community.vectorstores import Chroma
from app.services.google_embedding import GoogleTextEmbedding
import os

def create_vectorstore(chunks, doc_id):
    persist_dir = f"static/chroma_stores/{doc_id}"
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = GoogleTextEmbedding()

    for chunk in chunks:
        page_number = chunk.metadata.get("page_number", "Unknown")
        chunk.page_content = f"passage from page {page_number}: " + chunk.page_content.strip()

    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    vectorstore.persist()

    return persist_dir

