# app/services/vector_store.py

from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document
from app.core.config import settings


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.GEMINI_API_KEY
        )
        self.collection_name = settings.QDRANT_COLLECTION

    def ensure_collection(self):
        """Creates the collection if it doesn't exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,           # Gemini embedding-001 output dimension
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Collection '{self.collection_name}' créée")
        else:
            print(f"ℹ️  Collection '{self.collection_name}' already exists")

    def add_documents(self, chunks: List[Dict]) -> int:
        """
        Embeds and indexes chunks into Qdrant.
        Returns the number of chunks stored.
        """
        self.ensure_collection()

        if not chunks:
            return 0

        documents = [
            Document(
                page_content=chunk["content"],
                metadata=chunk["metadata"]
            )
            for chunk in chunks
        ]

        QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            client=self.client,
            collection_name=self.collection_name,
        )

        return len(documents)

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Semantic search — returns top-k most relevant chunks.
        Used by the RAG module.
        """
        db = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )
        return db.similarity_search(query, k=k)

    def search_by_language(self, query: str, language: str, k: int = 5) -> List[Document]:
        """
        Semantic search filtered by language.
        Useful when the chatbot knows which language the user is writing in.
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        db = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )
        return db.similarity_search(
            query,
            k=k,
            filter=Filter(
                must=[FieldCondition(
                    key="metadata.language",
                    match=MatchValue(value=language)
                )]
            )
        )

    def collection_count(self) -> int:
        """Returns the number of vectors currently stored."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0   # collection doesn't exist yet

    def delete_by_file(self, file_name: str) -> int:
        """
        Deletes all chunks belonging to a specific file.
        Useful when a document is re-uploaded and needs re-indexing.
        Returns the number of points deleted.
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        before = self.collection_count()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(
                    key="metadata.file_name",
                    match=MatchValue(value=file_name)
                )]
            )
        )
        after = self.collection_count()
        deleted = before - after
        print(f"🗑️  Deleted {deleted} chunks for '{file_name}'")
        return deleted


# Global instance — import this everywhere
vector_store = VectorStore()