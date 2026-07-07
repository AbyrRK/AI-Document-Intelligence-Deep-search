# app/services/vector_store.py

from datetime import datetime
from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_core.documents import Document
from app.core.config import settings
from functools import lru_cache
from app.services.llm_client import llm



class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        # Keep LangChain wrapper for QdrantVectorStore compatibility
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GEMINI_API_KEY
        )
        self.collection_name = settings.QDRANT_COLLECTION
        self.vector_size = len(self.embeddings.embed_documents(["test"])[0])
        self._embedding_cache: dict = {}

    def ensure_collection(self):
        try:
            info = self.client.get_collection(self.collection_name)
            current_size = info.config.params.vectors.size
            if current_size != self.vector_size:
                print(f"  Collection has {current_size} dims, need {self.vector_size}. Recreating...")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"  Collection '{self.collection_name}' already exists")
                return
        except Exception:
            pass

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )
        print(f"✅ Collection '{self.collection_name}' created ({self.vector_size} dims)")

    def collection_exists(self) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def collection_count(self) -> int:
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def add_documents(self, chunks: List[Dict], session_id: str = "") -> int:
        self.ensure_collection()
        if not chunks:
            return 0

        texts = [chunk["content"] for chunk in chunks]
        
        # Retry with backoff for Gemini 503
        embeddings = None
        for attempt in range(4):
            try:
                print(f"[VectorStore] Embedding {len(texts)} chunks (attempt {attempt+1})...")
                embeddings = llm.embed_batch_sync(texts)
                break
            except Exception as e:
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s
                print(f"[VectorStore] Embedding failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                import time
                time.sleep(wait)
        
        if embeddings is None:
            print("[VectorStore] ❌ All embedding attempts failed. Skipping indexing.")
            return 0

        documents = []
        now = datetime.utcnow().isoformat()

        for i, chunk in enumerate(chunks):
            metadata = chunk["metadata"].copy()
            if not metadata.get("session_id") and session_id:
                metadata["session_id"] = session_id
            if "created_at" not in metadata:
                metadata["created_at"] = now
            documents.append(Document(
                page_content=chunk["content"],
                metadata=metadata
            ))

        self._add_documents_with_embeddings(documents, embeddings)
        return len(documents)
    
    def _add_documents_with_embeddings(self, documents: List[Document], embeddings: List[List[float]]):
        from qdrant_client.http.models import PointStruct
        import uuid
        
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point_id = str(uuid.uuid4())  # UUID instead of float timestamp
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query: str, k: int = 5) -> List[Document]:
        cache_key = f"{query}:{k}:global"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        if not self.collection_exists():
            return []

        db = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        results = db.similarity_search(query, k=k)
        self._embedding_cache[cache_key] = results
        return results
    
    def search_by_session(self, query: str, session_id: str, k: int = 10) -> List[Document]:
        if not self.collection_exists():
            print("📄 RAG: Collection doesn't exist")
            return []

        total_in_collection = self.collection_count()
        print(f"📄 RAG: Searching session='{session_id}' | total vectors in Qdrant={total_in_collection}")

        if total_in_collection == 0:
            print("📄 RAG: ⚠️  Qdrant is empty")
            return []

        # Use LangChain wrapper (handles API compatibility)
        db = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        # Try nested metadata key first
        results = db.similarity_search(
            query, k=k,
            filter=Filter(must=[FieldCondition(
                key="metadata.session_id", match=MatchValue(value=session_id)
            )])
        )
        print(f"📄 RAG: metadata.session_id → {len(results)} results")
        if results:
            return results

        # Try flat key
        results = db.similarity_search(
            query, k=k,
            filter=Filter(must=[FieldCondition(
                key="session_id", match=MatchValue(value=session_id)
            )])
        )
        print(f"📄 RAG: session_id (flat) → {len(results)} results")
        if results:
            return results

        # Diagnosis
        print("📄 RAG: ⚠️  No results for this session. Inspecting Qdrant payloads...")
        try:
            sample, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=3,
                with_payload=True,
                with_vectors=False
            )
            for p in sample:
                print(f"  Sample payload: {p.payload}")
        except Exception as e:
            print(f"  Could not inspect payloads: {e}")

        return []
    
    def _results_to_documents(self, results) -> List[Document]:
        """Convert Qdrant search results to LangChain Document objects."""
        documents = []
        for r in results:
            payload = r.payload or {}
            meta = payload.get("metadata", {})
            documents.append(Document(
                page_content=payload.get("page_content", ""),
                metadata=meta
            ))
        return documents
    
    def search_by_language(self, query: str, language: str, k: int = 5) -> List[Document]:
        if not self.collection_exists():
            return []

        db = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        return db.similarity_search(
            query, k=k,
            filter=Filter(must=[FieldCondition(
                key="metadata.language", match=MatchValue(value=language)
            )])
        )

    def search_by_filename(self, query: str, file_name: str, k: int = 10) -> List[Document]:
        """Fallback: search by filename when session_id lookup returns nothing."""
        if not self.collection_exists():
            return []

        db = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        for key in ["metadata.file_name", "file_name"]:
            results = db.similarity_search(
                query, k=k,
                filter=Filter(must=[FieldCondition(
                    key=key, match=MatchValue(value=file_name)
                )])
            )
            if results:
                print(f"📄 RAG: filename fallback ({key}) → {len(results)} results")
                return results

        return []
    
    def search_by_doc(self, query: str, doc_id: str, k: int = 15) -> List[Document]:
        """Search only within chunks that belong to a specific document."""
        if not self.collection_exists():
            return []

        print(f"📄 RAG: Searching doc_id='{doc_id}' for '{query[:50]}...'")

        # Use LangChain wrapper (handles API compatibility)
        db = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        # Try nested key first
        results = db.similarity_search(
            query, k=k,
            filter=Filter(must=[FieldCondition(
                key="metadata.doc_id", match=MatchValue(value=doc_id)
            )])
        )
        print(f"📄 RAG: metadata.doc_id filter → {len(results)} results")
        if results:
            return results

        # Try flat key
        results = db.similarity_search(
            query, k=k,
            filter=Filter(must=[FieldCondition(
                key="doc_id", match=MatchValue(value=doc_id)
            )])
        )
        print(f"📄 RAG: doc_id (flat) filter → {len(results)} results")
        return results

    def delete_by_doc(self, doc_id: str) -> int:
        """Delete all chunks belonging to one document."""
        if not self.collection_exists():
            return 0

        before = self.collection_count()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=[
                FieldCondition(key="metadata.doc_id", match=MatchValue(value=doc_id))
            ])
        )
        after = self.collection_count()
        deleted = before - after
        print(f"🗑️  Deleted {deleted} chunks for doc_id='{doc_id}'")
        return deleted

    def delete_by_file(self, file_name: str, session_id: str = "") -> int:
        """Legacy method — prefer delete_by_doc when doc_id is available."""
        must_conditions = [
            FieldCondition(key="metadata.file_name", match=MatchValue(value=file_name))
        ]
        if session_id:
            must_conditions.append(
                FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id))
            )

        before = self.collection_count()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=must_conditions)
        )
        after = self.collection_count()
        deleted = before - after
        print(f"🗑️  Deleted {deleted} chunks for '{file_name}'")
        return deleted

    def list_by_session(self, session_id: str) -> List[str]:
        if not self.collection_exists():
            return []

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=[FieldCondition(
                key="metadata.session_id", match=MatchValue(value=session_id)
            )]),
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        filenames = list({
            p.payload.get("metadata", {}).get("file_name")
            for p in results
            if p.payload.get("metadata", {}).get("file_name")
        })
        return filenames

    def list_docs_by_session(self, session_id: str) -> List[Dict]:
        """Return [{doc_id, file_name}] for all documents in a session."""
        if not self.collection_exists():
            return []

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=[FieldCondition(
                key="metadata.session_id", match=MatchValue(value=session_id)
            )]),
            limit=1000,
            with_payload=True,
            with_vectors=False
        )

        seen = {}
        for p in results:
            meta = p.payload.get("metadata", {})
            doc_id = meta.get("doc_id", "")
            file_name = meta.get("file_name", "")
            if doc_id and doc_id not in seen:
                seen[doc_id] = file_name

        return [{"doc_id": k, "file_name": v} for k, v in seen.items()]

    def cleanup_expired(self, max_age_hours: int = 24) -> int:
        from datetime import datetime, timedelta

        if not self.collection_exists():
            return 0

        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()

        try:
            all_points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )

            expired_ids = []
            for point in all_points:
                meta = point.payload.get("metadata", {})
                created = meta.get("created_at", "")
                if created and created < cutoff:
                    expired_ids.append(point.id)

            if expired_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=expired_ids
                )
                print(f"🧹 TTL cleanup: removed {len(expired_ids)} expired chunks")
                return len(expired_ids)
            return 0
        except Exception as e:
            print(f"🧹 TTL cleanup failed: {e}")
            return 0


vector_store = VectorStore()