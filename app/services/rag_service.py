"""
app/services/rag_service.py
"""

import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.services.vector_store import vector_store
from app.services.chat_memory import get_session_history, save_session
from app.core.config import settings


class RAGService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
        )

        # Variant B — winner of prompt evaluation (87%)
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are a precise document assistant. Answer questions using ONLY the retrieved context below.\n"
            "Do not include any citation markers like [Excerpt N] in your answer.\n"
            "If the information is not found in the context, respond exactly with: \"Not found in document.\"\n"
            "Never hallucinate or add information not present in the context.\n\n"
            "Context:\n{context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        self.brief_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a document assistant. Based on the following excerpts, provide a brief summary "
             "(3-5 bullet points) of the document's main points and key findings. "
             "Be concise and structured. Respond in the same language as the document."),
            ("human", "{context}"),
        ])

        self.parser = StrOutputParser()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _format_docs(self, docs: list[Document]) -> str:
        return "\n\n".join(
            f"[Excerpt {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

    def _extract_sources(self, docs: list[Document]) -> list[dict]:
        sources = []
        seen = set()

        for doc in docs:
            meta = doc.metadata or {}
            file_name = meta.get("file_name", "")

            # Replace temp filenames with real filename from session metadata
            if file_name.startswith("tmp") and len(file_name) < 20:
                from app.services.chat_memory import get_session_metadata
                session_meta = get_session_metadata(meta.get("session_id", ""))
                files = session_meta.get("files", [])
                if files:
                    file_name = files[0]

            # Try every possible metadata key for page number
            page = (
                meta.get("page_number")
                or meta.get("page")
                or meta.get("page_num")
                or meta.get("source_page")
            )

            # Fallback: parse "[Page N]" from the chunk text itself
            if not page or page == "?":
                match = re.search(r"\[Page\s+(\d+)\]", doc.page_content)
                if match:
                    page = int(match.group(1))

            # Last resort: use chunk index as a human-readable label
            if not page or page == "?":
                idx = meta.get("chunk_index")
                page = f"chunk {idx + 1}" if idx is not None else "?"

            # Clean preview — strip "[Page N]" prefix and truncate
            preview = re.sub(r"^\[Page\s+\d+\]\s*", "", doc.page_content).strip()
            preview = preview[:300] + ("..." if len(preview) > 300 else "")

            # Deduplicate
            key = f"{file_name}:{page}:{preview[:50]}"
            if key in seen:
                continue
            seen.add(key)

            sources.append({
                "page":     page,
                "chunk":    preview,
                "file":     file_name,
                "language": meta.get("language", ""),
            })

        return sources

    def _get_context(self, question: str, session_id: str, k: int = 5) -> list[Document]:
        """
        Search Qdrant by session_id first.
        Falls back to filename search if session lookup returns nothing
        (handles server restarts where vectors exist but session_id cache is gone).
        """
        from app.services.chat_memory import get_session_files

        docs = vector_store.search_by_session(question, session_id=session_id, k=k)

        if not docs:
            files = get_session_files(session_id)
            for file_name in files:
                docs = vector_store.search_by_filename(question, file_name, k=k)
                if docs:
                    print(f"⚠️  Session lookup missed — recovered via filename: {file_name}")
                    break

        return docs
    import re

    def _clean_answer(self, text: str) -> str:
        """Remove [Excerpt N] citation markers from the answer text."""
        text = re.sub(r'\[Excerpt\s+\d+(?:,\s*Excerpt\s+\d+)*\]', '', text)
        text = re.sub(r'\[Excerpt\s+[\d,\s]+\]', '', text)
        return text.strip()


    def generate_brief(self, session_id: str, k: int = 8, fallback_text: str = "") -> str:
        """
        Generate a document brief from Qdrant chunks.
        Falls back to raw extracted text if Qdrant returns nothing yet.
        Uses unified LLM client with automatic fallback on rate limit.
        """
        from app.services.llm_client import llm
        
        docs    = vector_store.search_by_session(
            "main points key findings overview summary", session_id=session_id, k=k
        )
        context = self._format_docs(docs)
        print(f"📊 generate_brief: {len(docs)} chunks from Qdrant, fallback_text={len(fallback_text)} chars")

        if not context.strip():
            if fallback_text and fallback_text.strip():
                context = fallback_text[:4000]
                print("⚠️  Using fallback_text for brief (indexing still in progress)")
            else:
                print("⚠️  No context and no fallback — returning placeholder")
                answer = "Document indexed. Ask me a question to get started."
                history = get_session_history(session_id)
                history.add_ai_message(answer)
                save_session(session_id)
                return answer

        # Build prompt manually and use unified LLM client with fallback
        prompt = (
            "You are a document assistant. Based on the following excerpts, provide a brief summary "
            "(3-5 bullet points) of the document's main points and key findings. "
            "Be concise and structured. Respond in the same language as the document.\n\n"
            f"Context:\n{context}"
        )
        
        answer = llm.generate_sync(prompt=prompt, temperature=0.3, max_tokens=1024, use_cache=False)
        
        # If fallback returned error message, use placeholder
        if answer.startswith("[System") or answer.startswith("[Error"):
            print(f"⚠️  Brief generation failed: {answer}")
            answer = "Document uploaded successfully. You can now ask questions about it."

        history = get_session_history(session_id)
        history.add_ai_message(answer)
        save_session(session_id)
        return answer

    def ask(self, question: str, session_id: str, k: int = 5) -> dict:
        """Non-streaming RAG answer with conversation history."""
        docs    = self._get_context(question, session_id, k)
        context = self._format_docs(docs)
        sources = self._extract_sources(docs)

        if not context.strip():
            return {"answer": "Not found in document.", "sources": []}

        # Snapshot history BEFORE adding user message to avoid duplicate in prompt
        history = get_session_history(session_id)
        existing_messages = list(history.messages)
        history.add_user_message(question)
        save_session(session_id)

        chain  = self.rag_prompt | self.llm | self.parser
        answer = chain.invoke({
            "context":  context,
            "question": question,
            "history":  existing_messages,
        })

        answer = self._clean_answer(answer)
        history.add_ai_message(answer)
        save_session(session_id)
        return {"answer": answer, "sources": sources}

    def ask_stream(self, question: str, session_id: str, k: int = 5):
        """Streaming RAG answer with conversation history."""
        docs    = self._get_context(question, session_id, k)
        context = self._format_docs(docs)
        sources = self._extract_sources(docs)

        if not context.strip():
            yield {"sources": []}
            yield {"token": "Not found in document."}
            return

        # Snapshot history BEFORE adding user message to avoid duplicate in prompt
        history = get_session_history(session_id)
        existing_messages = list(history.messages)
        history.add_user_message(question)
        save_session(session_id)

        yield {"sources": sources}

        chain = self.rag_prompt | self.llm | self.parser
        full_response = []

        for token in chain.stream({
            "context":  context,
            "question": question,
            "history":  existing_messages,
        }):
            full_response.append(token)
            yield {"token": token}

        clean = self._clean_answer("".join(full_response))
        history.add_ai_message(clean)
        save_session(session_id)


rag_service = RAGService()