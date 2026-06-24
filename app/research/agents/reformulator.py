import os
from google import genai
from app.research.state import ResearchState

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

_SHORT_QUERY_THRESHOLD = 6  # words


def reformulator_node(state: ResearchState) -> dict:
    question = state["question"]
    language = state["language"]
    has_doc  = bool(state.get("doc_id") or state.get("session_has_documents"))

    # Skip if already a precise, short query
    if len(question.split()) <= _SHORT_QUERY_THRESHOLD:
        print(f"🔄 REFORMULATOR: skipped (short/precise) → '{question}'")
        return {"reformulated_query": question}

    # ── Choose prompt based on whether a document is in scope ────────────────
    if has_doc:
        prompt = (
            "You are a search query optimizer for a RAG system that searches "
            "inside business documents (invoices, contracts, devis, reports).\n\n"
            "Your query will be embedded and matched against document chunks. "
            "Financial documents contain tables with numbers, labels like "
            "'HT', 'TTC', 'TVA', 'montant', 'total', 'prix', 'taux', "
            "'amount', 'rate', 'balance', 'subtotal'.\n\n"
            "RULES:\n"
            "- Keep the SAME language as the original question. Do NOT translate.\n"
            "- Replace vague pronouns ('they', 'it', 'them') with explicit entities.\n"
            "- Include financial/document-specific terms that would appear in "
            "  the actual document (e.g. 'TVA montant total HT TTC' instead of "
            "  'check if tax is correct').\n"
            "- If the question is about validation/correctness, include both "
            "  the document term AND the regulation term.\n"
            "- Keep it under 20 words.\n"
            "- Output ONLY the rewritten query — no explanation, no quotes.\n\n"
            f"Language: {language}\n"
            f"Original: {question}\n\n"
            "Examples:\n"
            "  'is the TVA correct?' → 'TVA montant taux total HT TTC prix facture'\n"
            "  'what are the payment terms?' → 'conditions paiement délai montant échéance'\n"
            "  'who signed the contract?' → 'signature signataire partie contractante nom'\n\n"
            "Rewritten:"
        )
    else:
        prompt = (
            "You are a search query optimizer for a business intelligence system.\n"
            "Rewrite the user's question as a precise, entity-rich search query.\n\n"
            "RULES:\n"
            "- Keep the SAME language as the original question. Do NOT translate.\n"
            "- Replace vague pronouns ('they', 'it', 'them') with explicit entities.\n"
            "- Keep it under 20 words.\n"
            "- Output ONLY the rewritten query — no explanation, no quotes.\n\n"
            f"Language: {language}\n"
            f"Original: {question}\n\n"
            "Rewritten:"
        )

    try:
        result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        ).text.strip()

        reformulated = result if result else question
        print(f"🔄 REFORMULATOR: '{question[:50]}' → '{reformulated[:60]}'")
        return {"reformulated_query": reformulated}

    except Exception as e:
        print(f"🔄 REFORMULATOR: ⚠️ failed ({e}), using original")
        return {"reformulated_query": question}