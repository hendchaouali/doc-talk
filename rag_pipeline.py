

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from groq import Groq
from dotenv import load_dotenv
import tempfile, os, json
from datetime import datetime

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file!")

groq_client = Groq(api_key=GROQ_API_KEY)

# Pricing (per million tokens) — llama-3.1-8b-instant
INPUT_COST_PER_M = 0.05
OUTPUT_COST_PER_M = 0.08


# -----------------------------
# TOKEN + COST TRACKER
# -----------------------------
def estimate_cost(input_tokens: int, output_tokens: int) -> dict:
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_M
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(input_cost + output_cost, 6),
    }


# -----------------------------
# PDF LOADING + INDEXING
# -----------------------------


def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )


def load_and_index(uploaded_files) -> tuple:
    all_chunks = []
    file_metadata = {}

    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        os.unlink(tmp_path)

        for doc in documents:
            doc.metadata["source_file"] = uploaded_file.name

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        file_metadata[uploaded_file.name] = {
            "pages": len(documents),
            "chunks": len(chunks),
            "name": uploaded_file.name
        }
        all_chunks.extend(chunks)

    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    return vectorstore, len(all_chunks), file_metadata


# -----------------------------
# FORMAT DOCS WITH SOURCE TAGS
# -----------------------------
def format_docs_with_sources(docs):
    parts = []
    for doc in docs:
        src = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[Source: {src}, Page {int(page)+1 if page != '?' else '?'}]\n{doc.page_content}"
        )
    return "\n\n".join(parts)


# -----------------------------
# GROQ CALL — returns content + usage
# -----------------------------
def _groq_call(system_prompt: str, user_prompt: str, temperature=0.3, max_tokens=1024):
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage = resp.usage
    cost = estimate_cost(usage.prompt_tokens, usage.completion_tokens)
    return resp.choices[0].message.content.strip(), cost


# -----------------------------
# Q&A WITH CONVERSATIONAL MEMORY
# -----------------------------
def answer_question(question: str, context: str, history: list) -> tuple:
    messages = [
        {
            "role": "system",
            "content": """You are an intelligent assistant and expert tutor helping users understand their PDF documents deeply.

You are given relevant excerpts from a document and a question. Your job is to:

1. EXPLAIN the concept thoroughly in simple, clear language — don't just copy text from the document
2. Use the document as your PRIMARY source but feel free to add helpful context, examples, or analogies from your own knowledge to make the explanation richer
3. If the document mentions a term or concept, explain what it means even if the full definition isn't in the document
4. Structure your answers clearly — use bullet points, numbered steps, or sections when helpful
5. Always mention which file/page the core information came from
6. If the document doesn't cover something at all, say so — but still try to help with general knowledge
7. End with a "💡 Key Takeaway:" summarising the main point in one sentence""",
        }
    ]
    # Inject last 6 messages as memory
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})

    messages.append(
        {
            "role": "user",
            "content": f"Context from documents:\n{context}\n\nQuestion: {question}",
        }
    )

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    usage = resp.usage
    cost = estimate_cost(usage.prompt_tokens, usage.completion_tokens)
    return resp.choices[0].message.content.strip(), cost


# -----------------------------
# DOCUMENT SUMMARY
# -----------------------------
def summarise_documents(vectorstore: FAISS) -> tuple:
    docs = vectorstore.similarity_search(
        "main topic summary overview introduction", k=8
    )
    context = format_docs_with_sources(docs)
    return _groq_call(
        "You are a document analyst. Produce a structured summary with: Overview, Key Themes, Important Facts, and Conclusions.",
        f"Summarise these document excerpts:\n\n{context}",
    )


# -----------------------------
# KEY INSIGHTS
# -----------------------------
def generate_key_insights(vectorstore: FAISS) -> tuple:
    docs = vectorstore.similarity_search(
        "key insight finding important conclusion result", k=8
    )
    context = format_docs_with_sources(docs)
    return _groq_call(
        "You are an expert analyst. Extract the most important insights from the document.",
        f"""Extract 5-7 key insights from these excerpts. Format each as:
💡 Insight [N]: [Title]
[2-3 sentence explanation]
Source: [mention the file/page]

Excerpts:
{context}""",
    )


# -----------------------------
# MOST DISCUSSED TOPICS
# -----------------------------
def get_top_topics(vectorstore: FAISS) -> tuple:
    docs = vectorstore.similarity_search("topic theme subject concept", k=10)
    context = format_docs_with_sources(docs)
    return _groq_call(
        "You are a text analyst. Identify the most discussed topics in the document.",
        f"""Identify the top 7 most discussed topics. Format:
🔵 Topic [N]: [Topic Name]
Frequency: [High/Medium/Low]
Summary: [1-2 sentences]

Content:
{context}""",
    )


# -----------------------------
# DOCUMENT INSIGHTS DASHBOARD DATA
# -----------------------------
def get_dashboard_insights(vectorstore: FAISS) -> tuple:
    docs = vectorstore.similarity_search("overview summary main points", k=10)
    context = format_docs_with_sources(docs)
    return _groq_call(
        "You are a document analyst. Return ONLY valid JSON, no explanation, no markdown.",
        f"""Analyse the document and return ONLY this JSON structure:
{{
  "document_type": "...",
  "main_subject": "...",
  "tone": "...",
  "complexity": "Beginner/Intermediate/Advanced",
  "top_topics": ["topic1","topic2","topic3","topic4","topic5"],
  "key_facts": ["fact1","fact2","fact3"],
  "recommended_questions": ["q1","q2","q3"]
}}

Content:
{context}""",
    )


# -----------------------------
# QA CHAIN — MULTI DOC
# -----------------------------
def build_qa_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}  # increased from 4 — more context = better answers
    )

    def run_chain(inputs: dict):
        docs = retriever.invoke(inputs["question"])
        context = format_docs_with_sources(docs)
        answer, cost = answer_question(
            inputs["question"], context, inputs.get("history", [])
        )
        return answer, cost, docs

    return run_chain, retriever


# -----------------------------
# PER-DOC VECTORSTORE CACHE
# -----------------------------
_per_doc_cache: dict = {}  # filename -> vectorstore

def get_or_build_per_doc_vectorstore(uploaded_file) -> FAISS:
    """Build and cache a per-doc vectorstore — only indexes once per file."""
    name = uploaded_file.name
    if name in _per_doc_cache:
        return _per_doc_cache[name]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        # Reset file pointer in case it was already read
        uploaded_file.seek(0)
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.unlink(tmp_path)

    for doc in documents:
        doc.metadata["source_file"] = name

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = load_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    _per_doc_cache[name] = vs
    return vs


def query_single_doc(uploaded_file, question: str, history: list) -> tuple:
    """Query a single document's vectorstore directly."""
    vs = get_or_build_per_doc_vectorstore(uploaded_file)
    retriever = vs.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(question)
    context = format_docs_with_sources(docs)
    answer, cost = answer_question(question, context, history)
    return answer, cost, docs


def clear_per_doc_cache():
    """Call this when new files are uploaded."""
    global _per_doc_cache
    _per_doc_cache = {}