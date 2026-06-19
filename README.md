# DocTalk — Multi-PDF Intelligent Assistant

Ask questions across multiple PDF documents using RAG 
(Retrieval-Augmented Generation).

## What I built and why
I built DocTalk as a personal project to go beyond 
experimentation and build something complete end-to-end 
with the AI tools I had explored during my AI Topic Owner 
role (LangChain, FAISS, embeddings, LLMs).

## Features
- Multi-PDF upload and indexing with FAISS
- Chat across all documents or per-document with isolated vectorstores
- Conversational memory — last 6 exchanges kept for context-aware answers
- Document dashboard: type, tone, complexity, suggested questions
- Auto summary, key insights extraction, and topic analysis
- Token usage and cost tracking per session (Groq pricing)
- Download chat history, summaries, and insights

## RAG Architecture
Naive RAG pipeline with modular extensions:
- PDF loading → chunking (500 chars, 50 overlap) → HuggingFace embeddings → FAISS index
- Similarity search (k=6) → context formatting with source tags → Groq LLM answer
- Per-document isolation: dedicated cached vectorstore per file
- Multi-document fusion: unified FAISS index across all uploaded PDFs
- Conversational memory: last 6 messages injected per query

## What I learned
- How chunking strategy affects retrieval quality
- The difference between multi-doc and per-doc retrieval
- How to track and expose token costs transparently
- That a well-written system prompt matters more than 
  complex routing logic

## Stack
Python · LangChain · FAISS · HuggingFace Embeddings (all-MiniLM-L6-v2)
Groq API (LLaMA 3.1 8B Instant) · Streamlit

## Run locally
pip install -r requirements.txt

## Add GROQ_API_KEY=your_key to .env
streamlit run app.py