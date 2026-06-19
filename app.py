

import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import (
    load_and_index, build_qa_chain,
    summarise_documents,
    generate_key_insights, get_top_topics,
    get_dashboard_insights
)
from datetime import datetime
import json

load_dotenv()

MIME_TEXT_PLAIN = "text/plain"

MAX_QUESTIONS_PER_SESSION = 10

if st.session_state.qa_count >= MAX_QUESTIONS_PER_SESSION:
    st.error("Session limit reached (10 questions). Please restart the app.")
    st.stop()
    
def init_session_state() :
    for key, default in {
        "messages": [],
        "doc_messages": {},
        "total_tokens": 0,
        "total_cost": 0.0,
        "qa_count": 0,
        "file_names": [],
        "chunk_count": 0,
        "file_metadata": {},
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

st.set_page_config(page_title="DocTalk — Ask your PDFs", page_icon="📄", layout="wide")

# -----------------------------
# SESSION STATE INIT
# -----------------------------

init_session_state()
# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("📄 DocTalk")
    st.caption("Multi-PDF intelligent assistant · LLaMA 3.1 · Groq · FAISS")    
    st.markdown("---")

    # Upload
    st.header("📁 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        file_names  = sorted([f.name for f in uploaded_files])
        current_key = "_".join(file_names)
       
        if "file_key" not in st.session_state or st.session_state.file_key != current_key:
            with st.spinner(f"Indexing {len(uploaded_files)} PDF(s)..."):
                from rag_pipeline import clear_per_doc_cache
                clear_per_doc_cache() 
                vectorstore, chunk_count, file_metadata = load_and_index(uploaded_files)
                qa_fn, retriever = build_qa_chain(vectorstore)
                st.session_state.vectorstore   = vectorstore
                st.session_state.qa_fn         = qa_fn
                st.session_state.retriever     = retriever
                st.session_state.file_key      = current_key
                st.session_state.messages      = []
                st.session_state.doc_messages  = {name: [] for name in file_names}
                st.session_state.file_names    = file_names
                st.session_state.chunk_count   = chunk_count
                st.session_state.file_metadata = file_metadata
            st.success(f"✅ {len(uploaded_files)} file(s) ready!")

    # Stats
    if st.session_state.file_names:
        if len(uploaded_files) > 0:
            st.markdown("---")
            st.markdown("### 📊 Session Stats")
            col1, col2 = st.columns(2)
            col1.metric("PDFs", len(st.session_state.file_names))
            col2.metric("Chunks", st.session_state.chunk_count)
            col1.metric("Questions", st.session_state.qa_count)
            col2.metric("Tokens", f"{st.session_state.total_tokens:,}")
            st.metric("Est. Cost", f"${st.session_state.total_cost:.5f}")

            st.markdown("### 📄 Loaded Files")
            for name, meta in st.session_state.file_metadata.items():
                with st.expander(f"📄 {name}"):
                    st.caption(f"Pages: {meta['pages']} | Chunks: {meta['chunks']}")
        else:
            # init_session_state()
            from rag_pipeline import clear_per_doc_cache
            clear_per_doc_cache()  
            if "topics" in st.session_state  : del st.session_state.topics
            if "summary"in st.session_state : del st.session_state.summary
            if "insights" in st.session_state: del st.session_state.insights
            if "file_key" in st.session_state: del st.session_state.file_key
            st.session_state.file_names = []
            st.session_state.chunk_count = 0
            st.session_state.qa_count = 0
            st.session_state.total_tokens = 0
            st.session_state.total_cost = 0.0
            st.session_state.file_metadata = {}
            st.rerun()
    # Clear
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.doc_messages = {n: [] for n in st.session_state.file_names}
        st.rerun()
        
# -----------------------------
# MAIN TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
     "💬 Multi-PDF Chat",
     "📄 Chat per Document",
     "📊 Dashboard",
     "📝 Summary & Insights",
     "📈 Topics",
 ])

# ─────────────────────────────
# TAB 1 — MULTI-PDF CHAT
# ─────────────────────────────
with tab1:
    st.subheader("💬 Chat across all PDFs")

    if not st.session_state.file_names:
        st.info("⬅️ Upload PDFs from the sidebar to start chatting.")
    else:
        # Render history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander("📚 Sources used"):
                        for i, src in enumerate(msg["sources"], 1):
                            file = src.metadata.get("source_file", "?")
                            page = src.metadata.get("page", "?")
                            pg   = int(page) + 1 if page != "?" else "?"
                            # Highlight the chunk
                            st.markdown(
                                f"<div style='background:#f0f4ff;border-left:4px solid #4a6cf7;"
                                f"padding:8px 12px;border-radius:4px;margin:4px 0'>"
                                f"<b>Chunk {i}</b> — 📄 <code>{file}</code> | Page {pg}<br>"
                                f"<small>{src.page_content[:300]}...</small></div>",
                                unsafe_allow_html=True,
                            )
                if msg["role"] == "assistant" and "cost" in msg:
                    c = msg["cost"]
                    st.caption(
                        f"🔢 Tokens: {c['total_tokens']} "
                        f"(in:{c['input_tokens']} out:{c['output_tokens']}) | "
                        f"💰 Cost: ${c['total_cost']:.6f}"
                    )

        # Download chat
        if st.session_state.messages:
            chat_text = "\n\n".join(
                f"{'You' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in st.session_state.messages
            )
            st.download_button(
                "⬇️ Download Chat",
                data=chat_text,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime=MIME_TEXT_PLAIN
            )

        if "pending_question" in st.session_state and st.session_state.pending_question:
            question = st.session_state.pending_question
            st.session_state.pending_question = None
        # Input
        if question := st.chat_input("Ask anything across all your PDFs..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    
                    answer, cost, source_docs = st.session_state.qa_fn({
                        "question": question,
                        "history":  st.session_state.messages[:-1],
                    })
                st.markdown(answer)

                with st.expander("📚 Sources used"):
                    for i, doc in enumerate(source_docs, 1):
                        file = doc.metadata.get("source_file", "?")
                        page = doc.metadata.get("page", "?")
                        pg   = int(page) + 1 if page != "?" else "?"
                        st.markdown(
                            f"<div style='background:#f0f4ff;border-left:4px solid #4a6cf7;"
                            f"padding:8px 12px;border-radius:4px;margin:4px 0'>"
                            f"<b>Chunk {i}</b> — 📄 <code>{file}</code> | Page {pg}<br>"
                            f"<small>{doc.page_content[:300]}...</small></div>",
                            unsafe_allow_html=True,
                        )
                st.caption(
                    f"🔢 Tokens: {cost['total_tokens']} | 💰 Cost: ${cost['total_cost']:.6f}"
                )

            # Update state
            st.session_state.messages.append({
                "role": "assistant", "content": answer,
                "sources": source_docs, "cost": cost,
            })
            st.session_state.total_tokens += cost["total_tokens"]
            st.session_state.total_cost   += cost["total_cost"]
            st.session_state.qa_count     += 1

# ─────────────────────────────
# TAB 2 — CHAT PER DOCUMENT
# ─────────────────────────────
with tab2:
    st.subheader("📄 Chat with a specific document")

    if not st.session_state.file_names:
        st.info("⬅️ Upload PDFs from the sidebar first.")
    else:
        selected_doc = st.selectbox("Choose a document:", st.session_state.file_names)

        if selected_doc not in st.session_state.doc_messages:
            st.session_state.doc_messages[selected_doc] = []

        doc_history = st.session_state.doc_messages[selected_doc]

        # Render history
        for msg in doc_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(msg["sources"], 1):
                            page = src.metadata.get("page", "?")
                            pg = int(page) + 1 if page != "?" else "?"
                            st.markdown(
                                f"<div style='background:#f0f4ff;border-left:4px solid #4a6cf7;"
                                f"padding:8px 12px;border-radius:4px;margin:4px 0'>"
                                f"<b>Chunk {i}</b> — Page {pg}<br>"
                                f"<small>{src.page_content[:300]}...</small></div>",
                                unsafe_allow_html=True,
                            )
                if msg["role"] == "assistant" and "cost" in msg:
                    c = msg["cost"]
                    st.caption(f"🔢 Tokens: {c['total_tokens']} | 💰 Cost: ${c['total_cost']:.6f}")

        if doc_history:
            chat_text = "\n\n".join(
                f"{'You' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in doc_history
            )
            st.download_button(
                f"⬇️ Download chat for {selected_doc}",
                data=chat_text,
                file_name=f"chat_{selected_doc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime=MIME_TEXT_PLAIN,
            )

        if question := st.chat_input(f"Ask about {selected_doc}..."):
            doc_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Find the uploaded file object matching selected_doc
                    target_file = next(
                        (f for f in uploaded_files if f.name == selected_doc), None
                    )
                    if target_file is None:
                        st.error("Could not find the selected document. Please re-upload.")
                        st.stop()

                    from rag_pipeline import query_single_doc
                    answer, cost, source_docs = query_single_doc(
                        target_file, question, doc_history[:-1]
                    )

                st.markdown(answer)
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(source_docs, 1):
                        page = doc.metadata.get("page", "?")
                        pg = int(page) + 1 if page != "?" else "?"
                        st.markdown(
                            f"<div style='background:#f0f4ff;border-left:4px solid #4a6cf7;"
                            f"padding:8px 12px;border-radius:4px;margin:4px 0'>"
                            f"<b>Chunk {i}</b> — Page {pg}<br>"
                            f"<small>{doc.page_content[:300]}...</small></div>",
                            unsafe_allow_html=True,
                        )
                st.caption(f"🔢 Tokens: {cost['total_tokens']} | 💰 Cost: ${cost['total_cost']:.6f}")

            doc_history.append({
                "role": "assistant", "content": answer,
                "sources": source_docs, "cost": cost
            })
            st.session_state.total_tokens += cost["total_tokens"]
            st.session_state.total_cost += cost["total_cost"]
            st.session_state.qa_count += 1

# ─────────────────────────────
# TAB 3 — DASHBOARD
# ─────────────────────────────
with tab3:
    st.subheader("📊 Document Insights Dashboard")

    if "vectorstore" not in st.session_state or not st.session_state.file_names:
        st.info("⬅️ Upload PDFs to generate insights.")
    else:
        if st.button("🔍 Analyse Documents", use_container_width=True):
            with st.spinner("Analysing..."):
                raw, cost = get_dashboard_insights(st.session_state.vectorstore)
                st.session_state.total_tokens += cost["total_tokens"]
                st.session_state.total_cost   += cost["total_cost"]
            try:
                # Strip markdown fences if present
                clean = raw.strip().strip("```json").strip("```").strip()
                data  = json.loads(clean)

                col1, col2, col3 = st.columns(3)
                col1.metric("Document Type", data.get("document_type", "—"))
                col2.metric("Complexity",    data.get("complexity",    "—"))
                col3.metric("Tone",          data.get("tone",          "—"))

                st.markdown(f"**Main Subject:** {data.get('main_subject','—')}")

                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🔵 Top Topics")
                    for t in data.get("top_topics", []):
                        st.markdown(f"- {t}")

                    st.markdown("### 💡 Key Facts")
                    for f in data.get("key_facts", []):
                        st.markdown(f"- {f}")

                with col2:
                    st.markdown("### ❓ Suggested Questions")
                    for q in data.get("recommended_questions", []):
                        if st.button(q, key=f"sugg_{q}"):
                            st.session_state["pending_question"] = q
                            st.info("✅ Switching to chat tab — question queued.")
    # if st.button(q, key=f"sugg_{q}"):
    #                         st.session_state.messages.append({"role": "user", "content": q})
    #                         st.info("Question added to Multi-PDF Chat tab!")

            except Exception:
                st.markdown(raw)

        # File-level stats
        st.markdown("---")
        st.markdown("### 📄 Per-File Statistics")
        for name, meta in st.session_state.file_metadata.items():
            col1, col2, col3 = st.columns(3)
            col1.metric("File",   name[:30])
            col2.metric("Pages",  meta["pages"])
            col3.metric("Chunks", meta["chunks"])

        # Token usage
        st.markdown("---")
        st.markdown("### 💰 Token & Cost Tracker")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tokens Used", f"{st.session_state.total_tokens:,}")
        col2.metric("Total API Calls",   st.session_state.qa_count)
        col3.metric("Estimated Cost",    f"${st.session_state.total_cost:.5f}")
        st.caption(f"Model: {st.session_state.get('GROQ_MODEL','llama-3.1-8b-instant')} | Rates: $0.05/M input, $0.08/M output tokens")

# ─────────────────────────────
# TAB 4 — SUMMARY & INSIGHTS
# ─────────────────────────────
with tab4:
    st.subheader("📝 Summary & Key Insights")

    if "vectorstore" not in st.session_state or not st.session_state.file_names:
        st.info("⬅️ Upload PDFs first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✨ Generate Summary", use_container_width=True):
                with st.spinner("Summarising..."):
                    summary, cost = summarise_documents(st.session_state.vectorstore)
                    st.session_state.summary = summary
                    st.session_state.total_tokens += cost["total_tokens"]
                    st.session_state.total_cost   += cost["total_cost"]

        with col2:
            if st.button("💡 Generate Key Insights", use_container_width=True):
                with st.spinner("Extracting insights..."):
                    insights, cost = generate_key_insights(st.session_state.vectorstore)
                    st.session_state.insights = insights
                    st.session_state.total_tokens += cost["total_tokens"]
                    st.session_state.total_cost   += cost["total_cost"]

        if "summary" in st.session_state:
            st.markdown("### 📝 Summary")
            st.markdown(st.session_state.summary)
            st.download_button(
                "⬇️ Download Summary",
                data=st.session_state.summary,
                file_name="summary.txt", mime=MIME_TEXT_PLAIN,
            )

        if "insights" in st.session_state:
            st.markdown("---")
            st.markdown("### 💡 Key Insights")
            st.markdown(st.session_state.insights)
            st.download_button(
                "⬇️ Download Insights",
                data=st.session_state.insights,
                file_name="insights.txt", mime=MIME_TEXT_PLAIN,
            )

# ─────────────────────────────
# TAB 6 — TOPICS
# ─────────────────────────────
with tab5:
    st.subheader("📈 Most Discussed Topics")

    if "vectorstore" not in st.session_state or not st.session_state.file_names:
        st.info("⬅️ Upload PDFs first.")
    else:
        if st.button("🔎 Analyse Topics", use_container_width=True):
            with st.spinner("Analysing topics..."):
                topics, cost = get_top_topics(st.session_state.vectorstore)
                st.session_state.topics = topics
                st.session_state.total_tokens += cost["total_tokens"]
                st.session_state.total_cost   += cost["total_cost"]
        if "topics" in st.session_state:
            st.markdown(st.session_state.topics)
            st.download_button(
                "⬇️ Download Topics",
                data=st.session_state.topics,
                file_name="topics.txt", mime=MIME_TEXT_PLAIN)
