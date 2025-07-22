import streamlit as st
import requests
import pandas as pd
from urllib.parse import quote
import os
from dotenv import load_dotenv

load_dotenv()

mcp_client_endpoint = os.getenv("MCP_CLIENT_ENDPOINT")
mcp_server_endpoint = os.getenv("MCP_SERVER_ENDPOINT")

@st.cache_data(show_spinner=False)
def get_vector_stores():
    try:
        list_vector_store_endpoint = f"{mcp_server_endpoint}/retrieve_vector_stores"
        response = requests.get(list_vector_store_endpoint)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching vector stores: {e}")
        return []

# Page config
st.set_page_config(page_title="researchSoup", layout="wide", page_icon="🥣")

# Custom CSS
st.markdown(
    "<style>"
    ".stApp {max-width: 100% !important; padding: 0 1rem;}"
    "footer {padding-top: 2rem;}"
    "</style>",
    unsafe_allow_html=True
)

# Header
st.markdown(
    "<h1 style='text-align:center; color:#f5a623;'>🥣 researchSoup</h1>"
    "<p style='text-align:center; color:#e0c097;'>Serve your queries and sip on responses</p>",
    unsafe_allow_html=True
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "new_prompt" not in st.session_state:
    st.session_state.new_prompt = ''

# Tabs
tab_chat, tab_ingest = st.tabs(["🍽️ Chat", "📚 Ingest Documents"])

with tab_chat:
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg['role']):
            st.markdown(msg['content'], unsafe_allow_html=True)

            if msg['role'] == 'assistant':
                if 'docs_table' in msg and msg['docs_table']:
                    with st.expander("Reference Documents", expanded=False):
                        st.dataframe(pd.DataFrame(msg['docs_table']), use_container_width=True)

                        if 'pdf_paths' in msg and msg['pdf_paths']:
                            st.markdown("**Downloadable PDFs:**")
                            cols = st.columns(min(3, len(msg['pdf_paths'])))
                            for i, path in enumerate(msg['pdf_paths']):
                                label = os.path.basename(path).split('_', 1)[-1]
                                url = f"{mcp_server_endpoint}/download?file_path={quote(path)}"
                                with cols[i % 3]:
                                    try:
                                        data = requests.get(url, timeout=10).content
                                        st.download_button(
                                            label=label,
                                            data=data,
                                            file_name=path.rsplit('/', 1)[-1],
                                            mime="application/pdf",
                                            key=f"dl_{idx}_{i}"
                                        )
                                    except:
                                        st.markdown(f"[{label}]({url})")

                elif 'search_results' in msg and msg['search_results']:
                    with st.expander("🔍 Web Search Results", expanded=False):
                        st.dataframe(pd.DataFrame(msg['search_results']), use_container_width=True)

    input_col, send_col = st.columns([10, 1])
    st.session_state.new_prompt = input_col.text_input(
        label="Your Question",
        value=st.session_state.new_prompt,
        placeholder="What's cooking in your mind?",
        key="chat_input",
        label_visibility="collapsed",
        max_chars=500,
        help="Type your question here"
    )
    send = send_col.button("Send", use_container_width=True)

    if send and st.session_state.new_prompt.strip():
        prompt = st.session_state.new_prompt.strip()
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        try:
            r = requests.post(
                url= f"{mcp_client_endpoint}/chat",
                json={'query': prompt},
                timeout=30
            )
            r.raise_for_status()
            resp = r.json().get('response', {})
            answer = resp.get('response', 'No answer.')

            assistant_msg = {
                'role': 'assistant',
                'content': f"{answer}"
            }

            tool_calls = resp.get("tool_calls", [])
            documents_data = resp.get('documents', {})

            if tool_calls and tool_calls[0].get("tool_name") == "search_web":
                docs = documents_data.get("results", [])
                if docs:
                    web_results = []
                    for item in docs:
                        web_results.append({
                            "Title": item.get("title", "Untitled"),
                            "URL": item.get("url", "")
                        })
                    assistant_msg['search_results'] = web_results
            else:
                # Normal retrieved documents flow
                raw = documents_data.get('results', []) if documents_data else []

                docs = []
                if isinstance(raw, list) and raw:
                    if isinstance(raw[0], list) and len(raw[0]) > 1:
                        docs = raw[0][1]
                    elif isinstance(raw[0], dict):
                        docs = raw

                if docs:
                    table, pdfs, seen = [], [], set()

                    for idx, d in enumerate(docs, 1):
                        text = d.get('payload', {}).get('text', '') if d.get('payload') else ''
                        preview = text[:200] + ('...' if len(text) > 200 else '')
                        table.append({'Doc #': idx, 'Score': round(d.get('score', 0), 3), 'Preview': preview})

                        path = d.get('payload', {}).get('pdf_path')
                        if path and path not in seen:
                            pdfs.append(path)
                            seen.add(path)

                    assistant_msg['docs_table'] = table
                    assistant_msg['pdf_paths'] = pdfs

            st.session_state.messages.append(assistant_msg)

        except Exception as e:
            st.session_state.messages.append({'role': 'assistant', 'content': f"⚠️ {e}"})

        st.session_state.new_prompt = ''
        st.rerun()

with tab_ingest:
    st.markdown("### 📥 Upload Research Documents")
    st.markdown("Add new documents to your research collection")

    with st.form("document_ingestion", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            researcher = st.text_input(
                "Researcher Name",
                placeholder="Enter researcher name...",
                help="Name of the researcher or author"
            )

        with col2:
            vector_store_options = get_vector_stores()
            collection_name = st.selectbox(
                "Collection Name",
                options=vector_store_options or ["AI_store"],
                index=0,
                help="Select the document collection"
            )

        findings = st.text_area(
            "Research Findings",
            placeholder="Summarize key findings or context...",
            height=100,
            help="Brief summary of the research findings or document content"
        )

        embedding_model = st.selectbox(
            "Embedding Model (Optional)",
            options=["Default", "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"],
            index=0,
            help="Choose embedding model for document processing"
        )

        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to upload"
        )

        submitted = st.form_submit_button("🚀 Ingest Documents", type="primary")

        if submitted:
            if not researcher.strip():
                st.error("Please enter a researcher name")
            elif not findings.strip():
                st.error("Please enter research findings")
            elif not uploaded_files:
                st.error("Please upload at least one PDF file")
            else:
                files = [('pdfs', (f.name, f.getvalue(), 'application/pdf')) for f in uploaded_files]

                form_data = {
                    'researcher': researcher,
                    'findings': findings,
                    'collection_name': collection_name,
                }

                if embedding_model != "Default":
                    form_data['embedding_model'] = embedding_model

                with st.spinner("Ingesting documents... This may take a while ⏳"):
                    try:
                        response = requests.post(
                            url=f"{mcp_server_endpoint}/ingest_documents",
                            data=form_data,
                            files=files
                        )
                        response.raise_for_status()
                        result = response.json()

                        inserted_docs = result.get("inserted_documents", [])
                        skipped_docs = result.get("skipped_documents", [])

                        st.success("✅ Documents successfully processed!")
                        st.info(f"📁 Added {len(inserted_docs)} new chunk(s) to collection '{collection_name}'")
                        st.info(f"🔍 Skipped {len(skipped_docs)} due to similarity detection")

                        def format_doc_list(title, docs):
                            html = f"<details><summary>📄 {title} ({len(docs)})</summary><ul>"
                            for i, doc in enumerate(docs):
                                preview = doc.get("text", "No text available")[:300]
                                if len(doc.get("text", "")) > 300:
                                    preview += "..."
                                html += f"<li><strong>Document #{i+1}</strong><br><span style='font-size: 0.85em;'>{preview}</span></li><br>"
                            html += "</ul></details>"
                            return html

                        st.markdown(format_doc_list("Inserted Documents", inserted_docs), unsafe_allow_html=True)
                        st.markdown(format_doc_list("Skipped Documents", skipped_docs), unsafe_allow_html=True)

                    except requests.exceptions.Timeout:
                        st.error("⏰ Request timed out. The documents might still be processing.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Error ingesting documents: {e}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")

# Footer
st.markdown(
    "<footer style='text-align:center; color:#888; font-size:0.8rem;'>"
    "researchSoup 🍲 — brewed with 🤖 and ☕</footer>", unsafe_allow_html=True
)
