import streamlit as st
import requests
import pandas as pd
from urllib.parse import quote
import os
from dotenv import load_dotenv
import re
import html
from typing import List, Dict, Any
import random
import time
import json

# --- CONFIGURATION & SETUP ---
load_dotenv()
mcp_client_url = os.getenv("mcp_client_url")
mcp_server_url = os.getenv("mcp_server_url")

st.set_page_config(
    page_title="ResearchSoup: AI Research Assistant",
    layout="wide",
    page_icon="🥣",
    initial_sidebar_state="expanded",
)


# ---------- Utility helpers (improvements included) ----------


def safe_json(resp):
    """Try to decode JSON safely; return dict on failure."""
    try:
        return resp.json()
    except Exception:
        return {}


def clean_html_tags(text: str) -> str:
    """Strip simple HTML tags and html-escape sequences from assistant/user content."""
    if not text:
        return ""
    # Unescape common HTML entities
    text = html.unescape(text)
    # Remove basic tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_most_recent_name_from_history(messages: List[dict]) -> str:
    """
    Very simple heuristic: scan messages from newest to oldest and attempt to
    extract the first plausible name (one or two capitalized words).
    Falls back to empty string if nothing found.
    """
    name_pattern = re.compile(r"\b([A-Z][a-z]{1,}(?:\s[A-Z][a-z]{1,})?)\b")
    stopwords = set(
        [
            "The",
            "A",
            "I",
            "We",
            "You",
            "It",
            "This",
            "That",
            "Research",
            "Assistant",
            "Documents",
            "PDF",
            "Please",
            "Thanks",
        ]
    )
    for msg in reversed(messages):
        txt = clean_html_tags(msg.get("content", ""))
        # Find candidates
        candidates = name_pattern.findall(txt)
        # Filter small/obvious non-names
        for cand in candidates:
            if cand.split()[0] in stopwords:
                continue
            # Heuristic: require at least 2 letters and not just common word
            if len(cand) >= 2:
                return cand
    return ""


def rewrite_pronouns_with_name(prompt: str, name: str) -> str:
    """
    If prompt contains pronouns (his/her/he/she/their), replace them with the name.
    Very conservative: only rewrites if a name is available.
    """
    if not name:
        return prompt
    pronoun_pattern = re.compile(r"\b(his|her|their|he|she|they)\b")
    if pronoun_pattern.search(prompt):
        # Replace only the first occurrence (keeps the rest intact)
        return pronoun_pattern.sub(name, prompt, count=1)
    return prompt


@st.cache_data(show_spinner=False)
def get_vector_stores():
    """Fetch available vector stores from the server."""
    try:
        endpoint = f"{mcp_server_url.rstrip('/')}/retrieve_vector_stores"
        r = requests.get(endpoint, timeout=10)
        r.raise_for_status()
        data = safe_json(r)
        if isinstance(data, list):
            collections = (
                data
                if data
                else [
                    "AI_store",
                    "Team_Projects",
                    "Research_Papers",
                    "Internal_Audit",
                    "Client_Reports_Q3",
                ]
            )
            if "AI_store" not in collections:
                collections.append("AI_store")
            return collections

        if isinstance(data, dict):
            for key in ("collections", "stores", "vector_stores"):
                if key in data and isinstance(data[key], list):
                    collections = (
                        data[key]
                        if data[key]
                        else [
                            "AI_store",
                            "Team_Projects",
                            "Research_Papers",
                            "Internal_Audit",
                            "Client_Reports_Q3",
                        ]
                    )
                    if "AI_store" not in collections:
                        collections.append("AI_store")
                    return collections

        # Placeholder data if server connection fails or returns unexpected format
        return ["AI_store"]
    except requests.RequestException as e:
        print(f"Error fetching vector stores: {e}")
        return [f"AI_store (Error: {type(e).__name__})"]


# --- Placeholder function for simulation consistency (Ingest relies on clearing its cache) ---
@st.cache_data(show_spinner=False)
def get_collection_details(collection_name: str) -> List[Dict[str, Any]]:
    """
    Simulated function for document details (will be unused in the simplified view,
    but kept for cache clearing consistency with other tabs).
    """
    return []


def format_tool_name(tool):
    if tool is None:
        return "🧠 Auto-Select Tool"
    mapping = {
        "retrieve_documents": "Retrieve Documents (RAG)",
        "search_web": "Search Web (Current Events)",
        "retrieve_vector_stores": "List Collections",
    }
    return mapping.get(tool, tool.replace("_", " ").title())


# ---------- CSS (Removed .sidebar-summary and other unneeded classes) ----------
st.markdown(
    """
    <style>
    :root {
        --primary-accent: #008080;
        --background-color: #f7f9fc;
        --card-background: #ffffff;
        --text-color: #1e1e1e;
        --secondary-text: #6c757d;
        --hover-color: #e6f7f7;
    }
    .stApp { background-color: var(--background-color); color: var(--text-color); font-family: 'Arial', sans-serif; }
    .stApp > header { display: none; }
    .header-style { text-align: center; padding: 1.5rem 0 0.5rem; background-color: var(--card-background); border-bottom: 3px solid var(--primary-accent); margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .header-style h1 { color: var(--primary-accent) !important; font-size: 3rem; margin-bottom: 0.2rem; }
    .header-style p { color: var(--secondary-text); font-size: 1.2rem; margin-top: 0; }

    /* Collection Card Styling */
    .collection-card {
        background-color: var(--card-background);
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: default;
        height: 100%; 
    }
    .collection-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 10px rgba(0,0,0,0.1);
        background-color: var(--hover-color);
    }
    .collection-card h4 {
        color: var(--primary-accent);
        margin-top: 0;
        margin-bottom: 5px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .collection-card p {
        color: var(--secondary-text);
        margin-bottom: 0;
        font-size: 0.9rem;
    }

    /* Custom Info Box for Metric */
    .info-box {
        background-color: #e9f5f5; /* Light teal background */
        border: 1px solid var(--primary-accent);
        padding: 10px 15px;
        border-radius: 8px;
        display: inline-block;
        font-weight: 600;
        color: var(--primary-accent);
        margin-bottom: 15px;
        font-size: 1rem;
    }

    /* General Streamlit component styling (kept) */
    .stChatMessage[data-testid="stChatMessage-assistant"] { background-color: var(--card-background); border: 1px solid #e9ecef; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border-left: 5px solid var(--primary-accent); padding: 10px 15px; }
    .stChatMessage[data-testid="stChatMessage-user"] { background-color: #e9ecef; border: 1px solid #ced4da; border-radius: 10px; padding: 10px 15px; border-right: 5px solid var(--secondary-text); }
    .stExpander { border: 1px solid #ced4da; border-radius: 8px; margin-top: 1rem; background-color: #f8f9fa; }
    .main_chat_input_container { position: fixed; bottom: 0; left: 0; right: 0; padding: 1rem 1.5rem 0.5rem 1.5rem; background-color: var(--background-color); border-top: 1px solid #ced4da; z-index: 1000; max-width: 100%; box-sizing: border-box; }
    .main > div { padding-bottom: 120px; }
    .stTextInput > div > div > input { border-radius: 25px; padding: 0.8rem 1.5rem; border: 1px solid #adb5bd; }
    .stButton > button { border-radius: 25px; font-weight: bold; background-color: var(--primary-accent); color: white; transition: background-color 0.3s; border: none; }
    .stButton > button:hover { background-color: #005f5f !important; }
    .stForm { padding: 2rem; background-color: var(--card-background); border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
    .chat-form-container .stForm { padding: 0; box-shadow: none; background: transparent; }
    .custom-alert { border-left: 5px solid; padding: 1rem; margin: 1rem 0; border-radius: 5px; background-color: var(--card-background); font-weight: 500; }
    .success { border-color: #28a745; color: #28a745; background-color: #d4edda; }
    .info { border-color: #17a2b8; color: #17a2b8; background-color: #d1ecf1; }
    .error { border-color: #dc3545; color: #dc3545; background-color: #f8d7da; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='header-style'><h1>🥣 ResearchSoup</h1><p>Your AI-Powered Research Assistant</p></div>",
    unsafe_allow_html=True,
)

# ---------- Session state initialization ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_view" not in st.session_state:
    st.session_state.current_view = "💬 Chat"
if "last_ingested_collection" not in st.session_state:
    st.session_state.last_ingested_collection = "AI_store"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "query_mode" not in st.session_state:
    st.session_state.query_mode = "Web Search"
if "ingestion_results" not in st.session_state:
    st.session_state.ingestion_results = None
if "selected_collection_view" not in st.session_state:
    st.session_state.selected_collection_view = None


# ---------- API call + response parsing (kept) ----------
def _make_api_call_and_parse_response(prompt, tool_name, collection_name):
    """Send the prompt to the MCP client and parse RAG / web responses into a friendly assistant message."""
    params = {"query": prompt}
    if tool_name:
        params["tool_name"] = tool_name
    if tool_name == "retrieve_documents" and collection_name:
        params["collection_name"] = collection_name

    client_endpoint = f"{mcp_client_url.rstrip('/')}/chat"

    print("\n--- DEBUG LOG ---")
    print(f"Endpoint: {client_endpoint}")
    print(f"Params: {params}")
    print("-----------------\n")

    try:
        r = requests.post(url=client_endpoint, params=params, timeout=60)
        if r.status_code != 200:
            print(f"Server status: {r.status_code}; Body: {r.text[:500]}")
            r.raise_for_status()

        data = safe_json(r)
        resp = data.get("response", {}) if isinstance(data, dict) else {}
        print("📥 Parsed response keys:", list(resp.keys()))

        # --- Extract assistant text ---
        answer_text = ""
        if isinstance(resp, dict):
            answer_text = (
                    resp.get("response", "")
                    or resp.get("final_answer", "")
                    or resp.get("answer", "")
            )
        elif isinstance(resp, str):
            answer_text = resp

        assistant_msg = {
            "role": "assistant",
            "content": answer_text or "No answer found in server response.",
        }

        # --- Detect tool calls ---
        tool_calls = resp.get("tool_calls", []) if isinstance(resp, dict) else []
        documents_data = None

        # --- Handle retrieve_documents tool call ---
        if (
                tool_calls
                and len(tool_calls) > 0
                and tool_calls[0].get("tool_name") == "retrieve_documents"
        ):
            print("🧠 Detected retrieve_documents tool call")

            # Get JSON string inside the first result text
            tool_result = tool_calls[0].get("result", [])
            if tool_result and isinstance(tool_result[0], dict):
                text_block = tool_result[0].get("text", "")
                if text_block:
                    try:
                        documents_data = json.loads(text_block)
                    except Exception as e:
                        print(f"⚠️ Failed to parse nested text JSON: {e}")
                        documents_data = None

            # Now safely extract points
            docs = []
            if documents_data:
                raw_results = documents_data.get("results", [])
                if (
                        raw_results
                        and isinstance(raw_results[0], list)
                        and len(raw_results[0]) > 1
                ):
                    docs = raw_results[0][1]
                elif isinstance(raw_results[0], dict):
                    docs = raw_results

            # Build preview table
            if docs:
                table, pdfs, seen = [], [], set()
                for d in docs:
                    payload = d.get("payload", {}) or {}
                    text = payload.get("text", "") or payload.get("content", "")
                    researcher = payload.get("researcher", "Unknown")
                    preview = text[:200] + ("..." if len(text) > 200 else "")
                    score = round(d.get("score", 0), 3)

                    table.append(
                        {
                            "Researcher": researcher,
                            "Score": score,
                            "Chunk (Preview)": preview,
                        }
                    )

                    path = payload.get("pdf_path")
                    if path and path not in seen:
                        pdfs.append(path)
                        seen.add(path)

                assistant_msg["docs_table"] = table
                assistant_msg["pdf_paths"] = pdfs
                print(f"✅ Extracted {len(table)} document chunks.")
            else:
                print("⚠️ No documents found after parsing nested structure.")

        # --- Handle search_web tool call ---
        elif (
                tool_calls
                and len(tool_calls) > 0
                and tool_calls[0].get("tool_name") == "search_web"
        ):
            print("🌐 Detected search_web tool call")

            tool_result = tool_calls[0].get("result", [])
            if tool_result and isinstance(tool_result[0], dict):
                documents_data = tool_result[0].get("text", "")
                if documents_data:
                    try:
                        documents_data = json.loads(documents_data)
                    except Exception as e:
                        print(f"⚠️ Failed to parse search_web JSON: {e}")
                        documents_data = None

            docs = []
            if isinstance(documents_data, dict):
                docs = documents_data.get("results", [])
            elif isinstance(documents_data, list):
                docs = documents_data

            if docs:
                table = []
                for d in docs:
                    title = d.get("title", "Untitled")
                    url = d.get("url", "")
                    snippet = d.get("snippet", "") or d.get("text", "")
                    preview = snippet[:200] + ("..." if len(snippet) > 200 else "")
                    table.append(
                        {
                            "Title": title,
                            "URL": url,
                            "Snippet (Preview)": preview,
                        }
                    )

                assistant_msg["docs_table"] = table
                print(f"✅ Extracted {len(table)} web search results.")
            else:
                print("⚠️ No search_web results found.")

        return assistant_msg

    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return {"role": "assistant", "content": f"An error occurred: {e}"}


def render_chat_interface():
    st.markdown("## 📚 Research Chat & Document Query")
    st.markdown("### 💬 Live Chat Interface")

    # --- Initializations ---
    if "show_researchers" not in st.session_state:
        st.session_state.show_researchers = False
    if "selected_researcher" not in st.session_state:
        st.session_state.selected_researcher = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_input_key_form_value" not in st.session_state:
        st.session_state.chat_input_key_form_value = ""
    if "reset_chat_input" not in st.session_state:
        st.session_state.reset_chat_input = False

    # --- Reset chat input on rerun ---
    if st.session_state.reset_chat_input:
        st.session_state.chat_input_key_form_value = ""
        st.session_state.reset_chat_input = False

    all_collections = get_vector_stores()
    rag_system_error = all_collections and "Error" in all_collections[0]
    rag_generally_available = not rag_system_error and len(all_collections) > 0

    collection_to_query = st.session_state.get("last_ingested_collection", None)
    if rag_generally_available:
        if not collection_to_query or collection_to_query not in all_collections:
            collection_to_query = all_collections[0]
            st.session_state.last_ingested_collection = collection_to_query
    else:
        collection_to_query = None

    rag_label = f"Query Documents"
    rag_display_label = (
        rag_label if rag_generally_available else "Query Documents - (Not Available)"
    )

    col_mode, col_actions = st.columns([3, 1])
    options = [rag_display_label, "Web Search"]
    current_mode = st.session_state.get("query_mode", "Web Search")

    query_mode = col_mode.radio(
        "Select Query Mode",
        options=options,
        index=options.index(current_mode) if current_mode in options else 1,
        key="query_mode_selector_main",
        horizontal=True,
        help="RAG queries documents in the selected collection. Web Search queries current events.",
    )

    if query_mode != st.session_state.get("query_mode"):
        st.session_state.query_mode = query_mode
        st.session_state.selected_researcher = None
        st.session_state.show_researchers = False
        st.rerun()

    if col_actions.button("🗑️ Clear Chat History", key="clear_chat_btn_chat_main"):
        st.session_state.chat_messages = []
        st.session_state.reset_chat_input = True
        st.rerun()

    chat_input_disabled = False
    tool_name = "search_web"
    collection_name = None
    info_message = "Using Web Search"

    if query_mode.startswith("Query Documents"):
        tool_name = "retrieve_documents"
        collection_name = collection_to_query
        if not rag_generally_available:
            chat_input_disabled = True
            st.error("🚫 RAG Disabled: No valid document collections found or server error.")
        else:
            filter_status = st.session_state.get("selected_researcher") or "All Documents"
            info_message = f"Using RAG Mode. **Current Filter: {filter_status}** (Click 👤 to change)"

    st.info(info_message)

    # --- Researcher Dropdown ---
    if tool_name == "retrieve_documents" and st.session_state.show_researchers:
        try:
            response = requests.get(f"{mcp_server_url}/collection_details")
            response.raise_for_status()
            response_data = response.json()

            all_researchers_set = set()
            if isinstance(response_data, dict):
                for collection_data in response_data.values():
                    for r in collection_data.get("researchers", []):
                        if r:
                            all_researchers_set.add(r)
            all_researchers = sorted(list(all_researchers_set))

            current_selection = (
                st.session_state.selected_researcher
                if st.session_state.selected_researcher
                else "* All Documents (No Filter)"
            )

            options_list = ["* All Documents (No Filter)"] + [
                f"👨‍🔬 {r}" for r in all_researchers
            ]

            try:
                default_index = (
                    options_list.index(f"👨‍🔬 {current_selection}")
                    if current_selection and current_selection != "* All Documents (No Filter)"
                    else 0
                )
            except ValueError:
                default_index = 0

            st.markdown("---")
            st.markdown("#### 👥 **Select Researcher to Filter Results**")

            selected_option = st.selectbox(
                "",
                options=options_list,
                index=default_index,
                key="researcher_select_rag_mode",
            )

            new_selection = selected_option.replace("👨‍🔬 ", "").strip() if selected_option.startswith("👨‍🔬 ") else None

            if new_selection != st.session_state.selected_researcher:
                st.session_state.selected_researcher = new_selection
                st.rerun()

            st.markdown("---")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch researchers (Server Error): {str(e)}")
        except Exception as e:
            st.error(f"Failed to fetch researchers (Data Error): {str(e)}")

    elif tool_name != "retrieve_documents":
        st.session_state.selected_researcher = None

    # --- Chat History Display ---
    with st.container():
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

                #  Add RAG preview table
                if msg["role"] == "assistant" and "docs_table" in msg and msg["docs_table"]:
                    st.markdown("#### 📊 Retrieved Document Chunks")
                    df = pd.DataFrame(msg["docs_table"])

                    # Show the DataFrame in a scrollable table
                    st.dataframe(df, use_container_width=True, height=250)

                    # Optional: PDF download buttons
                    if "pdf_paths" in msg and msg["pdf_paths"]:
                        st.markdown("##### 📄 Download PDF Sources")
                        cols = st.columns(min(4, len(msg["pdf_paths"])))
                        for i, path in enumerate(msg["pdf_paths"]):
                            label = (
                                    os.path.basename(path).split("_", 1)[-1].strip()
                                    or "Download File"
                            )
                            url = f"{mcp_server_url.rstrip('/')}/download?file_path={quote(path)}"
                            with cols[i % 4]:
                                st.markdown(
                                    f'<a href="{url}" target="_self" download '
                                    f'style="display:block;text-decoration:none;padding:0.6rem 0.5rem;'
                                    f'margin-top:0.25rem;background-color:var(--primary-accent);color:white;'
                                    f'border-radius:5px;text-align:center;font-weight:bold;font-size:0.85rem;">⬇️ {label}</a>',
                                    unsafe_allow_html=True,
                                )

    # --- Sticky Footer Input ---
    form_key = f"chat_query_form_{collection_name}_{tool_name}"

    if tool_name == "retrieve_documents":
        plus_col, main_input_form_col = st.columns([1, 11])

        with plus_col:
            st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
            button_icon = "🧹" if st.session_state.show_researchers else "👤"
            if st.button(button_icon, key="plus_btn_regular",
                         help="Toggle Researcher Filter",
                         use_container_width=True,
                         disabled=chat_input_disabled):
                st.session_state.show_researchers = not st.session_state.show_researchers
                st.rerun()

        with main_input_form_col:
            with st.form(form_key, clear_on_submit=True):
                input_col, send_col = st.columns([9, 1])
                with input_col:
                    chat_prompt = st.text_input(
                        "Question",
                        placeholder="Ask about the research in the collection...",
                        key="chat_input_key_form_value",
                        label_visibility="collapsed",
                        max_chars=500,
                        disabled=chat_input_disabled,
                    )
                with send_col:
                    chat_send = st.form_submit_button("Send")

                if chat_send:
                    input_text = st.session_state.chat_input_key_form_value.strip()

                    if chat_input_disabled or not input_text:
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": "<div class='custom-alert error'>❌ Please type a valid question or check connection.</div>"
                        })
                        st.session_state.reset_chat_input = True
                        st.rerun()
                        return

                    # Add user message
                    st.session_state.chat_messages.append({"role": "user", "content": input_text})

                    # Build context for RAG
                    history_parts = []
                    for msg in st.session_state.chat_messages[:-1][-10:]:
                        role_label = "User" if msg["role"] == "user" else "Assistant"
                        content = msg.get("content", "")
                        content = content if len(content) <= 1000 else content[-1000:]
                        history_parts.append(f"<{role_label}>: {content}")
                    context_history = "\n".join(history_parts)

                    selected_researcher_filter = st.session_state.selected_researcher or "any researcher"

                    rag_directive = (
                        f"YOU ARE A KNOWLEDGE-AWARE RESEARCH ASSISTANT. "
                        f"Retrieve information from ALL documents in '{collection_name}'. "
                        f"Focus on documents uploaded by '{selected_researcher_filter}' if possible."
                    )

                    prompt_for_api = f"{rag_directive}\n\n{context_history}\n\n{input_text}"

                    with st.status("Searching for a response...", expanded=True):
                        assistant_msg = _make_api_call_and_parse_response(
                            prompt=prompt_for_api,
                            tool_name=tool_name,
                            collection_name=collection_name,
                        )
                        st.session_state.chat_messages.append(assistant_msg)
                        st.session_state.reset_chat_input = True
                        st.rerun()

    else:
        with st.form(form_key, clear_on_submit=True):
            input_col, send_col = st.columns([10, 1])
            with input_col:
                chat_prompt = st.text_input(
                    "Question",
                    placeholder="Ask a general question using Web Search...",
                    key="chat_input_key_form_value",
                    label_visibility="collapsed",
                    max_chars=500,
                    disabled=chat_input_disabled,
                )
            with send_col:
                chat_send = st.form_submit_button("Send", disabled=chat_input_disabled)

            if chat_send:
                input_text = st.session_state.chat_input_key_form_value.strip()

                if chat_input_disabled or not input_text:
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": "<div class='custom-alert error'>❌ Please type a valid question or check connection.</div>"
                    })
                    st.session_state.reset_chat_input = True
                    st.rerun()
                    return

                st.session_state.chat_messages.append({"role": "user", "content": input_text})

                with st.status("Searching for a response...", expanded=True):
                    assistant_msg = _make_api_call_and_parse_response(
                        prompt=input_text,
                        tool_name=tool_name,
                        collection_name=collection_name,
                    )
                    st.session_state.chat_messages.append(assistant_msg)
                    st.session_state.reset_chat_input = True
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ---------- Ingest interface (kept modified) ----------
def format_doc_list_html(title, docs):
    html_block = f"<details><summary>📄 {title} ({len(docs)})</summary><ul>"
    for i, doc in enumerate(docs):
        text = doc.get("text") if isinstance(doc, dict) else str(doc)
        preview = (text or "")[:300] + ("..." if len(text or "") > 300 else "")
        html_block += (
            f"<li><p style='color:var(--text-color); margin-bottom: 5px; font-weight: 600;'>Chunk #{i + 1}</p>"
            f"<span style='font-size: 0.9em; color: var(--secondary-text);'>{preview}</span></li>"
        )
    html_block += "</ul></details>"
    return html_block


def render_ingest_interface():
    st.markdown("## 📚 Document Ingestion")
    st.markdown("Upload your research documents to a specific collection.")

    st.markdown("---")
    st.markdown("### Upload Documents to an Existing Collection")

    all_collections = get_vector_stores()
    rag_generally_available = not (all_collections and "Error" in all_collections[0])

    if not rag_generally_available:
        st.error(
            "🚫 Document Ingestion is disabled because no valid document server connection was found."
        )

    collection_to_query = st.session_state.last_ingested_collection
    if collection_to_query not in all_collections:
        collection_to_query = all_collections[0]

    with st.form("document_ingestion", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            researcher = st.text_input(
                "Researcher Name",
                placeholder="Enter researcher name",
                key="ingest_researcher",
            )
        with col2:
            options_list = all_collections if rag_generally_available else ["AI_store"]
            default_collection_index = 0
            try:
                default_collection_index = options_list.index(collection_to_query)
            except ValueError:
                default_collection_index = 0
            # Select an existing collection
            collection_name_select = st.selectbox(
                "Existing Collection Name",
                options=options_list,
                index=default_collection_index,
                key="ingest_collection",
                help="Select an existing collection to add documents to.",
            )

        # Determine the final collection name (only using selectbox value now)
        final_collection_name = collection_name_select

        findings = st.text_area(
            "Research Findings",
            placeholder="Summarize key findings or context for better retrieval...",
            height=100,
            key="ingest_findings",
        )
        st.markdown("---")

        col3, col4 = st.columns(2)
        with col3:
            embedding_model = st.selectbox(
                "Embedding Model (Optional)",
                options=[
                    "Default",
                    "BAAI/bge-small-en-v1.5",
                    "sentence-transformers/all-MiniLM-L6-v2",
                ],
                index=0,
                key="ingest_model",
            )
        with col4:
            uploaded_files = st.file_uploader(
                "Upload PDF Documents",
                type=["pdf"],
                accept_multiple_files=True,
                key="ingest_files",
            )

        st.markdown("---")
        submitted = st.form_submit_button(
            "🚀 Ingest Documents",
            type="primary",
            use_container_width=True,
            disabled=not rag_generally_available,
        )

        if submitted and rag_generally_available:
            st.session_state.ingestion_results = None
            if not (researcher or "").strip():
                st.markdown(
                    "<div class='custom-alert error'>❌ Please enter a **Researcher Name**.</div>",
                    unsafe_allow_html=True,
                )
            elif not (findings or "").strip():
                st.markdown(
                    "<div class='custom-alert error'>❌ Please enter **Research Findings**.</div>",
                    unsafe_allow_html=True,
                )
            elif not uploaded_files:
                st.markdown(
                    "<div class='custom-alert error'>❌ Please upload at least one **PDF Document**.</div>",
                    unsafe_allow_html=True,
                )
            elif not (final_collection_name or "").strip():
                st.markdown(
                    "<div class='custom-alert error'>❌ Please select a **Collection Name**.</div>",
                    unsafe_allow_html=True,
                )
            else:
                files = [
                    ("pdfs", (f.name, f.getvalue(), "application/pdf"))
                    for f in uploaded_files
                ]
                form_data = {
                    "researcher": researcher,
                    "findings": findings,
                    "collection_name": final_collection_name,
                }
                if embedding_model != "Default":
                    form_data["embedding_model"] = embedding_model

                try:
                    with st.spinner(f"Ingesting documents"):
                        response = requests.post(
                            url=f"{mcp_server_url.rstrip('/')}/ingest_documents",
                            data=form_data,
                            files=files,
                            timeout=120,
                        )
                        response.raise_for_status()
                        result = safe_json(response)
                        inserted_docs = result.get("inserted_documents", [])
                        skipped_docs = result.get("skipped_documents", [])

                        st.session_state.last_ingested_collection = (
                            final_collection_name
                        )
                        st.session_state.chat_messages = []
                        st.session_state.ingestion_results = {
                            "collection_name": final_collection_name,
                            "inserted_docs": inserted_docs,
                            "skipped_docs": skipped_docs,
                        }
                        # Clear caches after successful ingest
                        get_collection_details.clear()
                        get_vector_stores.clear()
                        st.rerun()
                except requests.exceptions.Timeout:
                    st.markdown(
                        "<div class='custom-alert error'>⏰ Request timed out (120s). Documents might still be processing.</div>",
                        unsafe_allow_html=True,
                    )
                except requests.exceptions.RequestException as e:
                    status_code = getattr(e.response, "status_code", "Unknown")
                    st.markdown(
                        f"<div class='custom-alert error'>❌ **Error** ingesting documents. Status: {status_code}. Details: {e}</div>",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.markdown(
                        f"<div class='custom-alert error'>❌ **Unexpected error**: {e}</div>",
                        unsafe_allow_html=True,
                    )

    # --- INGESTION RESULTS MOVED BELOW THE FORM ---
    if st.session_state.ingestion_results:
        result = st.session_state.ingestion_results
        collection_name_input = result["collection_name"]
        inserted_docs = result["inserted_docs"]
        skipped_docs = result["skipped_docs"]
        # Add a visual separator before the results
        st.markdown("---")
        st.markdown(
            f"<div class='custom-alert success'>✅ Documents successfully processed!",
            unsafe_allow_html=True,
        )
        with st.expander(f"📥 Ingestion Details", expanded=True):
            st.markdown(
                f"<div classs='custom-alert info'>📁 Added {len(inserted_docs)} new chunks.</div>",
                unsafe_allow_html=True,
            )
            if skipped_docs:
                st.markdown(
                    f"<div class='custom-alert info'>🔍 Skipped {len(skipped_docs)} chunks due to similarity detection.</div>",
                    unsafe_allow_html=True,
                )
            if inserted_docs:
                st.markdown(
                    format_doc_list_html(
                        "Successfully Inserted Document Chunks", inserted_docs
                    ),
                    unsafe_allow_html=True,
                )
            if skipped_docs:
                st.markdown(
                    format_doc_list_html(
                        "Skipped Duplicate Document Chunks", skipped_docs
                    ),
                    unsafe_allow_html=True,
                )


# --- 2. MAIN STREAMLIT FUNCTION ---


def clean_filename(filename):
    """Removes the UUID prefix (e.g., 'uuid_') from a filename if it exists."""
    if not isinstance(filename, str):
        return str(filename)
    if "_" in filename:
        return filename.split("_", 1)[-1]
    return filename


# --- NEW FUNCTION TO FETCH ALL COLLECTION DATA AND NAMES ---
def fetch_all_collection_data():
    """Fetches all collection details from the backend, returning the raw data."""
    try:
        response = requests.get(f"{mcp_server_url}/collection_details")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to connect to collection backend at {mcp_server_url}: {e}")
        return {"Error": f"Connection Failed: {e}"}
    except json.JSONDecodeError:
        st.error("⚠️ Invalid JSON response from collection backend.")
        return {"Error": "Invalid JSON response"}


# -----------------------------------------------------------
def render_view_collections_interface():
    # Initialize session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "collections"
    if "selected_collection" not in st.session_state:
        st.session_state.selected_collection = "AI_store"

    def go_to_collection(collection_name):
        st.session_state.selected_collection = collection_name
        st.session_state.current_page = "collection_details"

    # ---------------- Collections Page ----------------
    if st.session_state.current_page == "collections":
        st.markdown("## 📚 Vector Store Collections")

        all_collections_data = fetch_all_collection_data()
        if "Error" in all_collections_data:
            return

        collections = list(all_collections_data.keys())

        if not collections:
            st.info(
                "No document collections found. Upload some PDFs in the **Ingest Documents** tab to create one."
            )
            return

        collection_names = [name.replace("_", " ").title() for name in collections]
        collection_map = {name.replace("_", " ").title(): name for name in collections}

        current_selection_key = st.session_state.selected_collection.replace(
            "_", " "
        ).title()
        default_index = (
            collection_names.index(current_selection_key)
            if current_selection_key in collection_names
            else 0
        )

        selected_name = st.selectbox(
            "Select a collection to view its contents:",
            options=collection_names,
            index=default_index,
            key="collection_selectbox",
        )

        st.button(
            f"View **{selected_name}** Details",
            type="primary",
            use_container_width=True,
            on_click=go_to_collection,
            args=(collection_map[selected_name],),
        )

        st.markdown("---")

        # Display table overview
        data_for_df = []
        for name in collections:
            display_name = name.replace("_", " ").title()
            details = all_collections_data.get(name, {})
            total_documents = details.get("total_documents", "N/A")
            status = (
                "Active" if name == st.session_state.selected_collection else "Inactive"
            )
            data_for_df.append(
                {
                    "Collection Name": display_name,
                    "Documents": total_documents,
                    "Status": status,
                }
            )

        df = pd.DataFrame(data_for_df, index=range(1, len(collections) + 1))
        df.index.name = "ID"

        st.markdown("### Collection Overview")
        st.dataframe(
            df,
            hide_index=False,
            use_container_width=True,
            column_config={
                "Collection Name": st.column_config.TextColumn(
                    "Collection Name", disabled=True
                ),
                "Documents": st.column_config.TextColumn(
                    "Total Documents", disabled=True
                ),
                "Status": st.column_config.TextColumn("Status", disabled=True),
            },
        )

    # ---------------- Collection Details Page ----------------
    elif st.session_state.current_page == "collection_details":
        selected_collection = st.session_state.selected_collection

        st.button(
            "⬅️ Back to Collections",
            on_click=lambda: setattr(st.session_state, "current_page", "collections"),
        )

        st.markdown(
            f"## 📂 Documents Collection — **{selected_collection.replace('_', ' ').title()}**"
        )
        st.markdown(
            "A centralized collection of documents uploaded by various researchers."
        )
        st.markdown("---")

        try:
            response = requests.get(f"{mcp_server_url}/collection_details")
            if response.status_code == 200:
                data = response.json()
                collection_data = data.get(selected_collection, {})

                if not collection_data:
                    st.info(f"No details found for {selected_collection} in backend.")
                    return

                # Extract information
                all_researchers = collection_data.get("researchers", [])
                files = collection_data.get("files", [])
                total_documents = collection_data.get(
                    "total_documents", len(files) if files else 0
                )

                # 📦 Professional Metric Card
                st.markdown(
                    f"""
                    <div style="
                        background-color: #f8f9fa;
                        padding: 25px;
                        border-radius: 15px;
                        border: 1px solid #e0e0e0;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                        text-align: center;
                        margin-bottom: 20px;
                    ">
                        <h3 style="color: #2c3e50; margin-bottom: 10px;">📄 Total Documents in Repository</h3>
                        <p style="font-size: 36px; font-weight: 700; color: #007bff; margin: 0;">
                             {total_documents}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Researcher search
                search_query = st.text_input(
                    "🔍 Search Researcher Name",
                    placeholder="search researcher",
                    key="researcher_search",
                )

                if search_query:
                    filtered_researchers = [
                        r
                        for r in all_researchers
                        if re.search(search_query, r, re.IGNORECASE)
                    ]
                    if not filtered_researchers:
                        st.info(f"No researchers found matching **'{search_query}'**.")
                        return
                else:
                    filtered_researchers = all_researchers

                if filtered_researchers:
                    file_details_map = {}
                    for f in files:
                        if isinstance(f, dict):
                            file_researcher = f.get("researcher")
                            if file_researcher:
                                file_details_map.setdefault(file_researcher, []).append(
                                    f
                                )

                    filtered_researchers.sort()

                    for researcher in filtered_researchers:
                        researcher_files = file_details_map.get(researcher, [])
                        file_count = len(researcher_files)

                        with st.expander(f"👤 {researcher} ({file_count})"):
                            if researcher_files:
                                file_df = pd.DataFrame(
                                    [
                                        {
                                            "File Name": clean_filename(
                                                f.get("filename", "Unknown File")
                                            ),
                                            "View": f"{mcp_server_url}/view/?file_path={f.get('filename', 'unknown_file')}",
                                            "Download": f"{mcp_server_url}/download/?file_path={f.get('filename', 'unknown_file')}",
                                        }
                                        for f in researcher_files
                                    ]
                                )
                                st.dataframe(
                                    file_df,
                                    hide_index=True,
                                    use_container_width=True,
                                    column_config={
                                        "View": st.column_config.LinkColumn(
                                            "👁️ View",
                                            display_text="👁️",
                                            help="Open and view the document.",
                                            width="small",
                                        ),
                                        "Download": st.column_config.LinkColumn(
                                            "⬇️ Download",
                                            display_text="⬇️",
                                            help="Click to download the source file.",
                                            width="small",
                                        ),
                                    },
                                )
                            else:
                                st.info("No files uploaded by this researcher.")
                else:
                    st.info("No researchers found in this collection.")

            else:
                st.error(
                    f"❌ Failed to fetch collection details (Status {response.status_code})."
                )

        except Exception as e:
            st.error(f"⚠️ Error retrieving details (Connection Error): {e}")

        st.markdown("---")
        st.markdown(
            "<p style='font-size: 0.9em; color: var(--secondary-text);'>"
            "The list above is dynamically pulled from the connected vector store service."
            "</p>",
            unsafe_allow_html=True,
        )


# ---------- Main ----------
def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")

        VIEW_OPTIONS = ["💬 Chat", "📚 Ingest Documents", "📊 View Collections"]

        # ✅ Directly use selectbox value (no manual session_state handling)
        selected_view = st.selectbox(
            "**Select Feature**", VIEW_OPTIONS, key="navigation_selector"
        )

        st.markdown("---")
        st.markdown("#### Application Overview")
        st.markdown(
            """
            **researchSoup** is a collaborative AI research platform designed to centralize knowledge 
            and enhance information retrieval for individuals and teams.
        """
        )
        st.markdown("**Core Capabilities:**")
        st.markdown(
            """
        * **Research Ingestion (RAG):** Upload, chunk, and embed research documents for precise retrieval.  
        * **Contextual Chat:** Ask questions using your ingested research data.  
        * **Collection Management:** View and manage your knowledge collections.
        """
        )

    # ---------- Render Selected Page Instantly ----------
    if selected_view == "📚 Ingest Documents":
        render_ingest_interface()
    elif selected_view == "📊 View Collections":
        render_view_collections_interface()
    elif selected_view == "💬 Chat":
        render_chat_interface()


# ---------- Run App ----------as
if __name__ == "__main__":
    main()
