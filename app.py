# app.py
# 📚 RAG AI Agent with Multilingual & Voice Support 🎙️
# Supports file uploads: PDF, TXT, CSV, HTML, XML, GitHub raw files.
# Uses Context-Augmented Generation (CAG) for history and cost minimization.

import streamlit as st
import os, sys, tempfile, uuid, time, io, asyncio, datetime, re
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import torch

# RAG dependencies (using placeholders for heavy libraries like pypdf)
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# sqlite fix for Chroma
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass

# Gemini SDK (python-genai)
try:
    from google import genai
    from google.genai.errors import APIError
    from google.genai import types
except ImportError:
    genai = None
    APIError = None
    types = None

# Optional TTS engines
try:
    import edge_tts
except Exception:
    edge_tts = None
try:
    from gtts import gTTS
except Exception:
    gTTS = None

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="RAG AI Agent 📚", page_icon="🤖", layout="wide")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

LANGUAGE_DICT = {
    "English": "en", "Spanish": "es", "Arabic": "ar", "French": "fr", "German": "de", "Hindi": "hi",
    "Tamil": "ta", "Bengali": "bn", "Japanese": "ja", "Korean": "ko", "Russian": "ru",
    "Chinese (Simplified)": "zh-Hans", "Portuguese": "pt", "Italian": "it", "Dutch": "nl", "Turkish": "tr"
}

COLLECTION_NAME = "uploaded_documents_rag"
CACHE_EXPIRY_SECONDS = 300 # 5 minutes for Conversation Caching (CAG)

# -------------------------
# RAG and Storage Helpers
# -------------------------
@st.cache_resource(show_spinner=False)
def initialize_rag_dependencies():
    # Use disk storage for persistence across re-runs (needed for file uploads)
    db_path = tempfile.mkdtemp()
    db_client = chromadb.PersistentClient(path=db_path)
    # Use a small, fast model for embedding
    model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu') 
    if GEMINI_API_KEY and genai:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        gemini_client = None
    return db_client, model, gemini_client

def get_collection():
    if 'db_client' not in st.session_state:
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client = initialize_rag_dependencies()
    return st.session_state.db_client.get_or_create_collection(name=COLLECTION_NAME)

def split_documents(text_data, chunk_size=500, chunk_overlap=100):
    # Enforcing overlapping chunking as requested
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        length_function=len, 
        is_separator_regex=False
    )
    return splitter.split_text(text_data)

def process_and_store_documents(documents: List[str]):
    collection = get_collection()
    model = st.session_state.model
    
    # Generate embeddings in batches for efficiency
    embeddings = model.encode(documents, convert_to_tensor=False).tolist()
    ids = [str(uuid.uuid4()) for _ in documents]
    
    # Store documents
    collection.add(documents=documents, embeddings=embeddings, ids=ids)
    return len(documents)

def retrieve_documents(query, n_results=5):
    collection = get_collection()
    model = st.session_state.model
    q_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=n_results, include=['documents', 'distances'])
    return results['documents'][0] if results['documents'] else []

# --- Document Loading Placeholder ---
def extract_text_from_upload(uploaded_file):
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    raw_text = ""
    
    # Simple reader for text-like files
    if file_details["FileType"] in ["text/plain", "application/csv", "text/csv", "application/json", 
                                    "text/html", "application/xml", "text/xml"]:
        raw_text = uploaded_file.getvalue().decode("utf-8")
    
    # Placeholder for PDF. Requires libraries like 'pypdf' or 'pdfminer.six'
    elif file_details["FileType"] == "application/pdf":
        try:
            # THIS IS A PLACEHOLDER. User must install pypdf
            from pypdf import PdfReader
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                raw_text += page.extract_text() or ""
        except ImportError:
            return f"Error: Cannot read PDF. Please install 'pypdf'. Content placeholder: {uploaded_file.name}", False
    
    # Basic attempt for other binary types (like images, etc. should be handled by specific tools)
    else:
        return f"File type {file_details['FileType']} not supported or requires external tools.", False

    return raw_text, True

def clear_rag_storage():
    db_client = st.session_state.db_client if 'db_client' in st.session_state else initialize_rag_dependencies()[0]
    try:
        db_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    get_collection() # Re-create the collection
    st.session_state.ingested_files = []
    st.session_state.cache = {}
    st.rerun()

# -------------------------
# TTS utilities (retained)
# -------------------------
def tts_gemini(text: str, voice_name="Kore"):
    """SDK-compatible Gemini TTS call. Returns (bytes, error)"""
    if not (genai and GEMINI_API_KEY): return None, "Gemini TTS not available."
    try:
        client = initialize_gemini_client()
        cfg = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            )
        )
        resp = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=cfg,
        )
        return resp.binary, None
    except Exception as e:
        return None, str(e)

async def _edge_async(text: str, voice="en-US-AriaNeural", rate=None):
    if not edge_tts: return None
    kwargs = {"text": text, "voice": voice}
    if rate:
        r = rate.strip()
        if r == "0%" or r == "0": r = "+0%"
        elif not r.startswith(("+","-")): r = f"+{r}"
        kwargs["rate"] = r
    comm = edge_tts.Communicate(**kwargs)
    out = io.BytesIO()
    async for chunk in comm.stream():
        if chunk[2]: out.write(chunk[2])
    return out.getvalue()

def tts_edge(text: str, voice="en-US-AriaNeural", rate=None):
    try: return asyncio.run(_edge_async(text, voice, rate)), None
    except Exception as e: return None, str(e)

def tts_gtts(text: str, lang="en"):
    if not gTTS: return None, "gTTS not available."
    try:
        buf = io.BytesIO()
        gTTS(text, lang=lang).write_to_fp(buf)
        return buf.getvalue(), None
    except Exception as e: return None, str(e)

def synthesize(text: str, engine: str, lang_code="en"):
    """Returns (audio_bytes, mime, error)"""
    edge_voice_map = {
        "en": "en-US-AriaNeural", "hi": "hi-IN-SwaraNeural", "ta": "ta-IN-PallaviNeural", "bn": "bn-IN-BashkarNeural",
        "es": "es-ES-AlvaroNeural", "fr": "fr-FR-DeniseNeural", "de": "de-DE-KatjaNeural", "ar": "ar-SA-HamedNeural",
        "zh-Hans": "zh-CN-XiaoxiaoNeural", "zh-cn": "zh-CN-XiaoxiaoNeural", "ja": "ja-JP-NanamiNeural",
        "ko": "ko-KR-SunHiNeural", "pt": "pt-PT-FernandaNeural", "it": "it-IT-ElsaNeural", "nl": "nl-NL-ColetteNeural",
        "tr": "tr-TR-AhmetNeural", "ru": "ru-RU-DariyaNeural"
    }
    if engine == "Gemini TTS":
        audio, err = tts_gemini(text, voice_name="Kore")
        return (audio, "audio/mp3", err)
    if engine == "Edge-TTS":
        voice = edge_voice_map.get(lang_code, "en-US-AriaNeural")
        audio, err = tts_edge(text, voice=voice, rate="+0%")
        if audio: return (audio, "audio/mp3", None)
        # Fallback to gTTS if edge_tts fails or is not available
        audio2, err2 = tts_gtts(text, lang=lang_code if lang_code else "en")
        return (audio2, "audio/mp3", err or err2)
    if engine == "gTTS":
        audio, err = tts_gtts(text, lang=lang_code if lang_code else "en")
        return (audio, "audio/mp3", err)
    return (None, None, "Unknown engine")

# -------------------------
# RAG Pipeline with CAG (Caching)
# -------------------------
def rag_pipeline(query, selected_language):
    # 1. CAG Check (Conversation Augmented Generation / Caching)
    cache = st.session_state.get('cache', {})
    if query in cache and (time.time() - cache[query]['timestamp'] < CACHE_EXPIRY_SECONDS):
        st.info("🔄 Serving response from cache (CAG/Cost Reduction).")
        return cache[query]['response']
    
    # 2. RAG Retrieval
    relevant_docs = retrieve_documents(query)
    kb_context = "\n".join(relevant_docs)
    
    # 3. Dynamic System Instruction
    file_count = get_collection().count()
    if file_count == 0 and not relevant_docs:
        system_instruction = (
            f"You are a helpful assistant. No documents have been uploaded. Answer the query generally. "
            f"Your response must be in {selected_language}."
        )
        prompt = query
    else:
        system_instruction = (
            f"You are an expert RAG system. Use the provided context from the uploaded documents to answer the user's question. "
            f"If the context is insufficient, state that you cannot fully answer based on the provided documents. "
            f"Your response must be accurate and formatted in {selected_language}. "
            f"Cite the documents where possible."
        )
        prompt = f"### CONTEXT FROM DOCUMENTS\n{kb_context}\n\n### USER QUESTION: {query}"
        
    # 4. API Call
    response_json = call_gemini_api(prompt, system_instruction=system_instruction)

    if 'error' in response_json:
        answer = f"Generation error: {response_json['error']}"
    else:
        answer = response_json.get('response', "No response text.")

    # 5. CAG Update
    cache[query] = {'response': answer, 'timestamp': time.time()}
    st.session_state.cache = cache
    
    return answer

# -------------------------
# Sidebar and state
# -------------------------
st.sidebar.title("RAG Settings ⚙️")
menu = st.sidebar.radio("Select Module", ["Document Loader", "RAG Chatbot"])

# File Uploader and RAG Status
st.sidebar.markdown("---")
st.sidebar.subheader("Document Uploads")
if 'db_client' not in st.session_state:
    st.session_state.db_client, st.session_state.model, st.session_state.gemini_client = initialize_rag_dependencies()
if 'ingested_files' not in st.session_state:
    st.session_state.ingested_files = []
if 'cache' not in st.session_state:
    st.session_state.cache = {}

kb_count = get_collection().count()
st.sidebar.info(f"Loaded Chunks: {kb_count}")
if st.sidebar.button("Clear RAG Storage"):
    clear_rag_storage()

st.sidebar.markdown("---")
st.sidebar.subheader("Response Options")
resp_mode = st.sidebar.selectbox("Response mode", ["Text", "Voice"])
tts_engine = st.sidebar.selectbox("TTS engine", ["Gemini TTS", "Edge-TTS", "gTTS"])

if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"
lang_display = st.sidebar.selectbox("Answer Language", list(LANGUAGE_DICT.keys()), index=0)
st.session_state.selected_language = lang_display
lang_code = LANGUAGE_DICT.get(lang_display, "en")

# -------------------------
# Modules
# -------------------------

if menu == "Document Loader":
    st.title("Document Loader 📄➡️🧠")
    st.markdown("""
        Upload documents (PDF, TXT, CSV, HTML, XML, etc.) to build the RAG knowledge base.
        Files are processed using **overlapping chunking** for better context retrieval.
        **Note:** For PDF parsing, you may need to install external libraries like `pypdf`.
    """)
    
    uploaded_files = st.file_uploader(
        "Upload Files (PDF, TXT, CSV, HTML, XML, etc.)",
        type=["pdf", "txt", "csv", "html", "xml", "json"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button(f"Process {len(uploaded_files)} File(s) and Ingest"):
            newly_ingested = 0
            ingested_names = set(st.session_state.ingested_files)
            
            with st.spinner("Processing and storing documents..."):
                for uploaded_file in uploaded_files:
                    if uploaded_file.name in ingested_names:
                        st.warning(f"Skipped: '{uploaded_file.name}' already processed.")
                        continue
                        
                    raw_text, success = extract_text_from_upload(uploaded_file)
                    
                    if success:
                        docs = split_documents(raw_text)
                        if docs:
                            newly_ingested += process_and_store_documents(docs)
                            st.session_state.ingested_files.append(uploaded_file.name)
                            st.success(f"Successfully processed '{uploaded_file.name}' into {len(docs)} chunks.")
                        else:
                            st.warning(f"No content found in '{uploaded_file.name}'.")
                    else:
                        st.error(f"Failed to process '{uploaded_file.name}': {raw_text}")
            
            st.success(f"Total new chunks added: {newly_ingested}")
            st.rerun() # Refresh status

    st.markdown("---")
    st.subheader("Currently Ingested Files")
    if st.session_state.ingested_files:
        st.json(st.session_state.ingested_files)
    else:
        st.info("No files have been loaded into the RAG system.")


elif menu == "RAG Chatbot":
    st.markdown("## RAG AI Agent 🧠 (Context-Aware Chat)")
    st.caption(f"""
        **Status:** Loaded Chunks: {kb_count} | **Language:** {st.session_state.selected_language} | **Mode:** {resp_mode} ({tts_engine})
    """)

    if not GEMINI_API_KEY:
        st.error("Set GEMINI_API_KEY in secrets.toml.")
        st.stop()
        
    if "messages_rag" not in st.session_state:
        st.session_state["messages_rag"] = [{"role": "assistant", "content": "Hello! I'm your RAG consultant. Upload files in the 'Document Loader' tab and ask me questions about them."}]
        
    for msg in st.session_state.messages_rag:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages_rag.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            # Ensure the response is quick as requested (within a minute is default for most models)
            with st.spinner("Retrieving context and generating..."):
                start_time = time.time()
                
                answer = rag_pipeline(prompt, st.session_state.selected_language)
                
                duration = time.time() - start_time
                st.write(answer)
                st.caption(f"Generation time: {duration:.2f} seconds.")
                
                if resp_mode == "Voice":
                    with st.spinner("Synthesizing speech..."):
                        audio, mime, err = synthesize(answer, tts_engine, lang_code)
                        if audio:
                            st.audio(io.BytesIO(audio), format=mime)
                        else:
                            st.warning(f"TTS failed: {err}")
                            
                st.session_state.messages_rag.append({"role": "assistant", "content": answer})
