import streamlit as st
import os
import sys
import tempfile
import uuid
import time
import io
import asyncio
import hashlib
import requests # Added for general use, if needed later
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
from io import BytesIO

# =====================================================================
# FIX: SQLITE3 PATCH for Streamlit Cloud (Ensures ChromaDB works)
# =====================================================================
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass

# --- Core RAG Imports ---
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FIX: Using pypdf instead of deprecated PyPDF2
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# --- Google Gemini Imports ---
try:
    from google import genai
    from google.genai.errors import APIError
    from google.genai import types
except ImportError:
    genai = None
    APIError = None
    types = None

# --- TTS Imports ---
try:
    from edge_tts import Communicate
except ImportError:
    Communicate = None
try:
    from gtts import gTTS
except ImportError:
    gTTS = None

# --- Configuration ---
st.set_page_config(page_title="📄 Task 4 RAG - Multilingual", page_icon="🚀", layout="wide")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
GEMINI_MODEL_ID = "gemini-2.5-flash"
COLLECTION_NAME = "rag_documents"

# --- Language Mappings ---
LANGUAGE_DICT = {
    "English": "en", "Hindi (हिन्दी)": "hi", "Tamil (தமிழ்)": "ta", "Bengali (বাংলা)": "bn",
    "Spanish (Español)": "es", "French (Français)": "fr", "German (Deutsch)": "de", "Arabic (العربية)": "ar",
    "Japanese (日本語)": "ja", "Korean (한국어)": "ko", "Russian (Русский)": "ru",
    "Chinese (Simplified)": "zh-Hans", "Portuguese": "pt", "Italian (Italiano)": "it",
    "Dutch (Nederlands)": "nl", "Turkish (Türkçe)": "tr"
}

EDGE_VOICE_MAP = {
    "en": "en-US-AriaNeural", "hi": "hi-IN-SwaraNeural", "ta": "ta-IN-ValluvarNeural",
    "bn": "bn-IN-BashkarNeural", "es": "es-ES-AlvaroNeural", "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural", "ar": "ar-SA-HamedNeural", "zh-Hans": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural", "ko": "ko-KR-SunHiNeural", "pt": "pt-PT-FernandaNeural",
    "it": "it-IT-ElsaNeural", "nl": "nl-NL-ColetteNeural", "tr": "tr-TR-AhmetNeural", "ru": "ru-RU-DariyaNeural"
}


# =====================================================================
# SESSION STATE & DEPENDENCY INITIALIZATION
# =====================================================================

if 'selected_language' not in st.session_state: st.session_state.selected_language = "English"
if 'messages_rag' not in st.session_state: st.session_state.messages_rag = []
if 'query_cache' not in st.session_state: st.session_state.query_cache = {}
if 'db_client' not in st.session_state: st.session_state.db_client = None
if 'gemini_client' not in st.session_state: st.session_state.gemini_client = None
if 'model' not in st.session_state: st.session_state.model = None
if 'google_search_tool' not in st.session_state: st.session_state.google_search_tool = None

@st.cache_resource(show_spinner=False)
def initialize_rag_dependencies():
    if not GEMINI_API_KEY:
        st.warning("🚨 GEMINI_API_KEY is not set.")
        st.stop()
        
    db_path = tempfile.mkdtemp()
    db_client = chromadb.PersistentClient(path=db_path)
    model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
    
    if genai:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        google_search_tool = [types.Tool(google_search={})]
    else:
        gemini_client = None
        google_search_tool = None
        
    return db_client, model, gemini_client, google_search_tool

# =====================================================================
# RAG CORE FUNCTIONS
# =====================================================================

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

def split_documents(text_data, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, is_separator_regex=False)
    return splitter.split_text(text_data)

def get_collection():
    if st.session_state.db_client is None:
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client, st.session_state.google_search_tool = initialize_rag_dependencies()
    return st.session_state.db_client.get_or_create_collection(name=COLLECTION_NAME)

def process_and_store_documents(documents):
    collection = get_collection()
    model = get_embedding_model()
    embeddings = model.encode(documents).tolist()
    ids = [str(uuid.uuid4()) for _ in documents]
    collection.add(documents=documents, embeddings=embeddings, ids=ids)

def retrieve_documents(query, n_results=5):
    collection = get_collection()
    model = get_embedding_model()
    
    if collection.count() == 0:
        return []
        
    q_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=n_results, include=['documents', 'distances'])
    return results['documents'][0] if results.get('documents') and results['documents'] else []

# FIX: Increased max_retries to 6 to mitigate API timeout errors
def call_gemini_api(prompt, model_name=GEMINI_MODEL_ID, system_instruction="You are helpful.", tools=None, max_retries=6): 
    if not st.session_state.get('gemini_client'):
        return {"error": "Gemini client not configured"}
    
    retry_delay = 1
    for i in range(max_retries):
        try:
            cfg = types.GenerateContentConfig(system_instruction=system_instruction, tools=tools)
            resp = st.session_state.gemini_client.models.generate_content(model=model_name, contents=prompt, config=cfg)
            return {"response": resp.text}
        except APIError as e:
            if i < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            return {"error": f"API Error after multiple retries: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

# =====================================================================
# Caching (CAG) Functions
# =====================================================================

def get_query_hash(query: str, lang_code: str) -> str:
    query_key = f"{query.lower().strip()}_{lang_code}"
    return hashlib.md5(query_key.encode()).hexdigest()

def check_cache(query: str, lang_code: str):
    cache_key = get_query_hash(query, lang_code)
    return st.session_state.query_cache.get(cache_key)

def store_cache(query: str, lang_code: str, response: str):
    cache_key = get_query_hash(query, lang_code)
    st.session_state.query_cache[cache_key] = response

def rag_pipeline(query: str, selected_language: str) -> str:
    lang_code = LANGUAGE_DICT.get(selected_language, "en")
    cached_response = check_cache(query, lang_code)
    
    if cached_response:
        st.info("📦 Cached (CAG)")
        return cached_response
        
    relevant_docs = retrieve_documents(query, n_results=5)
    
    if relevant_docs and len(relevant_docs) >= 1: # Reduced to 1 for flexibility
        kb_context = "\n".join(relevant_docs)
        system_instruction = f"You are a RAG assistant. Answer ONLY based on context.\n🚨 CRITICAL: Respond ONLY in {selected_language}.\nDo NOT use English. Be concise."
        prompt = f"Context:\n{kb_context}\n\nQuestion: {query}\n\nAnswer in {selected_language}:"
        response_json = call_gemini_api(prompt, system_instruction=system_instruction)
    else:
        # Fallback to general knowledge/search if no relevant RAG context found
        system_instruction = f"You are helpful. Use Google Search if needed.\n🚨 Respond ONLY in {selected_language}. Provide citations."
        prompt = f"Answer in {selected_language}: {query}"
        response_json = call_gemini_api(prompt, system_instruction=system_instruction, tools=st.session_state.google_search_tool)
        
    if 'error' in response_json:
        return f"Error: {response_json['error']}"
        
    answer = response_json.get('response', 'No response')
    store_cache(query, lang_code, answer)
    return answer

# =====================================================================
# TTS Functions
# =====================================================================

async def _edge_async(text: str, voice="en-US-AriaNeural", rate=None):
    if not Communicate: return None
    kwargs = {"text": text, "voice": voice}
    comm = Communicate(**kwargs)
    out = io.BytesIO()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            out.write(chunk["data"])
    return out.getvalue()

def tts_edge(text: str, voice="en-US-AriaNeural"):
    try:
        return asyncio.run(_edge_async(text, voice)), None
    except Exception as e:
        return None, str(e)

def tts_gtts(text: str, lang="en"):
    if not gTTS: return None, "gTTS not available"
    try:
        buf = io.BytesIO()
        gTTS(text, lang=lang).write_to_fp(buf)
        return buf.getvalue(), None
    except Exception as e:
        return None, str(e)

def synthesize(text: str, engine: str, lang_code="en"):
    voice = EDGE_VOICE_MAP.get(lang_code, "en-US-AriaNeural")
    
    if engine == "Edge-TTS":
        audio, err = tts_edge(text, voice=voice)
        if audio:
            return (audio, "audio/mp3", None)
        # Fallback to gTTS if Edge-TTS fails
        audio2, err2 = tts_gtts(text, lang=lang_code)
        return (audio2, "audio/mp3", err or err2)
    elif engine == "gTTS":
        audio, err = tts_gtts(text, lang=lang_code)
        return (audio, "audio/mp3", err)
        
    return (None, None, "Unknown engine")

# =====================================================================
# Streamlit UI
# =====================================================================

def main_ui():
    
    # Initialize Dependencies
    if st.session_state.db_client is None:
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client, st.session_state.google_search_tool = initialize_rag_dependencies()

    st.title("📄 Task 4 RAG Chatbot - Multilingual & Voice 🎙️")
    st.markdown("### ✨ Features:\n- **📁 RAG:** PDF, TXT, CSV support\n- **🌍 Multilingual:** 15+ languages\n- **🎙️ Voice:** Text-to-Speech\n- **💾 CAG:** Query caching\n- **⚡ Fast:** Mitigating long response times with increased retries")
    st.divider()

    index = None
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # --- File Upload and Indexing ---
        st.subheader("📁 Step 1: Upload Documents")
        uploaded_files = st.file_uploader("Upload files (PDF, TXT, CSV)", accept_multiple_files=True, key="file_uploader")
        
        # Process files when uploaded
        if uploaded_files:
            st.success(f"✅ Files: {len(uploaded_files)}")
            with st.spinner("Creating index..."):
                docs = []
                for file in uploaded_files:
                    try:
                        file.seek(0)
                        if file.name.lower().endswith(".pdf") and PdfReader:
                            reader = PdfReader(file)
                            text = "\n".join([page.extract_text() or "" for page in reader.pages])
                        elif file.name.lower().endswith(".csv"):
                            df = pd.read_csv(file)
                            text = df.to_markdown(index=False)
                        elif file.name.lower().endswith((".txt", ".md")):
                            text = file.read().decode('utf-8')
                        else:
                            # Catch-all for simple text content
                            text = file.read().decode('utf-8', errors='ignore')

                        if text:
                            docs.extend(split_documents(text))
                    except Exception as e:
                        st.warning(f"Failed to process {file.name}: {e}")
                        
                if docs:
                    process_and_store_documents(docs)
                    st.success(f"✅ Indexed {len(docs)} chunks")
                else:
                    st.warning("No usable text extracted from uploaded files.")
        
        index = get_collection()
        
        if index and index.count() > 0:
            st.success(f"📚 Index Ready! Chunks: {index.count()}")
        else:
            st.warning("⚠️ Upload documents first")
            
        # --- Language and Voice Settings ---
        st.subheader("🌍 Step 2: Select Language")
        selected_language = st.selectbox(f"Language ({len(LANGUAGE_DICT)} Options):", list(LANGUAGE_DICT.keys()), index=0, key="lang_select")
        st.session_state.selected_language = selected_language
        lang_code = LANGUAGE_DICT.get(selected_language, "en")
        st.info(f"🔤 {selected_language} | 🏷️ `{lang_code}`")
        
        st.subheader("🎤 Step 3: Response Mode")
        response_mode = st.radio("Choose format:", ("📝 Text Only", "🎤 Text + Voice"), index=0)
        tts_engine = st.selectbox("TTS Engine:", ["Edge-TTS", "gTTS"])
        st.divider()
        
        # --- Status ---
        st.subheader("💾 Cache Stats")
        st.metric("Cached Queries", len(st.session_state.query_cache))
        st.metric("Total Messages", len(st.session_state.messages_rag))
        
    # --- Chat Interface ---
    if index and index.count() > 0:
        if not st.session_state.messages_rag:
            st.session_state.messages_rag = [{"role": "assistant", "content": f"👋 Hello! I'm your RAG assistant. Ask me about your documents in **{selected_language}**!"}]
            
        for msg in st.session_state.messages_rag:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages_rag.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = rag_pipeline(prompt, st.session_state.selected_language)
                    
                st.markdown(answer)
                
                if "Text + Voice" in response_mode:
                    audio, mime, err = synthesize(answer, tts_engine, lang_code)
                    if audio:
                        st.audio(io.BytesIO(audio), format=mime)
                    else:
                        st.warning(f"🔊 TTS failed: {err}")
                        
                st.session_state.messages_rag.append({"role": "assistant", "content": answer})
    else:
        st.info("📂 Please upload documents in the sidebar to start the chat.")

if __name__ == "__main__":
    main_ui()
