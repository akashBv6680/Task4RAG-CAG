# app.py
# 📚 RAG AI Agent with Multilingual & Voice Support 🎙️
# Status: RAG System with file support (PDF, TXT, CSV, HTML, XML, GitHub raw).
# Features: Conversation Augmented Generation (CAG) for cost-saving, Voice Response Mode, Multilingual.
import streamlit as st
import os, sys, tempfile, uuid, time, io, asyncio, datetime, re, base64
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import torch

# RAG dependencies
import chromadb
# Note: Ensure sentence-transformers is installed for this to work
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("Please install sentence-transformers: pip install sentence-transformers")
    SentenceTransformer = None
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
# CAG/Caching setting: Cache hit if the exact query is repeated within 5 minutes
CACHE_EXPIRY_SECONDS = 300 

# -------------------------
# Cached init (Minimal - only for Gemini client and RAG components)
# -------------------------
@st.cache_resource(show_spinner=False)
def initialize_rag_dependencies():
    if not SentenceTransformer:
        return None, None, None

    # Use disk storage for persistence across re-runs
    db_path = os.path.join(tempfile.gettempdir(), "chroma_db_rag")
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    db_client = chromadb.PersistentClient(path=db_path)
    
    # Use a small, fast embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu') 
    
    if GEMINI_API_KEY and genai:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        gemini_client = None
        
    return db_client, model, gemini_client

@st.cache_resource(show_spinner=False)
def initialize_gemini_client():
    if GEMINI_API_KEY and genai:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        gemini_client = None
    return gemini_client

# -------------------------
# RAG and Storage Helpers
# -------------------------
def get_collection():
    if 'db_client' not in st.session_state:
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client = initialize_rag_dependencies()
        if not st.session_state.db_client:
            return None
    try:
        # Tries to get the collection, creates it if it doesn't exist
        return st.session_state.db_client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as e:
        st.error(f"Error accessing ChromaDB: {e}")
        return None

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
    if not collection: return 0

    model = st.session_state.model
    
    # Generate embeddings in batches for efficiency
    embeddings = model.encode(documents, convert_to_tensor=False).tolist()
    ids = [str(uuid.uuid4()) for _ in documents]
    
    # Store documents
    collection.add(documents=documents, embeddings=embeddings, ids=ids)
    return len(documents)

def retrieve_documents(query, n_results=5):
    collection = get_collection()
    if not collection or collection.count() == 0: return []

    model = st.session_state.model
    q_emb = model.encode(query).tolist()
    
    # Query the collection
    results = collection.query(query_embeddings=q_emb, n_results=n_results, include=['documents', 'distances'])
    return results['documents'][0] if results['documents'] else []

# --- Document Loading Placeholder ---
def extract_text_from_upload(uploaded_file):
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    raw_text = ""
    
    # Simple reader for text-like files
    text_types = ["text/plain", "application/csv", "text/csv", "application/json", 
                  "text/html", "application/xml", "text/xml"]
    
    if any(ft in file_details["FileType"] for ft in text_types) or uploaded_file.name.endswith(('.txt', '.csv', '.html', '.xml', '.json', '.raw')):
        try:
            raw_text = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            raw_text = uploaded_file.getvalue().decode("latin-1") # Fallback for odd encodings
    
    # Placeholder for PDF. Requires libraries like 'pypdf'
    elif file_details["FileType"] == "application/pdf":
        try:
            # THIS IS A PLACEHOLDER. User must install pypdf
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            for page in reader.pages:
                raw_text += page.extract_text() or ""
        except ImportError:
            return f"Error: Cannot read PDF. Please install 'pypdf' (pip install pypdf). Content placeholder: {uploaded_file.name}", False
        except Exception as e:
             return f"Error reading PDF: {e}", False
    
    # Basic attempt for other binary types
    else:
        return f"File type {file_details['FileType']} not explicitly supported or requires external tools.", False

    return raw_text, True

def clear_rag_storage():
    db_client = st.session_state.db_client if 'db_client' in st.session_state else initialize_rag_dependencies()[0]
    if db_client:
        try:
            db_client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        get_collection() # Re-create the collection
        st.session_state.ingested_files = []
        st.session_state.cache = {}
        st.rerun()

# -------------------------
# Utility functions (Generic model call)
# -------------------------
def call_gemini_api(prompt, model_name="gemini-2.5-flash", system_instruction="You are a helpful assistant.", max_retries=5):
    gemini_client = initialize_gemini_client()
    if not gemini_client:
        return {"error": "Gemini client not configured."}
    
    retry_delay = 1
    for i in range(max_retries):
        try:
            cfg = types.GenerateContentConfig(system_instruction=system_instruction)
            resp = gemini_client.models.generate_content(model=model_name, contents=prompt, config=cfg)
            return {"response": resp.text}
        except APIError as e:
            if i < max_retries - 1:
                time.sleep(retry_delay); retry_delay *= 2; continue
            return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

# -------------------------
# TTS utilities (retained)
# -------------------------

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
    
    if engine == "Edge-TTS":
        voice = edge_voice_map.get(lang_code, "en-US-AriaNeural")
        audio, err = tts_edge(text, voice=voice, rate="+0%")
        if audio: return (audio, "audio/mp3", None)
        audio2, err2 = tts_gtts(text, lang=lang_code if lang_code else "en")
        return (audio2, "audio/mp3", err or err2)
    if engine == "gTTS":
        audio, err = tts_gtts(text, lang=lang_code if lang_code else "en")
        return (audio, "audio/mp3", err)
    return (None, None, "Unknown engine")

# -------------------------
# RAG Pipeline with CAG (Caching) - THE KEY IMPLEMENTATION
# -------------------------
def rag_pipeline(query, selected_language):
    # 1. CAG Check (Conversation Augmented Generation / Response Caching)
    cache = st.session_state.get('cache', {})
    if query in cache and (time.time() - cache[query]['timestamp'] < CACHE_EXPIRY_SECONDS):
        st.info("🔄 Serving response from cache (CAG/Cost Reduction).")
        return cache[query]['response']
    
    # 2. RAG Retrieval (Skipped if cache hit)
    relevant_docs = retrieve_documents(query)
    kb_context = "\n".join(relevant_docs)
    
    # 3. Dynamic System Instruction
    file_count = get_collection().count() if get_collection() else 0
    if file_count == 0 and not relevant_docs:
        system_instruction = (
            f"You are a helpful assistant. No documents have been uploaded. Answer the query generally. "
            f"Your response must be concise and in {selected_language}."
        )
        prompt = query
    else:
        system_instruction = (
            f"You are an expert RAG system. Use the provided context from the uploaded documents to answer the user's question. "
            f"If the context is insufficient, state that you cannot fully answer based on the provided documents. "
            f"Your response must be accurate, concise, and formatted in {selected_language}. "
            f"Do NOT invent information outside of the context."
        )
        prompt = f"### CONTEXT FROM DOCUMENTS\n{kb_context}\n\n### USER QUESTION: {query}"
        
    # 4. API Call (Skipped if cache hit)
    response_json = call_gemini_api(prompt, model_name="gemini-2.5-flash", system_instruction=system_instruction)

    if 'error' in response_json:
        answer = f"Generation error: {response_json['error']}"
    else:
        answer = response_json.get('response', "No response text.")

    # 5. CAG Update (Store new answer for future cache hits)
    cache[query] = {'response': answer, 'timestamp': time.time()}
    st.session_state.cache = cache
    
    return answer

# -------------------------
# Sidebar and state
# -------------------------
st.sidebar.title("RAG Settings ⚙️")
menu = st.sidebar.radio("Select Module", ["Document Loader", "RAG Chatbot", "TTS Demo (Standalone)"])

# File Uploader and RAG Status
st.sidebar.markdown("---")
st.sidebar.subheader("Document Uploads")
# Initialize RAG components if not present
if 'db_client' not in st.session_state:
    st.session_state.db_client, st.session_state.model, st.session_state.gemini_client = initialize_rag_dependencies()
if 'ingested_files' not in st.session_state:
    st.session_state.ingested_files = []
if 'cache' not in st.session_state:
    st.session_state.cache = {}

kb_count = get_collection().count() if get_collection() else 0
st.sidebar.info(f"Loaded Chunks: {kb_count}")
st.sidebar.caption(f"CAG Cache Size: {len(st.session_state.cache)} entries")

if st.sidebar.button("Clear RAG Storage & Cache"):
    clear_rag_storage()

st.sidebar.markdown("---")
st.sidebar.subheader("Response Options")
resp_mode = st.sidebar.selectbox("Response mode", ["Text", "Voice"])
tts_engine = st.sidebar.selectbox("TTS engine", ["Edge-TTS", "gTTS"])

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
    st.markdown("## RAG System Status: Files Ingestion and Chunking")
    st.caption("Upload documents (PDF, TXT, CSV, HTML, XML, etc.) to build the RAG knowledge base. Files are processed using **overlapping chunking** for better context retrieval. **Note:** For PDF/CSV support, check `requirements.txt`.")
    
    uploaded_files = st.file_uploader(
        "Upload Files (PDF, TXT, CSV, HTML, XML, JSON, etc.)",
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
        # Display audio if it exists for this message
        if "audio" in msg and msg["audio"]:
            st.audio(io.BytesIO(base64.b64decode(msg["audio"])), format="audio/mp3")
        
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages_rag.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            # Ensure the response is quick as requested (within a minute is default for most models)
            audio = None
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
                            
            # **Corrected Indentation and Dict Update**
            new_assistant_msg = {"role": "assistant", "content": answer}
            if audio:
                new_assistant_msg["audio"] = base64.b64encode(audio).decode('utf-8')
                
            st.session_state.messages_rag.append(new_assistant_msg)
            
elif menu == "TTS Demo (Standalone)":
    st.title("Text-to-Speech Demo 🔊")
    st.info("Uses the selected TTS engine and language from the sidebar.")
    tts_text = st.text_area("Text to convert to speech", "This is a demonstration of the text-to-speech capabilities using the selected engine and language settings from the sidebar. You requested Edge-TTS and multilingual support.", height=150)
    if st.button("Generate Speech"):
        if tts_text.strip():
            with st.spinner(f"Generating audio with {tts_engine} in {st.session_state.selected_language}..."):
                audio, mime, err = synthesize(tts_text, tts_engine, lang_code)
                if audio:
                    st.audio(io.BytesIO(audio), format=mime)
                    st.success("Playback complete.")
                else:
                    st.error(f"TTS Generation Failed: {err}")
        else:
            st.warning("Please enter some text for TTS.")
