import streamlit as st
import os, sys, tempfile, uuid, time, re, io, asyncio, datetime
from typing import Dict, Any
from cachetools import LRUCache 

# LlamaIndex Dependencies
from google.genai.errors import APIError 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# TTS dependencies (ensure 'pip install edge-tts' is run)
try:
    import edge_tts
except ImportError:
    edge_tts = None


# --- 1. Configuration and Caching ---
MODEL_NAME = "gemini-2.5-flash"
EMBED_MODEL = "models/text-embedding-004" 
CHUNK_SIZE = 1024       
CHUNK_OVERLAP = 256     
CACHE_SIZE = 100        
CACHE_TTL = 3600        

# Multi-language support dictionary
LANGUAGE_DICT = {
    "English": "en", "Hindi (हिन्दी)": "hi", "Tamil (தமிழ்)": "ta", "Spanish": "es", 
    "French": "fr", "German": "de", "Arabic": "ar", "Bengali (বাংলা)": "bn", 
    "Japanese (日本語)": "ja", "Korean (한국어)": "ko", "Russian (русский)": "ru",
    "Chinese (Simplified)": "zh-Hans", "Portuguese": "pt", "Italian": "it", 
    "Dutch": "nl", "Turkish": "tr"
}

# Edge-TTS Voice Map
TTS_VOICE_MAP = {
    "en": "en-US-AriaNeural",      # English
    "es": "es-ES-ElviraNeural",      # Spanish
    "fr": "fr-FR-HenriNeural",      # French
    "hi": "hi-IN-SwaraNeural",      # Hindi
    "ta": "ta-IN-ValluvarNeural",    # Tamil
    "ja": "ja-JP-NanamiNeural",      # Japanese
    "ko": "ko-KR-JiMinNeural",      # Korean
    "zh-Hans": "zh-CN-XiaoxiaoNeural", # Simplified Chinese
    "pt": "pt-PT-FernandaNeural",    # Portuguese
    "ar": "ar-SA-HamedNeural",      # Arabic
    "de": "de-DE-KatjaNeural",      # German
    "bn": "bn-IN-BashkarNeural",    # Bengali
    "it": "it-IT-ElsaNeural",      # Italian
    "nl": "nl-NL-ColetteNeural",    # Dutch
    "tr": "tr-TR-AhmetNeural",      # Turkish
    "ru": "ru-RU-DariyaNeural"      # Russian
}

# Initialize a simple LRU cache for final responses (CAG cost reduction)
if 'response_cache' not in st.session_state:
    st.session_state.response_cache = LRUCache(maxsize=CACHE_SIZE)

# --- 2. Streamlit UI Components (Top Banner) ---
st.set_page_config(
    page_title="Multilingual RAG AI Agent with Gemini 2.5 Flash",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📄💬 Multi-Lingual RAG AI Agent with Gemini 2.5 Flash")
st.markdown("""
### ✨ System Status & Features
- **RAG System:** Supports diverse document formats.
- **Language Support:** **Multi-lingual** RAG response in 17+ languages, controlled by user selection.
- **Voice Mode:** Supports **Text-to-Speech (TTS)** using Edge-TTS in the selected language.
""")
st.divider()

# --- 3. Helper Functions ---
@st.cache_resource
def initialize_llm_and_embedding():
    """Initializes and configures Gemini LLM and Embedding Model via LlamaIndex Settings."""
    if "GEMINI_API_KEY" not in os.environ:
        try:
            # Check Streamlit secrets
            os.environ["GEMINI_API_KEY"] = st.secrets["gemini_api_key"]
        except Exception:
            st.error("🚨 Gemini API Key not found in environment variables or Streamlit secrets. Please set it.")
            st.stop()
            
    # Configure LlamaIndex global settings
    Settings.llm = Gemini(model=MODEL_NAME, api_key=os.environ["GEMINI_API_KEY"])
    Settings.embed_model = GeminiEmbedding(model_name=EMBED_MODEL, api_key=os.environ["GEMINI_API_KEY"])
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    return Settings.llm, Settings.embed_model

def load_documents(uploaded_files, temp_dir="temp_data"):
    """Saves uploaded files and loads them into LlamaIndex Document objects."""
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
        
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue()) 
            
    loader = SimpleDirectoryReader(input_dir=temp_dir, recursive=True)
    documents = loader.load_data()
    return documents

@st.cache_resource(hash_funcs={"_uploaded_files_hash": lambda x: tuple(f.size for f in x)}, ttl=CACHE_TTL)
def get_index(uploaded_files):
    """Creates or updates a VectorStoreIndex from uploaded documents."""
    if not uploaded_files:
        return None
    
    with st.spinner("⏳ Creating/Updating Vector Store Index..."):
        documents = load_documents(uploaded_files)
        index = VectorStoreIndex.from_documents(documents)
        return index

# Edge-TTS Functions
async def _edge_async(text: str, voice: str, rate="+0%"):
    """Asynchronous Edge TTS worker."""
    if not edge_tts:
        return None
    
    kwargs = {"text": text, "voice": voice, "rate": rate}
    comm = edge_tts.Communicate(**kwargs)
    out = io.BytesIO()
    
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            out.write(chunk["data"])
    
    if out.tell() == 0:
        return None
        
    out.seek(0)
    return out.getvalue()

def generate_voice_response(text: str, lang_code: str):
    """
    Generates audio using Edge-TTS in the specified language voice.
    Returns (audio_bytes, error_message)
    """
    if not edge_tts:
        return None, "Edge-TTS dependency not found. Please install 'edge-tts'."
    
    voice = TTS_VOICE_MAP.get(lang_code, "en-US-AriaNeural")
    
    try:
        audio_data = asyncio.run(_edge_async(text, voice))
        
        if audio_data is None:
            return None, f"No audio was received. Please verify that the voice '{voice}' supports the generated language text."
            
        return audio_data, None
    except Exception as e:
        return None, str(e)


# --- 4. Main Application Logic ---
# Initialize LLM/Embedding settings
llm, embed_model = initialize_llm_and_embedding()

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ RAG Settings")
    
    uploaded_files = st.file_uploader(
        "1. Upload Documents (PDF, TXT, CSV, etc.)",
        accept_multiple_files=True,
        key="file_uploader" 
    )
    
    # Multi-language selector
    selected_language = st.selectbox(
        "2. Select Response Language:",
        options=list(LANGUAGE_DICT.keys()),
        index=0, 
        key="language_select"
    )
    # Get the language code (e.g., 'hi' for Hindi)
    lang_code = LANGUAGE_DICT[selected_language]
    
    # Response Mode
    response_mode = st.radio(
        "3. Select Response Mode:",
        ("Text Only", "Text and Voice"),
        index=0,
        key="response_mode_radio"
    )

    st.subheader("Index Status")
    
    index = None
    if uploaded_files:
        st.success(f"Files uploaded: {len(uploaded_files)}")
        # Build/get the index (cached)
        index = get_index(uploaded_files)
        st.session_state.index_built = True
        st.success("✅ RAG Index is Ready/Cached!")
    else:
        st.warning("Please upload documents to build the RAG Index.")
        st.session_state.index_built = False


# Main Chat Interface
if st.session_state.get("index_built", False) and index:
    
    # --- Dynamic System Prompt ---
    # THIS LINE ENSURES THE LLM RESPONDS IN THE SELECTED LANGUAGE
    system_prompt_template = (
        f"You are a helpful RAG AI assistant. Answer the user's question based ONLY on the context provided. "
        f"The final response MUST be in the selected language: {selected_language}. "
        f"Be concise and respond within a minute."
    )
    
    query_engine = index.as_query_engine(
        response_mode="compact", 
        llm=llm,
        streaming=True, 
        system_prompt=system_prompt_template
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate agent response
        with st.chat_message("assistant"):
            
            # CAG: Check cache first based on query and language
            cache_key = (prompt, lang_code)
            cached_response = st.session_state.response_cache.get(cache_key)
            final_response_text = ""

            if cached_response:
                st.info("💡 **Cache Hit (CAG)**: Returning cached answer to save tokens.")
                final_response_text = cached_response
                st.markdown(final_response_text) 
            else:
                with st.spinner(f"Thinking in {selected_language} (RAG Query)..."):
                    try:
                        # Stream the RAG response
                        response_stream = query_engine.query(prompt)
                        
                        response_placeholder = st.empty()
                        for token in response_stream.response_gen:
                            final_response_text += token
                            response_placeholder.markdown(final_response_text + "▌")
                        response_placeholder.markdown(final_response_text)
                        
                        # Store in cache (CAG)
                        st.session_state.response_cache[cache_key] = final_response_text

                    except APIError as e:
                        final_response_text = f"An API Error occurred. Please check your Gemini API key. Details: {e}"
                        st.error(final_response_text)
                    except Exception as e:
                        final_response_text = f"An unexpected error occurred: {e}"
                        st.error(final_response_text)

            # TTS Generation (only if edge_tts is installed and mode is selected)
            if response_mode == "Text and Voice" and edge_tts and final_response_text and not final_response_text.startswith("An API Error"):
                audio_buffer, err = generate_voice_response(final_response_text, lang_code)
                if audio_buffer:
                    st.audio(io.BytesIO(audio_buffer), format="audio/mp3", autoplay=True)
                    st.info(f"🔊 Speaking response in {selected_language}...")
                else:
                    st.warning(f"TTS failed: {err}. If the issue persists, the TTS service may be temporarily unavailable or the specific voice may not support the generated text.")

            # Store final message in history
            st.session_state.messages.append({"role": "assistant", "content": final_response_text})
else:
    st.info("Please upload your documents in the sidebar (step 1) and confirm the index is ready before chatting.")

# Next Step Suggestion
st.sidebar.markdown("---")
st.sidebar.markdown("### Next Step")
st.sidebar.info("The application is now fully configured for multilingual RAG responses! Try uploading a file and select any language like **Tamil (தமிழ்)** or **Hindi (हिन्दी)**.")
