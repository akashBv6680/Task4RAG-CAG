import streamlit as st
import os
from google.genai.errors import APIError
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter

# ----------------- 🛠️ FIX 1: CORRECTED LLM & EMBEDDING IMPORTS 🛠️ -----------------
# Importing from 'llama_index.llms.gemini' and 'llama_index.embeddings.gemini' 
# requires the packages 'llama-index-llms-gemini' and 'llama-index-embeddings-gemini'
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
# ------------------------------------------------------------------------------------

from edge_tts import communicate
import asyncio
import io
import time
from cachetools import LRUCache

# --- 1. Configuration and Caching (Simulating CAG for cost reduction) ---
MODEL_NAME = "gemini-2.5-flash"
EMBED_MODEL = "models/text-embedding-004" 
CHUNK_SIZE = 1024       
CHUNK_OVERLAP = 256     # Overlapping chunking overlap
CACHE_SIZE = 100        # Size for the LRU (Least Recently Used) cache
CACHE_TTL = 3600        # Cache time-to-live in seconds (1 hour)

# Multi-language support dictionary
LANGUAGE_DICT = {
    "English": "en", "Spanish": "es", "Arabic": "ar", "French": "fr", "German": "de", "Hindi": "hi",
    "Tamil": "ta", "Bengali": "bn", "Japanese": "ja", "Korean": "ko", "Russian": "ru",
    "Chinese (Simplified)": "zh-Hans", "Portuguese": "pt", "Italian": "it", "Dutch": "nl", "Turkish": "tr"
}

# Edge-TTS voice mapping (simplified mapping for multiple languages)
TTS_VOICE_MAP = {
    "en": "en-US-Standard-C", "es": "es-ES-ElviraNeural", "fr": "fr-FR-HenriNeural", 
    "hi": "hi-IN-MadhurNeural", "ta": "ta-IN-ValluvarNeural", "ja": "ja-JP-NanamiNeural", 
    "ko": "ko-KR-JiMinNeural", "zh-Hans": "zh-CN-XiaochenNeural", "pt": "pt-PT-FernandaNeural", 
    "ar": "ar-SA-HassanNeural", "de": "de-DE-ConradNeural", "bn": "bn-IN-TanishaNeural", 
    "it": "it-IT-IsabellaNeural", "nl": "nl-NL-ChristelNeural", "tr": "tr-TR-AhmetNeural", 
    "ru": "ru-RU-SvetlanaNeural"
}

# Initialize a simple LRU cache for final responses (CAG cost reduction)
if 'response_cache' not in st.session_state:
    st.session_state.response_cache = LRUCache(maxsize=CACHE_SIZE)

# --- 2. Streamlit UI Components (Top Banner) ---
st.set_page_config(
    page_title="RAG AI Agent with Gemini 2.5 Flash",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📄💬 Multi-Format RAG AI Agent with Gemini 2.5 Flash")
st.markdown("""
### ✨ System Status & Features
- **RAG System:** Supports documents like **PDF, TXT, CSV, HTML, XML, GitHub `.raw` file**, and other formats via `unstructured` and LlamaIndex loaders.
- **Language Support:** **Multi-lingual** RAG response in 17+ languages.
- **Response Mode:** Capacity to respond in **Voice Mode (Text-to-Speech)** using Edge-TTS.
- **Cost Optimization (CAG):** Utilizes **Cache-Augmented Generation** for repeated queries to minimize token cost.
- **Chunking:** Uses **Overlapping Chunking** (Size: $1024$, Overlap: $256$) for improved context retrieval.
- **Model:** `gemini-2.5-flash`
""")
st.divider()

# --- 3. Helper Functions ---
@st.cache_resource
def initialize_llm_and_embedding():
    """Initializes and configures Gemini LLM and Embedding Model via LlamaIndex Settings."""
    if "GEMINI_API_KEY" not in os.environ:
        try:
            # Assumes the key is set in Streamlit Secrets
            os.environ["GEMINI_API_KEY"] = st.secrets["gemini_api_key"]
        except Exception:
            st.error("🚨 Gemini API Key not found in environment variables or Streamlit secrets. Please set it.")
            st.stop()
            
    # LlamaIndex Global Settings
    Settings.llm = Gemini(model=MODEL_NAME, api_key=os.environ["GEMINI_API_KEY"])
    Settings.embed_model = GeminiEmbedding(model_name=EMBED_MODEL, api_key=os.environ["GEMINI_API_KEY"])
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    return Settings.llm, Settings.embed_model

def load_documents(uploaded_files, temp_dir="temp_data"):
    """Saves uploaded files and loads them into LlamaIndex Document objects."""
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Clear previous files to avoid mixing contexts from different runs
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
        
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            # ----------------- 🛠️ FIX 2: CORRECTED FILE READING 🛠️ -----------------
            # Use .getvalue() to get the file's binary content
            f.write(file.getvalue()) 
            # -----------------------------------------------------------------------
            
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

def generate_voice_response(text, lang_code):
    """Generates audio for a given text using Edge-TTS."""
    voice = TTS_VOICE_MAP.get(lang_code, "en-US-Standard-C") # Fallback to English
    
    async def tts_main():
        comm = communicate(text, voice)
        audio_buffer = io.BytesIO()
        async for chunk in comm:
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
        audio_buffer.seek(0)
        return audio_buffer

    try:
        audio_data = asyncio.run(tts_main())
        return audio_data
    except Exception as e:
        # Note: Streamlit's environment can sometimes have issues with Edge-TTS
        st.error(f"TTS Error: Could not generate voice response. {e}")
        return None


# --- 4. Main Application Logic ---
# Initialize LLM/Embedding settings
llm, embed_model = initialize_llm_and_embedding()

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    uploaded_files = st.file_uploader(
        "Upload Documents for RAG (PDF, TXT, CSV, HTML, XML, etc.)",
        accept_multiple_files=True,
        key="file_uploader" 
    )
    
    # Response Mode
    response_mode = st.radio(
        "Select Response Mode:",
        ("Text Only", "Text and Voice"),
        index=0,
        key="response_mode_radio"
    )

    # Multi-language selector
    selected_language = st.selectbox(
        "Select Response Language:",
        options=list(LANGUAGE_DICT.keys()),
        index=0, 
        key="language_select"
    )
    lang_code = LANGUAGE_DICT[selected_language]
    
    st.subheader("RAG Index Status")
    
    if uploaded_files:
        st.success(f"Files uploaded: {len(uploaded_files)}")
        index = get_index(uploaded_files)
        st.session_state.index_built = True
        st.success("✅ RAG Index is Ready/Cached!")
    else:
        st.warning("Please upload documents to build the RAG Index.")
        st.session_state.index_built = False
        index = None

# Main Chat Interface
if st.session_state.get("index_built", False) and index:
    query_engine = index.as_query_engine(
        response_mode="compact", 
        llm=llm,
        streaming=True, 
        system_prompt=f"You are a helpful RAG AI assistant. Answer the user's question based ONLY on the context provided. The final response MUST be in the selected language: {selected_language}. Be concise and respond within a minute."
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

            # TTS Generation
            if response_mode == "Text and Voice" and final_response_text and not final_response_text.startswith("An API Error"):
                audio_buffer = generate_voice_response(final_response_text, lang_code)
                if audio_buffer:
                    st.audio(audio_buffer, format="audio/mp3", autoplay=True)
                    st.info(f"🔊 Speaking response in {selected_language}...")

            # Store final message in history
            st.session_state.messages.append({"role": "assistant", "content": final_response_text})
else:
    st.info("Please upload and process your documents in the sidebar to begin chatting.")

# Next Step Suggestion
st.sidebar.markdown("---")
st.sidebar.markdown("### Next Step")
st.sidebar.info("The application is running! Please upload your documents and try asking a question in any supported language.")
