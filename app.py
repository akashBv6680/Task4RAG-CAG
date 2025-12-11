import streamlit as st
import os
import sys
import tempfile
import uuid
import requests
import time
from datetime import datetime
import re
import shutil
import pandas as pd
from bs4 import BeautifulSoup
import io

# =====================================================================
# FIX 1: SQLITE3 PATCH (for Streamlit Cloud)
# =====================================================================
# MUST be at the very top before any other imports that might use sqlite3
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass

# =====================================================================
# TTS & CAG Imports
# =====================================================================
import asyncio
import edge_tts
import nest_asyncio # <-- CRITICAL FIX for TTS in Streamlit

# =====================================================================
# FIX 2: ASYNCIO PATCH (for Streamlit compatibility)
# Apply the nest_asyncio patch immediately to allow nested asyncio.run calls
# =====================================================================
try:
    nest_asyncio.apply()
except Exception:
    # Ignore if already applied or if environment doesn't support it (unlikely)
    pass 

# --- Core RAG Imports ---
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# --- Google Gemini Imports ---
from google import genai
from google.genai.errors import APIError

# --- LangSmith Imports (Keep for tracing) ---
from langsmith import traceable, tracing_context

# --- Constants and Configuration ---
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

# *** CRITICAL: Read API Key from Streamlit Secrets ***
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
GEMINI_MODEL_ID = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    st.warning("🚨 GEMINI_API_KEY is not set in Streamlit Secrets. Please set it to proceed.")
    st.stop()
    
# Dictionary of supported languages and their ISO 639-1 codes for the LLM
LANGUAGE_DICT = {
    "English": "en", "Spanish": "es", "Arabic": "ar", "French": "fr", 
    "German": "de", "Hindi": "hi", "Tamil": "ta", "Bengali": "bn", 
    "Japanese": "ja", "Korean": "ko", "Russian": "ru", 
    "Chinese (Simplified)": "zh-Hans", "Portuguese": "pt", 
    "Italian": "it", "Dutch": "nl", "Turkish": "tr"
}

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================
if 'selected_language' not in st.session_state: st.session_state['selected_language'] = 'English'
if 'messages' not in st.session_state: st.session_state.messages = []
if 'chat_history' not in st.session_state: st.session_state.chat_history = {}
if 'current_chat_id' not in st.session_state: st.session_state.current_chat_id = None
if 'response_mode' not in st.session_state: st.session_state['response_mode'] = 'Text' # 'Text' or 'Voice'
if 'tts_audio_bytes' not in st.session_state: st.session_state['tts_audio_bytes'] = None


@st.cache_resource
def initialize_dependencies():
    """
    Initializes and returns the ChromaDB client, SentenceTransformer model,
    and the Google GenAI Client.
    """
    try:
        # 1. Initialize ChromaDB in a temporary directory
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)
        
        # 2. Initialize Sentence Transformer (for embeddings)
        model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
        
        # 3. Initialize Google GenAI Client (for LLM)
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        
        return db_client, model, gemini_client
    except Exception as e:
        st.error(f"An error occurred during dependency initialization. Error: {e}")
        st.stop()
        
def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

# --- TTS Function ---
async def generate_tts(text, lang_code):
    """Generates speech audio from text using Edge-TTS."""
    # Mapping ISO 639-1 to a relevant Edge-TTS voice (simplified selection)
    voice_map = {
        "en": "en-US-JennyNeural", "es": "es-ES-ElviraNeural", 
        "fr": "fr-FR-DeniseNeural", "de": "de-DE-KatjaNeural", 
        "hi": "hi-IN-SwaraNeural", "ta": "ta-IN-ValluvarNeural",
        "ja": "ja-JP-NanamiNeural", "ko": "ko-KR-SunHiNeural",
        "pt": "pt-PT-FernandaNeural", "it": "it-IT-IsabellaNeural",
        "ar": "ar-SA-ZariyahNeural", "zh-Hans": "zh-CN-XiaochenNeural",
        "ru": "ru-RU-SvetlanaNeural", "bn": "bn-IN-TanishaaNeural",
        "nl": "nl-NL-ChristelNeural", "tr": "tr-TR-EmelNeural"
    }
    voice = voice_map.get(lang_code, "en-US-JennyNeural")
    
    communicate = edge_tts.Communicate(text, voice)
    audio_buffer = io.BytesIO()
    
    # Write the audio stream to the buffer
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["audio"])
    
    return audio_buffer.getvalue()

# --- Gemini API Call ---
@traceable(run_type="llm")
def call_gemini_api(prompt, max_retries=3):
    """
    Calls the Google Gemini API for text generation.
    """
    gemini_client = st.session_state.gemini_client
    
    retry_delay = 1
    for i in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL_ID,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                )
            )
            return response.text.strip()
            
        except APIError as e:
            st.warning(f"Gemini API error (try {i+1}/{max_retries}). Retrying in {retry_delay} seconds. Error: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2
        except Exception as e:
            st.error(f"An unexpected error occurred during the API call: {e}")
            return f"Error: {e}"
            
    return "Error: Failed to get a response from the model after multiple retries."


def clear_chroma_data():
    """Clears all data from the ChromaDB collection."""
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
        st.session_state.db_client.get_or_create_collection(name=COLLECTION_NAME)
        st.toast("Knowledge base cleared!", icon="🗑️")
    except Exception as e:
        st.error(f"Error clearing collection: {e}")

# --- Document Processing Functions (Enhanced) ---

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_csv(uploaded_file):
    """Extracts text from an uploaded CSV file."""
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    df = pd.read_csv(stringio)
    return df.to_markdown(index=False) 

def extract_text_from_html(uploaded_file):
    """Extracts main text content from an uploaded HTML file."""
    content = uploaded_file.getvalue().decode("utf-8")
    soup = BeautifulSoup(content, 'lxml')
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return '\n'.join(chunk for chunk in chunks if chunk)

def extract_text_from_xml(uploaded_file):
    """Extracts all text content from an uploaded XML file."""
    content = uploaded_file.getvalue().decode("utf-8")
    soup = BeautifulSoup(content, 'lxml')
    return soup.get_text(separator=' ', strip=True) 

def split_documents(text_data, chunk_size=1000, chunk_overlap=200): 
    """Splits a single string of text into chunks with overlapping."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text_data)

def process_and_store_documents(documents):
    """
    Processes a list of text documents, generates embeddings, and
    stores them in ChromaDB.
    """
    collection = get_collection()
    model = st.session_state.model

    embeddings = model.encode(documents).tolist()
    document_ids = [str(uuid.uuid4()) for _ in documents]
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=document_ids
    )

    st.toast("Documents processed and stored successfully!", icon="✅")

@traceable(run_type="retriever")
def retrieve_documents(query, n_results=5):
    """
    Retrieves the most relevant documents from ChromaDB based on a query.
    """
    collection = get_collection()
    model = st.session_state.model
    
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    if results and results.get('documents') and results['documents'][0]:
        return results['documents'][0]
    return []

# --- CAG (Context Augmentation Generation) Implementation ---
def get_chat_history_for_cag():
    """Compiles the last few turns of chat history for context."""
    history = st.session_state.messages[-4:] 
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    return history_str

@traceable(run_type="chain")
def rag_pipeline(query, selected_language_code):
    """
    Executes the full RAG pipeline with CAG (Chat History Context).
    """
    collection = get_collection()
    if collection.count() == 0:
        return "Hey there! I'm a chatbot that answers questions based on documents you provide. Please upload a supported file or enter a GitHub raw URL in the section above before asking me anything. I'm ready when you are! 😊"

    # Step 1: RAG Retrieval
    relevant_docs = retrieve_documents(query)
    context = "\n".join(relevant_docs)
    chat_history = get_chat_history_for_cag() # Step 2: CAG/History retrieval
    
    # Combined Prompt for RAG + CAG
    if relevant_docs:
        prompt = (
            f"You are an expert document assistant. Your task is to answer the 'Question' using ONLY the 'Context' provided below. "
            f"For conversational memory, use the 'Chat History', but **prioritize** information from the 'Context'. "
            f"Your final response MUST be in {st.session_state.selected_language}. "
            f"If the Context does not contain the answer, you must politely state that the information is missing. "
            f"\n\nChat History:\n{chat_history}"
            f"\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        )
    else:
        # Fallback prompt using only history/general knowledge if RAG retrieval fails
        prompt = (
            f"You are an expert document assistant. Your task is to answer the 'Question' below. "
            f"Your final response MUST be in {st.session_state.selected_language}. "
            f"Use the 'Chat History' to understand the context or recall a repeated question. "
            f"Since no relevant documents were found, if the answer is NOT clearly derivable from the Chat History or general knowledge, politely state that you cannot answer from the documents. "
            f"\n\nChat History:\n{chat_history}\n\nQuestion: {query}\n\nAnswer:"
        )

    response = call_gemini_api(prompt)

    if response.startswith("Error:"):
        return response
    
    return response

# --- UI Helper Functions (TTS integration) ---

def display_chat_messages():
    """Displays all chat messages in the Streamlit app, with TTS play option."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Add TTS playback button for assistant messages if Voice mode is enabled
            if message["role"] == "assistant" and st.session_state.response_mode == 'Voice':
                try:
                    # Reruns the main loop, triggering TTS generation in handle_user_input
                    if st.button("🔊 Play Voice", key=f"tts_{uuid.uuid4()}", type="secondary"):
                         st.session_state['tts_to_play'] = message["content"]
                         st.experimental_rerun()

                except Exception as e:
                    st.error(f"TTS button error: {e}")

def is_valid_github_raw_url(url):
    """Checks if the URL is a raw GitHub URL for supported files."""
    return re.match(r"https://raw\.githubusercontent\.com/.+\.(txt|md|csv|html|xml)$", url)

def handle_user_input():
    """
    Handles new user input, runs the RAG pipeline, updates chat history, 
    and triggers TTS generation.
    """
    # 1. TTS Playback Handler (runs first if a button was clicked)
    if 'tts_to_play' in st.session_state and st.session_state.tts_to_play:
        try:
            with st.spinner("Generating speech..."):
                selected_language_code = LANGUAGE_DICT[st.session_state.selected_language]
                # nest_asyncio allows this asynchronous call to work
                audio_bytes = asyncio.run(generate_tts(st.session_state.tts_to_play, selected_language_code))
                st.audio(audio_bytes, format='audio/mp3', start_time=0)
            del st.session_state['tts_to_play']
        except Exception as e:
            st.error(f"Failed to generate or play voice: {e}")
            del st.session_state['tts_to_play']
        return 

    # 2. Chat Input Handler
    if prompt := st.chat_input("Ask about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                selected_language_code = LANGUAGE_DICT[st.session_state.selected_language] 
                response = rag_pipeline(prompt, selected_language_code)
                st.markdown(response)

                if st.session_state.response_mode == 'Voice':
                    try:
                        # Auto-play TTS immediately after generation if Voice mode is on
                        with st.spinner("Generating voice response..."):
                            # nest_asyncio allows this asynchronous call to work
                            audio_bytes = asyncio.run(generate_tts(response, selected_language_code))
                            st.audio(audio_bytes, format='audio/mp3', start_time=0)
                    except Exception as e:
                        st.warning(f"Could not generate voice response: {e}")
                        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Update chat title if it's a new chat
        if st.session_state.current_chat_id and st.session_state.chat_history[st.session_state.current_chat_id]['title'] == "New Chat":
            title = prompt[:50] + ('...' if len(prompt) > 50 else '')
            st.session_state.chat_history[st.session_state.current_chat_id]['title'] = title


# --- Streamlit UI ---
def main_ui():
    """Sets up the main Streamlit UI for the RAG chatbot."""
    st.set_page_config(
        page_title="RAG Chat Flow (Gemini LLM)", 
        layout="wide",
        initial_sidebar_state="auto"
    )

    # 1. Initialize dependencies
    if 'db_client' not in st.session_state or 'model' not in st.session_state or 'gemini_client' not in st.session_state:
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client = initialize_dependencies()
        
    # 2. Handle New Chat / Initialization
    if 'current_chat_id' not in st.session_state or st.session_state.current_chat_id is None or not st.session_state.messages:
        new_chat_id = str(uuid.uuid4())
        
        # Check if we are coming from a 'New Chat' click where state might be partially reset
        if new_chat_id not in st.session_state.chat_history:
             st.session_state.messages = []
        
        st.session_state.current_chat_id = new_chat_id
        st.session_state.chat_history[new_chat_id] = {
            'messages': st.session_state.messages,
            'title': "New Chat",
            'date': datetime.now()
        }
        
    # Top-of-app details as requested
    st.header("🤖 RAG System with Context Augmentation (CAG)")
    st.markdown("""
        **Features:**
        * 📁 **File Support:** PDF, TXT, CSV, HTML, XML, GitHub Raw (.txt/.md/.csv/.html/.xml)
        * 🗣️ **Response Mode:** Text (Default) / **Voice Mode** (TTS via Edge-TTS)
        * 🌐 **Multi-Language Support:** 17+ Languages Available
        * 🧠 **Cost Optimization:** Context Augmentation Generation (CAG) for history recall.
    """)
    st.markdown("---")


    # Sidebar
    with st.sidebar:
        st.header("RAG Chat Flow")
        
        st.caption(f"LLM: **{GEMINI_MODEL_ID}**")
        
        # Language Selector
        st.session_state.selected_language = st.selectbox(
            "🌐 Select Response Language",
            options=list(LANGUAGE_DICT.keys()),
            key="language_selector"
        )
        
        # Response Mode Selector
        st.session_state['response_mode'] = st.radio(
            "🔊 Select Response Mode (TTS)",
            options=['Text', 'Voice'],
            index=0 if st.session_state.response_mode == 'Text' else 1,
            key="response_mode_selector",
            help="Switch to 'Voice' to hear the assistant's response."
        )

        # --- BUG FIX: New Chat Button ---
        if st.button("New Chat", use_container_width=True):
            clear_chroma_data() 
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.toast("Starting a new chat session!", icon="✨")

        st.subheader("Chat History")
        if 'chat_history' in st.session_state and st.session_state.chat_history:
             sorted_chat_ids = sorted(
                st.session_state.chat_history.keys(), 
                key=lambda x: st.session_state.chat_history[x]['date'], 
                reverse=True
            )
             for chat_id in sorted_chat_ids:
                chat_title = st.session_state.chat_history[chat_id]['title']
                date_str = st.session_state.chat_history[chat_id]['date'].strftime("%b %d, %I:%M %p")
                
                is_current = chat_id == st.session_state.current_chat_id
                style = "background-color: #262730; border-radius: 5px; padding: 10px;" if is_current else "padding: 10px;"
                
                st.markdown(
                    f"<div style='{style}'>",
                    unsafe_allow_html=True
                )
                if st.button(f"{chat_title}", key=f"btn_{chat_id}", use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.messages = st.session_state.chat_history[chat_id]['messages']
                    st.experimental_rerun()
                st.markdown(f"<small>{date_str}</small></div>", unsafe_allow_html=True)


    # Main content area - Document upload/processing section
    with st.container():
        st.subheader("Add Context Documents")
        uploaded_files = st.file_uploader(
            "Upload files (.txt, .pdf, .csv, .html, .xml)", 
            type=["txt", "pdf", "csv", "html", "xml"], 
            accept_multiple_files=True
        )
        github_url = st.text_input("Enter a GitHub raw URL (.txt, .md, .csv, .html, .xml):")

        # --- File Processing Logic ---
        if uploaded_files:
            if st.button(f"Process {len(uploaded_files)} File(s)"):
                with st.spinner("Processing files..."):
                    total_chunks = 0
                    for uploaded_file in uploaded_files:
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        file_contents = None
                        uploaded_file.seek(0) 

                        try:
                            if file_ext == "txt":
                                file_contents = uploaded_file.getvalue().decode("utf-8")
                            elif file_ext == "pdf":
                                file_contents = extract_text_from_pdf(uploaded_file)
                            elif file_ext == "csv":
                                file_contents = extract_text_from_csv(uploaded_file)
                            elif file_ext == "html":
                                file_contents = extract_text_from_html(uploaded_file)
                            elif file_ext == "xml":
                                file_contents = extract_text_from_xml(uploaded_file)
                            else:
                                st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
                                continue
                            
                            if file_contents and file_contents.strip():
                                documents = split_documents(file_contents) 
                                process_and_store_documents(documents)
                                total_chunks += len(documents)
                            else:
                                st.warning(f"File {uploaded_file.name} was empty or failed to extract content.")
                                
                        except Exception as e:
                            st.error(f"Failed to process {uploaded_file.name}: {e}")
                            continue

                    if total_chunks > 0:
                        st.success(f"Successfully processed {len(uploaded_files)} file(s) into {total_chunks} chunks! Start chatting now.")
                    else:
                        st.warning("No new content was added to the knowledge base.")


        # --- GitHub URL Processing Logic ---
        if github_url:
            if st.button("Process URL"):
                if not is_valid_github_raw_url(github_url):
                    st.error("Invalid URL format. Please use a raw GitHub URL ending in a supported extension (.txt, .md, .csv, .html, .xml).")
                else:
                    file_ext = github_url.split('.')[-1].lower()
                    with st.spinner("Fetching and processing file from URL..."):
                        try:
                            response = requests.get(github_url)
                            response.raise_for_status()
                            file_contents = response.text
                            
                            if file_ext == "csv":
                                file_obj = io.StringIO(file_contents)
                                file_contents = pd.read_csv(file_obj).to_markdown(index=False)
                            elif file_ext == "html":
                                soup = BeautifulSoup(file_contents, 'lxml')
                                file_contents = soup.get_text(separator=' ', strip=True)
                            elif file_ext == "xml":
                                soup = BeautifulSoup(file_contents, 'lxml')
                                file_contents = soup.get_text(separator=' ', strip=True)

                            documents = split_documents(file_contents)
                            process_and_store_documents(documents)
                            st.success("File from URL processed! You can now chat about its contents.")
                        except requests.exceptions.RequestException as e:
                            st.error(f"Error fetching URL: {e}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred during URL processing: {e}")
                            
    st.markdown("---")
    
    # Chat display and input
    display_chat_messages()
    handle_user_input()

if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except AttributeError:
            pass 
        
    main_ui()
