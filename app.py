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
# TTS & CAG Imports
# =====================================================================
import asyncio
import edge_tts

# =====================================================================
# FIX 1: SQLITE3 PATCH (for Streamlit Cloud)
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
from pypdf import PdfReader
# Ensure lxml is available for robust HTML/XML parsing
# from lxml import etree # Not explicitly imported but needed by bs4 for better performance

# --- Google Gemini Imports ---
from google import genai
from google.genai.errors import APIError

# --- LangSmith Imports (Keep for tracing) ---
from langsmith import traceable, tracing_context

# --- Constants and Configuration ---
COLLECTION_NAME = "rag_documents"

# *** CRITICAL FIX: Read API Key from Streamlit Secrets ***
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
GEMINI_MODEL_ID = "gemini-2.5-flash"  # Using the requested model_name

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
        # 1. Initialize ChromaDB
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)
        
        # 2. Initialize Sentence Transformer (for embeddings)
        # Using a fast, high-quality embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
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

# --- Gemini API Call (Unchanged logic) ---
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
    except Exception as e:
        st.error(f"Error clearing collection: {e}")

# --- Document Processing Functions (Enhanced for new file types) ---

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_csv(uploaded_file):
    """Extracts text from an uploaded CSV file."""
    df = pd.read_csv(uploaded_file)
    return df.to_markdown(index=False) # Convert to Markdown table for better LLM context

def extract_text_from_html(uploaded_file):
    """Extracts main text content from an uploaded HTML file."""
    content = uploaded_file.read().decode("utf-8")
    soup = BeautifulSoup(content, 'lxml')
    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    # Get text
    text = soup.get_text()
    # Break into lines and remove excess whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return '\n'.join(chunk for chunk in chunks if chunk)

def extract_text_from_xml(uploaded_file):
    """Extracts all text content from an uploaded XML file."""
    content = uploaded_file.read().decode("utf-8")
    soup = BeautifulSoup(content, 'lxml')
    return soup.get_text(separator=' ', strip=True) # Simple text extraction

def split_documents(text_data, chunk_size=1000, chunk_overlap=200): # Increased size for better context, 20% overlap
    """Splits a single string of text into chunks with overlapping."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, # Overlapping chunking implemented here
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
    history = st.session_state.messages[-4:] # Use last 4 messages (2 user/2 assistant turns)
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

    relevant_docs = retrieve_documents(query)
    
    if not relevant_docs:
        # Check for CAG possibility without RAG (for general questions or history recall)
        chat_history = get_chat_history_for_cag()
        if len(chat_history) > 0 and 'user' not in chat_history.lower(): # Only run RAG if new context is needed
            
             # The system will naturally reuse information from history if the question is repeated
             # without adding a new prompt for it. However, if the history is compiled, 
             # we can make a direct LLM call for a history-based answer to save RAG look-up tokens.
             # For simplicity and robustness (to ensure model handles repeated questions), we proceed with a focused prompt.
             
             prompt_template = (
                f"You are an expert document assistant. Your task is to answer the 'Question' below. "
                f"Your final response MUST be in {st.session_state.selected_language}. "
                f"Use the 'Chat History' to understand the context. "
                f"Since no relevant documents were found, if the answer is clearly present in the Chat History (e.g., a repeated question), you can use it. "
                f"If the answer is NOT in the history, politely state that you cannot answer. "
                f"\n\nChat History:\n{chat_history}\n\nQuestion: {query}\n\nAnswer:"
            )
             response = call_gemini_api(prompt_template)
             return response

        return "I couldn't find relevant information in the uploaded documents to answer your question."

    context = "\n".join(relevant_docs)
    chat_history = get_chat_history_for_cag() # Get history for CAG

    # Main RAG/CAG Prompt
    prompt = (
        f"You are an expert document assistant. Your task is to answer the 'Question' using ONLY the 'Context' provided below. "
        f"For better conversational flow and to answer repeated questions, use the 'Chat History' for memory, but prioritize information from the 'Context'. "
        f"Your final response MUST be in {st.session_state.selected_language}. "
        f"If the Context does not contain the answer, you must politely state that the information is missing. "
        f"\n\nChat History:\n{chat_history}"
        f"\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
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
    # Updated to support .csv, .html, .xml, .txt, .md
    return re.match(r"https://raw\.githubusercontent\.com/.+\.(txt|md|csv|html|xml)$", url)

def handle_user_input():
    """
    Handles new user input, runs the RAG pipeline, updates chat history, 
    and triggers TTS generation.
    """
    if 'tts_to_play' in st.session_state and st.session_state.tts_to_play:
        # This branch handles the TTS playback triggered by a button press
        try:
            with st.spinner("Generating speech..."):
                selected_language_code = LANGUAGE_DICT[st.session_state.selected_language]
                audio_bytes = asyncio.run(generate_tts(st.session_state.tts_to_play, selected_language_code))
                st.audio(audio_bytes, format='audio/mp3', start_time=0)
            del st.session_state['tts_to_play']
        except Exception as e:
            st.error(f"Failed to generate or play voice: {e}")
            del st.session_state['tts_to_play']
        return # Do not process chat input if playing TTS

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
                            audio_bytes = asyncio.run(generate_tts(response, selected_language_code))
                            st.audio(audio_bytes, format='audio/mp3', start_time=0)
                    except Exception as e:
                        st.warning(f"Could not generate voice response: {e}")
                        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
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

    # Initialize dependencies
    if 'db_client' not in st.session_state or 'model' not in st.session_state or 'gemini_client' not in st.session_state:
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client = initialize_dependencies()
        
    if 'current_chat_id' not in st.session_state or not st.session_state.messages:
        new_chat_id = str(uuid.uuid4())
        st.session_state.current_chat_id = new_chat_id
        st.session_state.messages = []
        st.session_state.chat_history[new_chat_id] = {
            'messages': st.session_state.messages,
            'title': "New Chat",
            'date': datetime.now()
        }

    # Top-of-app details as requested
    st.header("🤖 RAG System with Context Augmentation (CAG)")
    st.markdown("""
        **Features:**
        * 📁 **File Support:** PDF, TXT, CSV, HTML, XML, GitHub Raw (.txt/.md)
        * 🗣️ **Response Mode:** Text (Default) / **Voice Mode** (TTS via Edge-TTS)
        * 🌐 **Multi-Language Support:** 17+ Languages Available
        * 🧠 **Cost Optimization:** Context Augmentation (CAG) for history recall.
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

        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            clear_chroma_data()
            st.session_state.chat_history = {}
            st.session_state.current_chat_id = None
            st.experimental_rerun()

        st.subheader("Chat History")
        if 'chat_history' in st.session_state and st.session_state.chat_history:
             # ... (Chat history display logic remains the same for brevity)
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

                        try:
                            if file_ext == "txt":
                                file_contents = uploaded_file.read().decode("utf-8")
                            elif file_ext == "pdf":
                                file_contents = extract_text_from_pdf(uploaded_file)
                            elif file_ext == "csv":
                                # Need to reset pointer for multi-file processing
                                uploaded_file.seek(0)
                                file_contents = extract_text_from_csv(uploaded_file)
                            elif file_ext == "html":
                                uploaded_file.seek(0)
                                file_contents = extract_text_from_html(uploaded_file)
                            elif file_ext == "xml":
                                uploaded_file.seek(0)
                                file_contents = extract_text_from_xml(uploaded_file)
                            else:
                                st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
                                continue
                            
                            if file_contents and file_contents.strip(): # Check for non-empty content
                                documents = split_documents(file_contents) # Uses overlapping chunking
                                process_and_store_documents(documents)
                                total_chunks += len(documents)
                            else:
                                st.warning(f"File {uploaded_file.name} was empty or failed to extract content.")
                                
                        except Exception as e:
                            st.error(f"Failed to process {uploaded_file.name}: {e}")
                            continue

                    if total_chunks > 0:
                        st.success(f"Successfully processed {len(uploaded_files)} file(s) into {total_chunks} chunks!")
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
                            
                            # Simple processing for raw text/md, relies on requests/BeautifulSoup for others
                            if file_ext in ["csv", "html", "xml"]:
                                # Convert string content to file-like object for re-use of extractor functions
                                file_obj = io.StringIO(file_contents)
                                if file_ext == "csv":
                                    file_contents = extract_text_from_csv(file_obj)
                                elif file_ext == "html":
                                    file_contents = extract_text_from_html(file_obj)
                                elif file_ext == "xml":
                                    file_contents = extract_text_from_xml(file_obj)

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
    # Ensure asyncio is available for Edge-TTS
    if sys.platform == "win32" and sys.version_info >= (3, 8) and hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    main_ui()
