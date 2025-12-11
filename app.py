import streamlit as st
import os
import sys
import tempfile
import uuid
import requests
import time
from datetime import datetime
import re
import pandas as pd
from bs4 import BeautifulSoup
import asyncio
from io import BytesIO

# =====================================================================
# FIX 1: SQLITE3 PATCH for Streamlit Cloud (Ensures ChromaDB works)
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

# Using pypdf is preferred for PDF extraction
try:
    from pypdf import PdfReader
except ImportError:
    pass

# --- Google Gemini Imports ---
from google import genai
from google.genai.errors import APIError

# --- LangSmith Imports (Keep for tracing if you use LangSmith) ---
try:
    from langsmith import traceable
except ImportError:
    # Define a dummy traceable decorator if LangSmith is not installed
    def traceable(run_type=None):
        def decorator(func):
            return func
        return decorator

# --- TTS Import ---
try:
    from edge_tts import Communicate
except ImportError:
    pass


# --- Constants and Configuration ---
COLLECTION_NAME = "rag_documents"
VOICE_NAME = "en-US-JennyNeural" # Default voice for Edge-TTS

# *** CRITICAL FIX: Read API Key from Streamlit Secrets ***
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
GEMINI_MODEL_ID = "gemini-2.5-flash" 

if not GEMINI_API_KEY:
    st.warning("🚨 GEMINI_API_KEY is not set in Streamlit Secrets. Please set it to proceed.")
    st.stop()
    
# Dictionary of supported languages and their ISO 639-1 codes
LANGUAGE_DICT = {
    "English": "en", "Spanish": "es", "Arabic": "ar", "French": "fr",
    "German": "de", "Hindi": "hi", "Tamil": "ta", "Bengali": "bn",
    "Japanese": "ja", "Korean": "ko", "Russian": "ru",
    "Chinese (Simplified)": "zh-Hans", "Portuguese": "pt",
    "Italian": "it", "Dutch": "nl", "Turkish": "tr"
}

# =====================================================================
# SESSION STATE INITIALIZATION & HELPERS
# =====================================================================

def initialize_session_state():
    """Initializes all necessary session state variables."""
    if 'selected_language' not in st.session_state: st.session_state['selected_language'] = 'English'
    if 'response_mode' not in st.session_state: st.session_state['response_mode'] = 'Text'
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'chat_history' not in st.session_state: st.session_state.chat_history = {}
    if 'current_chat_id' not in st.session_state: st.session_state.current_chat_id = None
    if 'cag_cache' not in st.session_state: st.session_state.cag_cache = {}
    if 'documents_loaded' not in st.session_state: st.session_state.documents_loaded = False
    if 'processed_files' not in st.session_state: st.session_state.processed_files = set()
    if 'new_chat_triggered' not in st.session_state: st.session_state.new_chat_triggered = False
    # New state to track processed URLs
    if 'processed_urls' not in st.session_state: st.session_state.processed_urls = set()

def create_new_chat(save_current=True):
    """
    Resets session state for a new conversation, but keeps RAG data.
    
    Args:
        save_current (bool): Whether to save the current conversation 
                             before starting a new one.
    """
    # 1. Save the current chat before starting a new one
    if save_current and st.session_state.current_chat_id and st.session_state.messages:
        if st.session_state.current_chat_id in st.session_state.chat_history:
             st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = st.session_state.messages
    
    # 2. Create new chat data
    new_chat_id = str(uuid.uuid4())
    
    # 3. Update session state
    st.session_state.current_chat_id = new_chat_id
    st.session_state.messages = []
    st.session_state.chat_history[new_chat_id] = {
        'messages': [],
        'title': "New Chat",
        'date': datetime.now()
    }
    st.session_state.cag_cache = {}
    st.session_state.new_chat_triggered = True

@st.cache_resource
def initialize_dependencies():
    """Initializes and returns the ChromaDB client, SentenceTransformer model, and the Google GenAI Client."""
    try:
        # 1. Initialize ChromaDB (Use a unique temp directory for each Streamlit run)
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)
        
        # 2. Initialize Sentence Transformer (for embeddings)
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

def clear_chroma_data():
    """Clears all data from the ChromaDB collection, CAG cache, and resets load flag."""
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
            st.session_state.db_client.get_or_create_collection(name=COLLECTION_NAME) 
        st.session_state.cag_cache = {} 
        st.session_state.documents_loaded = False
        st.session_state.processed_files = set() 
        st.session_state.processed_urls = set() # Clear processed URLs
    except Exception as e:
        st.error(f"Error clearing collection: {e}")

# =====================================================================
# DOCUMENT PROCESSING FUNCTIONS
# =====================================================================

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except NameError:
        st.warning("PDF processing skipped. 'pypdf' library is required.")
        return ""

def extract_text_from_csv(uploaded_file):
    """Extracts text from an uploaded CSV file."""
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    return df.to_markdown(index=False)

def extract_text_from_html_xml(uploaded_file, file_type):
    """Extracts main text content from HTML or XML files."""
    uploaded_file.seek(0)
    content = uploaded_file.read().decode("utf-8")
    
    soup = BeautifulSoup(content, 'lxml')
    
    if file_type == 'html':
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        return soup.get_text(' ', strip=True)
    
    elif file_type == 'xml':
        return soup.get_text(' ', strip=True)

    return ""

def split_documents(text_data, chunk_size=1000, chunk_overlap=200):
    """Splits a single string of text into chunks with OVERLAPPING CHUNKING."""
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
    st.session_state.documents_loaded = True
    st.toast("Documents processed and stored successfully!", icon="✅")

# =====================================================================
# RAG, CAG, AND TTS FUNCTIONS
# =====================================================================

@traceable(run_type="llm")
def call_gemini_api(prompt, max_retries=3):
    """Calls the Google Gemini API for text generation."""
    gemini_client = st.session_state.gemini_client
    
    retry_delay = 1
    for i in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL_ID,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                )
            )
            # Add to CAG cache 
            st.session_state.cag_cache[prompt] = response.text.strip()
            return response.text.strip()
            
        except APIError as e:
            time.sleep(retry_delay)
            retry_delay *= 2
        except Exception as e:
            return f"Error: An unexpected error occurred during the API call: {e}"
    
    return "Error: Failed to get a response from the model after multiple retries."

@traceable(run_type="retriever")
def retrieve_documents(query, n_results=5):
    """Retrieves the most relevant documents from ChromaDB based on a query."""
    collection = get_collection()
    model = st.session_state.model
    
    if collection.count() == 0:
        return []

    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    if results and results.get('documents') and results['documents'][0]:
        return results['documents'][0]
    return []

@traceable(run_type="chain")
def rag_pipeline(query, selected_language):
    """
    Executes the full RAG pipeline, incorporating CAG for cost reduction.
    Returns (response_text, is_cached)
    """
    
    # --- 1. Construct Full Prompt Template ---
    context_placeholder = "[CONTEXT_PLACEHOLDER]"
    prompt_template = (
        f"You are an expert document assistant. Your task is to answer the 'Question' using ONLY the 'Context' provided below. "
        f"Your final response MUST be in **{selected_language}**. "
        f"If the Context does not contain the answer, you must politely state that the information is missing. "
        f"\n\nContext:\n---\n{context_placeholder}\n---\n\nQuestion: {query}\n\nAnswer:"
    )

    # --- 2. Retrieval ---
    relevant_docs = retrieve_documents(query)
    
    if not relevant_docs:
        # Not cached, no documents found
        return ("Hello! I am a RAG AI Agent. I could not find relevant information in the uploaded documents. Please ensure documents are uploaded and processed successfully.", False)

    context = "\n".join(relevant_docs)
    
    # --- 3. Final Prompt Assembly & CAG Check ---
    final_prompt = prompt_template.replace(context_placeholder, context)
    
    if final_prompt in st.session_state.cag_cache:
        # Cache Hit! Return the cached response and flag it as cached
        return (st.session_state.cag_cache[final_prompt], True)

    # --- 4. LLM Call
    response = call_gemini_api(final_prompt)
    
    # Not cached, LLM call performed
    return (response, False)

def generate_tts_audio(text, language_code):
    """Generates a TTS audio file using Edge-TTS."""
    voice = VOICE_NAME
    if language_code == "hi": voice = "hi-IN-MadhurNeural"
    elif language_code == "es": voice = "es-ES-ElviraNeural"
    elif language_code == "fr": voice = "fr-FR-HenriNeural"
    
    try:
        communicate = Communicate(text, voice)
        audio_buffer = BytesIO()
        
        async def generate():
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])

        asyncio.run(generate())
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        return None

# =====================================================================
# UI HANDLERS AND MAIN UI
# =====================================================================

def handle_upload_and_process(uploaded_files):
    """Handles the automatic processing of multiple uploaded files."""
    if not uploaded_files:
        return

    processed_files = st.session_state.get('processed_files', set())
    new_files_to_process = [f for f in uploaded_files if f.name not in processed_files]
    
    if not new_files_to_process:
        return 

    with st.spinner("Processing files and creating vector embeddings..."):
        total_chunks = 0
        current_processed_files = set(processed_files)
        
        for uploaded_file in new_files_to_process:
            chunks = handle_file_upload(uploaded_file)
            total_chunks += chunks
            if chunks > 0:
                current_processed_files.add(uploaded_file.name)
        
        st.session_state.processed_files = current_processed_files

        if total_chunks > 0:
            st.success(f"Successfully processed {len(new_files_to_process)} new file(s) into {total_chunks} chunks! You can now ask questions.")
        elif new_files_to_process:
            st.warning("Processing failed for the new files. Check file types.")


def handle_file_upload(uploaded_file):
    """Handles parsing and chunking for a single uploaded file."""
    file_ext = uploaded_file.name.split('.')[-1].lower()
    file_contents = None
    
    uploaded_file.seek(0) 

    try:
        if file_ext == "txt":
            file_contents = uploaded_file.read().decode("utf-8")
        elif file_ext == "pdf":
            file_contents = extract_text_from_pdf(uploaded_file)
        elif file_ext == "csv":
            file_contents = extract_text_from_csv(uploaded_file)
        elif file_ext in ["html", "xml"]:
            file_contents = extract_text_from_html_xml(uploaded_file, file_ext)
        else:
            st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
            return 0
        
        if file_contents:
            documents = split_documents(file_contents)
            process_and_store_documents(documents)
            return len(documents)
            
    except Exception as e:
        st.error(f"Failed to process {uploaded_file.name} ({file_ext}). Error: {e}")
        return 0
    return 0


def handle_url_upload(github_url):
    """Handles fetching and processing a file from a GitHub raw URL."""
    if not github_url: return 0
    
    if not is_valid_github_raw_url(github_url):
        st.error("Invalid URL format. Please use a raw GitHub URL ending in a supported extension.")
        return 0
    
    file_ext = github_url.split('.')[-1].lower()
    
    with st.spinner(f"Fetching and processing {file_ext} file from URL..."):
        try:
            response = requests.get(github_url)
            response.raise_for_status()
            file_contents = response.text
            
            if file_ext in ["html", "xml"]:
                soup = BeautifulSoup(file_contents, 'lxml')
                if file_ext == 'html':
                    for script_or_style in soup(['script', 'style']): script_or_style.decompose()
                file_contents = soup.get_text(' ', strip=True)
                
            elif file_ext == "csv":
                csv_buffer = BytesIO(file_contents.encode('utf-8'))
                df = pd.read_csv(csv_buffer)
                file_contents = df.to_markdown(index=False)

            documents = split_documents(file_contents)
            process_and_store_documents(documents)
            
            # Record the URL as processed
            st.session_state.processed_urls.add(github_url)
            
            st.success(f"File from URL processed into {len(documents)} chunks! You can now chat about its contents.")
            return len(documents)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching URL: {e}")
            return 0
        except Exception as e:
            st.error(f"An unexpected error occurred during URL processing: {e}")
            return 0


def handle_user_input():
    """Handles new user input, runs the RAG pipeline, and updates chat history."""
    if prompt := st.chat_input("Ask about your document..."):
        # Save the user message to the current chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            # Placeholder to display the CACHE HIT message or the final response content
            status_placeholder = st.empty()
            
            selected_language = st.session_state.selected_language
            
            # --- RAG/CAG Pipeline Call ---
            # Call rag_pipeline once to get the result.
            response_text, is_cached = rag_pipeline(prompt, selected_language)
            
            # Step 2: Show Visual Feedback (Cache Hit)
            if is_cached:
                # Cache Hit! 
                status_placeholder.info("⚡ **CACHE HIT!** Response retrieved from Context Augment Generation (CAG) cache. Token cost minimized.")
                time.sleep(0.1) 
            elif response_text.startswith("Hello!"):
                # No relevant documents found message
                pass
            else:
                # Cache Miss - Clear the placeholder if it's not being used for the cache hit message
                status_placeholder.empty()

            # --- TTS Generation (if requested) ---
            audio_buffer = None
            if st.session_state.response_mode == 'Voice' and not response_text.startswith("Error:"):
                language_code = LANGUAGE_DICT.get(selected_language, 'en')
                audio_buffer = generate_tts_audio(response_text, language_code)
                
            # Step 3: Show the final text response
            status_placeholder.markdown(response_text)
            
            # Step 4: Show the audio response
            if audio_buffer:
                st.audio(audio_buffer, format="audio/mp3")

        # Store response details in session state
        message_to_store = {"role": "assistant", "content": response_text}
        if audio_buffer:
            message_to_store["audio"] = audio_buffer
            
        st.session_state.messages.append(message_to_store)
        
        # Update chat history title
        current_chat_data = st.session_state.chat_history.get(st.session_state.current_chat_id)
        if current_chat_data and current_chat_data['title'] == "New Chat":
            title = prompt[:50] + ('...' if len(prompt) > 50 else '')
            st.session_state.chat_history[st.session_state.current_chat_id]['title'] = title
        
        # Ensure the entire chat messages list is saved back to history
        if st.session_state.current_chat_id:
            st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = st.session_state.messages

def display_chat_messages():
    """Displays all chat messages in the Streamlit app."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "audio" in message and message["audio"] is not None:
                st.audio(message["audio"], format="audio/mp3")

def is_valid_github_raw_url(url):
    """Checks if the URL is a raw GitHub URL for supported text files."""
    return re.match(r"https://raw\.githubusercontent\.com/.+\.(txt|md|csv|html|xml)$", url, re.IGNORECASE)

# --- Streamlit UI ---
def main_ui():
    """Sets up the main Streamlit UI for the RAG chatbot."""
    st.set_page_config(
        page_title="RAG Chat Flow (Gemini LLM)",
        layout="wide",
        initial_sidebar_state="auto"
    )

    initialize_session_state()
    
    # Initialize dependencies
    if 'db_client' not in st.session_state or 'model' not in st.session_state or 'gemini_client' not in st.session_state:
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client = initialize_dependencies()
        
    # Only create a new chat if one hasn't been set
    if st.session_state.current_chat_id is None:
        create_new_chat(save_current=False)
    
    st.session_state.new_chat_triggered = False

    # Sidebar
    with st.sidebar:
        st.header("⚙️ RAG System Configuration")
        
        st.caption(f"LLM: **{GEMINI_MODEL_ID}**")
        st.caption(f"Caching: **Context Augment Generation (CAG)**")

        # --- Multilingual Selector ---
        st.session_state.selected_language = st.selectbox(
            "🌍 Select Response Language",
            options=list(LANGUAGE_DICT.keys()),
            key="language_selector"
        )
        
        # --- Response Mode Selector ---
        st.session_state.response_mode = st.radio(
            "🗣️ Select Response Mode",
            options=['Text', 'Voice'],
            key="response_mode_selector",
            horizontal=True
        )
        st.caption(f"Voice Engine: Edge-TTS")
        
        st.markdown("---")
        
        # New Chat button with RAG Data clearing
        if st.button("🔄 Start New Chat & Clear RAG Data", use_container_width=True):
            clear_chroma_data() 
            st.session_state.chat_history = {} 
            st.session_state.messages = [] 
            st.session_state.current_chat_id = None 
            st.session_state.new_chat_triggered = True
            st.rerun() 

        # New Chat button (keeps RAG data)
        if st.button("➕ Start New Chat (Keep Documents)", use_container_width=True):
            create_new_chat(save_current=True) 
            st.rerun()
            
        st.subheader("Chat History")
        
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            # Sort chats by date in reverse chronological order
            sorted_chat_ids = sorted(
                st.session_state.chat_history.keys(), 
                key=lambda x: st.session_state.chat_history[x]['date'], 
                reverse=True
            )
            
            for chat_id in sorted_chat_ids:
                chat_data = st.session_state.chat_history[chat_id]
                chat_title = chat_data.get('title', "Untitled Chat")
                date_str = chat_data['date'].strftime("%b %d, %I:%M %p")
                
                is_current = chat_id == st.session_state.current_chat_id
                
                with st.container(border=not is_current):
                    if st.button(f"**{chat_title}**", key=f"btn_{chat_id}", use_container_width=True):
                        # Action to load a previous chat
                        # Save the current state before loading a new one
                        if st.session_state.current_chat_id and st.session_state.messages:
                            st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = st.session_state.messages
                        
                        st.session_state.current_chat_id = chat_id
                        st.session_state.messages = chat_data['messages'] 
                        st.session_state.cag_cache = {} 
                        st.rerun() 
                    st.caption(date_str)

    # Main content area
    st.title("🧠 Gemini RAG AI Agent")
    
    st.markdown("""
        **A powerful Retrieval-Augmented Generation (RAG) system with advanced features:**
        * **📂 File Support:** Process documents like **.pdf**, **.txt**, **.csv**, **.html**, **.xml**, and **GitHub raw files**.
        * **⚙️ Overlapping Chunking:** Documents are split into **1000 character chunks with 200 characters overlap** for better context retrieval.
        * **🗣️ Multilingual:** Supports **16 languages** (see sidebar).
        * **💰 Cost Optimized (CAG):** Utilizes **Context Augment Generation** caching to reduce token costs on repeated questions. *(Watch for the **CACHE HIT!** message!)*
        * **🔊 Response Mode:** Capacity to respond in **Text** or **Voice** mode (TTS Engine: Edge-TTS).
    """)
    st.markdown("---")
    
    # Document upload/processing section
    with st.container():
        st.subheader("Add Context Documents (Knowledge Base)")
        uploaded_files = st.file_uploader(
            "Upload files (Limit 200MB per file - TXT, PDF, CSV, HTML, XML)",
            type=["txt", "pdf", "csv", "html", "xml"],
            accept_multiple_files=True
        )

        # --- Automatic File Processing Logic ---
        if uploaded_files:
            handle_upload_and_process(uploaded_files)

        # --- GitHub URL Input ---
        github_url = st.text_input("Enter a GitHub raw URL (.txt, .md, .csv, .html, .xml):", key='github_url_input')
        
        # --- Automatic URL Processing Logic (Replaces the button) ---
        if github_url and github_url not in st.session_state.processed_urls:
             # This automatically processes the file when the user finishes typing/pasting
             # and the script reruns, as long as it hasn't been processed yet.
             handle_url_upload(github_url)
             
    st.markdown("---")
    
    # Chat display and input
    display_chat_messages()
    
    # Show a prompt to upload if no documents are loaded
    if not st.session_state.documents_loaded and get_collection().count() == 0:
        st.info("Please upload and process at least one document or enter a GitHub URL to activate the chat input.")
    else:
        handle_user_input()

if __name__ == "__main__":
    main_ui()
