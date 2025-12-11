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

# =====================================================================
# FIX 1: SQLITE3 PATCH for Streamlit Cloud
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

# --- Google Gemini Imports ---
from google import genai
from google.genai.errors import APIError

# --- LangSmith Imports ---
from langsmith import traceable, tracing_context

# --- TTS Import ---
from edge_tts import Communicate
import asyncio
from io import BytesIO

# --- Constants and Configuration ---
COLLECTION_NAME = "rag_documents"
VOICE_NAME = "en-US-JennyNeural" # A common, high-quality voice for Edge-TTS

# *** CRITICAL FIX: Read API Key from Streamlit Secrets ***
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
# Using the model ID you specified. Note: 'preview' suffix might change or be removed over time.
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
if 'response_mode' not in st.session_state: st.session_state['response_mode'] = 'Text'
if 'messages' not in st.session_state: st.session_state.messages = []
if 'chat_history' not in st.session_state: st.session_state.chat_history = {}
if 'current_chat_id' not in st.session_state: st.session_state.current_chat_id = None
# --- Context Augment Generation (CAG) Cache ---
if 'cag_cache' not in st.session_state: st.session_state.cag_cache = {}


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
        # Using a model that performs well and is efficient
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
        # 3. Initialize Google GenAI Client (for LLM)
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        
        return db_client, model, gemini_client
    except Exception as e:
        st.error(f"An error occurred during dependency initialization. Error: {e}")
        st.stop()
        
def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    # The embedding function is not explicitly passed here, so ChromaDB will use its default 
    # (which can be problematic), but we rely on pre-computed embeddings.
    # We must ensure the 'all-MiniLM-L6-v2' is used consistently.
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

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
                    temperature=0.3, # Lower temperature for RAG to stay grounded
                    max_output_tokens=1024,
                )
            )
            # Add to CAG cache for immediate repeats within the session
            st.session_state.cag_cache[prompt] = response.text.strip()
            return response.text.strip()
            
        except APIError as e:
            st.warning(f"Gemini API error (try {i+1}/{max_retries}). Retrying in {retry_delay} seconds. Error: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2
        except Exception as e:
            st.error(f"An unexpected error occurred during the API call: {e}")
            return f"Error: {e}"
    
    return "Error: Failed to get a response from the model after multiple retries."

# =====================================================================
# DOCUMENT PROCESSING FUNCTIONS (EXTENDED SUPPORT)
# =====================================================================

def clear_chroma_data():
    """Clears all data from the ChromaDB collection and the CAG cache."""
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
            # Recreate the collection after deletion
            st.session_state.db_client.get_or_create_collection(name=COLLECTION_NAME) 
        st.session_state.cag_cache = {} # Clear CAG cache on new document load
    except Exception as e:
        st.error(f"Error clearing collection: {e}")

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_csv(uploaded_file):
    """Extracts text from an uploaded CSV file."""
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    # Convert the entire DataFrame to a string format suitable for RAG context
    return df.to_markdown(index=False)

def extract_text_from_html_xml(uploaded_file, file_type):
    """Extracts main text content from HTML or XML files."""
    content = uploaded_file.read().decode("utf-8")
    # Use lxml parser for robustness with HTML and XML
    soup = BeautifulSoup(content, 'lxml')
    
    if file_type == 'html':
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        # Get text, strip whitespaces, and join with spaces
        return soup.get_text(' ', strip=True)
    
    elif file_type == 'xml':
        # For XML, just get the main text content
        return soup.get_text(' ', strip=True)

    return ""

def split_documents(text_data, chunk_size=1000, chunk_overlap=200):
    """Splits a single string of text into chunks with OVERLAPPING CHUNKING."""
    # Increased chunk size and overlap for better context flow
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
    
    # Check if the collection is empty before proceeding
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
    """
    # --- 1. CAG Check (Cost Minimization) ---
    # We use the raw user query as the key for simplicity
    if query in st.session_state.cag_cache:
        st.toast("⚡ Retrieved response from **CAG** cache (Token Cost Saved!)", icon="💰")
        return st.session_state.cag_cache[query]

    collection = get_collection()
    if collection.count() == 0:
        return f"Hello! I am a RAG AI Agent. Please upload a document ({', '.join(['.pdf', '.txt', '.csv', '.html', '.xml'])} files, or a GitHub raw URL) in the section above before asking me anything. I'm ready when you are! 😊"

    # --- 2. Retrieval ---
    relevant_docs = retrieve_documents(query)
    
    if not relevant_docs:
        return "I couldn't find relevant information in the uploaded documents to answer your question."

    context = "\n".join(relevant_docs)
    
    # --- 3. Generation Prompt (with Multilingual Instruction) ---
    prompt = (
        f"You are an expert document assistant. Your task is to answer the 'Question' using ONLY the 'Context' provided below. "
        f"Your final response MUST be in **{selected_language}**. "
        f"If the Context does not contain the answer, you must politely state that the information is missing. "
        f"\n\nContext:\n---\n{context}\n---\n\nQuestion: {query}\n\nAnswer:"
    )
    
    # --- 4. LLM Call ---
    response = call_gemini_api(prompt)

    if response.startswith("Error:"):
        return response
    
    return response

# =====================================================================
# TEXT-TO-SPEECH (TTS) FUNCTION
# =====================================================================

def generate_tts_audio(text, language_code):
    """Generates a TTS audio file using Edge-TTS in a Streamlit compatible way."""
    # Edge-TTS requires the language code to select the best voice.
    # We'll stick to a common English voice for simplicity here, but a full implementation
    # would map language_code to appropriate VOICE_NAME.
    
    voice = VOICE_NAME
    if language_code == "hi": voice = "hi-IN-MadhurNeural"
    elif language_code == "es": voice = "es-ES-ElviraNeural"
    # Add more voice mappings here if needed
    
    communicate = Communicate(text, voice)
    
    # Use a BytesIO buffer to store the audio data in memory
    audio_buffer = BytesIO()
    
    # Edge-TTS uses async, which needs to be handled in Streamlit's sync context
    async def generate():
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

    try:
        # Run the async function
        asyncio.run(generate())
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        st.error(f"TTS Error: Could not generate voice for response. {e}")
        return None

# =====================================================================
# UI AND HANDLERS
# =====================================================================

def display_chat_messages():
    """Displays all chat messages in the Streamlit app."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display audio player if it's an assistant message and an audio file exists
            if message["role"] == "assistant" and "audio" in message and message["audio"] is not None:
                st.audio(message["audio"], format="audio/mp3")


def is_valid_github_raw_url(url):
    """Checks if the URL is a raw GitHub URL for supported text files."""
    return re.match(r"https://raw\.githubusercontent\.com/.+\.(txt|md|csv|html|xml)$", url, re.IGNORECASE)

def handle_file_upload(uploaded_file):
    """Handles parsing and chunking for a single uploaded file."""
    file_ext = uploaded_file.name.split('.')[-1].lower()
    file_contents = None

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
        st.error(f"Failed to process {uploaded_file.name} ({file_ext}): {e}")
        return 0
    return 0

def handle_url_upload(github_url):
    """Handles fetching and processing a file from a GitHub raw URL."""
    if not is_valid_github_raw_url(github_url):
        st.error("Invalid URL format. Please use a raw GitHub URL ending in a supported extension.")
        return 0
    
    file_ext = github_url.split('.')[-1].lower()
    
    try:
        response = requests.get(github_url)
        response.raise_for_status()
        file_contents = response.text
        
        # Simple extraction for text files fetched via URL
        if file_ext in ["html", "xml"]:
            # Re-process to extract text if it's HTML/XML
            soup = BeautifulSoup(file_contents, 'lxml')
            if file_ext == 'html':
                for script_or_style in soup(['script', 'style']): script_or_style.decompose()
            file_contents = soup.get_text(' ', strip=True)
            
        elif file_ext == "csv":
            # Process CSV content fetched via URL (requires temporary file or string buffer)
            csv_buffer = BytesIO(file_contents.encode('utf-8'))
            df = pd.read_csv(csv_buffer)
            file_contents = df.to_markdown(index=False)

        documents = split_documents(file_contents)
        process_and_store_documents(documents)
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
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                selected_language = st.session_state.selected_language
                
                # --- RAG/CAG Pipeline ---
                response_text = rag_pipeline(prompt, selected_language)
                
                audio_buffer = None
                if st.session_state.response_mode == 'Voice' and not response_text.startswith("Error:"):
                    # Use the code, not the full name, for the TTS language selection/logic
                    language_code = LANGUAGE_DICT[selected_language]
                    audio_buffer = generate_tts_audio(response_text, language_code)
                    
                st.markdown(response_text)
                if audio_buffer:
                    st.audio(audio_buffer, format="audio/mp3")

        # Store response details in session state
        message_to_store = {"role": "assistant", "content": response_text}
        if audio_buffer:
            message_to_store["audio"] = audio_buffer
            
        st.session_state.messages.append(message_to_store)
        
        # Update chat history title
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

    # Initialize dependencies: db_client, model, and gemini_client
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
        
        if st.button("🔄 Start New Chat & Clear RAG Data", use_container_width=True):
            st.session_state.messages = []
            clear_chroma_data() # Clears documents AND CAG cache
            st.session_state.chat_history = {}
            st.session_state.current_chat_id = None
            st.experimental_rerun()

        st.subheader("Chat History")
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            # ... (chat history display logic remains the same)
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
                
    # Main content area
    st.title("🧠 Gemini RAG AI Agent")
    
    # --- Introductory Message (as requested) ---
    st.markdown("""
        **A powerful Retrieval-Augmented Generation (RAG) system with advanced features:**
        * **📂 File Support:** Process documents like **.pdf**, **.txt**, **.csv**, **.html**, **.xml**, and **GitHub raw files**.
        * **🗣️ Multilingual:** Supports **16 languages** (see sidebar).
        * **💰 Cost Optimized (CAG):** Utilizes **Context Augment Generation** caching to reduce token costs on repeated questions.
        * **🔊 Response Mode:** Capacity to respond in **Text** or **Voice** mode (TTS Engine: Edge-TTS).
    """)
    st.markdown("---")
    
    # Document upload/processing section
    with st.container():
        st.subheader("Add Context Documents (Knowledge Base)")
        uploaded_files = st.file_uploader(
            "Upload files (.txt, .pdf, .csv, .html, .xml)",
            type=["txt", "pdf", "csv", "html", "xml"],
            accept_multiple_files=True
        )
        github_url = st.text_input("Enter a GitHub raw URL (.txt, .md, .csv, .html, .xml):")

        # Process uploaded files
        if uploaded_files:
            if st.button(f"Process {len(uploaded_files)} Uploaded File(s)"):
                with st.spinner("Processing files and creating vector embeddings..."):
                    total_chunks = 0
                    for uploaded_file in uploaded_files:
                        total_chunks += handle_file_upload(uploaded_file)
                    
                    if total_chunks > 0:
                        st.success(f"Successfully processed {len(uploaded_files)} file(s) into {total_chunks} chunks!")
                    else:
                        st.warning("No new content was added to the knowledge base or processing failed.")

        # Process GitHub URL
        if github_url:
            if st.button("Process URL"):
                with st.spinner("Fetching, parsing, and processing file from URL..."):
                    total_chunks = handle_url_upload(github_url)
                    if total_chunks > 0:
                        st.success(f"File from URL processed into {total_chunks} chunks! You can now chat about its contents.")
                    # Error messages are handled inside handle_url_upload

    st.markdown("---")
    
    # Chat display and input
    display_chat_messages()
    handle_user_input()

if __name__ == "__main__":
    main_ui()
