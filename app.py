"""
🚀 Multi-Document RAG AI Agent with CAG & Voice Support
─────────────────────────────────────────────────────────
📋 Features:
  ✅ Multi-file support: PDF, TXT, CSV, HTML, XML, GitHub Raw files
  ✅ CAG (Context Augmented Generation) - Caches repeated queries for cost reduction
  ✅ Overlapping Chunking - Better context preservation
  ✅ 🌍 Multilingual support (15+ languages)
  ✅ 🎙️ Text-to-Speech (TTS) using Edge-TTS
  ✅ Response modes: Text & Voice
  ✅ Gemini 2.5 Flash API integration
  ✅ Fast responses within 1 minute
─────────────────────────────────────────────────────────
"""

import streamlit as st
import os
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import hashlib
import json
import io # <--- ADDED: Needed for BytesIO in TTS function

# PDF Processing
import PyPDF2

# Document Processing
import csv
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

# Vector DB & Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
# This line caused the error. Keeping it, but dependency must be correct.
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.schema import Document
from langchain.chains.retrieval_qa.base import RetrievalQA

# TTS
import edge_tts
import asyncio

# Web requests for GitHub raw files
import requests

# ─────────────────────────────────────────────────────────
# 🔑 CONFIGURATION & SETUP
# ─────────────────────────────────────────────────────────

# Get API key from Streamlit secrets
# Ensure you have a secrets.toml file in .streamlit/
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in Streamlit secrets!")
    st.stop()

# Language dictionary
LANGUAGE_DICT = {
    "English": "en", "Spanish": "es", "Arabic": "ar", "French": 
    "fr", "German": "de", "Hindi": "hi", "Tamil": "ta", "Bengali": "bn", 
    "Japanese": "ja", "Korean": "ko", "Russian": "ru",
    "Chinese (Simplified)": "zh-Hans", "Portuguese": "pt", "Italian": "it", 
    "Dutch": "nl", "Turkish": "tr"
}

# Session state initialization
if "documents" not in st.session_state:
    st.session_state.documents = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}  # CAG cache
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "response_mode" not in st.session_state:
    st.session_state.response_mode = "Text" # Default mode

# ─────────────────────────────────────────────────────────
# 📄 DOCUMENT LOADERS
# ─────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF: {str(e)}")
        return ""

def extract_text_from_txt(txt_file) -> str:
    """Extract text from TXT file"""
    try:
        # Reset file pointer and decode
        txt_file.seek(0)
        return txt_file.read().decode('utf-8')
    except Exception as e:
        st.error(f"❌ Error reading TXT: {str(e)}")
        return ""

def extract_text_from_csv(csv_file) -> str:
    """Extract text from CSV file"""
    try:
        csv_file.seek(0)
        # Read file content as string
        content_string = csv_file.read().decode('utf-8')
        # Use io.StringIO to treat the string as a file for the CSV reader
        reader = csv.reader(io.StringIO(content_string).readlines())
        text = ""
        for row in reader:
            text += " | ".join(row) + "\n"
        return text
    except Exception as e:
        st.error(f"❌ Error reading CSV: {str(e)}")
        return ""

def extract_text_from_html(html_file) -> str:
    """Extract text from HTML file"""
    try:
        html_file.seek(0)
        html_content = html_file.read().decode('utf-8')
        
        class HTMLTextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
            
            def handle_data(self, data):
                if data.strip():
                    self.text.append(data.strip())
            
        extractor = HTMLTextExtractor()
        extractor.feed(html_content)
        return " ".join(extractor.text)
    except Exception as e:
        st.error(f"❌ Error reading HTML: {str(e)}")
        return ""

def extract_text_from_xml(xml_file) -> str:
    """Extract text from XML file"""
    try:
        xml_file.seek(0)
        xml_content = xml_file.read().decode('utf-8')
        root = ET.fromstring(xml_content)
        text = ""
        
        def extract_all_text(elem):
            nonlocal text
            if elem.text:
                text += elem.text.strip() + " "
            for child in elem:
                extract_all_text(child)
            if elem.tail:
                text += elem.tail.strip() + " "
        
        extract_all_text(root)
        return text
    except Exception as e:
        st.error(f"❌ Error reading XML: {str(e)}")
        return ""

def extract_text_from_github_raw(url: str) -> str:
    """Extract text from GitHub raw file URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"❌ Error fetching GitHub raw file: {str(e)}")
        return ""

def process_uploaded_file(uploaded_file) -> Tuple[str, str]:
    """Process uploaded file and return (text, file_type)"""
    file_name = uploaded_file.name.lower()
    
    # Check if a file object is present (Streamlit handles file upload objects)
    if uploaded_file is None:
        return "", "NONE"

    if file_name.endswith('.pdf'):
        text = extract_text_from_pdf(uploaded_file)
        return text, "PDF"
    elif file_name.endswith('.txt'):
        text = extract_text_from_txt(uploaded_file)
        return text, "TXT"
    elif file_name.endswith('.csv'):
        text = extract_text_from_csv(uploaded_file)
        return text, "CSV"
    elif file_name.endswith('.html') or file_name.endswith('.htm'):
        text = extract_text_from_html(uploaded_file)
        return text, "HTML"
    elif file_name.endswith('.xml'):
        text = extract_text_from_xml(uploaded_file)
        return text, "XML"
    else:
        st.warning(f"⚠️ Unsupported file type: {file_name}")
        return "", "UNKNOWN"

# ─────────────────────────────────────────────────────────
# 🔄 OVERLAPPING CHUNKING STRATEGY
# ─────────────────────────────────────────────────────────

def create_overlapping_chunks(text: str, chunk_size: int = 1000, 
                             overlap: int = 200) -> List[Document]:
    """
    Create overlapping chunks to preserve context
    Better for semantic understanding
    """
    if not text:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

# ─────────────────────────────────────────────────────────
# 🧠 VECTOR STORE & EMBEDDINGS
# ─────────────────────────────────────────────────────────

def create_vector_store(documents: List[Document]) -> FAISS:
    """Create FAISS vector store from documents"""
    if not documents:
        return None
    
    try:
        with st.spinner("🔄 Creating embeddings... Please wait"):
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GEMINI_API_KEY
            )
            # FAISS.from_documents can handle an empty list if not checked, but 
            # we check 'documents' above anyway.
            vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return None

def handle_upload_and_processing(uploaded_files, github_url):
    """Handles file uploads, GitHub URL and processing to create the vector store."""
    all_chunks = []
    
    if uploaded_files:
        for file in uploaded_files:
            text, file_type = process_uploaded_file(file)
            if text:
                all_chunks.extend(create_overlapping_chunks(text))
                st.session_state.documents.append(file.name)
    
    if github_url and github_url.strip():
        text = extract_text_from_github_raw(github_url.strip())
        if text:
            all_chunks.extend(create_overlapping_chunks(text))
            st.session_state.documents.append(f"GitHub: {github_url[:30]}...")
            
    if all_chunks:
        st.session_state.vector_store = create_vector_store(all_chunks)
        if st.session_state.vector_store:
            st.success(f"✅ Successfully processed {len(st.session_state.documents)} documents into {len(all_chunks)} chunks.")
        else:
            st.error("❌ Failed to create vector store. Check API key and logs.")
    else:
        st.info("ℹ️ No content extracted or processed.")

# ─────────────────────────────────────────────────────────
# 💾 CAG (Context Augmented Generation) - Query Cache
# ─────────────────────────────────────────────────────────

def get_query_hash(query: str, language: str) -> str:
    """Generate hash for query caching"""
    query_key = f"{query.lower().strip()}_{language}"
    return hashlib.md5(query_key.encode()).hexdigest()

def check_query_cache(query: str, language: str) -> Optional[str]:
    """Check if query response is cached (CAG)"""
    cache_key = get_query_hash(query, language)
    if cache_key in st.session_state.query_cache:
        cached_response = st.session_state.query_cache[cache_key]
        st.info(f"📦 Using cached response (CAG - Cost optimized)")
        return cached_response
    return None

def cache_query_response(query: str, language: str, response: str):
    """Cache query response for future use (CAG)"""
    cache_key = get_query_hash(query, language)
    st.session_state.query_cache[cache_key] = response

# ─────────────────────────────────────────────────────────
# 🎙️ TEXT-TO-SPEECH (Edge-TTS)
# ─────────────────────────────────────────────────────────

async def text_to_speech_async(text: str, language_code: str) -> Optional[bytes]:
    """Convert text to speech using Edge-TTS"""
    # NOTE: The TTS function is wrapped in a try/except, which is good.
    try:
        voice_map = {
            "en": "en-US-AriaNeural",
            "es": "es-ES-AlvaroNeural",
            "fr": "fr-FR-DeniseNeural",
            "de": "de-DE-ConradNeural",
            "hi": "hi-IN-MadhurNeural",
            "ta": "ta-IN-ValluvarNeural",
            "ja": "ja-JP-NanamiNeural",
            "zh-Hans": "zh-CN-XiaoxiaoNeural",
            "pt": "pt-BR-BrendaNeural",
            "it": "it-IT-IsabellaNeural",
            "ar": "ar-SA-LelaNeural",
            "bn": "bn-IN-BashkarNeural",
            "ko": "ko-KR-SunHiNeural",
            "ru": "ru-RU-DariyaNeural",
            "tr": "tr-TR-EmelNeural",
            "nl": "nl-NL-ColetteNeural"
        }
        
        voice = voice_map.get(language_code, "en-US-AriaNeural")
        
        communicate = edge_tts.Communicate(text, voice)
        audio_file = io.BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
        
        audio_file.seek(0)
        return audio_file.getvalue()
    except Exception as e:
        # st.error is not for async context, but kept for consistency
        print(f"❌ TTS Error: {str(e)}") 
        return None

def generate_speech(text: str, language_code: str) -> Optional[bytes]:
    """Wrapper for async TTS"""
    # This wrapper is necessary because Streamlit and Edge-TTS use different async patterns
    try:
        # Create a new event loop for synchronous execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(text_to_speech_async(text, language_code))
        loop.close()
        return result
    except Exception as e:
        st.error(f"❌ Speech generation failed: {str(e)}")
        return None

# ─────────────────────────────────────────────────────────
# 🤖 RAG QUERY ENGINE
# ─────────────────────────────────────────────────────────

def query_rag(query: str, k: int = 5, language: str = "English") -> Dict:
    """
    Query RAG system with caching (CAG) support
    Returns: {"answer": str, "sources": List, "response_time": float, "from_cache": bool}
    """
    start_time = time.time()
    language_code = LANGUAGE_DICT.get(language, "en")
    
    # Check CAG cache first
    cached_response = check_query_cache(query, language)
    if cached_response:
        return {
            "answer": cached_response,
            "sources": ["📦 Cached Response (CAG)"],
            "response_time": time.time() - start_time,
            "from_cache": True
        }
    
    if st.session_state.vector_store is None:
        return {
            "answer": "❌ No documents loaded. Please upload documents first.",
            "sources": [],
            "response_time": 0,
            "from_cache": False
        }
    
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.7,
            max_output_tokens=1024
        )
        
        # Create RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True,
            verbose=False
        )
        
        # Query (The query is implicitly translated by the LLM's multi-lingual ability)
        # However, for RAG, the query should be in the document language. 
        # For multi-lingual RAG, the prompt should encourage the LLM to answer in the 
        # requested language.
        full_query = f"Answer the following question in {language}: {query}"
        
        result = qa_chain({"query": full_query})
        
        answer = result.get("result", "No answer found")
        # Extract the first 100 characters of the source document content for display
        sources = [doc.page_content[:100].replace('\n', ' ') + "..." for doc in result.get("source_documents", [])]
        
        # Cache the response
        cache_query_response(query, language, answer)
        
        response_time = time.time() - start_time
        
        return {
            "answer": answer,
            "sources": sources,
            "response_time": response_time,
            "from_cache": False
        }
    
    except Exception as e:
        return {
            "answer": f"❌ Error during RAG query: {str(e)}",
            "sources": [],
            "response_time": time.time() - start_time,
            "from_cache": False
        }

# ─────────────────────────────────────────────────────────
# 🎨 STREAMLIT UI
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Document RAG AI Agent",
    page_icon="🚀",
    layout="wide"
)

def main_ui():
    """Defines the main Streamlit UI and interaction logic."""
    st.title("🚀 Multi-Document RAG AI Agent")
    st.caption("Powered by Gemini 2.5 Flash, CAG, and Edge-TTS")

    # --- Sidebar for Configuration and Upload ---
    with st.sidebar:
        st.header("📂 Data & Configuration")
        
        # File Uploader
        uploaded_files = st.file_uploader(
            "Upload Files (PDF, TXT, CSV, HTML, XML)",
            type=['pdf', 'txt', 'csv', 'html', 'xml'],
            accept_multiple_files=True
        )
        
        # GitHub Raw URL Input
        github_url = st.text_input(
            "Or, enter GitHub Raw File URL:",
            placeholder="e.g., https://raw.githubusercontent.com/..."
        )

        if st.button("Process Documents"):
            st.session_state.documents = [] # Clear old list
            st.session_state.vector_store = None # Clear old store
            st.session_state.query_cache = {} # Clear cache for new documents
            
            handle_upload_and_processing(uploaded_files, github_url)
        
        st.markdown("---")
        st.subheader("Agent Settings")
        
        # Language Selection
        selected_language = st.selectbox(
            "Select Response Language (🌍 Multilingual)",
            options=list(LANGUAGE_DICT.keys()),
            index=0 # English default
        )
        
        # Response Mode
        st.session_state.response_mode = st.radio(
            "Response Mode",
            ["Text", "Voice"],
            index=0,
            horizontal=True,
            key="response_mode_radio"
        )
        
        st.markdown("---")
        st.subheader("System Status")
        st.info(f"Loaded Docs: **{len(st.session_state.documents)}**")
        st.info(f"Vector Store: **{'Ready' if st.session_state.vector_store else 'Empty'}**")
        st.info(f"CAG Cache Size: **{len(st.session_state.query_cache)}**")
        
    # --- Main Chat Interface ---
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")

    # Handle user input
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get RAG response
        with st.spinner("Thinking..."):
            rag_result = query_rag(user_query, language=selected_language)
        
        assistant_response = rag_result["answer"]
        sources = rag_result["sources"]
        response_time = rag_result["response_time"]
        from_cache = rag_result["from_cache"]

        # Format and display assistant response
        response_markdown = f"**{assistant_response}**\n\n"
        response_markdown += f"***\n"
        response_markdown += f"**Info:** ⏱️ Took **{response_time:.2f}s** ({'Cached' if from_cache else 'Live RAG'})"
        
        if not from_cache:
            response_markdown += "\n\n**Sources Used:**"
            for i, source in enumerate(sources):
                response_markdown += f"\n- *Chunk {i+1}:* {source}"

        audio_bytes = None
        if st.session_state.response_mode == "Voice" and not from_cache and "❌ Error" not in assistant_response:
            with st.spinner(f"Generating speech in {selected_language}..."):
                # Use only the main answer text for TTS
                audio_bytes = generate_speech(assistant_response, LANGUAGE_DICT[selected_language])

        # Display and record assistant message
        with st.chat_message("assistant"):
            st.markdown(response_markdown)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
        
        # Update chat history with full response details
        assistant_message = {
            "role": "assistant", 
            "content": response_markdown,
            "audio": audio_bytes
        }
        st.session_state.chat_history.append(assistant_message)

if __name__ == '__main__':
    main_ui()
