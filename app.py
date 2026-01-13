# app.py
# 📚 RAG AI Agent with Multilingual & Voice Support 🎤
# Status: RAG System with file support (PDF, TXT, CSV, HTML, XML, GitHub raw).
# Features: Conversation Augmented Generation (CAG) for cost-saving, Voice Response Mode, Multilingual.
import streamlit as st
import os, sys, tempfile, uuid, time, io, asyncio, datetime, re, base64
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import torch
# AgentOps Integration
from agentops_config import tracker
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
st.set_page_config(page_title="RAG AI Agent \ud83d\udcda", page_icon="🤖", layout="wide")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# Initialize AgentOps at app startup
if "agentops_initialized" not in st.session_state:
    tracker.initialize()
    st.session_state.agentops_initialized = True

LANGUAGE_DICT = {
    "English": "en", "Spanish": "es", "Arabic": "ar", "French": "fr", "German": "de", "Hindi": "hi",
    "Tamil": "ta", "Bengali": "bn", "Japanese": "ja", "Korean": "ko", "Russian": "ru",
    "Chinese (Simplified)": "zh-Hans", "Portuguese": "pt", "Italian": "it", "Dutch": "nl", "Turkish": "tr"
}
COLLECTION_NAME = "uploaded_documents_rag"
CACHE_EXPIRY_SECONDS = 300
