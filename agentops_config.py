import agentops
import os
import streamlit as st
from datetime import datetime
import json

class RAGAgentOpsTracker:
    """Tracks RAG operations with AgentOps"""
    
    def __init__(self):
        self.session_id = None
        self.is_initialized = False
        self.operations = []
    def initialize(self):
    """Initialize AgentOps client"""
    try:
        api_key = st.secrets.get("AGENTOPS_API_KEY")
        if not api_key:
            st.warning("⚠️ AgentOps API key not found in secrets. Set it in Streamlit Cloud.")
            return False
        
        # Initialize AgentOps and capture the session object
        session = agentops.init(api_key=api_key)
        self.is_initialized = True
        self.session_id = session.session_id
        st.success(f"✅ AgentOps initialized with session: {self.session_id}")
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize AgentOps: {str(e)}")
        return False
    
        except Exception as e:
            st.error(f"\u274c Failed to initialize AgentOps: {str(e)}")
            return False
    
    def track_document_upload(self, filename: str, file_size: int, chunk_count: int):
        """Track document upload operation"""
        try:
            operation_data = {
                "operation": "document_upload",
                "timestamp": datetime.now().isoformat(),
                "file_name": filename,
                "file_size_bytes": file_size,
                "chunks_created": chunk_count,
                "status": "completed"
            }
            self.operations.append(operation_data)
            
            if self.is_initialized:
                agentops.record(
                    {
                        "name": "document_upload",
                        "input": {"filename": filename, "size": file_size},
                        "output": {"chunks": chunk_count},
                        "success": True
                    }
                )
        except Exception as e:
            print(f"Error tracking document upload: {str(e)}")
    
    def track_rag_query(self, query: str, retrieved_chunks: int, answer: str, response_time: float):
        """Track RAG query operation"""
        try:
            operation_data = {
                "operation": "rag_query",
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "retrieved_chunks": retrieved_chunks,
                "response_time_seconds": response_time,
                "answer_length": len(answer),
                "status": "completed"
            }
            self.operations.append(operation_data)
            
            if self.is_initialized:
                agentops.record(
                    {
                        "name": "rag_query",
                        "input": {"query": query},
                        "output": {"answer_length": len(answer), "chunks_used": retrieved_chunks},
                        "execution_time": response_time,
                        "success": True
                    }
                )
        except Exception as e:
            print(f"Error tracking RAG query: {str(e)}")
    
    def track_tts_generation(self, text: str, language: str, engine: str, audio_duration: float):
        """Track TTS generation operation"""
        try:
            operation_data = {
                "operation": "tts_generation",
                "timestamp": datetime.now().isoformat(),
                "text_length": len(text),
                "language": language,
                "tts_engine": engine,
                "audio_duration_seconds": audio_duration,
                "status": "completed"
            }
            self.operations.append(operation_data)
            
            if self.is_initialized:
                agentops.record(
                    {
                        "name": "tts_generation",
                        "input": {"language": language, "engine": engine},
                        "output": {"audio_duration": audio_duration},
                        "success": True
                    }
                )
        except Exception as e:
            print(f"Error tracking TTS generation: {str(e)}")
    
    def track_error(self, operation: str, error_message: str):
        """Track error operations"""
        try:
            operation_data = {
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "error": error_message,
                "status": "failed"
            }
            self.operations.append(operation_data)
            
            if self.is_initialized:
                agentops.record(
                    {
                        "name": f"{operation}_error",
                        "error": error_message,
                        "success": False
                    }
                )
        except Exception as e:
            print(f"Error tracking error: {str(e)}")
    
    def get_session_dashboard_url(self):
        """Get AgentOps dashboard URL for current session"""
        if self.session_id:
            return f"https://app.agentops.ai/sessions/{self.session_id}"
        return None
    
    def end_session(self):
        """End the AgentOps session"""
        try:
            if self.is_initialized:
                agentops.end_session()
                st.info("\u2705 AgentOps session ended")
        except Exception as e:
            print(f"Error ending AgentOps session: {str(e)}")

# Initialize tracker globally
tracker = RAGAgentOpsTracker()
