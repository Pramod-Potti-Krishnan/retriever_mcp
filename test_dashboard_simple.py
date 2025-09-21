#!/usr/bin/env python3
"""
Simple Streamlit Dashboard for Testing Document Retrieval MCP Server
Uses direct table queries instead of RPC functions.
"""

import os
import sys
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Page configuration
st.set_page_config(
    page_title="MCP Server Test Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Initialize clients
@st.cache_resource
def init_clients():
    """Initialize Supabase and OpenAI clients."""
    supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return supabase, openai_client

def check_connection_status():
    """Check connection to Supabase and OpenAI."""
    status = {
        "supabase": False,
        "openai": False,
        "tables_exist": False,
        "documents_count": 0,
        "embeddings_count": 0
    }

    try:
        supabase, openai_client = init_clients()

        # Check Supabase connection
        try:
            # Try to query documents table
            result = supabase.table("documents").select("id").limit(1).execute()
            status["supabase"] = True
            status["tables_exist"] = True

            # Get counts
            docs = supabase.table("documents").select("*", count="exact").execute()
            status["documents_count"] = docs.count if docs.count else 0

            embeddings = supabase.table("document_chunks").select("*", count="exact").execute()
            status["embeddings_count"] = embeddings.count if embeddings.count else 0

        except Exception as e:
            if "documents" not in str(e):
                status["supabase"] = True  # Connection works, tables don't exist
            error_msg = str(e)
            if "does not exist" in error_msg:
                status["tables_exist"] = False

        # Check OpenAI
        try:
            test_response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input="test"
            )
            status["openai"] = True
        except:
            status["openai"] = False

    except Exception as e:
        st.error(f"Connection error: {str(e)}")

    return status

def simple_insert_test_data(user_id: str, session_id: str, project_id: str = "-"):
    """Insert simple test data without using RPC functions."""
    try:
        supabase, openai_client = init_clients()

        # Simple test document
        test_content = [
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "Neural networks are inspired by the human brain.",
            "Deep learning uses multiple layers of neural networks.",
            "Natural language processing helps computers understand text."
        ]

        # Create document
        doc_id = str(uuid.uuid4())
        doc_data = {
            "id": doc_id,
            "user_id": user_id,
            "session_id": session_id,
            "project_id": project_id,
            "filename": f"test_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "file_type": "text/plain",
            "total_chunks": len(test_content),
            "processing_status": "completed"
        }

        # Don't include file_size if it doesn't exist in schema
        try:
            supabase.table("documents").insert(doc_data).execute()
        except Exception as e:
            if "file_size" in str(e):
                # Try without file_size
                doc_data.pop("file_size", None)
                supabase.table("documents").insert(doc_data).execute()

        # Create embeddings for each chunk
        for idx, chunk_text in enumerate(test_content):
            # Generate embedding
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk_text
            )
            embedding = response.data[0].embedding

            embedding_data = {
                "document_id": doc_id,
                "user_id": user_id,
                "session_id": session_id,
                "project_id": project_id,
                "chunk_text": chunk_text,
                "embedding": embedding,
                "chunk_index": idx
            }

            supabase.table("document_embeddings").insert(embedding_data).execute()

        return True, doc_id, len(test_content)

    except Exception as e:
        return False, str(e), 0

def main():
    """Main dashboard application."""

    # Title and description
    st.title("🔍 Document Retrieval MCP Server - Simple Test Dashboard")
    st.markdown("Test your MCP server setup and database connection.")

    # Sidebar for connection status
    with st.sidebar:
        st.header("🔌 Connection Status")

        if st.button("Check Connections", type="primary"):
            with st.spinner("Checking..."):
                status = check_connection_status()

            col1, col2 = st.columns(2)
            with col1:
                if status["supabase"]:
                    st.success("✅ Supabase Connected")
                else:
                    st.error("❌ Supabase Failed")

                if status["tables_exist"]:
                    st.success("✅ Tables Found")
                else:
                    st.warning("⚠️ Tables Missing")
                    st.info("Run the SQL script in create_database_schema.sql")

            with col2:
                if status["openai"]:
                    st.success("✅ OpenAI Connected")
                else:
                    st.error("❌ OpenAI Failed")

                st.metric("Documents", status["documents_count"])
                st.metric("Embeddings", status["embeddings_count"])

        st.divider()

        # Test parameters
        st.header("🔧 Test Parameters")
        user_id = st.text_input("User ID", value="test-user-123")
        session_id = st.text_input("Session ID", value="test-session-001")
        project_id = st.text_input("Project ID", value="-")

    # Main content area
    tab1, tab2, tab3 = st.tabs([
        "📝 Database Setup",
        "➕ Insert Test Data",
        "📊 View Data"
    ])

    # Tab 1: Database Setup
    with tab1:
        st.header("Database Setup Instructions")

        st.markdown("""
        ### If tables don't exist, follow these steps:

        1. **Go to your Supabase Dashboard**
        2. **Navigate to SQL Editor**
        3. **Copy and run this SQL:**
        """)

        with st.expander("Show SQL Schema", expanded=True):
            st.code("""
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL,
    session_id text NOT NULL,
    project_id text DEFAULT '-',
    filename text NOT NULL,
    file_type text NOT NULL,
    total_chunks integer,
    upload_date timestamp with time zone DEFAULT now(),
    processing_status text DEFAULT 'completed',
    metadata jsonb
);

-- Create document_embeddings table
CREATE TABLE IF NOT EXISTS document_embeddings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
    user_id uuid NOT NULL,
    session_id text NOT NULL,
    project_id text DEFAULT '-',
    chunk_text text NOT NULL,
    embedding vector(1536),
    chunk_index integer NOT NULL,
    chunk_metadata jsonb,
    created_at timestamp with time zone DEFAULT now()
);

-- Create indexes
CREATE INDEX idx_embeddings_vector ON document_embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_embeddings_user_session ON document_embeddings(user_id, session_id);
CREATE INDEX idx_documents_user_session ON documents(user_id, session_id);
            """, language="sql")

        st.info("💡 The full SQL script is saved in: `create_database_schema.sql`")

    # Tab 2: Insert Test Data
    with tab2:
        st.header("Insert Test Data")
        st.markdown("Create sample documents and embeddings for testing.")

        if st.button("Insert Simple Test Data", type="primary"):
            with st.spinner("Creating test data..."):
                success, result, count = simple_insert_test_data(user_id, session_id, project_id)

            if success:
                st.success(f"✅ Created test document with ID: {result}")
                st.info(f"Added {count} chunks with embeddings")
            else:
                st.error(f"Failed to insert data: {result}")
                if "does not exist" in str(result):
                    st.warning("Please create the database tables first (see Database Setup tab)")

    # Tab 3: View Data
    with tab3:
        st.header("View Existing Data")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("List Documents"):
                try:
                    supabase, _ = init_clients()

                    # Query documents
                    result = supabase.table("documents") \
                        .select("*") \
                        .eq("user_id", user_id) \
                        .eq("session_id", session_id) \
                        .execute()

                    if result.data:
                        st.success(f"Found {len(result.data)} documents")

                        for doc in result.data:
                            with st.expander(f"📄 {doc.get('filename', 'Unknown')}"):
                                st.json(doc)
                    else:
                        st.info("No documents found")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with col2:
            if st.button("List Embeddings"):
                try:
                    supabase, _ = init_clients()

                    # Query embeddings (without the actual vectors)
                    result = supabase.table("document_chunks") \
                        .select("id, document_id, text_content, chunk_index") \
                        .eq("user_id", user_id) \
                        .eq("session_id", session_id) \
                        .limit(10) \
                        .execute()

                    if result.data:
                        st.success(f"Found {len(result.data)} embeddings (showing first 10)")

                        # Debug: Show the raw data structure
                        st.json(result.data)

                        for idx, emb in enumerate(result.data):
                            chunk_text = emb.get('text_content', 'No text available')
                            chunk_idx = emb.get('chunk_index', idx)
                            doc_id = emb.get('document_id', 'Unknown doc')

                            with st.expander(f"📄 Chunk #{chunk_idx} (Doc: {doc_id[:8]}...)"):
                                st.write(chunk_text)
                                st.caption(f"Document ID: {doc_id}")
                                st.caption(f"Chunk Index: {chunk_idx}")
                    else:
                        st.info("No embeddings found")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Footer
    st.divider()
    st.markdown(
        """
        ---
        **Simple Test Dashboard** | Document Retrieval MCP Server

        This simplified dashboard helps you:
        1. Check your database connection
        2. Create the required tables
        3. Insert test data
        4. Verify everything is working

        Once setup is complete, you can use the main dashboard or test the MCP server directly.
        """
    )

if __name__ == "__main__":
    main()