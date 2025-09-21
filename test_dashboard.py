#!/usr/bin/env python3
"""
Streamlit Dashboard for Testing Document Retrieval MCP Server

A simple web interface to test all MCP server functionality.
"""

import os
import sys
import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
import asyncpg

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
        "tables": False,
        "embeddings_count": 0
    }

    try:
        supabase, openai_client = init_clients()

        # Check Supabase
        result = supabase.table("documents").select("id").limit(1).execute()
        status["supabase"] = True

        # Check tables exist
        docs_count = supabase.table("documents").select("id", count="exact").execute()
        embeddings_count = supabase.table("document_embeddings").select("id", count="exact").execute()
        status["tables"] = True
        status["embeddings_count"] = embeddings_count.count if embeddings_count.count else 0

        # Check OpenAI
        test_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input="test"
        )
        status["openai"] = True

    except Exception as e:
        st.error(f"Connection error: {str(e)}")

    return status

def search_documents(query: str, user_id: str, session_id: str, project_id: str = "-", top_k: int = 5, threshold: float = 0.7):
    """Test the search_documents tool."""
    try:
        supabase, openai_client = init_clients()

        # Generate embedding for query
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding

        # Search in Supabase
        # Note: This is a simplified version - actual MCP server uses asyncpg
        results = supabase.rpc(
            'search_embeddings',
            {
                'query_embedding': query_embedding,
                'user_id': user_id,
                'session_id': session_id,
                'project_id': project_id,
                'similarity_threshold': threshold,
                'match_count': top_k
            }
        ).execute()

        return {"success": True, "results": results.data if results.data else []}

    except Exception as e:
        return {"success": False, "error": str(e)}

def get_document_context(document_id: str, user_id: str, session_id: str):
    """Test the get_document_context tool."""
    try:
        supabase, _ = init_clients()

        # Get document chunks
        result = supabase.table("document_embeddings") \
            .select("*, documents(filename, file_type, total_chunks)") \
            .eq("document_id", document_id) \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .order("chunk_index") \
            .execute()

        return {"success": True, "chunks": result.data if result.data else []}

    except Exception as e:
        return {"success": False, "error": str(e)}

def list_user_documents(user_id: str, session_id: str, project_id: Optional[str] = None):
    """Test the list_user_documents tool."""
    try:
        supabase, _ = init_clients()

        query = supabase.table("documents") \
            .select("*") \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .eq("processing_status", "completed")

        if project_id:
            query = query.eq("project_id", project_id)

        result = query.execute()

        return {"success": True, "documents": result.data if result.data else []}

    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    """Main dashboard application."""

    # Title and description
    st.title("🔍 Document Retrieval MCP Server - Test Dashboard")
    st.markdown("Test all MCP server tools and verify functionality with your Supabase database.")

    # Sidebar for connection status
    with st.sidebar:
        st.header("🔌 Connection Status")

        if st.button("Check Connections", type="primary"):
            with st.spinner("Checking..."):
                status = check_connection_status()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Supabase", "✅ Connected" if status["supabase"] else "❌ Failed")
                st.metric("Tables", "✅ Found" if status["tables"] else "❌ Missing")
            with col2:
                st.metric("OpenAI", "✅ Connected" if status["openai"] else "❌ Failed")
                st.metric("Embeddings", f"{status['embeddings_count']:,}")

        st.divider()

        # Test parameters
        st.header("🔧 Test Parameters")
        user_id = st.text_input("User ID", value="550e8400-e29b-41d4-a716-446655440000")
        session_id = st.text_input("Session ID", value="session-123")
        project_id = st.text_input("Project ID", value="-")

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Search Documents",
        "📄 Get Document Context",
        "📋 List Documents",
        "🔗 Similar Chunks",
        "➕ Insert Test Data"
    ])

    # Tab 1: Search Documents
    with tab1:
        st.header("Search Documents")
        st.markdown("Test semantic search across document embeddings.")

        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Search Query", placeholder="Enter your search query...")
        with col2:
            top_k = st.number_input("Top K Results", min_value=1, max_value=20, value=5)
            threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)

        if st.button("Search", type="primary", key="search"):
            if query:
                with st.spinner("Searching..."):
                    result = search_documents(query, user_id, session_id, project_id, top_k, threshold)

                if result["success"]:
                    if result["results"]:
                        st.success(f"Found {len(result['results'])} results")

                        for idx, doc in enumerate(result["results"], 1):
                            with st.expander(f"Result {idx} - Similarity: {doc.get('similarity', 'N/A'):.3f}"):
                                st.write(f"**Chunk Text:** {doc.get('chunk_text', 'N/A')}")
                                st.write(f"**Document ID:** {doc.get('document_id', 'N/A')}")
                                st.write(f"**Chunk Index:** {doc.get('chunk_index', 'N/A')}")
                    else:
                        st.warning("No results found matching your query.")
                else:
                    st.error(f"Search failed: {result.get('error', 'Unknown error')}")
            else:
                st.warning("Please enter a search query.")

    # Tab 2: Get Document Context
    with tab2:
        st.header("Get Document Context")
        st.markdown("Retrieve full document content or specific chunks.")

        document_id = st.text_input("Document ID", placeholder="Enter document UUID...")

        if st.button("Get Context", type="primary", key="context"):
            if document_id:
                with st.spinner("Retrieving..."):
                    result = get_document_context(document_id, user_id, session_id)

                if result["success"]:
                    if result["chunks"]:
                        st.success(f"Found {len(result['chunks'])} chunks")

                        # Display document info
                        if result["chunks"]:
                            doc_info = result["chunks"][0].get("documents", {})
                            if doc_info:
                                st.info(f"**File:** {doc_info.get('filename', 'N/A')} | "
                                       f"**Type:** {doc_info.get('file_type', 'N/A')} | "
                                       f"**Total Chunks:** {doc_info.get('total_chunks', 'N/A')}")

                        # Display chunks
                        for chunk in result["chunks"]:
                            with st.expander(f"Chunk {chunk.get('chunk_index', 'N/A')}"):
                                st.write(chunk.get("chunk_text", "N/A"))
                    else:
                        st.warning("No document found with this ID.")
                else:
                    st.error(f"Retrieval failed: {result.get('error', 'Unknown error')}")
            else:
                st.warning("Please enter a document ID.")

    # Tab 3: List User Documents
    with tab3:
        st.header("List User Documents")
        st.markdown("View all documents accessible to the user.")

        if st.button("List Documents", type="primary", key="list"):
            with st.spinner("Loading..."):
                result = list_user_documents(user_id, session_id, project_id if project_id != "-" else None)

            if result["success"]:
                if result["documents"]:
                    st.success(f"Found {len(result['documents'])} documents")

                    # Create DataFrame for better display
                    df_data = []
                    for doc in result["documents"]:
                        df_data.append({
                            "ID": doc.get("id", "N/A"),
                            "Filename": doc.get("filename", "N/A"),
                            "Type": doc.get("file_type", "N/A"),
                            "Size": doc.get("file_size", 0),
                            "Chunks": doc.get("total_chunks", 0),
                            "Upload Date": doc.get("upload_date", "N/A"),
                            "Status": doc.get("processing_status", "N/A")
                        })

                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No documents found for this user.")
            else:
                st.error(f"List failed: {result.get('error', 'Unknown error')}")

    # Tab 4: Similar Chunks
    with tab4:
        st.header("Get Similar Chunks")
        st.markdown("Find chunks similar to a reference chunk.")

        chunk_id = st.text_input("Reference Chunk ID", placeholder="Enter chunk UUID...")
        similar_top_k = st.number_input("Number of Similar Chunks", min_value=1, max_value=10, value=3)

        if st.button("Find Similar", type="primary", key="similar"):
            if chunk_id:
                st.info("Note: This requires direct database access. Implement get_similar_chunks function if needed.")
            else:
                st.warning("Please enter a chunk ID.")

    # Tab 5: Insert Test Data
    with tab5:
        st.header("Insert Test Data")
        st.markdown("Create sample documents and embeddings for testing.")

        st.warning("⚠️ This will insert test data into your database.")

        test_docs = st.text_area(
            "Test Document Content",
            value="Machine learning is a subset of artificial intelligence.\nNeural networks are inspired by biological neurons.\nDeep learning uses multiple layers of processing.",
            height=150
        )

        if st.button("Insert Test Data", type="secondary", key="insert"):
            try:
                supabase, openai_client = init_clients()

                # Create test document
                doc_id = str(uuid.uuid4())
                doc_data = {
                    "id": doc_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "project_id": project_id,
                    "filename": "test_document.txt",
                    "file_type": "text/plain",
                    "file_size": len(test_docs),
                    "total_chunks": len(test_docs.split("\n")),
                    "processing_status": "completed"
                }

                # Insert document
                supabase.table("documents").insert(doc_data).execute()

                # Create embeddings for each chunk
                chunks = test_docs.split("\n")
                embedding_data = []

                for idx, chunk_text in enumerate(chunks):
                    if chunk_text.strip():
                        # Generate embedding
                        response = openai_client.embeddings.create(
                            model="text-embedding-3-small",
                            input=chunk_text
                        )
                        embedding = response.data[0].embedding

                        embedding_data.append({
                            "document_id": doc_id,
                            "user_id": user_id,
                            "session_id": session_id,
                            "project_id": project_id,
                            "chunk_text": chunk_text,
                            "embedding": embedding,
                            "chunk_index": idx,
                            "chunk_metadata": {"test": True}
                        })

                # Insert embeddings
                if embedding_data:
                    supabase.table("document_embeddings").insert(embedding_data).execute()
                    st.success(f"✅ Inserted test document with {len(embedding_data)} chunks")
                    st.info(f"Document ID: {doc_id}")

            except Exception as e:
                st.error(f"Failed to insert test data: {str(e)}")

    # Footer
    st.divider()
    st.markdown(
        """
        ---
        **Document Retrieval MCP Server** | Test Dashboard v1.0

        This dashboard tests the MCP server functionality by directly accessing your Supabase database.
        For full MCP protocol testing, use the `test_mcp_direct.py` script.
        """
    )

if __name__ == "__main__":
    main()