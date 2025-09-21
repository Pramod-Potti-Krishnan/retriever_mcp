#!/usr/bin/env python3
"""
Insert Test Data Script

Creates sample documents and embeddings in Supabase for testing.
"""

import os
import sys
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# Load environment variables
load_dotenv()

# Sample documents for testing
SAMPLE_DOCUMENTS = [
    {
        "filename": "ai_fundamentals.txt",
        "content": """Artificial Intelligence Fundamentals

Machine learning is a subset of artificial intelligence that enables computers to learn from data.
Deep learning is a specialized form of machine learning using neural networks with multiple layers.
Natural language processing allows machines to understand and generate human language.
Computer vision enables machines to interpret and understand visual information from the world.
Reinforcement learning trains agents through reward and punishment mechanisms."""
    },
    {
        "filename": "python_basics.txt",
        "content": """Python Programming Basics

Python is a high-level, interpreted programming language known for its simplicity.
Variables in Python are dynamically typed and don't require explicit declaration.
Functions are defined using the def keyword and support default parameters.
Python uses indentation to define code blocks instead of curly braces.
Lists, dictionaries, and tuples are fundamental data structures in Python."""
    },
    {
        "filename": "web_development.txt",
        "content": """Modern Web Development

HTML provides the structure and content of web pages.
CSS is used for styling and layout of web elements.
JavaScript adds interactivity and dynamic behavior to websites.
React is a popular library for building user interfaces.
RESTful APIs enable communication between frontend and backend systems."""
    },
    {
        "filename": "data_science.txt",
        "content": """Data Science Essentials

Data preprocessing is crucial for preparing raw data for analysis.
Exploratory data analysis helps understand patterns and relationships in data.
Statistical modeling provides frameworks for making predictions from data.
Data visualization communicates insights effectively through charts and graphs.
Feature engineering creates new variables to improve model performance."""
    },
    {
        "filename": "cloud_computing.txt",
        "content": """Cloud Computing Overview

Infrastructure as a Service provides virtualized computing resources.
Platform as a Service offers development and deployment environments.
Software as a Service delivers applications over the internet.
Containerization with Docker ensures consistent deployment across environments.
Kubernetes orchestrates container deployment and scaling in production."""
    }
]

def create_test_data(user_id: str = None, session_id: str = None, project_id: str = "-"):
    """Create test documents and embeddings."""

    # Use default test values if not provided
    if not user_id:
        user_id = "550e8400-e29b-41d4-a716-446655440000"
    if not session_id:
        session_id = "session-" + str(uuid.uuid4())[:8]

    print(f"📝 Creating test data for:")
    print(f"   User ID: {user_id}")
    print(f"   Session ID: {session_id}")
    print(f"   Project ID: {project_id}")
    print()

    # Initialize clients
    try:
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_API_KEY")
        )
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("✅ Clients initialized")
    except Exception as e:
        print(f"❌ Failed to initialize clients: {e}")
        return False

    documents_created = 0
    embeddings_created = 0

    for doc_data in SAMPLE_DOCUMENTS:
        try:
            # Create document ID
            doc_id = str(uuid.uuid4())

            # Split content into chunks (by lines for simplicity)
            chunks = [line.strip() for line in doc_data["content"].split("\n") if line.strip()]

            # Create document record
            document = {
                "id": doc_id,
                "user_id": user_id,
                "session_id": session_id,
                "project_id": project_id,
                "filename": doc_data["filename"],
                "file_type": "text/plain",
                "file_size": len(doc_data["content"]),
                "total_chunks": len(chunks),
                "processing_status": "completed",
                "metadata": {
                    "test_data": True,
                    "created_by": "insert_test_data.py"
                }
            }

            # Insert document
            supabase.table("documents").insert(document).execute()
            documents_created += 1
            print(f"📄 Created document: {doc_data['filename']}")

            # Create embeddings for each chunk
            embedding_records = []
            for idx, chunk_text in enumerate(chunks):
                # Generate embedding
                response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk_text
                )
                embedding = response.data[0].embedding

                embedding_record = {
                    "document_id": doc_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "project_id": project_id,
                    "chunk_text": chunk_text,
                    "embedding": embedding,
                    "chunk_index": idx,
                    "chunk_metadata": {
                        "chunk_number": idx + 1,
                        "total_chunks": len(chunks)
                    }
                }
                embedding_records.append(embedding_record)

            # Batch insert embeddings
            if embedding_records:
                supabase.table("document_embeddings").insert(embedding_records).execute()
                embeddings_created += len(embedding_records)
                print(f"   ✅ Created {len(embedding_records)} embeddings")

        except Exception as e:
            print(f"   ❌ Error processing {doc_data['filename']}: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Data Creation Summary")
    print("=" * 50)
    print(f"✅ Documents created: {documents_created}")
    print(f"✅ Embeddings created: {embeddings_created}")
    print(f"\n🎉 Test data successfully created!")
    print(f"\nYou can now test with:")
    print(f"  - User ID: {user_id}")
    print(f"  - Session ID: {session_id}")
    print(f"  - Project ID: {project_id}")

    return True

def cleanup_test_data(user_id: str = None, session_id: str = None):
    """Remove test data from database."""

    if not user_id:
        user_id = "550e8400-e29b-41d4-a716-446655440000"

    print("🗑️ Cleaning up test data...")

    try:
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_API_KEY")
        )

        # Delete documents (cascades to embeddings)
        query = supabase.table("documents").delete().eq("user_id", user_id)

        if session_id:
            query = query.eq("session_id", session_id)

        result = query.execute()

        print("✅ Test data cleaned up")
        return True

    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return False

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Insert test data for MCP server testing")
    parser.add_argument("--user-id", help="User ID (UUID format)")
    parser.add_argument("--session-id", help="Session ID")
    parser.add_argument("--project-id", default="-", help="Project ID")
    parser.add_argument("--cleanup", action="store_true", help="Remove test data instead of creating")

    args = parser.parse_args()

    # Check environment
    required_vars = ["SUPABASE_URL", "SUPABASE_API_KEY", "OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"❌ Missing environment variables: {missing}")
        print("Please set them in your .env file")
        return 1

    print("=" * 50)
    print("📦 Document Retrieval MCP - Test Data Inserter")
    print("=" * 50)
    print()

    if args.cleanup:
        cleanup_test_data(args.user_id, args.session_id)
    else:
        create_test_data(args.user_id, args.session_id, args.project_id)

    return 0

if __name__ == "__main__":
    sys.exit(main())