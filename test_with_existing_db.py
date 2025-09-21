#!/usr/bin/env python3
"""
Test script for your existing Supabase database with document_chunks table.
"""

import os
import asyncio
import asyncpg
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection details
DB_CONFIG = {
    "host": "aws-0-us-east-2.pooler.supabase.com",
    "port": 5432,
    "database": "postgres",
    "user": "postgres.eshvntffcestlfuofwhv",
    "password": "pramodpotti"
}

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

async def test_database_connection():
    """Test the database connection and explore schema."""
    print("🔍 Testing Database Connection...")
    print("=" * 50)

    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        print("✅ Connected to database successfully!")

        # Check documents table
        docs_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
        print(f"📄 Documents table: {docs_count} records")

        # Check document_chunks table
        chunks_count = await conn.fetchval("SELECT COUNT(*) FROM document_chunks")
        print(f"📦 Document chunks table: {chunks_count} records")

        # Check if embeddings exist
        embeddings_count = await conn.fetchval(
            "SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL"
        )
        print(f"🎯 Chunks with embeddings: {embeddings_count}")

        # Get sample data
        print("\n📋 Sample Document:")
        sample_doc = await conn.fetchrow("""
            SELECT id, filename, user_id, session_id, project_id
            FROM documents
            LIMIT 1
        """)
        if sample_doc:
            for key, value in sample_doc.items():
                print(f"  {key}: {value}")

        print("\n📋 Sample Chunk:")
        sample_chunk = await conn.fetchrow("""
            SELECT
                dc.id,
                dc.document_id,
                dc.chunk_index,
                LEFT(dc.text_content, 100) as text_preview,
                dc.user_id,
                dc.session_id
            FROM document_chunks dc
            WHERE dc.embedding IS NOT NULL
            LIMIT 1
        """)
        if sample_chunk:
            for key, value in sample_chunk.items():
                print(f"  {key}: {value}")

        await conn.close()
        return True

    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

async def test_semantic_search(query: str = "machine learning"):
    """Test semantic search on the existing database."""
    print(f"\n🔍 Testing Semantic Search for: '{query}'")
    print("=" * 50)

    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

        # Generate query embedding
        print("📊 Generating query embedding...")
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding
        print(f"✅ Generated {len(query_embedding)}-dimensional embedding")

        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)

        # Perform semantic search
        print("\n🔎 Searching for similar documents...")

        # Convert embedding list to PostgreSQL array string format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        results = await conn.fetch("""
            SELECT
                dc.text_content,
                d.filename,
                dc.chunk_index,
                1 - (dc.embedding <=> $1::vector) AS similarity
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.embedding IS NOT NULL
            ORDER BY dc.embedding <=> $1::vector
            LIMIT 5
        """, embedding_str)

        if results:
            print(f"\n✅ Found {len(results)} relevant chunks:\n")
            for idx, row in enumerate(results, 1):
                print(f"{idx}. File: {row['filename']}")
                print(f"   Chunk: {row['chunk_index']}")
                print(f"   Similarity: {row['similarity']:.3f}")
                print(f"   Text: {row['text_content'][:150]}...")
                print()
        else:
            print("❌ No results found")

        await conn.close()
        return True

    except Exception as e:
        print(f"❌ Search error: {e}")
        return False

async def test_user_documents(user_id: str = None, session_id: str = None):
    """Test listing documents for a specific user/session."""
    print(f"\n📋 Testing Document Listing")
    print("=" * 50)

    try:
        conn = await asyncpg.connect(**DB_CONFIG)

        # First, get all unique users/sessions
        print("👥 Finding users and sessions in database...")
        users = await conn.fetch("""
            SELECT DISTINCT user_id, session_id, COUNT(*) as doc_count
            FROM documents
            GROUP BY user_id, session_id
            ORDER BY doc_count DESC
            LIMIT 5
        """)

        if users:
            print(f"Found {len(users)} user/session combinations:\n")
            for user in users:
                print(f"  User: {user['user_id']}")
                print(f"  Session: {user['session_id']}")
                print(f"  Documents: {user['doc_count']}")
                print()

            # Use the first user/session if not specified
            if not user_id:
                user_id = str(users[0]['user_id'])
                session_id = users[0]['session_id']

        if user_id:
            print(f"\n📄 Documents for user {user_id}:")
            docs = await conn.fetch("""
                SELECT id, filename, created_at, total_chunks
                FROM documents
                WHERE user_id = $1
                AND ($2 IS NULL OR session_id = $2)
                LIMIT 10
            """, user_id, session_id)

            if docs:
                for doc in docs:
                    print(f"  • {doc['filename']} ({doc['total_chunks']} chunks)")
            else:
                print("  No documents found")

        await conn.close()
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 TESTING EXISTING SUPABASE DATABASE")
    print("=" * 60)

    # Test database connection
    if await test_database_connection():
        # Test semantic search
        await test_semantic_search("machine learning")
        await test_semantic_search("python programming")

        # Test user documents
        await test_user_documents()

    print("\n" + "=" * 60)
    print("✅ Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())