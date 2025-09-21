"""
Test configuration and fixtures for Document Retrieval MCP Server.
"""

import os
import pytest
import asyncio
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, AsyncGenerator

import asyncpg
import numpy as np
from mcp.types import ToolRequest, GetPromptResult
from cachetools import TTLCache

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from server import DocumentRetrievalServer


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_API_KEY": "test-api-key",
        "OPENAI_API_KEY": "test-openai-key",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "VECTOR_DIMENSIONS": "1536",
        "MAX_RESULTS_LIMIT": "20",
        "DEFAULT_SIMILARITY_THRESHOLD": "0.7",
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
        "DB_POOL_MIN_SIZE": "5",
        "DB_POOL_MAX_SIZE": "20",
        "CACHE_TTL": "300",
        "CACHE_MAX_SIZE": "1000"
    }

    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def sample_user_id():
    """Sample user ID for testing."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_session_id():
    """Sample session ID for testing."""
    return "test-session-123"


@pytest.fixture
def sample_project_id():
    """Sample project ID for testing."""
    return "test-project"


@pytest.fixture
def sample_document_id():
    """Sample document ID for testing."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_chunk_id():
    """Sample chunk ID for testing."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing."""
    return np.random.rand(1536).tolist()


@pytest.fixture
def sample_document_metadata():
    """Sample document metadata for testing."""
    return {
        "id": str(uuid.uuid4()),
        "filename": "test_document.pdf",
        "file_type": "pdf",
        "file_size": 1024,
        "total_chunks": 5,
        "upload_date": datetime.now(),
        "processing_status": "completed",
        "metadata": {"author": "Test Author", "title": "Test Document"}
    }


@pytest.fixture
def sample_chunk_data():
    """Sample chunk data for testing."""
    return {
        "id": str(uuid.uuid4()),
        "document_id": str(uuid.uuid4()),
        "chunk_text": "This is a sample chunk of text for testing purposes.",
        "chunk_index": 0,
        "chunk_metadata": {"page": 1, "section": "introduction"},
        "embedding": np.random.rand(1536).tolist(),
        "created_at": datetime.now()
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = AsyncMock()

    # Mock embedding response
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock()]
    embedding_response.data[0].embedding = np.random.rand(1536).tolist()

    client.embeddings.create.return_value = embedding_response
    return client


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing."""
    return MagicMock()


@pytest.fixture
def mock_db_pool():
    """Mock database connection pool for testing."""
    pool = AsyncMock()

    # Mock connection context manager
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    # Mock connection methods
    conn.fetch.return_value = []
    conn.fetchrow.return_value = None
    conn.fetchval.return_value = 0

    pool.get_size.return_value = 5
    pool.close = AsyncMock()

    return pool, conn


@pytest.fixture
async def mock_server(mock_env_vars, mock_openai_client, mock_supabase_client, mock_db_pool):
    """Create a mock DocumentRetrievalServer for testing."""
    server = DocumentRetrievalServer()

    # Mock external dependencies
    server.openai_client = mock_openai_client
    server.supabase_client = mock_supabase_client
    server.db_pool, _ = mock_db_pool
    server.metadata_cache = TTLCache(maxsize=1000, ttl=300)

    return server


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            "id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "chunk_text": "Sample text chunk 1",
            "chunk_index": 0,
            "chunk_metadata": {"page": 1},
            "filename": "test1.pdf",
            "file_type": "pdf",
            "document_metadata": {"title": "Test Document 1"},
            "similarity": 0.85
        },
        {
            "id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "chunk_text": "Sample text chunk 2",
            "chunk_index": 1,
            "chunk_metadata": {"page": 2},
            "filename": "test2.pdf",
            "file_type": "pdf",
            "document_metadata": {"title": "Test Document 2"},
            "similarity": 0.75
        }
    ]


@pytest.fixture
def sample_document_list():
    """Sample document list for testing."""
    return [
        {
            "id": str(uuid.uuid4()),
            "filename": "document1.pdf",
            "file_type": "pdf",
            "file_size": 1024,
            "total_chunks": 3,
            "upload_date": datetime.now(),
            "project_id": "test-project",
            "metadata": {"title": "Document 1"}
        },
        {
            "id": str(uuid.uuid4()),
            "filename": "document2.docx",
            "file_type": "docx",
            "file_size": 2048,
            "total_chunks": 5,
            "upload_date": datetime.now(),
            "project_id": "test-project",
            "metadata": {"title": "Document 2"}
        }
    ]


@pytest.fixture
def valid_tool_requests():
    """Valid tool request examples for testing."""
    return {
        "search_documents": ToolRequest(
            tool="search_documents",
            params={
                "query": "test query",
                "user_id": str(uuid.uuid4()),
                "session_id": "test-session",
                "project_id": "test-project",
                "top_k": 5,
                "similarity_threshold": 0.7
            }
        ),
        "get_document_context": ToolRequest(
            tool="get_document_context",
            params={
                "document_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "session_id": "test-session",
                "chunk_ids": [str(uuid.uuid4()), str(uuid.uuid4())]
            }
        ),
        "list_user_documents": ToolRequest(
            tool="list_user_documents",
            params={
                "user_id": str(uuid.uuid4()),
                "session_id": "test-session",
                "project_id": "test-project",
                "page": 1,
                "per_page": 20
            }
        ),
        "get_similar_chunks": ToolRequest(
            tool="get_similar_chunks",
            params={
                "chunk_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "session_id": "test-session",
                "top_k": 3
            }
        )
    }


@pytest.fixture
def invalid_tool_requests():
    """Invalid tool request examples for testing."""
    return {
        "search_documents_missing_query": ToolRequest(
            tool="search_documents",
            params={
                "user_id": str(uuid.uuid4()),
                "session_id": "test-session"
            }
        ),
        "search_documents_invalid_user_id": ToolRequest(
            tool="search_documents",
            params={
                "query": "test",
                "user_id": "invalid-uuid",
                "session_id": "test-session"
            }
        ),
        "get_document_context_missing_required": ToolRequest(
            tool="get_document_context",
            params={
                "document_id": str(uuid.uuid4())
            }
        ),
        "list_user_documents_invalid_pagination": ToolRequest(
            tool="list_user_documents",
            params={
                "user_id": str(uuid.uuid4()),
                "session_id": "test-session",
                "page": 0,  # Invalid page number
                "per_page": 200  # Exceeds maximum
            }
        )
    }


@pytest.fixture
def database_error_scenarios():
    """Database error scenarios for testing."""
    return {
        "connection_timeout": asyncpg.exceptions.ConnectionTimeoutError("Connection timeout"),
        "query_timeout": asyncpg.exceptions.QueryTimeoutError("Query timeout"),
        "invalid_sql": asyncpg.exceptions.InvalidSQLStatementNameError("Invalid SQL"),
        "connection_failed": asyncpg.exceptions.ConnectionFailureError("Connection failed"),
        "insufficient_privilege": asyncpg.exceptions.InsufficientPrivilegeError("Insufficient privilege")
    }


@pytest.fixture
def openai_error_scenarios():
    """OpenAI API error scenarios for testing."""
    return {
        "rate_limit": Exception("Rate limit exceeded"),
        "invalid_api_key": Exception("Invalid API key"),
        "model_not_found": Exception("Model not found"),
        "request_timeout": Exception("Request timeout")
    }


# Test data generators
def generate_mock_search_results(count: int = 5) -> List[Dict[str, Any]]:
    """Generate mock search results for testing."""
    results = []
    for i in range(count):
        results.append({
            "id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "chunk_text": f"This is sample chunk text {i + 1}",
            "chunk_index": i,
            "chunk_metadata": {"page": i + 1, "section": f"section_{i}"},
            "filename": f"document_{i + 1}.pdf",
            "file_type": "pdf",
            "document_metadata": {"title": f"Document {i + 1}"},
            "similarity": 0.9 - (i * 0.1)
        })
    return results


def generate_mock_documents(count: int = 10) -> List[Dict[str, Any]]:
    """Generate mock documents for testing."""
    documents = []
    for i in range(count):
        documents.append({
            "id": str(uuid.uuid4()),
            "filename": f"document_{i + 1}.pdf",
            "file_type": "pdf",
            "file_size": 1024 * (i + 1),
            "total_chunks": (i + 1) * 2,
            "upload_date": datetime.now(),
            "project_id": "test-project",
            "metadata": {"title": f"Test Document {i + 1}"}
        })
    return documents


# Performance test helpers
@pytest.fixture
def performance_test_config():
    """Configuration for performance tests."""
    return {
        "concurrent_requests": 10,
        "request_timeout": 5.0,
        "max_response_time": 1.0,
        "min_throughput": 50  # requests per second
    }


# Schema validation helpers
@pytest.fixture
def schema_test_cases():
    """Test cases for schema validation."""
    return {
        "valid_schemas": {
            "search_query": {
                "query": "test search",
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "project_id": "project-456",
                "top_k": 5,
                "similarity_threshold": 0.7
            },
            "document_context": {
                "document_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123"
            }
        },
        "invalid_schemas": {
            "search_query_missing_required": {
                "query": "test"
                # Missing required fields
            },
            "search_query_invalid_types": {
                "query": 123,  # Should be string
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "top_k": "five"  # Should be integer
            },
            "search_query_out_of_range": {
                "query": "test",
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "top_k": 50,  # Exceeds maximum
                "similarity_threshold": 1.5  # Exceeds maximum
            }
        }
    }


# Async test utilities
async def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
    """Wait for a condition to become true."""
    import time
    start_time = time.time()
    while time.time() - start_time < timeout:
        if await condition_func():
            return True
        await asyncio.sleep(interval)
    return False


# Mock data consistency helpers
@pytest.fixture
def consistent_test_data():
    """Consistent test data set for cross-test validation."""
    base_user_id = str(uuid.uuid4())
    base_session_id = "consistent-session-123"
    base_document_id = str(uuid.uuid4())
    base_chunk_id = str(uuid.uuid4())

    return {
        "user_id": base_user_id,
        "session_id": base_session_id,
        "document_id": base_document_id,
        "chunk_id": base_chunk_id,
        "project_id": "consistent-project",
        "embedding": np.random.rand(1536).tolist(),
        "document_metadata": {
            "filename": "consistent_document.pdf",
            "file_type": "pdf",
            "file_size": 2048,
            "total_chunks": 3,
            "metadata": {"title": "Consistent Test Document"}
        },
        "chunk_data": {
            "chunk_text": "This is consistent chunk text for testing.",
            "chunk_index": 0,
            "chunk_metadata": {"page": 1, "section": "intro"}
        }
    }