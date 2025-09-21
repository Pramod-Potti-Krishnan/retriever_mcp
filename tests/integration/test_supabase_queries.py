"""
Integration tests for Supabase database queries in Document Retrieval MCP Server.
"""

import pytest
import uuid
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, List, Any

import asyncpg
import numpy as np

from server import DocumentRetrievalServer


class TestSupabaseIntegration:
    """Test suite for Supabase database integration."""

    @pytest.fixture
    async def mock_supabase_server(self, mock_env_vars):
        """Create server with mocked Supabase connection."""
        server = DocumentRetrievalServer()

        # Mock the database pool
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()

        # Setup connection context manager
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_pool.get_size.return_value = 5
        mock_pool.close = AsyncMock()

        server.db_pool = mock_pool
        server.openai_client = AsyncMock()
        server.supabase_client = MagicMock()

        return server, mock_conn

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_search_documents_query_structure(
        self,
        mock_supabase_server,
        sample_user_id,
        sample_session_id,
        sample_project_id
    ):
        """Test search documents SQL query structure and parameters."""
        server, mock_conn = mock_supabase_server

        # Mock embedding generation
        test_embedding = np.random.rand(1536).tolist()
        server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        server.openai_client.embeddings.create.return_value.data[0].embedding = test_embedding

        # Mock database response
        mock_conn.fetch.return_value = [
            {
                'id': uuid.uuid4(),
                'document_id': uuid.uuid4(),
                'chunk_text': "Sample chunk text",
                'chunk_index': 0,
                'chunk_metadata': {"page": 1},
                'filename': "test.pdf",
                'file_type': "pdf",
                'document_metadata': {"title": "Test Document"},
                'similarity': 0.85
            }
        ]

        # Execute search
        await server._search_documents(
            query="test query",
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=sample_project_id,
            top_k=5,
            similarity_threshold=0.7
        )

        # Verify query was called with correct parameters
        mock_conn.fetch.assert_called_once()
        call_args = mock_conn.fetch.call_args

        # Verify SQL query structure
        sql_query = call_args[0][0]
        assert "SELECT" in sql_query
        assert "document_embeddings de" in sql_query
        assert "documents d" in sql_query
        assert "JOIN" in sql_query
        assert "WHERE" in sql_query
        assert "ORDER BY" in sql_query
        assert "LIMIT" in sql_query
        assert "embedding <=> $1::vector" in sql_query

        # Verify parameters
        params = call_args[0][1:]
        assert params[0] == test_embedding  # Embedding vector
        assert str(params[1]) == sample_user_id  # User ID as UUID
        assert params[2] == sample_session_id  # Session ID
        assert params[3] == sample_project_id  # Project ID
        assert params[4] == 0.7  # Similarity threshold
        assert params[5] == 5  # Top K limit

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_document_context_with_chunk_ids_query(
        self,
        mock_supabase_server,
        sample_user_id,
        sample_session_id,
        sample_document_id
    ):
        """Test document context query with specific chunk IDs."""
        server, mock_conn = mock_supabase_server

        chunk_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

        # Mock database response
        mock_conn.fetch.return_value = [
            {
                'id': uuid.UUID(chunk_ids[0]),
                'chunk_text': "Chunk 1 text",
                'chunk_index': 0,
                'chunk_metadata': {"page": 1},
                'filename': "test.pdf",
                'file_type': "pdf",
                'document_metadata': {"title": "Test"},
                'total_chunks': 3
            },
            {
                'id': uuid.UUID(chunk_ids[1]),
                'chunk_text': "Chunk 2 text",
                'chunk_index': 1,
                'chunk_metadata': {"page": 2},
                'filename': "test.pdf",
                'file_type': "pdf",
                'document_metadata': {"title": "Test"},
                'total_chunks': 3
            }
        ]

        # Execute document context retrieval
        await server._get_document_context(
            document_id=sample_document_id,
            user_id=sample_user_id,
            session_id=sample_session_id,
            chunk_ids=chunk_ids
        )

        # Verify query structure
        mock_conn.fetch.assert_called_once()
        call_args = mock_conn.fetch.call_args
        sql_query = call_args[0][0]

        assert "ANY($4::uuid[])" in sql_query
        assert "document_embeddings de" in sql_query
        assert "documents d ON de.document_id = d.id" in sql_query

        # Verify parameters
        params = call_args[0][1:]
        assert str(params[0]) == sample_document_id
        assert str(params[1]) == sample_user_id
        assert params[2] == sample_session_id
        assert len(params[3]) == 2  # Chunk UUIDs array

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_user_documents_pagination_query(
        self,
        mock_supabase_server,
        sample_user_id,
        sample_session_id,
        sample_project_id
    ):
        """Test list user documents pagination queries."""
        server, mock_conn = mock_supabase_server

        # Mock count query
        mock_conn.fetchval.return_value = 25

        # Mock documents query
        mock_conn.fetch.return_value = [
            {
                'id': uuid.uuid4(),
                'filename': f"doc_{i}.pdf",
                'file_type': "pdf",
                'file_size': 1024 * (i + 1),
                'total_chunks': i + 1,
                'upload_date': datetime.now(),
                'project_id': sample_project_id,
                'metadata': {"title": f"Document {i}"}
            }
            for i in range(10)
        ]

        # Execute list documents with pagination
        await server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=sample_project_id,
            page=2,
            per_page=10
        )

        # Verify two queries were made (count + list)
        assert mock_conn.fetchval.call_count == 1
        assert mock_conn.fetch.call_count == 1

        # Verify count query
        count_call = mock_conn.fetchval.call_args
        count_sql = count_call[0][0]
        assert "SELECT COUNT(*) as total" in count_sql
        assert "processing_status = 'completed'" in count_sql

        # Verify list query
        list_call = mock_conn.fetch.call_args
        list_sql = list_call[0][0]
        assert "LIMIT" in list_sql
        assert "OFFSET" in list_sql
        assert "ORDER BY upload_date DESC" in list_sql

        # Verify pagination parameters
        list_params = list_call[0][1:]
        assert 10 in list_params  # per_page
        assert 10 in list_params  # offset (page 2 * per_page)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_similar_chunks_vector_query(
        self,
        mock_supabase_server,
        sample_user_id,
        sample_session_id,
        sample_chunk_id
    ):
        """Test similar chunks vector similarity query."""
        server, mock_conn = mock_supabase_server

        reference_embedding = np.random.rand(1536).tolist()

        # Mock reference chunk query
        mock_conn.fetchrow.return_value = {
            'embedding': reference_embedding,
            'chunk_text': "Reference chunk text",
            'document_id': uuid.uuid4(),
            'chunk_index': 0
        }

        # Mock similar chunks query
        mock_conn.fetch.return_value = [
            {
                'id': uuid.uuid4(),
                'document_id': uuid.uuid4(),
                'chunk_text': f"Similar chunk {i}",
                'chunk_index': i,
                'chunk_metadata': {"page": i + 1},
                'filename': f"doc_{i}.pdf",
                'file_type': "pdf",
                'similarity': 0.9 - (i * 0.1)
            }
            for i in range(3)
        ]

        # Execute similar chunks search
        await server._get_similar_chunks(
            chunk_id=sample_chunk_id,
            user_id=sample_user_id,
            session_id=sample_session_id,
            top_k=3
        )

        # Verify reference chunk query
        fetchrow_call = mock_conn.fetchrow.call_args
        ref_sql = fetchrow_call[0][0]
        assert "SELECT" in ref_sql
        assert "embedding" in ref_sql
        assert "chunk_text" in ref_sql
        assert "document_id" in ref_sql
        assert "WHERE id = $1::uuid" in ref_sql

        # Verify similar chunks query
        fetch_call = mock_conn.fetch.call_args
        similar_sql = fetch_call[0][0]
        assert "embedding <=> $1::vector" in similar_sql
        assert "ORDER BY de.embedding <=> $1::vector" in similar_sql
        assert "de.id != $4::uuid" in similar_sql  # Exclude reference chunk
        assert "LIMIT $5" in similar_sql

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_database_connection_error_handling(
        self,
        mock_supabase_server,
        sample_user_id,
        sample_session_id
    ):
        """Test database connection error handling."""
        server, mock_conn = mock_supabase_server

        # Mock connection acquisition failure
        server.db_pool.acquire.side_effect = asyncpg.exceptions.ConnectionFailureError("Connection failed")

        # Test each tool's error handling
        tools_and_params = [
            ("_search_documents", {
                "query": "test",
                "user_id": sample_user_id,
                "session_id": sample_session_id
            }),
            ("_get_document_context", {
                "document_id": str(uuid.uuid4()),
                "user_id": sample_user_id,
                "session_id": sample_session_id
            }),
            ("_list_user_documents", {
                "user_id": sample_user_id,
                "session_id": sample_session_id
            }),
            ("_get_similar_chunks", {
                "chunk_id": str(uuid.uuid4()),
                "user_id": sample_user_id,
                "session_id": sample_session_id
            })
        ]

        for tool_name, params in tools_and_params:
            tool_method = getattr(server, tool_name)
            result = await tool_method(**params)

            # All tools should handle errors gracefully
            assert "error" in result
            assert "connection failed" in result["error"].lower()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_database_query_timeout_handling(
        self,
        mock_supabase_server,
        sample_user_id,
        sample_session_id
    ):
        """Test database query timeout handling."""
        server, mock_conn = mock_supabase_server

        # Mock query timeout
        mock_conn.fetch.side_effect = asyncpg.exceptions.QueryTimeoutError("Query timeout")
        mock_conn.fetchval.side_effect = asyncpg.exceptions.QueryTimeoutError("Query timeout")
        mock_conn.fetchrow.side_effect = asyncpg.exceptions.QueryTimeoutError("Query timeout")

        # Test search documents timeout
        result = await server._search_documents(
            query="test",
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert "error" in result
        assert "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_database_transaction_handling(
        self,
        mock_supabase_server,
        sample_user_id,
        sample_session_id
    ):
        """Test database transaction handling."""
        server, mock_conn = mock_supabase_server

        # Mock successful transaction
        mock_conn.fetch.return_value = []

        # Execute operation
        await server._search_documents(
            query="test",
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        # Verify connection was acquired and released
        server.db_pool.acquire.assert_called_once()
        server.db_pool.acquire.return_value.__aenter__.assert_called_once()
        server.db_pool.acquire.return_value.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_vector_operations_performance(
        self,
        mock_supabase_server,
        sample_user_id,
        sample_session_id
    ):
        """Test vector operations performance characteristics."""
        server, mock_conn = mock_supabase_server

        # Mock large result set
        large_result_set = [
            {
                'id': uuid.uuid4(),
                'document_id': uuid.uuid4(),
                'chunk_text': f"Chunk {i} with substantial text content",
                'chunk_index': i,
                'chunk_metadata': {"page": i + 1, "section": f"section_{i}"},
                'filename': f"document_{i}.pdf",
                'file_type': "pdf",
                'document_metadata': {"title": f"Document {i}"},
                'similarity': 0.9 - (i * 0.01)
            }
            for i in range(20)  # Maximum result limit
        ]

        mock_conn.fetch.return_value = large_result_set

        # Mock embedding generation
        test_embedding = np.random.rand(1536).tolist()
        server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        server.openai_client.embeddings.create.return_value.data[0].embedding = test_embedding

        # Measure execution time
        import time
        start_time = time.time()

        result = await server._search_documents(
            query="performance test query",
            user_id=sample_user_id,
            session_id=sample_session_id,
            top_k=20,
            similarity_threshold=0.5
        )

        execution_time = time.time() - start_time

        # Verify result handling
        assert len(result["results"]) == 20
        assert result["count"] == 20

        # Performance should be reasonable (under 1 second for mocked operations)
        assert execution_time < 1.0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_database_operations(
        self,
        mock_supabase_server,
        sample_user_id,
        sample_session_id
    ):
        """Test concurrent database operations."""
        server, mock_conn = mock_supabase_server

        # Mock responses for concurrent operations
        mock_conn.fetch.return_value = []
        mock_conn.fetchval.return_value = 0
        mock_conn.fetchrow.return_value = None

        # Create multiple concurrent operations
        tasks = []

        # Different types of operations
        operations = [
            server._search_documents(
                query=f"query_{i}",
                user_id=sample_user_id,
                session_id=f"session_{i}"
            )
            for i in range(5)
        ]

        operations.extend([
            server._list_user_documents(
                user_id=sample_user_id,
                session_id=f"session_{i}",
                page=i + 1
            )
            for i in range(3)
        ])

        # Execute all operations concurrently
        results = await asyncio.gather(*operations, return_exceptions=True)

        # Verify all operations completed
        assert len(results) == 8

        # Check that no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_database_pool_management(self, mock_env_vars):
        """Test database connection pool management."""
        server = DocumentRetrievalServer()

        # Mock asyncpg pool creation
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool

            # Initialize server
            await server.initialize()

            # Verify pool was created with correct parameters
            mock_create_pool.assert_called_once()
            call_kwargs = mock_create_pool.call_args.kwargs
            assert call_kwargs['min_size'] == 5
            assert call_kwargs['max_size'] == 20
            assert call_kwargs['timeout'] == 10
            assert call_kwargs['command_timeout'] == 10

            # Test cleanup
            await server.cleanup()
            mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_supabase_url_parsing(self, mock_env_vars):
        """Test Supabase URL parsing for database connection."""
        server = DocumentRetrievalServer()

        # Test valid Supabase URL parsing
        with patch.dict('os.environ', {'SUPABASE_URL': 'https://abc123.supabase.co', 'DATABASE_URL': ''}):
            db_config = server._parse_supabase_url()

            assert db_config['host'] == 'db.abc123.supabase.co'
            assert db_config['port'] == 5432
            assert db_config['database'] == 'postgres'
            assert db_config['user'] == 'postgres'
            assert db_config['password'] == mock_env_vars['SUPABASE_API_KEY']

        # Test invalid Supabase URL
        with patch.dict('os.environ', {'SUPABASE_URL': 'invalid-url'}):
            with pytest.raises(ValueError, match="Invalid Supabase URL format"):
                server._parse_supabase_url()


class TestSupabaseDataConsistency:
    """Test suite for data consistency in Supabase operations."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_data_type_consistency(
        self,
        mock_supabase_server,
        consistent_test_data
    ):
        """Test data type consistency across operations."""
        server, mock_conn = mock_supabase_server

        # Mock search results with consistent data types
        mock_conn.fetch.return_value = [
            {
                'id': uuid.UUID(consistent_test_data['chunk_id']),
                'document_id': uuid.UUID(consistent_test_data['document_id']),
                'chunk_text': consistent_test_data['chunk_data']['chunk_text'],
                'chunk_index': consistent_test_data['chunk_data']['chunk_index'],
                'chunk_metadata': consistent_test_data['chunk_data']['chunk_metadata'],
                'filename': consistent_test_data['document_metadata']['filename'],
                'file_type': consistent_test_data['document_metadata']['file_type'],
                'document_metadata': consistent_test_data['document_metadata']['metadata'],
                'similarity': 0.85
            }
        ]

        # Mock embedding
        server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        server.openai_client.embeddings.create.return_value.data[0].embedding = consistent_test_data['embedding']

        # Execute search
        result = await server._search_documents(
            query="test",
            user_id=consistent_test_data['user_id'],
            session_id=consistent_test_data['session_id'],
            project_id=consistent_test_data['project_id']
        )

        # Verify data type consistency
        assert len(result['results']) == 1
        search_result = result['results'][0]

        assert isinstance(search_result['chunk_id'], str)
        assert isinstance(search_result['document_id'], str)
        assert isinstance(search_result['chunk_text'], str)
        assert isinstance(search_result['chunk_index'], int)
        assert isinstance(search_result['chunk_metadata'], dict)
        assert isinstance(search_result['similarity_score'], float)

        # Verify UUID format
        uuid.UUID(search_result['chunk_id'])
        uuid.UUID(search_result['document_id'])

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_metadata_handling_consistency(
        self,
        mock_supabase_server,
        sample_user_id,
        sample_session_id
    ):
        """Test consistent metadata handling across operations."""
        server, mock_conn = mock_supabase_server

        # Test cases with different metadata structures
        metadata_cases = [
            {},  # Empty metadata
            {"key": "value"},  # Simple metadata
            {"nested": {"key": "value"}},  # Nested metadata
            {"array": [1, 2, 3]},  # Array in metadata
            None  # Null metadata
        ]

        for metadata in metadata_cases:
            mock_conn.fetch.return_value = [
                {
                    'id': uuid.uuid4(),
                    'document_id': uuid.uuid4(),
                    'chunk_text': "Test chunk",
                    'chunk_index': 0,
                    'chunk_metadata': metadata,
                    'filename': "test.pdf",
                    'file_type': "pdf",
                    'document_metadata': metadata,
                    'similarity': 0.8
                }
            ]

            server.openai_client.embeddings.create.return_value.data = [MagicMock()]
            server.openai_client.embeddings.create.return_value.data[0].embedding = np.random.rand(1536).tolist()

            result = await server._search_documents(
                query="test",
                user_id=sample_user_id,
                session_id=sample_session_id
            )

            # Verify metadata is consistently handled
            search_result = result['results'][0]
            expected_metadata = metadata if metadata is not None else {}
            assert search_result['chunk_metadata'] == expected_metadata
            assert search_result['document_metadata'] == expected_metadata