"""
Unit tests for error handling and edge cases in Document Retrieval MCP Server.
"""

import pytest
import uuid
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import asyncpg
import numpy as np
from openai import (
    RateLimitError, AuthenticationError, APIConnectionError,
    InvalidRequestError, InternalServerError
)

from server import DocumentRetrievalServer


class TestErrorHandling:
    """Test suite for error handling scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_database_connection_failures(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test handling of various database connection failures."""
        error_scenarios = [
            (asyncpg.exceptions.ConnectionFailureError("Connection failed"), "connection failed"),
            (asyncpg.exceptions.ConnectionTimeoutError("Connection timeout"), "connection timeout"),
            (asyncpg.exceptions.TooManyConnectionsError("Too many connections"), "too many connections"),
            (ConnectionError("Network error"), "network error"),
            (OSError("System error"), "system error")
        ]

        for exception, expected_text in error_scenarios:
            # Mock connection acquisition failure
            mock_server.db_pool.acquire.side_effect = exception

            # Test search_documents
            result = await mock_server._search_documents(
                query="test",
                user_id=sample_user_id,
                session_id=sample_session_id
            )

            assert "error" in result
            assert result["results"] == []
            assert expected_text in result["error"].lower()

            # Reset for next test
            mock_server.db_pool.acquire.side_effect = None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_database_query_failures(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test handling of database query failures."""
        # Setup mock connection
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value

        query_error_scenarios = [
            (asyncpg.exceptions.QueryTimeoutError("Query timeout"), "query timeout"),
            (asyncpg.exceptions.InvalidSQLStatementNameError("Invalid SQL"), "invalid sql"),
            (asyncpg.exceptions.SyntaxOrAccessError("Syntax error"), "syntax"),
            (asyncpg.exceptions.DataError("Data type error"), "data"),
            (RuntimeError("Unexpected database error"), "unexpected")
        ]

        for exception, expected_text in query_error_scenarios:
            mock_conn.fetch.side_effect = exception
            mock_conn.fetchval.side_effect = exception
            mock_conn.fetchrow.side_effect = exception

            # Test each tool
            tools_and_params = [
                ("_search_documents", {
                    "query": "test",
                    "user_id": sample_user_id,
                    "session_id": sample_session_id
                }),
                ("_list_user_documents", {
                    "user_id": sample_user_id,
                    "session_id": sample_session_id
                }),
                ("_get_document_context", {
                    "document_id": str(uuid.uuid4()),
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
                tool_method = getattr(mock_server, tool_name)
                result = await tool_method(**params)

                assert "error" in result
                # Each tool has different empty result structures
                if "results" in result:
                    assert result["results"] == []
                elif "documents" in result:
                    assert result["documents"] == []
                elif "chunks" in result:
                    assert result["chunks"] == []
                elif "similar_chunks" in result:
                    assert result["similar_chunks"] == []

            # Reset for next test
            mock_conn.fetch.side_effect = None
            mock_conn.fetchval.side_effect = None
            mock_conn.fetchrow.side_effect = None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_openai_api_failures(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test handling of OpenAI API failures."""
        openai_error_scenarios = [
            (RateLimitError("Rate limit exceeded", response=MagicMock(), body={}), "rate limit"),
            (AuthenticationError("Invalid API key", response=MagicMock(), body={}), "authentication"),
            (InvalidRequestError("Invalid request", response=MagicMock(), body={}), "invalid request"),
            (APIConnectionError("Connection error"), "connection"),
            (InternalServerError("Internal server error", response=MagicMock(), body={}), "internal"),
            (Exception("Unexpected OpenAI error"), "unexpected")
        ]

        for exception, expected_text in openai_error_scenarios:
            mock_server.openai_client.embeddings.create.side_effect = exception

            # Test search_documents (which requires embedding generation)
            result = await mock_server._search_documents(
                query="test",
                user_id=sample_user_id,
                session_id=sample_session_id
            )

            assert "error" in result
            assert result["results"] == []

            # Reset for next test
            mock_server.openai_client.embeddings.create.side_effect = None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_invalid_uuid_handling(self, mock_server):
        """Test handling of invalid UUID parameters."""
        invalid_uuids = [
            "not-a-uuid",
            "12345",
            "",
            None,
            "123e4567-e89b-12d3-a456-42661417400",  # Too short
            "123e4567-e89b-12d3-a456-4266141740000"  # Too long
        ]

        for invalid_uuid in invalid_uuids:
            # Test with invalid user_id
            with pytest.raises((ValueError, TypeError, AttributeError)):
                await mock_server._search_documents(
                    query="test",
                    user_id=invalid_uuid,
                    session_id="session-123"
                )

            # Test with invalid document_id
            with pytest.raises((ValueError, TypeError, AttributeError)):
                await mock_server._get_document_context(
                    document_id=invalid_uuid,
                    user_id=str(uuid.uuid4()),
                    session_id="session-123"
                )

            # Test with invalid chunk_id
            with pytest.raises((ValueError, TypeError, AttributeError)):
                await mock_server._get_similar_chunks(
                    chunk_id=invalid_uuid,
                    user_id=str(uuid.uuid4()),
                    session_id="session-123"
                )

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_memory_exhaustion_simulation(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test handling of memory exhaustion scenarios."""
        # Mock very large result set
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value

        def create_large_result():
            # Simulate memory error during large result processing
            raise MemoryError("Insufficient memory")

        mock_conn.fetch.side_effect = create_large_result

        # Test search with memory error
        result = await mock_server._search_documents(
            query="test",
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert "error" in result
        assert result["results"] == []
        assert "memory" in result["error"].lower() or "insufficient" in result["error"].lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_concurrent_request_failures(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test handling of failures in concurrent requests."""
        # Setup mock to fail intermittently
        call_count = 0

        def intermittent_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise asyncpg.exceptions.ConnectionTimeoutError("Timeout")
            else:
                return []

        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.side_effect = intermittent_failure

        # Mock embedding generation
        mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

        # Create multiple concurrent requests
        tasks = [
            mock_server._search_documents(
                query=f"test {i}",
                user_id=sample_user_id,
                session_id=f"session-{i}"
            )
            for i in range(10)
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify handling of mixed success/failure
        success_count = 0
        error_count = 0

        for result in results:
            if isinstance(result, Exception):
                error_count += 1
            elif "error" in result:
                error_count += 1
            else:
                success_count += 1

        # Should have both successes and failures
        assert success_count > 0
        assert error_count > 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_malformed_database_response_handling(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test handling of malformed database responses."""
        # Mock malformed responses
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value

        malformed_responses = [
            [{"missing_required_field": "value"}],  # Missing required fields
            [{"id": "not-a-uuid", "chunk_text": None}],  # Invalid field types
            [{"id": uuid.uuid4(), "chunk_metadata": "not-a-dict"}],  # Wrong metadata type
            None,  # Null response
            "not-a-list",  # Wrong response type
            [None],  # List with null items
        ]

        for malformed_response in malformed_responses:
            mock_conn.fetch.return_value = malformed_response
            mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
            mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

            # Test should handle malformed response gracefully
            try:
                result = await mock_server._search_documents(
                    query="test",
                    user_id=sample_user_id,
                    session_id=sample_session_id
                )

                # Should either return empty results or proper error
                assert isinstance(result, dict)
                if "error" not in result:
                    assert "results" in result
                    assert isinstance(result["results"], list)

            except Exception as e:
                # If exception is raised, it should be handled gracefully
                assert isinstance(e, (TypeError, KeyError, ValueError, AttributeError))


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_empty_query_handling(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test handling of empty and whitespace-only queries."""
        edge_case_queries = [
            "",  # Empty string
            " ",  # Single space
            "\n",  # Newline
            "\t",  # Tab
            "   \n\t   ",  # Mixed whitespace
        ]

        for query in edge_case_queries:
            # Mock embedding generation for empty/whitespace queries
            mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
            mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.0] * 1536

            # Mock database response
            _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
            mock_conn.fetch.return_value = []

            # Should handle gracefully
            result = await mock_server._search_documents(
                query=query,
                user_id=sample_user_id,
                session_id=sample_session_id
            )

            assert "results" in result
            assert isinstance(result["results"], list)
            assert result["query"] == query

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_maximum_length_parameters(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test handling of maximum length parameters."""
        # Test very long query (at limit)
        max_query = "x" * 1000  # Maximum allowed length

        # Mock successful response
        mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = []

        # Should handle maximum length query
        result = await mock_server._search_documents(
            query=max_query,
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert "results" in result
        assert result["query"] == max_query

        # Test very long session ID
        long_session_id = "session-" + "x" * 100
        result = await mock_server._list_user_documents(
            user_id=sample_user_id,
            session_id=long_session_id
        )

        assert "documents" in result

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_boundary_value_parameters(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test boundary values for numeric parameters."""
        # Mock responses
        mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = []
        mock_conn.fetchval.return_value = 0

        # Test boundary values for top_k
        boundary_cases = [
            {"top_k": 1, "should_work": True},  # Minimum
            {"top_k": 20, "should_work": True},  # Maximum
            {"similarity_threshold": 0.0, "should_work": True},  # Minimum
            {"similarity_threshold": 1.0, "should_work": True},  # Maximum
        ]

        for case in boundary_cases:
            params = {k: v for k, v in case.items() if k != "should_work"}

            result = await mock_server._search_documents(
                query="test",
                user_id=sample_user_id,
                session_id=sample_session_id,
                **params
            )

            if case["should_work"]:
                assert "results" in result
                assert "error" not in result
            # Note: Invalid boundaries would be caught by schema validation

        # Test pagination boundaries
        pagination_cases = [
            {"page": 1, "per_page": 1},  # Minimum values
            {"page": 1, "per_page": 100},  # Maximum per_page
            {"page": 1000, "per_page": 20},  # Large page number
        ]

        for case in pagination_cases:
            result = await mock_server._list_user_documents(
                user_id=sample_user_id,
                session_id=sample_session_id,
                **case
            )

            assert "documents" in result
            assert "pagination" in result

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_unicode_and_special_character_handling(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test handling of Unicode and special characters."""
        special_queries = [
            "émojis 🚀 and ünicode",
            "中文测试",
            "Русский текст",
            "العربية",
            "עברית",
            "🎉🔥💯⭐🌟",  # Only emojis
            "Math: ∑∫∆π∞ ≤≥≠≈",
            "Special: @#$%^&*()_+-={}[]|\\:;\"'<>?,./"
        ]

        for query in special_queries:
            # Mock embedding generation
            mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
            mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

            # Mock database response
            _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
            mock_conn.fetch.return_value = []

            # Should handle Unicode gracefully
            result = await mock_server._search_documents(
                query=query,
                user_id=sample_user_id,
                session_id=sample_session_id
            )

            assert "results" in result
            assert result["query"] == query

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_large_result_set_handling(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test handling of large result sets."""
        # Create maximum size result set
        large_results = [
            {
                'id': uuid.uuid4(),
                'document_id': uuid.uuid4(),
                'chunk_text': f"Large chunk text {i} " + "x" * 1000,  # Large text
                'chunk_index': i,
                'chunk_metadata': {f"key_{j}": f"value_{j}" for j in range(100)},  # Large metadata
                'filename': f"large_document_{i}.pdf",
                'file_type': "pdf",
                'document_metadata': {f"doc_key_{j}": f"doc_value_{j}" for j in range(50)},
                'similarity': 0.95 - (i * 0.01)
            }
            for i in range(20)  # Maximum result limit
        ]

        # Mock database response
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = large_results

        # Mock embedding generation
        mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

        # Should handle large result set
        result = await mock_server._search_documents(
            query="test",
            user_id=sample_user_id,
            session_id=sample_session_id,
            top_k=20
        )

        assert "results" in result
        assert len(result["results"]) == 20
        assert result["count"] == 20

        # Verify large text and metadata are preserved
        for res in result["results"]:
            assert len(res["chunk_text"]) > 1000
            assert len(res["chunk_metadata"]) == 100
            assert len(res["document_metadata"]) == 50

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_zero_and_negative_similarity_scores(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test handling of zero and negative similarity scores."""
        # Mock results with edge case similarity scores
        edge_case_results = [
            {
                'id': uuid.uuid4(),
                'document_id': uuid.uuid4(),
                'chunk_text': "Perfect match",
                'chunk_index': 0,
                'chunk_metadata': {},
                'filename': "perfect.pdf",
                'file_type': "pdf",
                'document_metadata': {},
                'similarity': 1.0  # Perfect similarity
            },
            {
                'id': uuid.uuid4(),
                'document_id': uuid.uuid4(),
                'chunk_text': "Zero similarity",
                'chunk_index': 1,
                'chunk_metadata': {},
                'filename': "zero.pdf",
                'file_type': "pdf",
                'document_metadata': {},
                'similarity': 0.0  # Zero similarity
            },
            {
                'id': uuid.uuid4(),
                'document_id': uuid.uuid4(),
                'chunk_text': "Negative similarity",
                'chunk_index': 2,
                'chunk_metadata': {},
                'filename': "negative.pdf",
                'file_type': "pdf",
                'document_metadata': {},
                'similarity': -0.1  # Negative similarity (edge case)
            }
        ]

        # Mock database response
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = edge_case_results

        # Mock embedding generation
        mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

        # Should handle edge case similarity scores
        result = await mock_server._search_documents(
            query="test",
            user_id=sample_user_id,
            session_id=sample_session_id,
            similarity_threshold=0.0  # Allow all scores
        )

        assert "results" in result
        assert len(result["results"]) == 3

        # Verify similarity scores are preserved
        similarities = [res["similarity_score"] for res in result["results"]]
        assert 1.0 in similarities
        assert 0.0 in similarities
        assert -0.1 in similarities

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_null_and_empty_metadata_handling(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test handling of null and empty metadata."""
        # Mock results with various metadata states
        metadata_cases = [
            {
                'id': uuid.uuid4(),
                'document_id': uuid.uuid4(),
                'chunk_text': "Normal metadata",
                'chunk_index': 0,
                'chunk_metadata': {"key": "value"},
                'filename': "normal.pdf",
                'file_type': "pdf",
                'document_metadata': {"title": "Normal Doc"},
                'similarity': 0.8
            },
            {
                'id': uuid.uuid4(),
                'document_id': uuid.uuid4(),
                'chunk_text': "Empty metadata",
                'chunk_index': 1,
                'chunk_metadata': {},
                'filename': "empty.pdf",
                'file_type': "pdf",
                'document_metadata': {},
                'similarity': 0.7
            },
            {
                'id': uuid.uuid4(),
                'document_id': uuid.uuid4(),
                'chunk_text': "Null metadata",
                'chunk_index': 2,
                'chunk_metadata': None,
                'filename': "null.pdf",
                'file_type': "pdf",
                'document_metadata': None,
                'similarity': 0.6
            }
        ]

        # Mock database response
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = metadata_cases

        # Mock embedding generation
        mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

        # Should handle all metadata cases
        result = await mock_server._search_documents(
            query="test",
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert "results" in result
        assert len(result["results"]) == 3

        # Verify metadata handling
        for res in result["results"]:
            assert "chunk_metadata" in res
            assert "document_metadata" in res
            assert isinstance(res["chunk_metadata"], dict)
            assert isinstance(res["document_metadata"], dict)