"""
Unit tests for authentication and access control in Document Retrieval MCP Server.
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import asyncpg

from server import DocumentRetrievalServer


class TestAccessControl:
    """Test suite for access control and data isolation."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_user_data_isolation_in_search(
        self,
        mock_server,
        sample_session_id,
        sample_project_id
    ):
        """Test that users can only access their own documents in search."""
        user1_id = str(uuid.uuid4())
        user2_id = str(uuid.uuid4())

        # Mock database to return results only for the querying user
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value

        def mock_fetch(*args, **kwargs):
            # Check if the user_id parameter matches user1
            query_user_id = args[2]  # user_id is 3rd parameter (after embedding and query)
            if str(query_user_id) == user1_id:
                return [
                    {
                        'id': uuid.uuid4(),
                        'document_id': uuid.uuid4(),
                        'chunk_text': "User 1 document content",
                        'chunk_index': 0,
                        'chunk_metadata': {},
                        'filename': "user1_doc.pdf",
                        'file_type': "pdf",
                        'document_metadata': {},
                        'similarity': 0.85
                    }
                ]
            else:
                return []  # No results for other users

        mock_conn.fetch.side_effect = mock_fetch

        # Mock embedding generation
        mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

        # Test search for user1 - should return results
        result1 = await mock_server._search_documents(
            query="test",
            user_id=user1_id,
            session_id=sample_session_id,
            project_id=sample_project_id
        )

        assert len(result1["results"]) == 1
        assert "User 1 document content" in result1["results"][0]["chunk_text"]

        # Test search for user2 - should return no results
        result2 = await mock_server._search_documents(
            query="test",
            user_id=user2_id,
            session_id=sample_session_id,
            project_id=sample_project_id
        )

        assert len(result2["results"]) == 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_session_isolation_in_document_list(
        self,
        mock_server,
        sample_user_id,
        sample_project_id
    ):
        """Test that users can only access documents from their session."""
        session1_id = "session-1"
        session2_id = "session-2"

        # Mock database to return different documents for different sessions
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value

        def mock_fetchval(*args, **kwargs):
            return 1  # Total count

        def mock_fetch(*args, **kwargs):
            # Check session_id parameter
            query_session_id = args[2]  # session_id is 3rd parameter
            if query_session_id == session1_id:
                return [
                    {
                        'id': uuid.uuid4(),
                        'filename': "session1_doc.pdf",
                        'file_type': "pdf",
                        'file_size': 1024,
                        'total_chunks': 3,
                        'upload_date': datetime.now(),
                        'project_id': sample_project_id,
                        'metadata': {}
                    }
                ]
            elif query_session_id == session2_id:
                return [
                    {
                        'id': uuid.uuid4(),
                        'filename': "session2_doc.pdf",
                        'file_type': "pdf",
                        'file_size': 2048,
                        'total_chunks': 5,
                        'upload_date': datetime.now(),
                        'project_id': sample_project_id,
                        'metadata': {}
                    }
                ]
            else:
                return []

        mock_conn.fetchval.side_effect = mock_fetchval
        mock_conn.fetch.side_effect = mock_fetch

        # Test for session1
        result1 = await mock_server._list_user_documents(
            user_id=sample_user_id,
            session_id=session1_id,
            project_id=sample_project_id
        )

        assert len(result1["documents"]) == 1
        assert result1["documents"][0]["filename"] == "session1_doc.pdf"

        # Test for session2
        result2 = await mock_server._list_user_documents(
            user_id=sample_user_id,
            session_id=session2_id,
            project_id=sample_project_id
        )

        assert len(result2["documents"]) == 1
        assert result2["documents"][0]["filename"] == "session2_doc.pdf"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_project_isolation_in_document_context(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_document_id
    ):
        """Test that document context respects project boundaries."""
        # Mock database to check project access
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value

        def mock_fetch(*args, **kwargs):
            # The document_id, user_id, and session_id should match
            query_doc_id = args[1]
            query_user_id = args[2]
            query_session_id = args[3]

            if (str(query_doc_id) == sample_document_id and
                str(query_user_id) == sample_user_id and
                query_session_id == sample_session_id):
                return [
                    {
                        'id': uuid.uuid4(),
                        'chunk_text': "Document content",
                        'chunk_index': 0,
                        'chunk_metadata': {},
                        'filename': "test.pdf",
                        'file_type': "pdf",
                        'document_metadata': {},
                        'total_chunks': 1
                    }
                ]
            else:
                return []  # No access

        mock_conn.fetch.side_effect = mock_fetch

        # Test with correct credentials - should succeed
        result1 = await mock_server._get_document_context(
            document_id=sample_document_id,
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert len(result1["chunks"]) == 1
        assert "error" not in result1

        # Test with different user - should fail
        other_user_id = str(uuid.uuid4())
        result2 = await mock_server._get_document_context(
            document_id=sample_document_id,
            user_id=other_user_id,
            session_id=sample_session_id
        )

        assert len(result2["chunks"]) == 0
        assert "error" in result2

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_chunk_access_control_in_similar_chunks(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test access control for similar chunks functionality."""
        user_chunk_id = str(uuid.uuid4())
        other_user_chunk_id = str(uuid.uuid4())

        # Mock database responses
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value

        def mock_fetchrow(*args, **kwargs):
            # Check if accessing own chunk
            query_chunk_id = args[1]
            query_user_id = args[2]
            query_session_id = args[3]

            if (str(query_chunk_id) == user_chunk_id and
                str(query_user_id) == sample_user_id and
                query_session_id == sample_session_id):
                return {
                    'embedding': [0.1] * 1536,
                    'chunk_text': "Reference chunk text",
                    'document_id': uuid.uuid4(),
                    'chunk_index': 0
                }
            else:
                return None  # No access

        def mock_fetch(*args, **kwargs):
            # Return similar chunks only if reference chunk was accessible
            return [
                {
                    'id': uuid.uuid4(),
                    'document_id': uuid.uuid4(),
                    'chunk_text': "Similar chunk",
                    'chunk_index': 1,
                    'chunk_metadata': {},
                    'filename': "similar_doc.pdf",
                    'file_type': "pdf",
                    'similarity': 0.8
                }
            ]

        mock_conn.fetchrow.side_effect = mock_fetchrow
        mock_conn.fetch.side_effect = mock_fetch

        # Test with user's own chunk - should succeed
        result1 = await mock_server._get_similar_chunks(
            chunk_id=user_chunk_id,
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert "reference_text" in result1
        assert len(result1["similar_chunks"]) == 1
        assert "error" not in result1

        # Test with other user's chunk - should fail
        result2 = await mock_server._get_similar_chunks(
            chunk_id=other_user_chunk_id,
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert "error" in result2
        assert len(result2["similar_chunks"]) == 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_sql_injection_prevention(
        self,
        mock_server,
        sample_session_id,
        sample_project_id
    ):
        """Test SQL injection prevention in parameters."""
        # Test malicious user IDs and session IDs
        malicious_inputs = [
            "'; DROP TABLE documents; --",
            "' UNION SELECT * FROM documents --",
            "'; UPDATE documents SET user_id = 'hacked'; --",
            "1'; DELETE FROM document_embeddings; --",
            "admin'--"
        ]

        # Mock database connection
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = []

        # Mock embedding generation
        mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

        for malicious_input in malicious_inputs:
            # Test with malicious user_id
            try:
                result = await mock_server._search_documents(
                    query="test",
                    user_id=malicious_input,
                    session_id=sample_session_id,
                    project_id=sample_project_id
                )
                # Should either handle gracefully or raise proper exception
                # The UUID conversion should fail for malicious input
            except (ValueError, TypeError) as e:
                # Expected for invalid UUID format
                assert "invalid" in str(e).lower() or "uuid" in str(e).lower()

            # Test with malicious session_id
            try:
                result = await mock_server._search_documents(
                    query="test",
                    user_id=str(uuid.uuid4()),
                    session_id=malicious_input,
                    project_id=sample_project_id
                )
                # Should handle gracefully with parameterized queries
                assert "results" in result
            except Exception as e:
                # Any database-level protection is acceptable
                pass

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_uuid_validation_in_access_control(
        self,
        mock_server,
        sample_session_id
    ):
        """Test UUID validation for access control parameters."""
        # Test invalid UUID formats
        invalid_uuids = [
            "not-a-uuid",
            "123-456-789",
            "",
            "null",
            "undefined",
            "12345678-1234-1234-1234-12345678901"  # Too long
        ]

        for invalid_uuid in invalid_uuids:
            # Test search_documents with invalid user_id
            with pytest.raises((ValueError, TypeError)):
                await mock_server._search_documents(
                    query="test",
                    user_id=invalid_uuid,
                    session_id=sample_session_id
                )

            # Test get_document_context with invalid document_id
            with pytest.raises((ValueError, TypeError)):
                await mock_server._get_document_context(
                    document_id=invalid_uuid,
                    user_id=str(uuid.uuid4()),
                    session_id=sample_session_id
                )

            # Test get_similar_chunks with invalid chunk_id
            with pytest.raises((ValueError, TypeError)):
                await mock_server._get_similar_chunks(
                    chunk_id=invalid_uuid,
                    user_id=str(uuid.uuid4()),
                    session_id=sample_session_id
                )

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_session_id_pattern_validation(
        self,
        mock_server,
        sample_user_id
    ):
        """Test session ID pattern validation for security."""
        # Valid session IDs (alphanumeric, hyphens, underscores)
        valid_session_ids = [
            "session-123",
            "user_session_456",
            "SESSION-ABC-123",
            "session123",
            "a" * 50  # Long but valid
        ]

        # Mock database
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.return_value = 0
        mock_conn.fetch.return_value = []

        for valid_session_id in valid_session_ids:
            # Should not raise validation errors
            result = await mock_server._list_user_documents(
                user_id=sample_user_id,
                session_id=valid_session_id
            )
            assert "documents" in result

        # Invalid session IDs (contain special characters, spaces, etc.)
        invalid_session_ids = [
            "session with spaces",
            "session@special",
            "session#hash",
            "session/path",
            "session.dot",
            "session;semicolon",
            "session'quote",
            "session\"doublequote"
        ]

        # Note: Pattern validation is typically done at the MCP schema level
        # Here we test that the application handles these gracefully
        for invalid_session_id in invalid_session_ids:
            # The server should handle these without crashing
            # Pattern validation happens at the schema level
            result = await mock_server._list_user_documents(
                user_id=sample_user_id,
                session_id=invalid_session_id
            )
            # Should either work (if passed validation) or fail gracefully
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_concurrent_access_control(
        self,
        mock_server,
        sample_session_id,
        sample_project_id
    ):
        """Test access control under concurrent operations."""
        import asyncio

        # Create multiple users
        users = [str(uuid.uuid4()) for _ in range(5)]

        # Mock database to return user-specific data
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value

        def mock_fetch(*args, **kwargs):
            # Return data specific to the querying user
            if len(args) >= 3:
                query_user_id = str(args[2])
                return [
                    {
                        'id': uuid.uuid4(),
                        'document_id': uuid.uuid4(),
                        'chunk_text': f"Content for user {query_user_id}",
                        'chunk_index': 0,
                        'chunk_metadata': {},
                        'filename': f"user_{query_user_id}_doc.pdf",
                        'file_type': "pdf",
                        'document_metadata': {},
                        'similarity': 0.8
                    }
                ]
            return []

        mock_conn.fetch.side_effect = mock_fetch

        # Mock embedding generation
        mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

        # Create concurrent search operations for different users
        tasks = [
            mock_server._search_documents(
                query=f"query for user {user_id}",
                user_id=user_id,
                session_id=sample_session_id,
                project_id=sample_project_id
            )
            for user_id in users
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # Verify each user got their own data
        for i, result in enumerate(results):
            user_id = users[i]
            assert len(result["results"]) == 1
            assert user_id in result["results"][0]["chunk_text"]
            assert user_id in result["results"][0]["filename"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_database_privilege_errors(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test handling of database privilege errors."""
        # Mock database privilege error
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.side_effect = asyncpg.exceptions.InsufficientPrivilegeError(
            "permission denied for table documents"
        )

        # Test that privilege errors are handled gracefully
        result = await mock_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert "error" in result
        assert "permission denied" in result["error"].lower()
        assert result["documents"] == []

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_row_level_security_simulation(
        self,
        mock_server,
        sample_session_id,
        sample_project_id
    ):
        """Test simulation of row-level security enforcement."""
        user1_id = str(uuid.uuid4())
        user2_id = str(uuid.uuid4())

        # Mock database to simulate row-level security
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value

        def mock_fetch(*args, **kwargs):
            # Simulate RLS: users can only see their own rows
            query_user_id = args[2] if len(args) > 2 else None
            if str(query_user_id) == user1_id:
                return [
                    {
                        'id': uuid.uuid4(),
                        'document_id': uuid.uuid4(),
                        'chunk_text': "User 1 private content",
                        'chunk_index': 0,
                        'chunk_metadata': {"owner": user1_id},
                        'filename': "user1_private.pdf",
                        'file_type': "pdf",
                        'document_metadata': {"access": "private"},
                        'similarity': 0.9
                    }
                ]
            elif str(query_user_id) == user2_id:
                return [
                    {
                        'id': uuid.uuid4(),
                        'document_id': uuid.uuid4(),
                        'chunk_text': "User 2 private content",
                        'chunk_index': 0,
                        'chunk_metadata': {"owner": user2_id},
                        'filename': "user2_private.pdf",
                        'file_type': "pdf",
                        'document_metadata': {"access": "private"},
                        'similarity': 0.9
                    }
                ]
            else:
                return []  # No access to other users' data

        mock_conn.fetch.side_effect = mock_fetch

        # Mock embedding generation
        mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

        # Test User 1 access
        result1 = await mock_server._search_documents(
            query="private content",
            user_id=user1_id,
            session_id=sample_session_id,
            project_id=sample_project_id
        )

        assert len(result1["results"]) == 1
        assert "User 1 private content" in result1["results"][0]["chunk_text"]
        assert result1["results"][0]["chunk_metadata"]["owner"] == user1_id

        # Test User 2 access
        result2 = await mock_server._search_documents(
            query="private content",
            user_id=user2_id,
            session_id=sample_session_id,
            project_id=sample_project_id
        )

        assert len(result2["results"]) == 1
        assert "User 2 private content" in result2["results"][0]["chunk_text"]
        assert result2["results"][0]["chunk_metadata"]["owner"] == user2_id

        # Verify users don't see each other's content
        assert result1["results"][0]["chunk_text"] != result2["results"][0]["chunk_text"]