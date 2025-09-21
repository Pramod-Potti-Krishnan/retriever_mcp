"""
Unit tests for caching functionality in Document Retrieval MCP Server.
"""

import pytest
import uuid
import time
from unittest.mock import AsyncMock, patch
from datetime import datetime

from cachetools import TTLCache
from server import DocumentRetrievalServer


class TestCacheFunctionality:
    """Test suite for caching functionality."""

    @pytest.fixture
    def cache_server(self, mock_env_vars):
        """Create server instance with controlled cache settings."""
        with patch('server.CACHE_TTL', 1), patch('server.CACHE_MAX_SIZE', 10):
            server = DocumentRetrievalServer()
            server.metadata_cache = TTLCache(maxsize=10, ttl=1)
            return server

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_cache_hit_list_documents(
        self,
        cache_server,
        sample_user_id,
        sample_session_id,
        sample_project_id
    ):
        """Test cache hit for list_user_documents."""
        # Mock database pool
        cache_server.db_pool = AsyncMock()
        mock_conn = AsyncMock()
        cache_server.db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        cache_server.db_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        # First call - should hit database
        mock_conn.fetchval.return_value = 2
        mock_conn.fetch.return_value = [
            {
                'id': uuid.uuid4(),
                'filename': "doc1.pdf",
                'file_type': "pdf",
                'file_size': 1024,
                'total_chunks': 3,
                'upload_date': datetime.now(),
                'project_id': sample_project_id,
                'metadata': {}
            },
            {
                'id': uuid.uuid4(),
                'filename': "doc2.pdf",
                'file_type': "pdf",
                'file_size': 2048,
                'total_chunks': 5,
                'upload_date': datetime.now(),
                'project_id': sample_project_id,
                'metadata': {}
            }
        ]

        # First call
        result1 = await cache_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=sample_project_id,
            page=1,
            per_page=20
        )

        # Verify database was called
        assert mock_conn.fetchval.call_count == 1
        assert mock_conn.fetch.call_count == 1

        # Reset mock call counts
        mock_conn.fetchval.reset_mock()
        mock_conn.fetch.reset_mock()

        # Second call - should hit cache
        result2 = await cache_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=sample_project_id,
            page=1,
            per_page=20
        )

        # Verify database was not called on second request
        assert mock_conn.fetchval.call_count == 0
        assert mock_conn.fetch.call_count == 0

        # Verify results are identical
        assert result1 == result2
        assert len(result1["documents"]) == 2

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_cache_miss_different_parameters(
        self,
        cache_server,
        sample_user_id,
        sample_session_id,
        sample_project_id
    ):
        """Test cache miss with different parameters."""
        # Mock database pool
        cache_server.db_pool = AsyncMock()
        mock_conn = AsyncMock()
        cache_server.db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        cache_server.db_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_conn.fetchval.return_value = 1
        mock_conn.fetch.return_value = [
            {
                'id': uuid.uuid4(),
                'filename': "doc.pdf",
                'file_type': "pdf",
                'file_size': 1024,
                'total_chunks': 3,
                'upload_date': datetime.now(),
                'project_id': sample_project_id,
                'metadata': {}
            }
        ]

        # Call with page 1
        result1 = await cache_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=sample_project_id,
            page=1,
            per_page=20
        )

        # Call with page 2 - should miss cache
        result2 = await cache_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=sample_project_id,
            page=2,
            per_page=20
        )

        # Verify database was called twice
        assert mock_conn.fetchval.call_count == 2
        assert mock_conn.fetch.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_cache_expiration(self, cache_server, sample_user_id, sample_session_id):
        """Test cache expiration with TTL."""
        # Mock database pool
        cache_server.db_pool = AsyncMock()
        mock_conn = AsyncMock()
        cache_server.db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        cache_server.db_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_conn.fetchval.return_value = 1
        mock_conn.fetch.return_value = [
            {
                'id': uuid.uuid4(),
                'filename': "doc.pdf",
                'file_type': "pdf",
                'file_size': 1024,
                'total_chunks': 3,
                'upload_date': datetime.now(),
                'project_id': "test",
                'metadata': {}
            }
        ]

        # First call
        result1 = await cache_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            page=1,
            per_page=20
        )

        # Verify database was called
        assert mock_conn.fetchval.call_count == 1

        # Wait for cache to expire (TTL = 1 second)
        await asyncio.sleep(1.1)

        # Reset mock call counts
        mock_conn.fetchval.reset_mock()
        mock_conn.fetch.reset_mock()

        # Second call after expiration
        result2 = await cache_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            page=1,
            per_page=20
        )

        # Verify database was called again after expiration
        assert mock_conn.fetchval.call_count == 1
        assert mock_conn.fetch.call_count == 1

    @pytest.mark.unit
    def test_cache_key_generation(self, cache_server):
        """Test cache key generation for different parameter combinations."""
        # Test cases for cache key uniqueness
        test_cases = [
            {
                "user_id": "user1",
                "session_id": "session1",
                "project_id": "project1",
                "page": 1,
                "per_page": 20
            },
            {
                "user_id": "user1",
                "session_id": "session1",
                "project_id": "project2",  # Different project
                "page": 1,
                "per_page": 20
            },
            {
                "user_id": "user1",
                "session_id": "session1",
                "project_id": "project1",
                "page": 2,  # Different page
                "per_page": 20
            },
            {
                "user_id": "user2",  # Different user
                "session_id": "session1",
                "project_id": "project1",
                "page": 1,
                "per_page": 20
            }
        ]

        cache_keys = []
        for case in test_cases:
            cache_key = f"docs_{case['user_id']}_{case['session_id']}_{case['project_id']}_{case['page']}_{case['per_page']}"
            cache_keys.append(cache_key)

        # Verify all cache keys are unique
        assert len(cache_keys) == len(set(cache_keys))

    @pytest.mark.unit
    def test_cache_size_limit(self, cache_server):
        """Test cache size limit enforcement."""
        # Fill cache beyond capacity (capacity = 10)
        for i in range(15):
            cache_key = f"test_key_{i}"
            cache_server.metadata_cache[cache_key] = {"data": f"value_{i}"}

        # Verify cache size doesn't exceed maximum
        assert len(cache_server.metadata_cache) <= 10

        # Verify some early entries were evicted
        assert "test_key_0" not in cache_server.metadata_cache
        assert "test_key_1" not in cache_server.metadata_cache

        # Verify recent entries are still present
        assert "test_key_14" in cache_server.metadata_cache
        assert "test_key_13" in cache_server.metadata_cache

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_cache_with_none_project_id(
        self,
        cache_server,
        sample_user_id,
        sample_session_id
    ):
        """Test caching behavior when project_id is None."""
        # Mock database pool
        cache_server.db_pool = AsyncMock()
        mock_conn = AsyncMock()
        cache_server.db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        cache_server.db_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_conn.fetchval.return_value = 1
        mock_conn.fetch.return_value = [
            {
                'id': uuid.uuid4(),
                'filename': "doc.pdf",
                'file_type': "pdf",
                'file_size': 1024,
                'total_chunks': 3,
                'upload_date': datetime.now(),
                'project_id': "any",
                'metadata': {}
            }
        ]

        # First call with None project_id
        result1 = await cache_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=None,
            page=1,
            per_page=20
        )

        # Reset mock call counts
        mock_conn.fetchval.reset_mock()
        mock_conn.fetch.reset_mock()

        # Second call with None project_id - should hit cache
        result2 = await cache_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=None,
            page=1,
            per_page=20
        )

        # Verify database was not called on second request
        assert mock_conn.fetchval.call_count == 0
        assert mock_conn.fetch.call_count == 0

        # Verify results are identical
        assert result1 == result2

    @pytest.mark.unit
    def test_cache_thread_safety(self, cache_server):
        """Test cache thread safety with concurrent access."""
        import threading
        import concurrent.futures

        results = []

        def cache_operation(key_suffix):
            """Simulate cache operations from different threads."""
            try:
                cache_key = f"thread_test_{key_suffix}"
                test_data = {"thread_id": key_suffix, "data": f"value_{key_suffix}"}

                # Write to cache
                cache_server.metadata_cache[cache_key] = test_data

                # Read from cache
                retrieved_data = cache_server.metadata_cache.get(cache_key)

                results.append((key_suffix, retrieved_data == test_data))
                return True
            except Exception as e:
                results.append((key_suffix, False))
                return False

        # Execute concurrent cache operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(cache_operation, i) for i in range(10)]
            concurrent.futures.wait(futures)

        # Verify all operations completed successfully
        assert len(results) == 10
        for thread_id, success in results:
            assert success, f"Thread {thread_id} operation failed"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_cache_error_handling(
        self,
        cache_server,
        sample_user_id,
        sample_session_id,
        database_error_scenarios
    ):
        """Test cache behavior during database errors."""
        # Mock database pool to raise error
        cache_server.db_pool = AsyncMock()
        mock_conn = AsyncMock()
        cache_server.db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        cache_server.db_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_conn.fetchval.side_effect = database_error_scenarios["connection_timeout"]

        # Call should handle error gracefully
        result = await cache_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            page=1,
            per_page=20
        )

        # Verify error is handled and cache remains functional
        assert "error" in result
        assert result["documents"] == []

        # Verify cache is still operational
        test_key = "error_test_key"
        test_data = {"test": "data"}
        cache_server.metadata_cache[test_key] = test_data
        assert cache_server.metadata_cache[test_key] == test_data

    @pytest.mark.unit
    def test_cache_memory_usage(self, cache_server):
        """Test cache memory usage patterns."""
        import sys

        # Get initial cache size
        initial_size = sys.getsizeof(cache_server.metadata_cache)

        # Add data to cache
        large_data = {"documents": [{"id": str(uuid.uuid4()), "data": "x" * 1000} for _ in range(100)]}

        for i in range(5):
            cache_key = f"large_data_{i}"
            cache_server.metadata_cache[cache_key] = large_data

        # Verify cache handles large data appropriately
        assert len(cache_server.metadata_cache) <= 10  # Max size limit

        # Verify memory usage is reasonable (cache should evict old entries)
        final_size = sys.getsizeof(cache_server.metadata_cache)
        # Memory should not grow unbounded
        assert final_size < initial_size + (1000 * 1000)  # Reasonable upper bound


import asyncio  # Import needed for sleep in test_cache_expiration