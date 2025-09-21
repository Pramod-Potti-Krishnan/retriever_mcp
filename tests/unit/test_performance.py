"""
Performance tests for Document Retrieval MCP Server.
"""

import pytest
import asyncio
import uuid
import time
from unittest.mock import AsyncMock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor
import statistics

import numpy as np

from server import DocumentRetrievalServer


class TestPerformanceBaselines:
    """Test suite for performance baselines and benchmarks."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_search_documents_response_time(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        performance_test_config
    ):
        """Test search_documents response time under normal load."""
        # Mock fast responses
        mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = [
            {
                'id': uuid.uuid4(),
                'document_id': uuid.uuid4(),
                'chunk_text': "Sample result",
                'chunk_index': 0,
                'chunk_metadata': {},
                'filename': "test.pdf",
                'file_type': "pdf",
                'document_metadata': {},
                'similarity': 0.8
            }
        ]

        # Measure response times
        response_times = []
        for i in range(10):
            start_time = time.time()

            result = await mock_server._search_documents(
                query=f"test query {i}",
                user_id=sample_user_id,
                session_id=sample_session_id
            )

            response_time = time.time() - start_time
            response_times.append(response_time)

            assert "results" in result
            assert len(result["results"]) == 1

        # Verify performance metrics
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        percentile_95 = statistics.quantiles(response_times, n=20)[18]  # 95th percentile

        # Performance assertions
        assert avg_response_time < performance_test_config["max_response_time"]
        assert max_response_time < performance_test_config["max_response_time"] * 2
        assert percentile_95 < performance_test_config["max_response_time"] * 1.5

        print(f"Search performance - Avg: {avg_response_time:.3f}s, Max: {max_response_time:.3f}s, 95th: {percentile_95:.3f}s")

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_concurrent_request_throughput(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        performance_test_config
    ):
        """Test throughput under concurrent load."""
        # Mock responses
        mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = []

        concurrent_requests = performance_test_config["concurrent_requests"]
        request_timeout = performance_test_config["request_timeout"]

        # Create concurrent requests
        tasks = [
            mock_server._search_documents(
                query=f"concurrent query {i}",
                user_id=sample_user_id,
                session_id=f"session-{i}"
            )
            for i in range(concurrent_requests)
        ]

        # Measure throughput
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Verify all requests completed successfully
        successful_requests = sum(1 for result in results if isinstance(result, dict) and "error" not in result)
        throughput = successful_requests / total_time

        # Performance assertions
        assert successful_requests == concurrent_requests
        assert throughput >= performance_test_config["min_throughput"]
        assert total_time < request_timeout

        print(f"Throughput: {throughput:.1f} requests/second ({successful_requests} requests in {total_time:.3f}s)")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_database_connection_pool_efficiency(self, mock_env_vars):
        """Test database connection pool efficiency."""
        server = DocumentRetrievalServer()

        # Mock pool with tracking
        connection_acquisitions = []
        connection_releases = []

        class MockConnection:
            def __init__(self, conn_id):
                self.conn_id = conn_id

            async def fetch(self, *args, **kwargs):
                return []

            async def fetchval(self, *args, **kwargs):
                return 0

            async def fetchrow(self, *args, **kwargs):
                return None

        class MockPool:
            def __init__(self):
                self.connections = [MockConnection(i) for i in range(5)]
                self.available = list(self.connections)
                self.in_use = []

            def acquire(self):
                return MockAcquireContext(self)

            def get_size(self):
                return len(self.connections)

            async def close(self):
                pass

        class MockAcquireContext:
            def __init__(self, pool):
                self.pool = pool
                self.connection = None

            async def __aenter__(self):
                acquire_time = time.time()
                if self.pool.available:
                    self.connection = self.pool.available.pop(0)
                    self.pool.in_use.append(self.connection)
                    connection_acquisitions.append(acquire_time)
                    return self.connection
                else:
                    raise Exception("No connections available")

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                release_time = time.time()
                if self.connection in self.pool.in_use:
                    self.pool.in_use.remove(self.connection)
                    self.pool.available.append(self.connection)
                    connection_releases.append(release_time)

        # Mock external dependencies
        server.db_pool = MockPool()
        server.openai_client = AsyncMock()
        server.supabase_client = MagicMock()

        # Mock embedding response
        server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

        # Test concurrent operations
        users = [str(uuid.uuid4()) for _ in range(10)]
        tasks = [
            server._search_documents(
                query=f"query {i}",
                user_id=users[i % len(users)],
                session_id=f"session-{i}"
            )
            for i in range(20)  # More requests than connections
        ]

        # Execute with timing
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Verify connection pool efficiency
        successful_requests = sum(1 for result in results if isinstance(result, dict))
        assert successful_requests > 0

        # Verify connections were reused
        assert len(connection_acquisitions) == len(connection_releases)
        assert len(connection_acquisitions) >= 20  # At least one per request

        # Calculate connection utilization metrics
        if connection_acquisitions and connection_releases:
            avg_hold_time = statistics.mean([
                connection_releases[i] - connection_acquisitions[i]
                for i in range(len(connection_acquisitions))
            ])
            print(f"Pool efficiency - Avg connection hold time: {avg_hold_time:.3f}s")
            print(f"Pool efficiency - Total acquisitions: {len(connection_acquisitions)}")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage_under_load(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test memory usage patterns under sustained load."""
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Mock large result sets
        large_results = [
            {
                'id': uuid.uuid4(),
                'document_id': uuid.uuid4(),
                'chunk_text': "Large chunk text " * 100,  # ~1.7KB per result
                'chunk_index': i,
                'chunk_metadata': {f"key_{j}": f"value_{j}" for j in range(50)},
                'filename': f"doc_{i}.pdf",
                'file_type': "pdf",
                'document_metadata': {f"meta_{j}": f"data_{j}" for j in range(25)},
                'similarity': 0.9 - (i * 0.01)
            }
            for i in range(20)
        ]

        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = large_results

        mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
        mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

        # Run multiple cycles of large queries
        memory_samples = []
        for cycle in range(10):
            # Execute multiple searches
            tasks = [
                mock_server._search_documents(
                    query=f"memory test {cycle}-{i}",
                    user_id=sample_user_id,
                    session_id=f"session-{cycle}-{i}"
                )
                for i in range(5)
            ]

            results = await asyncio.gather(*tasks)

            # Sample memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory)

            # Verify results
            for result in results:
                assert len(result["results"]) == 20

            # Small delay to allow garbage collection
            await asyncio.sleep(0.1)

        # Analyze memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_samples)
        avg_memory = statistics.mean(memory_samples)

        print(f"Memory usage - Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB")
        print(f"Memory usage - Growth: {memory_growth:.1f}MB, Max: {max_memory:.1f}MB, Avg: {avg_memory:.1f}MB")

        # Memory growth should be reasonable (less than 100MB for this test)
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f}MB"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cache_performance_impact(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_project_id
    ):
        """Test cache performance impact on repeated requests."""
        # Mock database responses
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.return_value = 5
        mock_conn.fetch.return_value = [
            {
                'id': uuid.uuid4(),
                'filename': f"doc_{i}.pdf",
                'file_type': "pdf",
                'file_size': 1024,
                'total_chunks': 3,
                'upload_date': time.time(),
                'project_id': sample_project_id,
                'metadata': {}
            }
            for i in range(5)
        ]

        # First request (cache miss)
        start_time = time.time()
        result1 = await mock_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=sample_project_id
        )
        cache_miss_time = time.time() - start_time

        # Verify database was called
        assert mock_conn.fetchval.call_count == 1
        assert mock_conn.fetch.call_count == 1

        # Reset call counts
        mock_conn.fetchval.reset_mock()
        mock_conn.fetch.reset_mock()

        # Second request (cache hit)
        start_time = time.time()
        result2 = await mock_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=sample_project_id
        )
        cache_hit_time = time.time() - start_time

        # Verify cache was used (no database calls)
        assert mock_conn.fetchval.call_count == 0
        assert mock_conn.fetch.call_count == 0

        # Verify results are identical
        assert result1 == result2

        # Cache hit should be significantly faster
        performance_improvement = cache_miss_time / cache_hit_time if cache_hit_time > 0 else float('inf')

        print(f"Cache performance - Miss: {cache_miss_time:.4f}s, Hit: {cache_hit_time:.4f}s")
        print(f"Cache performance - Improvement: {performance_improvement:.1f}x")

        # Cache should provide at least 2x improvement
        assert performance_improvement >= 2.0 or cache_hit_time < 0.001

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_embedding_generation_performance(self, mock_server):
        """Test embedding generation performance characteristics."""
        # Test queries of different lengths
        test_queries = [
            "short",
            "medium length query with several words",
            "long query " * 50,  # ~500 words
            "very long query " * 200,  # ~2000 words
        ]

        embedding_times = []

        for query in test_queries:
            # Mock OpenAI response with delay simulation
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = np.random.rand(1536).tolist()

            async def mock_create_with_delay(**kwargs):
                # Simulate network delay proportional to input length
                delay = min(len(kwargs['input']) / 10000, 0.1)  # Max 100ms delay
                await asyncio.sleep(delay)
                return mock_response

            mock_server.openai_client.embeddings.create.side_effect = mock_create_with_delay

            # Measure embedding generation time
            start_time = time.time()
            result = await mock_server._generate_embedding(query)
            embedding_time = time.time() - start_time

            embedding_times.append((len(query), embedding_time))

            assert len(result) == 1536

        # Analyze performance characteristics
        for query_length, embedding_time in embedding_times:
            print(f"Embedding performance - Length: {query_length}, Time: {embedding_time:.3f}s")

        # All embeddings should complete within reasonable time
        max_time = max(time for _, time in embedding_times)
        assert max_time < 1.0, f"Embedding generation too slow: {max_time:.3f}s"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_vector_similarity_search_performance(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test vector similarity search performance with different result sizes."""
        # Test with different result set sizes
        result_sizes = [1, 5, 10, 20]  # Top-K values
        search_times = []

        for size in result_sizes:
            # Mock results of specified size
            mock_results = [
                {
                    'id': uuid.uuid4(),
                    'document_id': uuid.uuid4(),
                    'chunk_text': f"Result {i} with some content",
                    'chunk_index': i,
                    'chunk_metadata': {},
                    'filename': f"doc_{i}.pdf",
                    'file_type': "pdf",
                    'document_metadata': {},
                    'similarity': 0.9 - (i * 0.01)
                }
                for i in range(size)
            ]

            _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
            mock_conn.fetch.return_value = mock_results

            # Mock embedding
            mock_server.openai_client.embeddings.create.return_value.data = [MagicMock()]
            mock_server.openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536

            # Measure search time
            start_time = time.time()
            result = await mock_server._search_documents(
                query="performance test",
                user_id=sample_user_id,
                session_id=sample_session_id,
                top_k=size
            )
            search_time = time.time() - start_time

            search_times.append((size, search_time))

            assert len(result["results"]) == size

        # Analyze scaling characteristics
        for result_size, search_time in search_times:
            print(f"Vector search performance - Size: {result_size}, Time: {search_time:.3f}s")

        # Search time should scale reasonably with result size
        max_time = max(time for _, time in search_times)
        assert max_time < 1.0, f"Vector search too slow: {max_time:.3f}s"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_database_query_optimization(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test database query performance with different data patterns."""
        # Test different query patterns
        query_patterns = [
            ("simple_filter", {"user_id": sample_user_id}),
            ("compound_filter", {"user_id": sample_user_id, "session_id": sample_session_id}),
            ("pagination", {"user_id": sample_user_id, "page": 1, "per_page": 10}),
            ("large_pagination", {"user_id": sample_user_id, "page": 100, "per_page": 50}),
        ]

        query_times = []

        for pattern_name, params in query_patterns:
            # Mock database with query timing simulation
            _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value

            async def mock_query_with_timing(*args, **kwargs):
                # Simulate query execution time based on complexity
                if "page" in str(args):
                    page = params.get("page", 1)
                    delay = min(page * 0.001, 0.05)  # Larger page numbers take slightly longer
                else:
                    delay = 0.005  # Base query time
                await asyncio.sleep(delay)
                return []

            mock_conn.fetchval.side_effect = lambda *args, **kwargs: asyncio.create_task(mock_query_with_timing(*args, **kwargs)).result() or 0
            mock_conn.fetch.side_effect = mock_query_with_timing

            # Measure query time
            start_time = time.time()
            result = await mock_server._list_user_documents(**params)
            query_time = time.time() - start_time

            query_times.append((pattern_name, query_time))

            assert "documents" in result

        # Analyze query performance
        for pattern_name, query_time in query_times:
            print(f"Query performance - Pattern: {pattern_name}, Time: {query_time:.3f}s")

        # All queries should complete within reasonable time
        max_time = max(time for _, time in query_times)
        assert max_time < 0.5, f"Database query too slow: {max_time:.3f}s"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_resource_cleanup_performance(self, mock_env_vars):
        """Test resource cleanup performance during server shutdown."""
        server = DocumentRetrievalServer()

        # Mock initialized state with multiple resources
        mock_pools = [AsyncMock() for _ in range(5)]
        server.db_pool = mock_pools[0]  # Main pool

        # Mock cleanup operations with timing
        cleanup_times = []

        async def timed_cleanup():
            start_time = time.time()
            await server.cleanup()
            cleanup_time = time.time() - start_time
            cleanup_times.append(cleanup_time)

        # Test cleanup multiple times
        for i in range(3):
            await timed_cleanup()

        # Verify cleanup performance
        avg_cleanup_time = statistics.mean(cleanup_times)
        max_cleanup_time = max(cleanup_times)

        print(f"Cleanup performance - Avg: {avg_cleanup_time:.3f}s, Max: {max_cleanup_time:.3f}s")

        # Cleanup should be fast
        assert avg_cleanup_time < 1.0
        assert max_cleanup_time < 2.0

        # Verify cleanup was called on resources
        for pool in mock_pools[:1]:  # Only main pool should be cleaned
            pool.close.assert_called()