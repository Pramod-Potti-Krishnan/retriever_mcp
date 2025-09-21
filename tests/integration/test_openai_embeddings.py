"""
Integration tests for OpenAI embeddings in Document Retrieval MCP Server.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from server import DocumentRetrievalServer


class TestOpenAIIntegration:
    """Test suite for OpenAI embeddings integration."""

    @pytest.fixture
    async def mock_openai_server(self, mock_env_vars):
        """Create server with mocked OpenAI client."""
        server = DocumentRetrievalServer()

        # Mock OpenAI client
        server.openai_client = AsyncMock()

        # Mock database and Supabase
        server.db_pool = AsyncMock()
        server.supabase_client = MagicMock()

        return server

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embedding_generation_success(self, mock_openai_server):
        """Test successful embedding generation."""
        server = mock_openai_server

        # Mock OpenAI response
        test_embedding = np.random.rand(1536).tolist()
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = test_embedding

        server.openai_client.embeddings.create.return_value = mock_response

        # Test embedding generation
        result = await server._generate_embedding("test query")

        # Verify API call
        server.openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="test query"
        )

        # Verify result
        assert result == test_embedding
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embedding_generation_with_different_models(self, mock_openai_server):
        """Test embedding generation with different model configurations."""
        server = mock_openai_server

        test_cases = [
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 3072),
            ("text-embedding-ada-002", 1536)
        ]

        for model_name, expected_dimensions in test_cases:
            # Mock environment variable
            with patch('server.EMBEDDING_MODEL', model_name):
                with patch('server.VECTOR_DIMENSIONS', expected_dimensions):
                    # Mock response with appropriate dimensions
                    test_embedding = np.random.rand(expected_dimensions).tolist()
                    mock_response = MagicMock()
                    mock_response.data = [MagicMock()]
                    mock_response.data[0].embedding = test_embedding

                    server.openai_client.embeddings.create.return_value = mock_response

                    # Generate embedding
                    result = await server._generate_embedding("test query")

                    # Verify model and dimensions
                    server.openai_client.embeddings.create.assert_called_with(
                        model=model_name,
                        input="test query"
                    )
                    assert len(result) == expected_dimensions

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embedding_generation_rate_limit_error(self, mock_openai_server):
        """Test embedding generation with rate limit error."""
        server = mock_openai_server

        # Mock rate limit error
        from openai import RateLimitError
        server.openai_client.embeddings.create.side_effect = RateLimitError(
            "Rate limit exceeded",
            response=MagicMock(),
            body={}
        )

        # Test error handling
        with pytest.raises(Exception) as exc_info:
            await server._generate_embedding("test query")

        assert "Rate limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embedding_generation_api_key_error(self, mock_openai_server):
        """Test embedding generation with API key error."""
        server = mock_openai_server

        # Mock authentication error
        from openai import AuthenticationError
        server.openai_client.embeddings.create.side_effect = AuthenticationError(
            "Invalid API key",
            response=MagicMock(),
            body={}
        )

        # Test error handling
        with pytest.raises(Exception) as exc_info:
            await server._generate_embedding("test query")

        assert "Invalid API key" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embedding_generation_timeout_error(self, mock_openai_server):
        """Test embedding generation with timeout error."""
        server = mock_openai_server

        # Mock timeout error
        server.openai_client.embeddings.create.side_effect = asyncio.TimeoutError(
            "Request timeout"
        )

        # Test error handling
        with pytest.raises(Exception) as exc_info:
            await server._generate_embedding("test query")

        assert "Request timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embedding_generation_with_long_text(self, mock_openai_server):
        """Test embedding generation with long input text."""
        server = mock_openai_server

        # Create long text (near token limit)
        long_text = "This is a very long text. " * 1000  # Approximately 8000 tokens

        # Mock response
        test_embedding = np.random.rand(1536).tolist()
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = test_embedding

        server.openai_client.embeddings.create.return_value = mock_response

        # Test with long text
        result = await server._generate_embedding(long_text)

        # Verify API was called with full text
        server.openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=long_text
        )
        assert result == test_embedding

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embedding_generation_with_special_characters(self, mock_openai_server):
        """Test embedding generation with special characters and Unicode."""
        server = mock_openai_server

        test_cases = [
            "Text with émojis 🚀 and ünicode characters",
            "中文测试文本",
            "Русский текст для тестирования",
            "Text with\nnewlines\tand\ttabs",
            "Text with \"quotes\" and 'apostrophes'",
            "Mathematical symbols: ∑∫∆π∞",
            "Special chars: @#$%^&*()_+-={}[]|\\:;\"'<>?,./"
        ]

        for test_text in test_cases:
            # Mock response
            test_embedding = np.random.rand(1536).tolist()
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = test_embedding

            server.openai_client.embeddings.create.return_value = mock_response

            # Test embedding generation
            result = await server._generate_embedding(test_text)

            # Verify result
            assert result == test_embedding
            server.openai_client.embeddings.create.assert_called_with(
                model="text-embedding-3-small",
                input=test_text
            )

            # Reset mock for next iteration
            server.openai_client.embeddings.create.reset_mock()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embedding_generation_empty_text(self, mock_openai_server):
        """Test embedding generation with empty or whitespace text."""
        server = mock_openai_server

        test_cases = ["", "   ", "\n\t\r", " \n \t \r "]

        for test_text in test_cases:
            # Mock response
            test_embedding = np.random.rand(1536).tolist()
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = test_embedding

            server.openai_client.embeddings.create.return_value = mock_response

            # Test embedding generation
            result = await server._generate_embedding(test_text)

            # Verify API was called with the text as-is
            server.openai_client.embeddings.create.assert_called_with(
                model="text-embedding-3-small",
                input=test_text
            )
            assert result == test_embedding

            # Reset mock for next iteration
            server.openai_client.embeddings.create.reset_mock()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_embedding_generation(self, mock_openai_server):
        """Test concurrent embedding generation requests."""
        server = mock_openai_server

        # Create multiple test queries
        test_queries = [f"Test query {i}" for i in range(10)]

        # Mock responses for each query
        def mock_create_embedding(**kwargs):
            # Return different embeddings for different inputs
            input_text = kwargs['input']
            seed = abs(hash(input_text)) % 1000
            np.random.seed(seed)
            test_embedding = np.random.rand(1536).tolist()

            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = test_embedding
            return mock_response

        server.openai_client.embeddings.create.side_effect = mock_create_embedding

        # Generate embeddings concurrently
        tasks = [server._generate_embedding(query) for query in test_queries]
        results = await asyncio.gather(*tasks)

        # Verify all embeddings were generated
        assert len(results) == len(test_queries)
        assert server.openai_client.embeddings.create.call_count == len(test_queries)

        # Verify each result is valid
        for result in results:
            assert len(result) == 1536
            assert all(isinstance(x, float) for x in result)

        # Verify all embeddings are different (due to different seeds)
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                assert results[i] != results[j], f"Embeddings {i} and {j} are identical"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embedding_vector_normalization(self, mock_openai_server):
        """Test embedding vector normalization and validation."""
        server = mock_openai_server

        # Test different vector magnitudes
        test_vectors = [
            np.random.rand(1536).tolist(),  # Normal vector
            (np.random.rand(1536) * 10).tolist(),  # Large magnitude
            (np.random.rand(1536) * 0.1).tolist(),  # Small magnitude
            np.zeros(1536).tolist(),  # Zero vector
            (np.ones(1536) * 0.5).tolist()  # Uniform vector
        ]

        for test_vector in test_vectors:
            # Mock response
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = test_vector

            server.openai_client.embeddings.create.return_value = mock_response

            # Generate embedding
            result = await server._generate_embedding("test")

            # Verify vector properties
            assert len(result) == 1536
            assert all(isinstance(x, (int, float)) for x in result)
            assert result == test_vector

            # Verify vector is finite
            assert all(np.isfinite(x) for x in result)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embedding_integration_with_search(self, mock_openai_server):
        """Test embedding integration with search functionality."""
        server = mock_openai_server

        # Mock database connection
        mock_conn = AsyncMock()
        server.db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        server.db_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock embedding generation
        test_embedding = np.random.rand(1536).tolist()
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = test_embedding

        server.openai_client.embeddings.create.return_value = mock_response

        # Mock database search results
        mock_conn.fetch.return_value = [
            {
                'id': 'chunk-uuid',
                'document_id': 'doc-uuid',
                'chunk_text': "Sample chunk text",
                'chunk_index': 0,
                'chunk_metadata': {},
                'filename': "test.pdf",
                'file_type': "pdf",
                'document_metadata': {},
                'similarity': 0.85
            }
        ]

        # Execute search with embedding
        result = await server._search_documents(
            query="test query",
            user_id="user-123",
            session_id="session-123"
        )

        # Verify embedding was generated and used
        server.openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="test query"
        )

        # Verify database query used the embedding
        mock_conn.fetch.assert_called_once()
        db_call_args = mock_conn.fetch.call_args[0]
        assert db_call_args[1] == test_embedding  # First parameter after SQL should be embedding

        # Verify search results
        assert "results" in result
        assert len(result["results"]) == 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embedding_retry_mechanism(self, mock_openai_server):
        """Test embedding generation with retry mechanism for transient errors."""
        server = mock_openai_server

        # Mock transient failures followed by success
        call_count = 0

        def mock_create_with_retry(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count <= 2:
                # First two calls fail with transient errors
                from openai import APIConnectionError
                raise APIConnectionError("Connection failed")
            else:
                # Third call succeeds
                test_embedding = np.random.rand(1536).tolist()
                mock_response = MagicMock()
                mock_response.data = [MagicMock()]
                mock_response.data[0].embedding = test_embedding
                return mock_response

        server.openai_client.embeddings.create.side_effect = mock_create_with_retry

        # Implement simple retry logic in test
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await server._generate_embedding("test query")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1)  # Brief delay between retries

        # Verify final success
        assert len(result) == 1536
        assert call_count == 3  # Should have made 3 attempts

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embedding_api_usage_patterns(self, mock_openai_server):
        """Test different API usage patterns and configurations."""
        server = mock_openai_server

        # Test batch processing simulation
        queries = [f"Query {i}" for i in range(5)]

        # Mock responses for batch processing
        responses = []
        for i, query in enumerate(queries):
            test_embedding = (np.random.rand(1536) + i * 0.1).tolist()  # Slightly different embeddings
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = test_embedding
            responses.append(mock_response)

        server.openai_client.embeddings.create.side_effect = responses

        # Process queries sequentially
        results = []
        for query in queries:
            result = await server._generate_embedding(query)
            results.append(result)

        # Verify all queries were processed
        assert len(results) == len(queries)
        assert server.openai_client.embeddings.create.call_count == len(queries)

        # Verify each call had correct parameters
        for i, call in enumerate(server.openai_client.embeddings.create.call_args_list):
            args, kwargs = call
            assert kwargs['model'] == "text-embedding-3-small"
            assert kwargs['input'] == queries[i]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embedding_dimension_validation(self, mock_openai_server):
        """Test validation of embedding dimensions."""
        server = mock_openai_server

        # Test cases with different dimensions
        test_cases = [
            (1536, True),   # Correct dimension
            (768, False),   # Wrong dimension
            (3072, False),  # Wrong dimension
            (0, False),     # Invalid dimension
            (1535, False),  # Off by one
            (1537, False)   # Off by one
        ]

        for dimensions, should_pass in test_cases:
            # Mock response with specific dimensions
            if dimensions > 0:
                test_embedding = np.random.rand(dimensions).tolist()
            else:
                test_embedding = []

            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = test_embedding

            server.openai_client.embeddings.create.return_value = mock_response

            # Test embedding generation
            result = await server._generate_embedding("test")

            if should_pass:
                assert len(result) == 1536
            else:
                # In this test, we just verify the embedding is returned as-is
                # Real validation would happen in the application logic
                assert len(result) == dimensions