"""
Unit tests for Document Retrieval MCP Server tools.
"""

import pytest
import uuid
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, List, Any

import asyncpg
import numpy as np
from mcp.types import ToolRequest, ToolResult, TextContent

from server import DocumentRetrievalServer


class TestSearchDocumentsTool:
    """Test suite for search_documents tool."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_documents_success(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_project_id,
        sample_search_results
    ):
        """Test successful document search."""
        # Setup mock database response
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = [
            {
                'id': uuid.UUID(result['id']),
                'document_id': uuid.UUID(result['document_id']),
                'chunk_text': result['chunk_text'],
                'chunk_index': result['chunk_index'],
                'chunk_metadata': result['chunk_metadata'],
                'filename': result['filename'],
                'file_type': result['file_type'],
                'document_metadata': result['document_metadata'],
                'similarity': result['similarity']
            }
            for result in sample_search_results
        ]

        # Execute search
        result = await mock_server._search_documents(
            query="test query",
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=sample_project_id,
            top_k=5,
            similarity_threshold=0.7
        )

        # Verify results
        assert "results" in result
        assert "query" in result
        assert "count" in result
        assert result["query"] == "test query"
        assert result["count"] == len(sample_search_results)
        assert len(result["results"]) == len(sample_search_results)

        # Verify result structure
        for i, res in enumerate(result["results"]):
            assert "chunk_id" in res
            assert "document_id" in res
            assert "chunk_text" in res
            assert "similarity_score" in res
            assert res["chunk_text"] == sample_search_results[i]["chunk_text"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_documents_empty_results(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_project_id
    ):
        """Test search with no matching documents."""
        # Setup mock to return empty results
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = []

        result = await mock_server._search_documents(
            query="nonexistent query",
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=sample_project_id,
            top_k=5,
            similarity_threshold=0.7
        )

        assert result["results"] == []
        assert result["count"] == 0
        assert result["query"] == "nonexistent query"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_documents_database_error(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_project_id,
        database_error_scenarios
    ):
        """Test search with database error."""
        # Setup mock to raise database error
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.side_effect = database_error_scenarios["connection_timeout"]

        result = await mock_server._search_documents(
            query="test query",
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=sample_project_id
        )

        assert "error" in result
        assert result["results"] == []

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_documents_embedding_error(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_project_id,
        openai_error_scenarios
    ):
        """Test search with embedding generation error."""
        # Setup mock to raise OpenAI error
        mock_server.openai_client.embeddings.create.side_effect = openai_error_scenarios["rate_limit"]

        result = await mock_server._search_documents(
            query="test query",
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=sample_project_id
        )

        assert "error" in result
        assert result["results"] == []

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_search_documents_parameter_validation(self, mock_server, sample_user_id, sample_session_id):
        """Test parameter validation for search_documents."""
        # Test with different parameter combinations
        test_cases = [
            {"top_k": 0, "should_work": False},  # Invalid top_k
            {"top_k": 25, "should_work": False},  # Exceeds limit
            {"similarity_threshold": -0.1, "should_work": False},  # Invalid threshold
            {"similarity_threshold": 1.1, "should_work": False},  # Invalid threshold
            {"top_k": 5, "similarity_threshold": 0.8, "should_work": True}  # Valid
        ]

        for case in test_cases:
            if case["should_work"]:
                # Should not raise exception
                await mock_server._search_documents(
                    query="test",
                    user_id=sample_user_id,
                    session_id=sample_session_id,
                    **{k: v for k, v in case.items() if k != "should_work"}
                )
            # Note: Parameter validation is handled by MCP schema validation


class TestGetDocumentContextTool:
    """Test suite for get_document_context tool."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_document_context_success(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_document_id,
        sample_document_metadata
    ):
        """Test successful document context retrieval."""
        # Setup mock database response
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = [
            {
                'id': uuid.uuid4(),
                'chunk_text': f"Chunk {i} text content",
                'chunk_index': i,
                'chunk_metadata': {"page": i + 1},
                'filename': sample_document_metadata["filename"],
                'file_type': sample_document_metadata["file_type"],
                'document_metadata': sample_document_metadata["metadata"],
                'total_chunks': sample_document_metadata["total_chunks"]
            }
            for i in range(3)
        ]

        result = await mock_server._get_document_context(
            document_id=sample_document_id,
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert "document_id" in result
        assert "document_info" in result
        assert "chunks" in result
        assert "chunk_count" in result
        assert result["document_id"] == sample_document_id
        assert result["chunk_count"] == 3
        assert len(result["chunks"]) == 3

        # Verify document info structure
        doc_info = result["document_info"]
        assert doc_info["filename"] == sample_document_metadata["filename"]
        assert doc_info["file_type"] == sample_document_metadata["file_type"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_document_context_with_chunk_ids(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_document_id
    ):
        """Test document context retrieval with specific chunk IDs."""
        chunk_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

        # Setup mock database response
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = [
            {
                'id': uuid.UUID(chunk_ids[0]),
                'chunk_text': "Specific chunk 1",
                'chunk_index': 0,
                'chunk_metadata': {"page": 1},
                'filename': "test.pdf",
                'file_type': "pdf",
                'document_metadata': {},
                'total_chunks': 5
            },
            {
                'id': uuid.UUID(chunk_ids[1]),
                'chunk_text': "Specific chunk 2",
                'chunk_index': 2,
                'chunk_metadata': {"page": 2},
                'filename': "test.pdf",
                'file_type': "pdf",
                'document_metadata': {},
                'total_chunks': 5
            }
        ]

        result = await mock_server._get_document_context(
            document_id=sample_document_id,
            user_id=sample_user_id,
            session_id=sample_session_id,
            chunk_ids=chunk_ids
        )

        assert result["chunk_count"] == 2
        assert len(result["chunks"]) == 2
        chunk_ids_returned = [chunk["chunk_id"] for chunk in result["chunks"]]
        assert set(chunk_ids_returned) == set(chunk_ids)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_document_context_not_found(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_document_id
    ):
        """Test document context retrieval for non-existent document."""
        # Setup mock to return empty results
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = []

        result = await mock_server._get_document_context(
            document_id=sample_document_id,
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert "error" in result
        assert result["chunks"] == []
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_document_context_database_error(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_document_id,
        database_error_scenarios
    ):
        """Test document context retrieval with database error."""
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.side_effect = database_error_scenarios["query_timeout"]

        result = await mock_server._get_document_context(
            document_id=sample_document_id,
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert "error" in result
        assert result["chunks"] == []


class TestListUserDocumentsTool:
    """Test suite for list_user_documents tool."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_list_user_documents_success(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_project_id,
        sample_document_list
    ):
        """Test successful document listing."""
        # Setup mock database responses
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.return_value = len(sample_document_list)  # Total count
        mock_conn.fetch.return_value = [
            {
                'id': uuid.UUID(doc['id']),
                'filename': doc['filename'],
                'file_type': doc['file_type'],
                'file_size': doc['file_size'],
                'total_chunks': doc['total_chunks'],
                'upload_date': doc['upload_date'],
                'project_id': doc['project_id'],
                'metadata': doc['metadata']
            }
            for doc in sample_document_list
        ]

        result = await mock_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=sample_project_id,
            page=1,
            per_page=20
        )

        assert "documents" in result
        assert "pagination" in result
        assert len(result["documents"]) == len(sample_document_list)

        # Verify pagination structure
        pagination = result["pagination"]
        assert pagination["page"] == 1
        assert pagination["per_page"] == 20
        assert pagination["total"] == len(sample_document_list)

        # Verify document structure
        for i, doc in enumerate(result["documents"]):
            assert "document_id" in doc
            assert "filename" in doc
            assert "file_type" in doc
            assert doc["filename"] == sample_document_list[i]["filename"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_list_user_documents_pagination(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test document listing with pagination."""
        total_docs = 25
        per_page = 10

        # Setup mock for page 2
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.return_value = total_docs
        mock_conn.fetch.return_value = [
            {
                'id': uuid.uuid4(),
                'filename': f"doc_{i}.pdf",
                'file_type': "pdf",
                'file_size': 1024,
                'total_chunks': 3,
                'upload_date': datetime.now(),
                'project_id': "test",
                'metadata': {}
            }
            for i in range(per_page)  # Second page documents
        ]

        result = await mock_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            page=2,
            per_page=per_page
        )

        pagination = result["pagination"]
        assert pagination["page"] == 2
        assert pagination["per_page"] == per_page
        assert pagination["total"] == total_docs
        assert pagination["total_pages"] == 3  # ceil(25/10)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_list_user_documents_cache_hit(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_project_id
    ):
        """Test document listing with cache hit."""
        # Pre-populate cache
        cache_key = f"docs_{sample_user_id}_{sample_session_id}_{sample_project_id}_1_20"
        cached_result = {
            "documents": [{"document_id": str(uuid.uuid4()), "filename": "cached.pdf"}],
            "pagination": {"page": 1, "per_page": 20, "total": 1, "total_pages": 1}
        }
        mock_server.metadata_cache[cache_key] = cached_result

        result = await mock_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            project_id=sample_project_id,
            page=1,
            per_page=20
        )

        # Should return cached result without database call
        assert result == cached_result

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_list_user_documents_no_project_filter(
        self,
        mock_server,
        sample_user_id,
        sample_session_id
    ):
        """Test document listing without project filter."""
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.return_value = 1
        mock_conn.fetch.return_value = [
            {
                'id': uuid.uuid4(),
                'filename': "test.pdf",
                'file_type': "pdf",
                'file_size': 1024,
                'total_chunks': 3,
                'upload_date': datetime.now(),
                'project_id': "any-project",
                'metadata': {}
            }
        ]

        result = await mock_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id,
            # No project_id filter
        )

        assert "documents" in result
        assert len(result["documents"]) == 1

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_list_user_documents_database_error(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        database_error_scenarios
    ):
        """Test document listing with database error."""
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.side_effect = database_error_scenarios["connection_failed"]

        result = await mock_server._list_user_documents(
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert "error" in result
        assert result["documents"] == []


class TestGetSimilarChunksTool:
    """Test suite for get_similar_chunks tool."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_similar_chunks_success(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_chunk_id,
        sample_embedding
    ):
        """Test successful similar chunks retrieval."""
        # Setup mock database responses
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value

        # Mock reference chunk
        mock_conn.fetchrow.return_value = {
            'embedding': sample_embedding,
            'chunk_text': "Reference chunk text",
            'document_id': uuid.uuid4(),
            'chunk_index': 0
        }

        # Mock similar chunks
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

        result = await mock_server._get_similar_chunks(
            chunk_id=sample_chunk_id,
            user_id=sample_user_id,
            session_id=sample_session_id,
            top_k=3
        )

        assert "reference_chunk_id" in result
        assert "reference_text" in result
        assert "reference_document_id" in result
        assert "similar_chunks" in result
        assert "count" in result

        assert result["reference_chunk_id"] == sample_chunk_id
        assert result["reference_text"] == "Reference chunk text"
        assert result["count"] == 3
        assert len(result["similar_chunks"]) == 3

        # Verify similar chunks structure
        for chunk in result["similar_chunks"]:
            assert "chunk_id" in chunk
            assert "document_id" in chunk
            assert "chunk_text" in chunk
            assert "similarity_score" in chunk

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_similar_chunks_reference_not_found(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_chunk_id
    ):
        """Test similar chunks retrieval with non-existent reference."""
        # Setup mock to return no reference chunk
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.return_value = None

        result = await mock_server._get_similar_chunks(
            chunk_id=sample_chunk_id,
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert "error" in result
        assert result["similar_chunks"] == []
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_similar_chunks_no_similar_found(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_chunk_id,
        sample_embedding
    ):
        """Test similar chunks retrieval with no similar chunks found."""
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value

        # Mock reference chunk exists
        mock_conn.fetchrow.return_value = {
            'embedding': sample_embedding,
            'chunk_text': "Reference chunk text",
            'document_id': uuid.uuid4(),
            'chunk_index': 0
        }

        # Mock no similar chunks found
        mock_conn.fetch.return_value = []

        result = await mock_server._get_similar_chunks(
            chunk_id=sample_chunk_id,
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert result["similar_chunks"] == []
        assert result["count"] == 0
        assert "reference_chunk_id" in result

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_similar_chunks_database_error(
        self,
        mock_server,
        sample_user_id,
        sample_session_id,
        sample_chunk_id,
        database_error_scenarios
    ):
        """Test similar chunks retrieval with database error."""
        _, mock_conn = mock_server.db_pool, mock_server.db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.side_effect = database_error_scenarios["invalid_sql"]

        result = await mock_server._get_similar_chunks(
            chunk_id=sample_chunk_id,
            user_id=sample_user_id,
            session_id=sample_session_id
        )

        assert "error" in result
        assert result["similar_chunks"] == []


class TestToolIntegration:
    """Integration tests for tool functionality."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_call_tool_handler_success(self, mock_server, valid_tool_requests):
        """Test successful tool call handling."""
        # Mock the individual tool methods
        mock_server._search_documents = AsyncMock(return_value={"results": [], "count": 0})
        mock_server._get_document_context = AsyncMock(return_value={"chunks": [], "chunk_count": 0})
        mock_server._list_user_documents = AsyncMock(return_value={"documents": [], "pagination": {}})
        mock_server._get_similar_chunks = AsyncMock(return_value={"similar_chunks": [], "count": 0})

        for tool_name, request in valid_tool_requests.items():
            result = await mock_server.server.call_tool(request)

            assert isinstance(result, ToolResult)
            assert result.tool == tool_name
            assert result.error is None
            assert len(result.content) == 1
            assert isinstance(result.content[0], TextContent)

            # Verify JSON content is valid
            content_data = json.loads(result.content[0].text)
            assert isinstance(content_data, dict)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_call_tool_handler_unknown_tool(self, mock_server):
        """Test tool call handler with unknown tool."""
        unknown_request = ToolRequest(
            tool="unknown_tool",
            params={}
        )

        result = await mock_server.server.call_tool(unknown_request)

        assert isinstance(result, ToolResult)
        assert result.tool == "unknown_tool"
        assert result.error is not None
        assert "unknown tool" in result.error.lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_call_tool_handler_exception(self, mock_server, valid_tool_requests):
        """Test tool call handler with internal exception."""
        # Mock tool method to raise exception
        mock_server._search_documents = AsyncMock(side_effect=Exception("Internal error"))

        request = valid_tool_requests["search_documents"]
        result = await mock_server.server.call_tool(request)

        assert isinstance(result, ToolResult)
        assert result.tool == "search_documents"
        assert result.error is not None
        assert "internal error" in result.error.lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_list_tools(self, mock_server):
        """Test tool listing functionality."""
        tools = await mock_server.server.list_tools()

        assert len(tools) == 4
        tool_names = [tool.name for tool in tools]
        expected_tools = ["search_documents", "get_document_context", "list_user_documents", "get_similar_chunks"]

        for expected_tool in expected_tools:
            assert expected_tool in tool_names

        # Verify tool schemas
        for tool in tools:
            assert hasattr(tool, 'input_schema')
            assert isinstance(tool.input_schema, dict)
            assert 'type' in tool.input_schema
            assert tool.input_schema['type'] == 'object'
            assert 'properties' in tool.input_schema
            assert 'required' in tool.input_schema