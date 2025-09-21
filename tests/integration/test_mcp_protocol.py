"""
Integration tests for MCP protocol compliance in Document Retrieval MCP Server.
"""

import pytest
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import (
    Tool, TextContent, ToolRequest, ToolResult,
    Resource, ResourceContent, TextResourceContent
)

from server import DocumentRetrievalServer


class TestMCPProtocolCompliance:
    """Test suite for MCP protocol compliance."""

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_server_initialization(self, mock_env_vars):
        """Test MCP server initialization compliance."""
        server = DocumentRetrievalServer()

        # Verify server instance
        assert server.server is not None
        assert server.server.info.name == "document-retrieval-mcp"

        # Verify handlers are registered
        assert hasattr(server.server, '_tool_handlers')
        assert hasattr(server.server, '_resource_handlers')

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_list_tools_compliance(self, mock_server):
        """Test list_tools MCP compliance."""
        tools = await mock_server.server.list_tools()

        # Verify return type
        assert isinstance(tools, list)
        assert len(tools) == 4

        # Verify each tool is a Tool instance
        for tool in tools:
            assert isinstance(tool, Tool)
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'input_schema')

        # Verify tool names
        tool_names = [tool.name for tool in tools]
        expected_tools = ["search_documents", "get_document_context", "list_user_documents", "get_similar_chunks"]
        assert set(tool_names) == set(expected_tools)

        # Verify input schemas are valid JSON schemas
        for tool in tools:
            schema = tool.input_schema
            assert isinstance(schema, dict)
            assert schema.get('type') == 'object'
            assert 'properties' in schema
            assert 'required' in schema

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_call_tool_compliance(self, mock_server, valid_tool_requests):
        """Test call_tool MCP compliance."""
        # Mock tool implementations
        mock_server._search_documents = AsyncMock(return_value={"results": [], "count": 0})
        mock_server._get_document_context = AsyncMock(return_value={"chunks": [], "chunk_count": 0})
        mock_server._list_user_documents = AsyncMock(return_value={"documents": [], "pagination": {}})
        mock_server._get_similar_chunks = AsyncMock(return_value={"similar_chunks": [], "count": 0})

        for tool_name, request in valid_tool_requests.items():
            result = await mock_server.server.call_tool(request)

            # Verify return type
            assert isinstance(result, ToolResult)

            # Verify required fields
            assert hasattr(result, 'tool')
            assert hasattr(result, 'content')
            assert result.tool == tool_name

            # Verify content format
            if result.error is None:
                assert isinstance(result.content, list)
                assert len(result.content) == 1
                assert isinstance(result.content[0], TextContent)

                # Verify JSON content is valid
                content_text = result.content[0].text
                try:
                    parsed_content = json.loads(content_text)
                    assert isinstance(parsed_content, dict)
                except json.JSONDecodeError:
                    pytest.fail(f"Tool {tool_name} returned invalid JSON content")

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_error_handling_compliance(self, mock_server):
        """Test MCP error handling compliance."""
        # Test unknown tool
        unknown_request = ToolRequest(
            tool="unknown_tool",
            params={}
        )

        result = await mock_server.server.call_tool(unknown_request)

        # Verify error response format
        assert isinstance(result, ToolResult)
        assert result.tool == "unknown_tool"
        assert result.error is not None
        assert isinstance(result.error, str)
        assert result.content is None

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_tool_parameter_validation_compliance(self, mock_server):
        """Test tool parameter validation compliance."""
        # Test missing required parameters
        invalid_request = ToolRequest(
            tool="search_documents",
            params={"query": "test"}  # Missing required user_id and session_id
        )

        # Mock the tool to raise validation error
        mock_server._search_documents = AsyncMock(side_effect=ValueError("Missing required parameters"))

        result = await mock_server.server.call_tool(invalid_request)

        # Verify error is handled properly
        assert isinstance(result, ToolResult)
        assert result.error is not None
        assert "missing required parameters" in result.error.lower()

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_list_resources_compliance(self, mock_server):
        """Test list_resources MCP compliance."""
        resources = await mock_server.server.list_resources()

        # Verify return type
        assert isinstance(resources, list)
        assert len(resources) == 2

        # Verify each resource is a Resource instance
        for resource in resources:
            assert isinstance(resource, Resource)
            assert hasattr(resource, 'uri')
            assert hasattr(resource, 'name')
            assert hasattr(resource, 'description')

        # Verify resource URIs
        resource_uris = [resource.uri for resource in resources]
        expected_uris = ["resource://server-info", "resource://schema-info"]
        assert set(resource_uris) == set(expected_uris)

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_get_resource_compliance(self, mock_server):
        """Test get_resource MCP compliance."""
        test_cases = [
            "resource://server-info",
            "resource://schema-info"
        ]

        for uri in test_cases:
            result = await mock_server.server.get_resource(uri)

            # Verify return type
            assert isinstance(result, ResourceContent)
            assert isinstance(result, TextResourceContent)

            # Verify required fields
            assert hasattr(result, 'uri')
            assert hasattr(result, 'text')
            assert result.uri == uri

            # Verify content is valid JSON
            try:
                parsed_content = json.loads(result.text)
                assert isinstance(parsed_content, dict)
            except json.JSONDecodeError:
                pytest.fail(f"Resource {uri} returned invalid JSON content")

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_unknown_resource_error_compliance(self, mock_server):
        """Test unknown resource error handling compliance."""
        with pytest.raises(ValueError) as exc_info:
            await mock_server.server.get_resource("resource://unknown")

        assert "unknown resource" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_tool_result_content_compliance(self, mock_server):
        """Test tool result content format compliance."""
        # Mock different return value types
        test_cases = [
            {"results": [], "count": 0},
            {"documents": [], "pagination": {"page": 1}},
            {"chunks": [], "chunk_count": 0},
            {"similar_chunks": [], "count": 0}
        ]

        for return_value in test_cases:
            mock_server._search_documents = AsyncMock(return_value=return_value)

            request = ToolRequest(
                tool="search_documents",
                params={
                    "query": "test",
                    "user_id": str(uuid.uuid4()),
                    "session_id": "session-123"
                }
            )

            result = await mock_server.server.call_tool(request)

            # Verify content structure
            assert isinstance(result.content, list)
            assert len(result.content) == 1
            assert isinstance(result.content[0], TextContent)

            # Verify JSON serialization
            content_text = result.content[0].text
            parsed_content = json.loads(content_text)
            assert parsed_content == return_value

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_server_info_resource_compliance(self, mock_server):
        """Test server-info resource compliance."""
        result = await mock_server.server.get_resource("resource://server-info")

        # Parse server info
        server_info = json.loads(result.text)

        # Verify required fields
        required_fields = ["server", "version", "status", "embedding_model", "vector_dimensions"]
        for field in required_fields:
            assert field in server_info, f"Missing required field: {field}"

        # Verify field types
        assert isinstance(server_info["server"], str)
        assert isinstance(server_info["version"], str)
        assert isinstance(server_info["status"], str)
        assert isinstance(server_info["embedding_model"], str)
        assert isinstance(server_info["vector_dimensions"], int)

        # Verify values
        assert server_info["server"] == "document-retrieval-mcp"
        assert server_info["status"] == "healthy"

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_schema_info_resource_compliance(self, mock_server):
        """Test schema-info resource compliance."""
        result = await mock_server.server.get_resource("resource://schema-info")

        # Parse schema info
        schema_info = json.loads(result.text)

        # Verify schema structure
        assert "documents_table" in schema_info
        assert "document_embeddings_table" in schema_info

        # Verify table schemas
        for table_name, table_schema in schema_info.items():
            assert "columns" in table_schema
            assert isinstance(table_schema["columns"], list)
            assert len(table_schema["columns"]) > 0

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_async_compliance(self, mock_server):
        """Test async operation compliance."""
        import asyncio

        # Test concurrent tool calls
        requests = [
            ToolRequest(
                tool="search_documents",
                params={
                    "query": f"test {i}",
                    "user_id": str(uuid.uuid4()),
                    "session_id": f"session-{i}"
                }
            )
            for i in range(5)
        ]

        # Mock tool implementations
        mock_server._search_documents = AsyncMock(return_value={"results": [], "count": 0})

        # Execute concurrent calls
        tasks = [mock_server.server.call_tool(request) for request in requests]
        results = await asyncio.gather(*tasks)

        # Verify all calls completed successfully
        assert len(results) == 5
        for result in results:
            assert isinstance(result, ToolResult)
            assert result.error is None

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_tool_schema_validation_compliance(self, mock_server):
        """Test tool input schema validation compliance."""
        tools = await mock_server.server.list_tools()

        for tool in tools:
            schema = tool.input_schema

            # Verify schema structure compliance
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema

            # Verify all required fields are in properties
            for required_field in schema["required"]:
                assert required_field in schema["properties"]

            # Verify property definitions
            for prop_name, prop_def in schema["properties"].items():
                assert "type" in prop_def
                assert "description" in prop_def

                # Verify specific field constraints
                if prop_name in ["user_id", "document_id", "chunk_id"]:
                    if "format" in prop_def:
                        assert prop_def["format"] == "uuid"

                if prop_name in ["user_id", "session_id", "project_id"]:
                    if "pattern" in prop_def:
                        assert prop_def["pattern"] == "^[a-zA-Z0-9_-]+$"

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_json_serialization_compliance(self, mock_server):
        """Test JSON serialization compliance for all outputs."""
        # Test data with various types
        test_outputs = [
            {"string": "test", "number": 42, "float": 3.14, "boolean": True, "null": None},
            {"array": [1, 2, 3], "object": {"nested": "value"}},
            {"uuid": str(uuid.uuid4()), "datetime": "2024-01-01T00:00:00"},
            {"empty_array": [], "empty_object": {}},
        ]

        for test_output in test_outputs:
            mock_server._search_documents = AsyncMock(return_value=test_output)

            request = ToolRequest(
                tool="search_documents",
                params={
                    "query": "test",
                    "user_id": str(uuid.uuid4()),
                    "session_id": "session-123"
                }
            )

            result = await mock_server.server.call_tool(request)

            # Verify JSON serialization works
            content_text = result.content[0].text
            parsed_content = json.loads(content_text)
            assert parsed_content == test_output

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_error_message_compliance(self, mock_server):
        """Test error message format compliance."""
        # Test different error scenarios
        error_scenarios = [
            (ValueError("Invalid parameter"), "invalid parameter"),
            (RuntimeError("Internal error"), "internal error"),
            (Exception("Generic error"), "generic error")
        ]

        for exception, expected_text in error_scenarios:
            mock_server._search_documents = AsyncMock(side_effect=exception)

            request = ToolRequest(
                tool="search_documents",
                params={
                    "query": "test",
                    "user_id": str(uuid.uuid4()),
                    "session_id": "session-123"
                }
            )

            result = await mock_server.server.call_tool(request)

            # Verify error format
            assert isinstance(result, ToolResult)
            assert result.error is not None
            assert isinstance(result.error, str)
            assert expected_text in result.error.lower()
            assert result.content is None

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_large_response_compliance(self, mock_server):
        """Test handling of large responses compliance."""
        # Create large response data
        large_results = [
            {
                "chunk_id": str(uuid.uuid4()),
                "document_id": str(uuid.uuid4()),
                "chunk_text": "Large chunk text " * 100,  # Large text
                "chunk_index": i,
                "chunk_metadata": {"data": "x" * 1000},  # Large metadata
                "filename": f"document_{i}.pdf",
                "file_type": "pdf",
                "document_metadata": {"title": f"Document {i}"},
                "similarity_score": 0.9 - (i * 0.01)
            }
            for i in range(20)  # Maximum results
        ]

        large_response = {
            "results": large_results,
            "query": "test query",
            "count": len(large_results)
        }

        mock_server._search_documents = AsyncMock(return_value=large_response)

        request = ToolRequest(
            tool="search_documents",
            params={
                "query": "test",
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "top_k": 20
            }
        )

        result = await mock_server.server.call_tool(request)

        # Verify large response is handled correctly
        assert isinstance(result, ToolResult)
        assert result.error is None
        assert len(result.content) == 1

        # Verify JSON serialization works for large data
        content_text = result.content[0].text
        parsed_content = json.loads(content_text)
        assert parsed_content == large_response
        assert len(parsed_content["results"]) == 20


class TestMCPServerLifecycle:
    """Test suite for MCP server lifecycle compliance."""

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_server_initialization_sequence(self, mock_env_vars):
        """Test proper server initialization sequence."""
        server = DocumentRetrievalServer()

        # Verify initial state
        assert server.openai_client is None
        assert server.supabase_client is None
        assert server.db_pool is None
        assert server.metadata_cache is not None

        # Mock external dependencies
        with patch('asyncpg.create_pool') as mock_create_pool, \
             patch('server.AsyncOpenAI') as mock_openai, \
             patch('server.create_client') as mock_supabase:

            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool

            # Initialize server
            await server.initialize()

            # Verify initialization completed
            assert server.openai_client is not None
            assert server.supabase_client is not None
            assert server.db_pool is not None

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_server_cleanup_sequence(self, mock_env_vars):
        """Test proper server cleanup sequence."""
        server = DocumentRetrievalServer()

        # Mock initialized state
        mock_pool = AsyncMock()
        server.db_pool = mock_pool

        # Test cleanup
        await server.cleanup()

        # Verify cleanup was called
        mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_server_error_recovery(self, mock_env_vars):
        """Test server error recovery compliance."""
        server = DocumentRetrievalServer()

        # Mock initialization failure
        with patch('server.AsyncOpenAI', side_effect=Exception("OpenAI init failed")):
            with pytest.raises(Exception, match="OpenAI init failed"):
                await server.initialize()

        # Verify server remains in valid state after error
        assert server.openai_client is None
        assert server.server is not None  # MCP server should still be valid