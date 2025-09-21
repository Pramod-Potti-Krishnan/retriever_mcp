"""
Unit tests for validation functionality in Document Retrieval MCP Server.
"""

import pytest
import uuid
import json
from unittest.mock import patch, MagicMock

import jsonschema
from mcp.types import Tool

from server import DocumentRetrievalServer


class TestSchemaValidation:
    """Test suite for JSON schema validation."""

    @pytest.mark.unit
    async def test_search_documents_schema_validation(self, mock_server):
        """Test search_documents input schema validation."""
        tools = await mock_server.server.list_tools()
        search_tool = next(tool for tool in tools if tool.name == "search_documents")
        schema = search_tool.input_schema

        # Valid input cases
        valid_inputs = [
            {
                "query": "test query",
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "project_id": "project-456",
                "top_k": 5,
                "similarity_threshold": 0.7
            },
            {
                "query": "minimal query",
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123"
                # Optional parameters omitted
            },
            {
                "query": "edge case query",
                "user_id": str(uuid.uuid4()),
                "session_id": "session_123",
                "top_k": 1,  # Minimum value
                "similarity_threshold": 0.0  # Minimum value
            },
            {
                "query": "max values query",
                "user_id": str(uuid.uuid4()),
                "session_id": "session_123",
                "top_k": 20,  # Maximum value
                "similarity_threshold": 1.0  # Maximum value
            }
        ]

        for valid_input in valid_inputs:
            try:
                jsonschema.validate(valid_input, schema)
            except jsonschema.ValidationError as e:
                pytest.fail(f"Valid input failed validation: {valid_input}, Error: {e}")

        # Invalid input cases
        invalid_inputs = [
            {
                # Missing required query
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123"
            },
            {
                # Missing required user_id
                "query": "test",
                "session_id": "session-123"
            },
            {
                # Missing required session_id
                "query": "test",
                "user_id": str(uuid.uuid4())
            },
            {
                "query": "",  # Empty query (minLength: 1)
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123"
            },
            {
                "query": "x" * 1001,  # Query too long (maxLength: 1000)
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123"
            },
            {
                "query": "test",
                "user_id": "invalid-uuid",  # Invalid UUID format
                "session_id": "session-123"
            },
            {
                "query": "test",
                "user_id": str(uuid.uuid4()),
                "session_id": "session with spaces!",  # Invalid pattern
            },
            {
                "query": "test",
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "top_k": 0  # Below minimum
            },
            {
                "query": "test",
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "top_k": 25  # Above maximum
            },
            {
                "query": "test",
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "similarity_threshold": -0.1  # Below minimum
            },
            {
                "query": "test",
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "similarity_threshold": 1.1  # Above maximum
            }
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(invalid_input, schema)

    @pytest.mark.unit
    async def test_get_document_context_schema_validation(self, mock_server):
        """Test get_document_context input schema validation."""
        tools = await mock_server.server.list_tools()
        doc_tool = next(tool for tool in tools if tool.name == "get_document_context")
        schema = doc_tool.input_schema

        # Valid input cases
        valid_inputs = [
            {
                "document_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123"
            },
            {
                "document_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "chunk_ids": [str(uuid.uuid4()), str(uuid.uuid4())]
            },
            {
                "document_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "chunk_ids": []  # Empty array is valid
            }
        ]

        for valid_input in valid_inputs:
            try:
                jsonschema.validate(valid_input, schema)
            except jsonschema.ValidationError as e:
                pytest.fail(f"Valid input failed validation: {valid_input}, Error: {e}")

        # Invalid input cases
        invalid_inputs = [
            {
                # Missing required document_id
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123"
            },
            {
                "document_id": "invalid-uuid",  # Invalid UUID
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123"
            },
            {
                "document_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "chunk_ids": ["invalid-uuid"]  # Invalid UUID in array
            },
            {
                "document_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "chunk_ids": [str(uuid.uuid4())] * 51  # Too many items (max 50)
            }
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(invalid_input, schema)

    @pytest.mark.unit
    async def test_list_user_documents_schema_validation(self, mock_server):
        """Test list_user_documents input schema validation."""
        tools = await mock_server.server.list_tools()
        list_tool = next(tool for tool in tools if tool.name == "list_user_documents")
        schema = list_tool.input_schema

        # Valid input cases
        valid_inputs = [
            {
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123"
            },
            {
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "project_id": "project-456",
                "page": 1,
                "per_page": 20
            },
            {
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "page": 1,  # Minimum page
                "per_page": 1  # Minimum per_page
            },
            {
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "per_page": 100  # Maximum per_page
            }
        ]

        for valid_input in valid_inputs:
            try:
                jsonschema.validate(valid_input, schema)
            except jsonschema.ValidationError as e:
                pytest.fail(f"Valid input failed validation: {valid_input}, Error: {e}")

        # Invalid input cases
        invalid_inputs = [
            {
                # Missing required user_id
                "session_id": "session-123"
            },
            {
                # Missing required session_id
                "user_id": str(uuid.uuid4())
            },
            {
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "page": 0  # Below minimum
            },
            {
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "per_page": 0  # Below minimum
            },
            {
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "per_page": 101  # Above maximum
            },
            {
                "user_id": "invalid-uuid",  # Invalid UUID
                "session_id": "session-123"
            }
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(invalid_input, schema)

    @pytest.mark.unit
    async def test_get_similar_chunks_schema_validation(self, mock_server):
        """Test get_similar_chunks input schema validation."""
        tools = await mock_server.server.list_tools()
        similar_tool = next(tool for tool in tools if tool.name == "get_similar_chunks")
        schema = similar_tool.input_schema

        # Valid input cases
        valid_inputs = [
            {
                "chunk_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123"
            },
            {
                "chunk_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "top_k": 1  # Minimum value
            },
            {
                "chunk_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "top_k": 10  # Maximum value
            }
        ]

        for valid_input in valid_inputs:
            try:
                jsonschema.validate(valid_input, schema)
            except jsonschema.ValidationError as e:
                pytest.fail(f"Valid input failed validation: {valid_input}, Error: {e}")

        # Invalid input cases
        invalid_inputs = [
            {
                # Missing required chunk_id
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123"
            },
            {
                "chunk_id": "invalid-uuid",  # Invalid UUID
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123"
            },
            {
                "chunk_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "top_k": 0  # Below minimum
            },
            {
                "chunk_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "session_id": "session-123",
                "top_k": 11  # Above maximum
            }
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(invalid_input, schema)

    @pytest.mark.unit
    def test_schema_completeness(self, mock_server):
        """Test that all schemas have required structural elements."""
        async def run_test():
            tools = await mock_server.server.list_tools()

            for tool in tools:
                schema = tool.input_schema

                # Verify basic schema structure
                assert "type" in schema
                assert schema["type"] == "object"
                assert "properties" in schema
                assert "required" in schema

                # Verify properties have types
                for prop_name, prop_schema in schema["properties"].items():
                    assert "type" in prop_schema, f"Property {prop_name} missing type"
                    assert "description" in prop_schema, f"Property {prop_name} missing description"

                # Verify required fields are defined in properties
                for required_field in schema["required"]:
                    assert required_field in schema["properties"], f"Required field {required_field} not in properties"

        import asyncio
        asyncio.create_task(run_test())


class TestOutputValidation:
    """Test suite for output validation."""

    @pytest.mark.unit
    def test_search_documents_output_structure(self):
        """Test search_documents output structure validation."""
        # Expected output structure
        expected_structure = {
            "results": list,
            "query": str,
            "count": int
        }

        # Sample output
        sample_output = {
            "results": [
                {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": str(uuid.uuid4()),
                    "chunk_text": "Sample text",
                    "chunk_index": 0,
                    "chunk_metadata": {},
                    "filename": "test.pdf",
                    "file_type": "pdf",
                    "document_metadata": {},
                    "similarity_score": 0.85
                }
            ],
            "query": "test query",
            "count": 1
        }

        # Validate structure
        for key, expected_type in expected_structure.items():
            assert key in sample_output, f"Missing key: {key}"
            assert isinstance(sample_output[key], expected_type), f"Wrong type for {key}"

        # Validate result item structure
        if sample_output["results"]:
            result_item = sample_output["results"][0]
            required_fields = [
                "chunk_id", "document_id", "chunk_text", "chunk_index",
                "chunk_metadata", "filename", "file_type", "document_metadata",
                "similarity_score"
            ]
            for field in required_fields:
                assert field in result_item, f"Missing field in result: {field}"

    @pytest.mark.unit
    def test_get_document_context_output_structure(self):
        """Test get_document_context output structure validation."""
        expected_structure = {
            "document_id": str,
            "document_info": dict,
            "chunks": list,
            "chunk_count": int
        }

        sample_output = {
            "document_id": str(uuid.uuid4()),
            "document_info": {
                "filename": "test.pdf",
                "file_type": "pdf",
                "total_chunks": 3,
                "metadata": {}
            },
            "chunks": [
                {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_text": "Sample chunk",
                    "chunk_index": 0,
                    "chunk_metadata": {}
                }
            ],
            "chunk_count": 1
        }

        # Validate structure
        for key, expected_type in expected_structure.items():
            assert key in sample_output, f"Missing key: {key}"
            assert isinstance(sample_output[key], expected_type), f"Wrong type for {key}"

    @pytest.mark.unit
    def test_list_user_documents_output_structure(self):
        """Test list_user_documents output structure validation."""
        expected_structure = {
            "documents": list,
            "pagination": dict
        }

        sample_output = {
            "documents": [
                {
                    "document_id": str(uuid.uuid4()),
                    "filename": "test.pdf",
                    "file_type": "pdf",
                    "file_size": 1024,
                    "total_chunks": 3,
                    "upload_date": "2024-01-01T00:00:00",
                    "project_id": "test",
                    "metadata": {}
                }
            ],
            "pagination": {
                "page": 1,
                "per_page": 20,
                "total": 1,
                "total_pages": 1
            }
        }

        # Validate structure
        for key, expected_type in expected_structure.items():
            assert key in sample_output, f"Missing key: {key}"
            assert isinstance(sample_output[key], expected_type), f"Wrong type for {key}"

        # Validate pagination structure
        pagination = sample_output["pagination"]
        pagination_fields = ["page", "per_page", "total", "total_pages"]
        for field in pagination_fields:
            assert field in pagination, f"Missing pagination field: {field}"
            assert isinstance(pagination[field], int), f"Wrong type for pagination.{field}"

    @pytest.mark.unit
    def test_get_similar_chunks_output_structure(self):
        """Test get_similar_chunks output structure validation."""
        expected_structure = {
            "reference_chunk_id": str,
            "reference_text": str,
            "reference_document_id": str,
            "similar_chunks": list,
            "count": int
        }

        sample_output = {
            "reference_chunk_id": str(uuid.uuid4()),
            "reference_text": "Reference chunk text",
            "reference_document_id": str(uuid.uuid4()),
            "similar_chunks": [
                {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": str(uuid.uuid4()),
                    "chunk_text": "Similar chunk",
                    "chunk_index": 1,
                    "chunk_metadata": {},
                    "filename": "test.pdf",
                    "file_type": "pdf",
                    "similarity_score": 0.75
                }
            ],
            "count": 1
        }

        # Validate structure
        for key, expected_type in expected_structure.items():
            assert key in sample_output, f"Missing key: {key}"
            assert isinstance(sample_output[key], expected_type), f"Wrong type for {key}"

    @pytest.mark.unit
    def test_error_output_structure(self):
        """Test error output structure validation."""
        # All tools should return consistent error structure
        error_output = {
            "results": [],  # or appropriate empty container
            "error": "Error message"
        }

        assert "error" in error_output
        assert isinstance(error_output["error"], str)
        assert len(error_output["error"]) > 0


class TestDataTypeValidation:
    """Test suite for data type validation."""

    @pytest.mark.unit
    def test_uuid_validation(self):
        """Test UUID format validation."""
        valid_uuids = [
            str(uuid.uuid4()),
            "123e4567-e89b-12d3-a456-426614174000",
            "550e8400-e29b-41d4-a716-446655440000"
        ]

        invalid_uuids = [
            "not-a-uuid",
            "123-456-789",
            "123e4567-e89b-12d3-a456-42661417400",  # Too short
            "123e4567-e89b-12d3-a456-4266141740000",  # Too long
            "ggge4567-e89b-12d3-a456-426614174000",  # Invalid characters
        ]

        for valid_uuid in valid_uuids:
            try:
                uuid.UUID(valid_uuid)
            except ValueError:
                pytest.fail(f"Valid UUID failed validation: {valid_uuid}")

        for invalid_uuid in invalid_uuids:
            with pytest.raises(ValueError):
                uuid.UUID(invalid_uuid)

    @pytest.mark.unit
    def test_pattern_validation(self):
        """Test pattern validation for session_id and user_id."""
        pattern = r"^[a-zA-Z0-9_-]+$"
        import re

        valid_patterns = [
            "session-123",
            "user_456",
            "test-session-abc",
            "session123",
            "USER_ABC_123"
        ]

        invalid_patterns = [
            "session with spaces",
            "session@special",
            "session#123",
            "session/path",
            "session.dot",
            ""  # Empty string
        ]

        for valid_pattern in valid_patterns:
            assert re.match(pattern, valid_pattern), f"Valid pattern failed: {valid_pattern}"

        for invalid_pattern in invalid_patterns:
            assert not re.match(pattern, invalid_pattern), f"Invalid pattern passed: {invalid_pattern}"

    @pytest.mark.unit
    def test_numeric_range_validation(self):
        """Test numeric range validation."""
        # Test top_k ranges
        top_k_valid = [1, 5, 10, 20]  # Within range 1-20
        top_k_invalid = [0, -1, 21, 100]  # Outside range

        for value in top_k_valid:
            assert 1 <= value <= 20, f"Valid top_k failed range check: {value}"

        for value in top_k_invalid:
            assert not (1 <= value <= 20), f"Invalid top_k passed range check: {value}"

        # Test similarity_threshold ranges
        similarity_valid = [0.0, 0.5, 0.7, 1.0]  # Within range 0.0-1.0
        similarity_invalid = [-0.1, 1.1, 2.0, -1.0]  # Outside range

        for value in similarity_valid:
            assert 0.0 <= value <= 1.0, f"Valid similarity failed range check: {value}"

        for value in similarity_invalid:
            assert not (0.0 <= value <= 1.0), f"Invalid similarity passed range check: {value}"

    @pytest.mark.unit
    def test_string_length_validation(self):
        """Test string length validation."""
        # Query length validation (1-1000 characters)
        valid_queries = [
            "a",  # Minimum length
            "test query",
            "x" * 1000  # Maximum length
        ]

        invalid_queries = [
            "",  # Empty string
            "x" * 1001  # Too long
        ]

        for query in valid_queries:
            assert 1 <= len(query) <= 1000, f"Valid query failed length check: {query[:50]}..."

        for query in invalid_queries:
            assert not (1 <= len(query) <= 1000), f"Invalid query passed length check: len={len(query)}"

    @pytest.mark.unit
    def test_array_validation(self):
        """Test array validation for chunk_ids."""
        # Valid chunk_ids arrays
        valid_arrays = [
            [],  # Empty array is valid
            [str(uuid.uuid4())],  # Single item
            [str(uuid.uuid4()) for _ in range(50)]  # Maximum items
        ]

        invalid_arrays = [
            [str(uuid.uuid4()) for _ in range(51)],  # Too many items
            ["invalid-uuid"],  # Invalid UUID in array
            [str(uuid.uuid4()), "invalid-uuid"]  # Mixed valid/invalid
        ]

        for array in valid_arrays:
            assert len(array) <= 50, f"Valid array failed length check: len={len(array)}"
            for item in array:
                try:
                    uuid.UUID(item)
                except ValueError:
                    pytest.fail(f"Valid array contains invalid UUID: {item}")

        for array in invalid_arrays:
            if len(array) > 50:
                assert len(array) > 50, f"Invalid array passed length check: len={len(array)}"
            else:
                # Check for invalid UUIDs
                has_invalid_uuid = False
                for item in array:
                    try:
                        uuid.UUID(item)
                    except ValueError:
                        has_invalid_uuid = True
                        break
                assert has_invalid_uuid, f"Invalid array passed UUID validation: {array}"