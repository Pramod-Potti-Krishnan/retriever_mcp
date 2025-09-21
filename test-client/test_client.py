#!/usr/bin/env python3
"""
Test client for Document Retrieval MCP Server.

This client can be used to test the MCP server functionality manually
or as part of integration testing.
"""

import asyncio
import json
import uuid
import sys
import os
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mcp.types import Tool, ToolRequest
from mcp.client.stdio import stdio_transport
from mcp.client import Client


class DocumentRetrievalTestClient:
    """Test client for Document Retrieval MCP Server."""

    def __init__(self):
        self.client = None
        self.transport = None

    async def connect(self, server_command: List[str]):
        """Connect to the MCP server."""
        try:
            self.transport = stdio_transport(server_command)
            self.client = Client(self.transport)
            await self.client.start()
            print("✅ Connected to Document Retrieval MCP Server")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.client:
            await self.client.close()
        if self.transport:
            await self.transport.close()
        print("🔌 Disconnected from server")

    async def list_tools(self) -> List[Tool]:
        """List available tools."""
        try:
            tools = await self.client.list_tools()
            print(f"📋 Available tools ({len(tools)}):")
            for tool in tools:
                print(f"  • {tool.name}: {tool.description}")
            return tools
        except Exception as e:
            print(f"❌ Failed to list tools: {e}")
            return []

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a tool with parameters."""
        try:
            request = ToolRequest(tool=tool_name, params=params)
            result = await self.client.call_tool(request)

            if result.error:
                print(f"❌ Tool error: {result.error}")
                return None

            if result.content and len(result.content) > 0:
                content = result.content[0].text
                parsed_result = json.loads(content)
                print(f"✅ Tool '{tool_name}' completed successfully")
                return parsed_result
            else:
                print(f"⚠️ Tool '{tool_name}' returned no content")
                return {}

        except Exception as e:
            print(f"❌ Failed to call tool '{tool_name}': {e}")
            return None

    async def test_search_documents(self, query: str = "test query") -> bool:
        """Test search_documents tool."""
        print(f"\n🔍 Testing search_documents with query: '{query}'")

        params = {
            "query": query,
            "user_id": str(uuid.uuid4()),
            "session_id": "test-session-123",
            "project_id": "test-project",
            "top_k": 5,
            "similarity_threshold": 0.7
        }

        result = await self.call_tool("search_documents", params)
        if result is None:
            return False

        # Validate response structure
        required_fields = ["results", "query", "count"]
        for field in required_fields:
            if field not in result:
                print(f"❌ Missing field '{field}' in response")
                return False

        print(f"   📊 Found {result['count']} results")
        if result["results"]:
            first_result = result["results"][0]
            print(f"   📄 First result: {first_result.get('filename', 'N/A')}")

        return True

    async def test_get_document_context(self) -> bool:
        """Test get_document_context tool."""
        print(f"\n📄 Testing get_document_context")

        params = {
            "document_id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "session_id": "test-session-123"
        }

        result = await self.call_tool("get_document_context", params)
        if result is None:
            return False

        # Validate response structure
        required_fields = ["document_id", "chunks", "chunk_count"]
        for field in required_fields:
            if field not in result:
                print(f"❌ Missing field '{field}' in response")
                return False

        print(f"   📊 Found {result['chunk_count']} chunks")
        if "document_info" in result:
            doc_info = result["document_info"]
            print(f"   📋 Document: {doc_info.get('filename', 'N/A')}")

        return True

    async def test_list_user_documents(self) -> bool:
        """Test list_user_documents tool."""
        print(f"\n📚 Testing list_user_documents")

        params = {
            "user_id": str(uuid.uuid4()),
            "session_id": "test-session-123",
            "project_id": "test-project",
            "page": 1,
            "per_page": 10
        }

        result = await self.call_tool("list_user_documents", params)
        if result is None:
            return False

        # Validate response structure
        required_fields = ["documents", "pagination"]
        for field in required_fields:
            if field not in result:
                print(f"❌ Missing field '{field}' in response")
                return False

        pagination = result["pagination"]
        print(f"   📊 Page {pagination['page']}/{pagination['total_pages']}")
        print(f"   📋 {len(result['documents'])} documents on this page")

        return True

    async def test_get_similar_chunks(self) -> bool:
        """Test get_similar_chunks tool."""
        print(f"\n🔗 Testing get_similar_chunks")

        params = {
            "chunk_id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "session_id": "test-session-123",
            "top_k": 3
        }

        result = await self.call_tool("get_similar_chunks", params)
        if result is None:
            return False

        # Validate response structure
        required_fields = ["reference_chunk_id", "similar_chunks", "count"]
        for field in required_fields:
            if field not in result:
                print(f"❌ Missing field '{field}' in response")
                return False

        print(f"   📊 Found {result['count']} similar chunks")
        if "reference_text" in result:
            ref_text = result["reference_text"][:50] + "..." if len(result["reference_text"]) > 50 else result["reference_text"]
            print(f"   📝 Reference: {ref_text}")

        return True

    async def test_invalid_parameters(self) -> bool:
        """Test error handling with invalid parameters."""
        print(f"\n⚠️ Testing error handling with invalid parameters")

        # Test with missing required parameters
        invalid_params = {
            "query": "test"
            # Missing user_id and session_id
        }

        result = await self.call_tool("search_documents", invalid_params)

        # Should fail gracefully
        if result is not None:
            print("❌ Expected tool to fail with invalid parameters")
            return False

        print("✅ Tool properly rejected invalid parameters")
        return True

    async def test_large_query(self) -> bool:
        """Test with large query string."""
        print(f"\n📏 Testing with large query")

        large_query = "large query text " * 100  # ~1800 characters

        params = {
            "query": large_query,
            "user_id": str(uuid.uuid4()),
            "session_id": "test-session-123",
            "top_k": 1
        }

        result = await self.call_tool("search_documents", params)
        if result is None:
            return False

        print(f"   ✅ Handled large query ({len(large_query)} characters)")
        return True

    async def test_boundary_values(self) -> bool:
        """Test boundary values for parameters."""
        print(f"\n📐 Testing boundary values")

        # Test minimum values
        params_min = {
            "query": "a",  # Minimum length
            "user_id": str(uuid.uuid4()),
            "session_id": "test-session-123",
            "top_k": 1,  # Minimum
            "similarity_threshold": 0.0  # Minimum
        }

        result = await self.call_tool("search_documents", params_min)
        if result is None:
            print("❌ Failed with minimum boundary values")
            return False

        # Test maximum values
        params_max = {
            "query": "x" * 1000,  # Maximum length
            "user_id": str(uuid.uuid4()),
            "session_id": "test-session-123",
            "top_k": 20,  # Maximum
            "similarity_threshold": 1.0  # Maximum
        }

        result = await self.call_tool("search_documents", params_max)
        if result is None:
            print("❌ Failed with maximum boundary values")
            return False

        print("✅ All boundary values handled correctly")
        return True

    async def test_performance_baseline(self) -> bool:
        """Test basic performance characteristics."""
        print(f"\n⏱️ Testing performance baseline")

        params = {
            "query": "performance test query",
            "user_id": str(uuid.uuid4()),
            "session_id": "test-session-123"
        }

        import time
        start_time = time.time()

        result = await self.call_tool("search_documents", params)

        response_time = time.time() - start_time

        if result is None:
            return False

        print(f"   ⏱️ Response time: {response_time:.3f} seconds")

        # Basic performance check (should complete within 5 seconds)
        if response_time > 5.0:
            print(f"⚠️ Slow response time: {response_time:.3f}s")
        else:
            print(f"✅ Good response time: {response_time:.3f}s")

        return True

    async def run_comprehensive_tests(self) -> Dict[str, bool]:
        """Run all test scenarios."""
        print("🧪 Running comprehensive MCP server tests...\n")

        test_results = {}

        # Test individual tools
        test_results["list_tools"] = len(await self.list_tools()) == 4
        test_results["search_documents"] = await self.test_search_documents()
        test_results["get_document_context"] = await self.test_get_document_context()
        test_results["list_user_documents"] = await self.test_list_user_documents()
        test_results["get_similar_chunks"] = await self.test_get_similar_chunks()

        # Test error handling
        test_results["invalid_parameters"] = await self.test_invalid_parameters()

        # Test edge cases
        test_results["large_query"] = await self.test_large_query()
        test_results["boundary_values"] = await self.test_boundary_values()

        # Test performance
        test_results["performance_baseline"] = await self.test_performance_baseline()

        return test_results

    def print_test_summary(self, test_results: Dict[str, bool]):
        """Print test summary."""
        print("\n" + "="*50)
        print("📊 TEST SUMMARY")
        print("="*50)

        passed = sum(1 for result in test_results.values() if result)
        total = len(test_results)

        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")

        print("-"*50)
        print(f"📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        if passed == total:
            print("🎉 All tests passed!")
        else:
            print(f"⚠️ {total - passed} test(s) failed")

        return passed == total


async def main():
    """Main test runner."""
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <server_command>")
        print("Example: python test_client.py python ../src/server.py")
        sys.exit(1)

    server_command = sys.argv[1:]
    client = DocumentRetrievalTestClient()

    try:
        # Connect to server
        if not await client.connect(server_command):
            sys.exit(1)

        # Run tests
        test_results = await client.run_comprehensive_tests()

        # Print summary
        all_passed = client.print_test_summary(test_results)

        # Disconnect
        await client.disconnect()

        # Exit with appropriate code
        sys.exit(0 if all_passed else 1)

    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        await client.disconnect()
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test runner error: {e}")
        await client.disconnect()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())