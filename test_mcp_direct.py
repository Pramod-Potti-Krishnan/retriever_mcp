#!/usr/bin/env python3
"""
Direct MCP Protocol Tester

Tests the MCP server using the actual MCP protocol communication.
"""

import os
import sys
import json
import asyncio
import subprocess
from typing import Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MCPTester:
    """Direct MCP protocol tester."""

    def __init__(self):
        self.server_process = None
        self.reader = None
        self.writer = None
        self.request_id = 0

    async def start_server(self):
        """Start the MCP server as a subprocess."""
        env = os.environ.copy()

        # Start server process
        self.server_process = await asyncio.create_subprocess_exec(
            sys.executable,
            "src/server.py",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        print("✅ MCP Server started")
        return True

    async def send_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        """Send a JSON-RPC request to the MCP server."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self.request_id
        }

        if params:
            request["params"] = params

        # Send request
        request_str = json.dumps(request) + "\n"
        self.server_process.stdin.write(request_str.encode())
        await self.server_process.stdin.drain()

        # Read response
        response_line = await self.server_process.stdout.readline()
        response = json.loads(response_line.decode())

        return response

    async def test_list_tools(self):
        """Test listing available tools."""
        print("\n📋 Testing: List Tools")
        print("-" * 40)

        response = await self.send_request("list_tools")

        if "result" in response:
            tools = response["result"]
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description'][:50]}...")
            return True
        else:
            print(f"❌ Error: {response.get('error', 'Unknown error')}")
            return False

    async def test_list_resources(self):
        """Test listing available resources."""
        print("\n📚 Testing: List Resources")
        print("-" * 40)

        response = await self.send_request("list_resources")

        if "result" in response:
            resources = response["result"]
            print(f"✅ Found {len(resources)} resources:")
            for resource in resources:
                print(f"   - {resource['uri']}: {resource['name']}")
            return True
        else:
            print(f"❌ Error: {response.get('error', 'Unknown error')}")
            return False

    async def test_search_documents(self):
        """Test the search_documents tool."""
        print("\n🔍 Testing: Search Documents")
        print("-" * 40)

        params = {
            "tool": "search_documents",
            "params": {
                "query": "machine learning",
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "session_id": "session-123",
                "project_id": "-",
                "top_k": 5,
                "similarity_threshold": 0.7
            }
        }

        response = await self.send_request("call_tool", params)

        if "result" in response:
            result = response["result"]
            if result.get("content"):
                content = json.loads(result["content"][0]["text"])
                print(f"✅ Search completed:")
                print(f"   - Query: {content.get('query', 'N/A')}")
                print(f"   - Results found: {content.get('count', 0)}")

                if content.get("results"):
                    for idx, doc in enumerate(content["results"][:3], 1):
                        print(f"   {idx}. Similarity: {doc.get('similarity_score', 'N/A'):.3f}")
            return True
        else:
            error = response.get("error", {})
            print(f"❌ Error: {error.get('message', 'Unknown error')}")
            return False

    async def test_list_user_documents(self):
        """Test the list_user_documents tool."""
        print("\n📋 Testing: List User Documents")
        print("-" * 40)

        params = {
            "tool": "list_user_documents",
            "params": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "session_id": "session-123",
                "page": 1,
                "per_page": 10
            }
        }

        response = await self.send_request("call_tool", params)

        if "result" in response:
            result = response["result"]
            if result.get("content"):
                content = json.loads(result["content"][0]["text"])
                docs = content.get("documents", [])
                print(f"✅ Found {len(docs)} documents")

                for doc in docs[:5]:
                    print(f"   - {doc.get('filename', 'N/A')} ({doc.get('file_type', 'N/A')})")
            return True
        else:
            error = response.get("error", {})
            print(f"❌ Error: {error.get('message', 'Unknown error')}")
            return False

    async def test_get_server_info(self):
        """Test getting server info resource."""
        print("\n🖥️ Testing: Server Info Resource")
        print("-" * 40)

        params = {
            "uri": "resource://server-info"
        }

        response = await self.send_request("get_resource", params)

        if "result" in response:
            result = response["result"]
            if result.get("text"):
                info = json.loads(result["text"])
                print("✅ Server Information:")
                print(f"   - Server: {info.get('server', 'N/A')}")
                print(f"   - Version: {info.get('version', 'N/A')}")
                print(f"   - Status: {info.get('status', 'N/A')}")
                print(f"   - Model: {info.get('embedding_model', 'N/A')}")
                print(f"   - Cache Size: {info.get('cache_size', 0)}")
            return True
        else:
            print(f"❌ Error: {response.get('error', 'Unknown error')}")
            return False

    async def stop_server(self):
        """Stop the MCP server."""
        if self.server_process:
            self.server_process.terminate()
            await self.server_process.wait()
            print("\n🛑 MCP Server stopped")

    async def run_all_tests(self):
        """Run all tests."""
        print("=" * 50)
        print("🚀 MCP Server Direct Protocol Testing")
        print("=" * 50)

        try:
            # Start server
            await self.start_server()
            await asyncio.sleep(2)  # Give server time to initialize

            # Run tests
            results = []
            results.append(await self.test_list_tools())
            results.append(await self.test_list_resources())
            results.append(await self.test_search_documents())
            results.append(await self.test_list_user_documents())
            results.append(await self.test_get_server_info())

            # Summary
            print("\n" + "=" * 50)
            print("📊 Test Summary")
            print("=" * 50)
            passed = sum(results)
            total = len(results)
            print(f"✅ Passed: {passed}/{total}")
            print(f"❌ Failed: {total-passed}/{total}")

            if passed == total:
                print("\n🎉 All tests passed!")
            else:
                print(f"\n⚠️ {total-passed} test(s) failed")

        except Exception as e:
            print(f"\n❌ Test failed with error: {str(e)}")
        finally:
            await self.stop_server()


async def main():
    """Main entry point."""
    # Check environment variables
    required_vars = ["SUPABASE_URL", "SUPABASE_API_KEY", "OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"❌ Missing environment variables: {missing}")
        print("Please set them in your .env file")
        return

    # Run tests
    tester = MCPTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())