#!/usr/bin/env python3
"""
Document Retrieval MCP Server - Optimized for Fly.io Deployment

This MCP server provides document retrieval capabilities from Supabase
vector embeddings, designed for Fly.io deployment and multi-MCP architecture.
"""

import os
import sys
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import asyncpg
import numpy as np
from cachetools import TTLCache
from openai import AsyncOpenAI
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolRequest,
    CallToolResult,
    GetPromptResult,
    Resource,
    TextResourceContents
)

# Configure logging for Fly.io
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Ensure logs go to stderr for Fly.io
)
logger = logging.getLogger("presentation-retrieval-mcp")

# Database configuration - Using environment variables for Fly.io secrets
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is required")
    sys.exit(1)

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is required")
    sys.exit(1)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_DIMENSIONS = int(os.getenv("VECTOR_DIMENSIONS", "1536"))

# Query configuration
MAX_RESULTS_LIMIT = int(os.getenv("MAX_RESULTS_LIMIT", "20"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.7"))

# Connection pool configuration
DB_POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "5"))
DB_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "20"))

# Cache configuration
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))

# MCP Server metadata
MCP_SERVER_NAME = "presentation-retrieval-mcp"
MCP_SERVER_VERSION = "1.0.0"


class PresentationRetrievalMCP:
    """MCP server for document retrieval optimized for Fly.io deployment."""

    def __init__(self):
        self.server = Server(MCP_SERVER_NAME)
        self.openai_client = None
        self.db_pool = None
        self.metadata_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL)
        self._setup_handlers()

    async def initialize(self):
        """Initialize external connections and resources."""
        try:
            # Initialize OpenAI client
            self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

            # Initialize database connection pool with Fly.io optimized settings
            self.db_pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=DB_POOL_MIN_SIZE,
                max_size=DB_POOL_MAX_SIZE,
                timeout=10,
                command_timeout=10,
                server_settings={
                    'application_name': MCP_SERVER_NAME,
                    'jit': 'off'  # Disable JIT for consistent performance
                }
            )

            logger.info(f"{MCP_SERVER_NAME} v{MCP_SERVER_VERSION} initialized successfully")
            logger.info(f"Connected to database with pool size {DB_POOL_MIN_SIZE}-{DB_POOL_MAX_SIZE}")
            logger.info(f"Running in Fly.io region: {os.getenv('FLY_REGION', 'unknown')}")

        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            raise

    async def cleanup(self):
        """Clean up resources before shutdown."""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("Database connection pool closed")

    def _setup_handlers(self):
        """Set up MCP tool and resource handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available MCP tools."""
            return [
                Tool(
                    name="search_documents",
                    description="Search document embeddings using semantic similarity for presentation content",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text for presentation content",
                                "minLength": 1,
                                "maxLength": 1000
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "default": "test-user-123"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session identifier",
                                "default": "test-session-456"
                            },
                            "project_id": {
                                "type": "string",
                                "description": "Project identifier",
                                "default": "presentation-project"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "minimum": 1,
                                "maximum": MAX_RESULTS_LIMIT,
                                "default": 5
                            },
                            "similarity_threshold": {
                                "type": "number",
                                "description": "Minimum similarity score",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "default": DEFAULT_SIMILARITY_THRESHOLD
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_document_context",
                    description="Retrieve specific document sections for presentation slides",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "document_id": {
                                "type": "string",
                                "description": "Document ID (UUID format)"
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "default": "test-user-123"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session identifier",
                                "default": "test-session-456"
                            },
                            "chunk_limit": {
                                "type": "integer",
                                "description": "Maximum number of chunks to retrieve",
                                "minimum": 1,
                                "maximum": 50,
                                "default": 10
                            }
                        },
                        "required": ["document_id"]
                    }
                ),
                Tool(
                    name="list_user_documents",
                    description="List all documents available for a user's presentations",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "default": "test-user-123"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session identifier",
                                "default": "test-session-456"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of documents to list",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 20
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Pagination offset",
                                "minimum": 0,
                                "default": 0
                            }
                        }
                    }
                ),
                Tool(
                    name="get_similar_chunks",
                    description="Find similar content chunks for presentation consistency",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "chunk_id": {
                                "type": "string",
                                "description": "Source chunk ID to find similar content"
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "default": "test-user-123"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of similar chunks to return",
                                "minimum": 1,
                                "maximum": 20,
                                "default": 5
                            },
                            "similarity_threshold": {
                                "type": "number",
                                "description": "Minimum similarity score",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "default": 0.8
                            }
                        },
                        "required": ["chunk_id"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(request: CallToolRequest) -> CallToolResult:
            """Handle tool execution requests."""
            tool_name = request.params.name
            arguments = request.params.arguments or {}

            logger.info(f"Executing tool: {tool_name}")

            try:
                if tool_name == "search_documents":
                    result = await self._search_documents(**arguments)
                elif tool_name == "get_document_context":
                    result = await self._get_document_context(**arguments)
                elif tool_name == "list_user_documents":
                    result = await self._list_user_documents(**arguments)
                elif tool_name == "get_similar_chunks":
                    result = await self._get_similar_chunks(**arguments)
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")

                return CallToolResult(
                    content=[TextContent(text=json.dumps(result, indent=2))]
                )

            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return CallToolResult(
                    content=[TextContent(text=json.dumps({
                        "error": str(e),
                        "tool": tool_name
                    }))]
                )

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for query text using OpenAI."""
        try:
            response = await self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def _search_documents(
        self,
        query: str,
        user_id: str = "test-user-123",
        session_id: str = "test-session-456",
        project_id: str = "presentation-project",
        top_k: int = 5,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> Dict[str, Any]:
        """Search documents using vector similarity."""
        try:
            # Generate embedding for query
            query_embedding = await self._generate_embedding(query)

            async with self.db_pool.acquire() as conn:
                # Perform vector similarity search
                results = await conn.fetch("""
                    SELECT
                        dc.id,
                        dc.document_id,
                        dc.text_content,
                        dc.chunk_index,
                        dc.metadata,
                        d.title as document_title,
                        d.file_name,
                        1 - (dc.embedding <=> $1::vector) as similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE
                        d.user_id = $2
                        AND 1 - (dc.embedding <=> $1::vector) > $3
                    ORDER BY dc.embedding <=> $1::vector
                    LIMIT $4
                """, query_embedding, user_id, similarity_threshold, top_k)

                return {
                    "query": query,
                    "user_id": user_id,
                    "session_id": session_id,
                    "project_id": project_id,
                    "results": [
                        {
                            "chunk_id": str(r['id']),
                            "document_id": str(r['document_id']),
                            "document_title": r['document_title'],
                            "file_name": r['file_name'],
                            "content": r['text_content'],
                            "chunk_index": r['chunk_index'],
                            "similarity": float(r['similarity']),
                            "metadata": r['metadata']
                        }
                        for r in results
                    ],
                    "count": len(results)
                }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def _get_document_context(
        self,
        document_id: str,
        user_id: str = "test-user-123",
        session_id: str = "test-session-456",
        chunk_limit: int = 10
    ) -> Dict[str, Any]:
        """Get document context by retrieving its chunks."""
        try:
            async with self.db_pool.acquire() as conn:
                # Verify document ownership
                doc = await conn.fetchrow("""
                    SELECT id, title, file_name, metadata, created_at
                    FROM documents
                    WHERE id = $1 AND user_id = $2
                """, document_id, user_id)

                if not doc:
                    raise ValueError(f"Document {document_id} not found or access denied")

                # Get document chunks
                chunks = await conn.fetch("""
                    SELECT id, text_content, chunk_index, metadata
                    FROM document_chunks
                    WHERE document_id = $1
                    ORDER BY chunk_index
                    LIMIT $2
                """, document_id, chunk_limit)

                return {
                    "document_id": str(doc['id']),
                    "title": doc['title'],
                    "file_name": doc['file_name'],
                    "metadata": doc['metadata'],
                    "created_at": doc['created_at'].isoformat(),
                    "chunks": [
                        {
                            "chunk_id": str(c['id']),
                            "content": c['text_content'],
                            "chunk_index": c['chunk_index'],
                            "metadata": c['metadata']
                        }
                        for c in chunks
                    ],
                    "chunk_count": len(chunks)
                }

        except Exception as e:
            logger.error(f"Failed to get document context: {e}")
            raise

    async def _list_user_documents(
        self,
        user_id: str = "test-user-123",
        session_id: str = "test-session-456",
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List all documents for a user."""
        try:
            # Check cache first
            cache_key = f"docs:{user_id}:{limit}:{offset}"
            if cache_key in self.metadata_cache:
                logger.debug(f"Cache hit for user documents: {user_id}")
                return self.metadata_cache[cache_key]

            async with self.db_pool.acquire() as conn:
                # Get documents with chunk count
                documents = await conn.fetch("""
                    SELECT
                        d.id,
                        d.title,
                        d.file_name,
                        d.metadata,
                        d.created_at,
                        d.updated_at,
                        COUNT(dc.id) as chunk_count
                    FROM documents d
                    LEFT JOIN document_chunks dc ON d.id = dc.document_id
                    WHERE d.user_id = $1
                    GROUP BY d.id
                    ORDER BY d.updated_at DESC
                    LIMIT $2 OFFSET $3
                """, user_id, limit, offset)

                # Get total count
                total = await conn.fetchval("""
                    SELECT COUNT(*) FROM documents WHERE user_id = $1
                """, user_id)

                result = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "documents": [
                        {
                            "document_id": str(d['id']),
                            "title": d['title'],
                            "file_name": d['file_name'],
                            "chunk_count": d['chunk_count'],
                            "metadata": d['metadata'],
                            "created_at": d['created_at'].isoformat(),
                            "updated_at": d['updated_at'].isoformat()
                        }
                        for d in documents
                    ],
                    "count": len(documents),
                    "total": total,
                    "limit": limit,
                    "offset": offset
                }

                # Cache the result
                self.metadata_cache[cache_key] = result

                return result

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise

    async def _get_similar_chunks(
        self,
        chunk_id: str,
        user_id: str = "test-user-123",
        top_k: int = 5,
        similarity_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """Find similar chunks to a given chunk."""
        try:
            async with self.db_pool.acquire() as conn:
                # Get the source chunk embedding
                source = await conn.fetchrow("""
                    SELECT dc.embedding, dc.text_content, d.user_id
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE dc.id = $1 AND d.user_id = $2
                """, chunk_id, user_id)

                if not source:
                    raise ValueError(f"Chunk {chunk_id} not found or access denied")

                # Find similar chunks
                similar = await conn.fetch("""
                    SELECT
                        dc.id,
                        dc.document_id,
                        dc.text_content,
                        dc.chunk_index,
                        d.title as document_title,
                        1 - (dc.embedding <=> $1::vector) as similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE
                        d.user_id = $2
                        AND dc.id != $3
                        AND 1 - (dc.embedding <=> $1::vector) > $4
                    ORDER BY dc.embedding <=> $1::vector
                    LIMIT $5
                """, source['embedding'], user_id, chunk_id, similarity_threshold, top_k)

                return {
                    "source_chunk_id": chunk_id,
                    "source_content": source['text_content'][:200] + "...",
                    "similar_chunks": [
                        {
                            "chunk_id": str(s['id']),
                            "document_id": str(s['document_id']),
                            "document_title": s['document_title'],
                            "content": s['text_content'],
                            "chunk_index": s['chunk_index'],
                            "similarity": float(s['similarity'])
                        }
                        for s in similar
                    ],
                    "count": len(similar)
                }

        except Exception as e:
            logger.error(f"Failed to find similar chunks: {e}")
            raise

    async def run(self):
        """Run the MCP server."""
        try:
            await self.initialize()

            # Register shutdown handler
            async def shutdown():
                logger.info("Shutting down MCP server...")
                await self.cleanup()

            # Run the stdio server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )

        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main entry point for Fly.io deployment."""
    server = PresentationRetrievalMCP()
    await server.run()


if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    import signal

    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Run the server
    asyncio.run(main())