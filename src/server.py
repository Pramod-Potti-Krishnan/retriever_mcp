#!/usr/bin/env python3
"""
Document Retrieval MCP Server

Queries existing document embeddings stored in Supabase vector database.
Enables AI agents to search and retrieve relevant document content using
semantic similarity search without generating new embeddings.
"""

import os
import sys
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

import asyncpg
import numpy as np
from cachetools import TTLCache
from openai import AsyncOpenAI
from supabase import create_client, Client
from mcp.server import Server
from mcp.server.stdio import stdio_transport
from mcp.types import (
    Tool,
    TextContent,
    ToolRequest,
    ToolResult,
    GetPromptResult,
    Resource,
    ResourceContent,
    TextResourceContent
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("document-retrieval-mcp")

# Environment configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_DIMENSIONS = int(os.getenv("VECTOR_DIMENSIONS", "1536"))
MAX_RESULTS_LIMIT = int(os.getenv("MAX_RESULTS_LIMIT", "20"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.7"))

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "5"))
DB_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "20"))

# Cache configuration
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))


class DocumentRetrievalServer:
    """Main MCP server for document retrieval operations."""

    def __init__(self):
        self.server = Server("document-retrieval-mcp")
        self.openai_client = None
        self.supabase_client = None
        self.db_pool = None
        self.metadata_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL)
        self._setup_handlers()

    async def initialize(self):
        """Initialize external connections and resources."""
        try:
            # Initialize OpenAI client
            self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

            # Initialize Supabase client
            self.supabase_client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

            # Initialize database connection pool
            if DATABASE_URL:
                self.db_pool = await asyncpg.create_pool(
                    DATABASE_URL,
                    min_size=DB_POOL_MIN_SIZE,
                    max_size=DB_POOL_MAX_SIZE,
                    timeout=10,
                    command_timeout=10
                )
            else:
                # Use Supabase connection string
                db_config = self._parse_supabase_url()
                self.db_pool = await asyncpg.create_pool(
                    **db_config,
                    min_size=DB_POOL_MIN_SIZE,
                    max_size=DB_POOL_MAX_SIZE,
                    timeout=10,
                    command_timeout=10
                )

            logger.info("MCP server initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            raise

    def _parse_supabase_url(self) -> Dict[str, Any]:
        """Parse Supabase URL to extract database configuration."""
        # Extract host from Supabase URL
        import re
        match = re.match(r'https?://([^\.]+)\.', SUPABASE_URL)
        if match:
            project_ref = match.group(1)
            return {
                "host": f"db.{project_ref}.supabase.co",
                "port": 5432,
                "database": "postgres",
                "user": "postgres",
                "password": SUPABASE_API_KEY
            }
        raise ValueError("Invalid Supabase URL format")

    def _setup_handlers(self):
        """Set up MCP tool and resource handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available MCP tools."""
            return [
                Tool(
                    name="search_documents",
                    description="Search existing document embeddings using semantic similarity",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text",
                                "minLength": 1,
                                "maxLength": 1000
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "pattern": "^[a-zA-Z0-9_-]+$"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session/chat tab identifier",
                                "pattern": "^[a-zA-Z0-9_-]+$"
                            },
                            "project_id": {
                                "type": "string",
                                "description": "Project identifier (defaults to '-')",
                                "pattern": "^[a-zA-Z0-9_-]+$",
                                "default": "-"
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
                        "required": ["query", "user_id", "session_id"]
                    }
                ),
                Tool(
                    name="get_document_context",
                    description="Retrieve specific document sections from stored embeddings",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "document_id": {
                                "type": "string",
                                "description": "Unique document identifier",
                                "format": "uuid"
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "pattern": "^[a-zA-Z0-9_-]+$"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session identifier",
                                "pattern": "^[a-zA-Z0-9_-]+$"
                            },
                            "chunk_ids": {
                                "type": "array",
                                "description": "Specific chunk IDs to retrieve",
                                "items": {
                                    "type": "string",
                                    "format": "uuid"
                                },
                                "maxItems": 50
                            }
                        },
                        "required": ["document_id", "user_id", "session_id"]
                    }
                ),
                Tool(
                    name="list_user_documents",
                    description="List all documents with stored embeddings accessible to user",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "pattern": "^[a-zA-Z0-9_-]+$"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session identifier",
                                "pattern": "^[a-zA-Z0-9_-]+$"
                            },
                            "project_id": {
                                "type": "string",
                                "description": "Project filter",
                                "pattern": "^[a-zA-Z0-9_-]+$"
                            },
                            "page": {
                                "type": "integer",
                                "description": "Page number for pagination",
                                "minimum": 1,
                                "default": 1
                            },
                            "per_page": {
                                "type": "integer",
                                "description": "Items per page",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 20
                            }
                        },
                        "required": ["user_id", "session_id"]
                    }
                ),
                Tool(
                    name="get_similar_chunks",
                    description="Find similar document chunks to a given chunk using existing embeddings",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "chunk_id": {
                                "type": "string",
                                "description": "Reference chunk ID",
                                "format": "uuid"
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "pattern": "^[a-zA-Z0-9_-]+$"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session identifier",
                                "pattern": "^[a-zA-Z0-9_-]+$"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of similar chunks to return",
                                "minimum": 1,
                                "maximum": 10,
                                "default": 3
                            }
                        },
                        "required": ["chunk_id", "user_id", "session_id"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(request: ToolRequest) -> ToolResult:
            """Handle tool invocation requests."""
            try:
                tool_name = request.tool
                params = request.params or {}

                if tool_name == "search_documents":
                    result = await self._search_documents(**params)
                elif tool_name == "get_document_context":
                    result = await self._get_document_context(**params)
                elif tool_name == "list_user_documents":
                    result = await self._list_user_documents(**params)
                elif tool_name == "get_similar_chunks":
                    result = await self._get_similar_chunks(**params)
                else:
                    return ToolResult(
                        tool=tool_name,
                        error=f"Unknown tool: {tool_name}"
                    )

                return ToolResult(
                    tool=tool_name,
                    content=[TextContent(text=json.dumps(result, indent=2))]
                )
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                return ToolResult(
                    tool=request.tool,
                    error=str(e)
                )

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources."""
            return [
                Resource(
                    uri="resource://server-info",
                    name="Server Information",
                    description="Current server status and configuration"
                ),
                Resource(
                    uri="resource://schema-info",
                    name="Database Schema",
                    description="Database schema information for document embeddings"
                )
            ]

        @self.server.get_resource()
        async def get_resource(uri: str) -> ResourceContent:
            """Get resource content."""
            if uri == "resource://server-info":
                info = {
                    "server": "document-retrieval-mcp",
                    "version": "1.0.0",
                    "status": "healthy",
                    "embedding_model": EMBEDDING_MODEL,
                    "vector_dimensions": VECTOR_DIMENSIONS,
                    "cache_size": len(self.metadata_cache),
                    "db_pool_size": self.db_pool.get_size() if self.db_pool else 0
                }
                return TextResourceContent(
                    uri=uri,
                    text=json.dumps(info, indent=2)
                )
            elif uri == "resource://schema-info":
                schema = {
                    "documents_table": {
                        "columns": [
                            "id (uuid)",
                            "user_id (uuid)",
                            "session_id (text)",
                            "project_id (text)",
                            "filename (text)",
                            "file_type (text)",
                            "file_size (integer)",
                            "total_chunks (integer)",
                            "upload_date (timestamp)",
                            "processing_status (text)",
                            "metadata (jsonb)"
                        ]
                    },
                    "document_embeddings_table": {
                        "columns": [
                            "id (uuid)",
                            "document_id (uuid)",
                            "user_id (uuid)",
                            "session_id (text)",
                            "project_id (text)",
                            "chunk_text (text)",
                            "embedding (vector(1536))",
                            "chunk_index (integer)",
                            "chunk_metadata (jsonb)",
                            "created_at (timestamp)"
                        ]
                    }
                }
                return TextResourceContent(
                    uri=uri,
                    text=json.dumps(schema, indent=2)
                )
            else:
                raise ValueError(f"Unknown resource: {uri}")

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for query text using OpenAI."""
        try:
            response = await self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def _search_documents(
        self,
        query: str,
        user_id: str,
        session_id: str,
        project_id: str = "-",
        top_k: int = 5,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> Dict[str, Any]:
        """Search documents using semantic similarity."""
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)

            # Perform vector similarity search - using document_chunks table
            async with self.db_pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT
                        dc.id,
                        dc.document_id,
                        dc.text_content as chunk_text,
                        dc.chunk_index,
                        dc.metadata as chunk_metadata,
                        d.filename,
                        d.file_type,
                        d.metadata as document_metadata,
                        1 - (dc.embedding <=> $1::vector) as similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE dc.user_id = $2::uuid
                        AND dc.session_id = $3
                        AND dc.project_id = $4
                        AND 1 - (dc.embedding <=> $1::vector) > $5
                    ORDER BY dc.embedding <=> $1::vector
                    LIMIT $6
                """, query_embedding, uuid.UUID(user_id), session_id, project_id,
                similarity_threshold, top_k)

                # Format results
                formatted_results = []
                for row in results:
                    formatted_results.append({
                        "chunk_id": str(row['id']),
                        "document_id": str(row['document_id']),
                        "chunk_text": row['chunk_text'],
                        "chunk_index": row['chunk_index'],
                        "chunk_metadata": dict(row['chunk_metadata']) if row['chunk_metadata'] else {},
                        "filename": row['filename'],
                        "file_type": row['file_type'],
                        "document_metadata": dict(row['document_metadata']) if row['document_metadata'] else {},
                        "similarity_score": float(row['similarity'])
                    })

                return {
                    "results": formatted_results,
                    "query": query,
                    "count": len(formatted_results)
                }
        except Exception as e:
            logger.error(f"Search documents failed: {e}")
            return {"results": [], "error": str(e)}

    async def _get_document_context(
        self,
        document_id: str,
        user_id: str,
        session_id: str,
        chunk_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Retrieve document context from stored embeddings."""
        try:
            async with self.db_pool.acquire() as conn:
                # Build query based on whether chunk_ids are specified
                if chunk_ids:
                    chunk_uuids = [uuid.UUID(cid) for cid in chunk_ids]
                    results = await conn.fetch("""
                        SELECT
                            de.id,
                            de.chunk_text,
                            de.chunk_index,
                            de.chunk_metadata,
                            d.filename,
                            d.file_type,
                            d.metadata as document_metadata,
                            d.total_chunks
                        FROM document_embeddings de
                        JOIN documents d ON de.document_id = d.id
                        WHERE de.document_id = $1::uuid
                            AND de.user_id = $2::uuid
                            AND de.session_id = $3
                            AND de.id = ANY($4::uuid[])
                        ORDER BY de.chunk_index
                    """, uuid.UUID(document_id), uuid.UUID(user_id), session_id, chunk_uuids)
                else:
                    # Get all chunks for the document
                    results = await conn.fetch("""
                        SELECT
                            de.id,
                            de.chunk_text,
                            de.chunk_index,
                            de.chunk_metadata,
                            d.filename,
                            d.file_type,
                            d.metadata as document_metadata,
                            d.total_chunks
                        FROM document_embeddings de
                        JOIN documents d ON de.document_id = d.id
                        WHERE de.document_id = $1::uuid
                            AND de.user_id = $2::uuid
                            AND de.session_id = $3
                        ORDER BY de.chunk_index
                    """, uuid.UUID(document_id), uuid.UUID(user_id), session_id)

                if not results:
                    return {
                        "document_id": document_id,
                        "chunks": [],
                        "error": "Document not found or access denied"
                    }

                # Format results
                chunks = []
                doc_info = None
                for row in results:
                    if not doc_info:
                        doc_info = {
                            "filename": row['filename'],
                            "file_type": row['file_type'],
                            "total_chunks": row['total_chunks'],
                            "metadata": dict(row['document_metadata']) if row['document_metadata'] else {}
                        }
                    chunks.append({
                        "chunk_id": str(row['id']),
                        "chunk_text": row['chunk_text'],
                        "chunk_index": row['chunk_index'],
                        "chunk_metadata": dict(row['chunk_metadata']) if row['chunk_metadata'] else {}
                    })

                return {
                    "document_id": document_id,
                    "document_info": doc_info,
                    "chunks": chunks,
                    "chunk_count": len(chunks)
                }
        except Exception as e:
            logger.error(f"Get document context failed: {e}")
            return {"document_id": document_id, "chunks": [], "error": str(e)}

    async def _list_user_documents(
        self,
        user_id: str,
        session_id: str,
        project_id: Optional[str] = None,
        page: int = 1,
        per_page: int = 20
    ) -> Dict[str, Any]:
        """List documents accessible to the user."""
        try:
            # Check cache first
            cache_key = f"docs_{user_id}_{session_id}_{project_id}_{page}_{per_page}"
            if cache_key in self.metadata_cache:
                return self.metadata_cache[cache_key]

            async with self.db_pool.acquire() as conn:
                # Build query conditions
                conditions = ["user_id = $1::uuid", "session_id = $2"]
                params = [uuid.UUID(user_id), session_id]

                if project_id:
                    conditions.append("project_id = $3")
                    params.append(project_id)

                # Count total documents
                count_query = f"""
                    SELECT COUNT(*) as total
                    FROM documents
                    WHERE {' AND '.join(conditions)}
                        AND processing_status = 'completed'
                """
                total_count = await conn.fetchval(count_query, *params)

                # Get paginated results
                offset = (page - 1) * per_page
                params.extend([per_page, offset])

                list_query = f"""
                    SELECT
                        id,
                        filename,
                        file_type,
                        file_size,
                        total_chunks,
                        upload_date,
                        project_id,
                        metadata
                    FROM documents
                    WHERE {' AND '.join(conditions)}
                        AND processing_status = 'completed'
                    ORDER BY upload_date DESC
                    LIMIT ${len(params) - 1}
                    OFFSET ${len(params)}
                """

                results = await conn.fetch(list_query, *params)

                # Format results
                documents = []
                for row in results:
                    documents.append({
                        "document_id": str(row['id']),
                        "filename": row['filename'],
                        "file_type": row['file_type'],
                        "file_size": row['file_size'],
                        "total_chunks": row['total_chunks'],
                        "upload_date": row['upload_date'].isoformat() if row['upload_date'] else None,
                        "project_id": row['project_id'],
                        "metadata": dict(row['metadata']) if row['metadata'] else {}
                    })

                result = {
                    "documents": documents,
                    "pagination": {
                        "page": page,
                        "per_page": per_page,
                        "total": total_count,
                        "total_pages": (total_count + per_page - 1) // per_page
                    }
                }

                # Cache the result
                self.metadata_cache[cache_key] = result
                return result

        except Exception as e:
            logger.error(f"List user documents failed: {e}")
            return {"documents": [], "error": str(e)}

    async def _get_similar_chunks(
        self,
        chunk_id: str,
        user_id: str,
        session_id: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """Find similar chunks using existing embeddings."""
        try:
            async with self.db_pool.acquire() as conn:
                # First get the reference chunk's embedding
                ref_chunk = await conn.fetchrow("""
                    SELECT
                        embedding,
                        chunk_text,
                        document_id,
                        chunk_index
                    FROM document_embeddings
                    WHERE id = $1::uuid
                        AND user_id = $2::uuid
                        AND session_id = $3
                """, uuid.UUID(chunk_id), uuid.UUID(user_id), session_id)

                if not ref_chunk:
                    return {
                        "reference_chunk_id": chunk_id,
                        "similar_chunks": [],
                        "error": "Reference chunk not found or access denied"
                    }

                # Find similar chunks
                results = await conn.fetch("""
                    SELECT
                        de.id,
                        de.document_id,
                        de.chunk_text,
                        de.chunk_index,
                        de.chunk_metadata,
                        d.filename,
                        d.file_type,
                        1 - (de.embedding <=> $1::vector) as similarity
                    FROM document_embeddings de
                    JOIN documents d ON de.document_id = d.id
                    WHERE de.user_id = $2::uuid
                        AND de.session_id = $3
                        AND de.id != $4::uuid
                    ORDER BY de.embedding <=> $1::vector
                    LIMIT $5
                """, ref_chunk['embedding'], uuid.UUID(user_id), session_id,
                uuid.UUID(chunk_id), top_k)

                # Format results
                similar_chunks = []
                for row in results:
                    similar_chunks.append({
                        "chunk_id": str(row['id']),
                        "document_id": str(row['document_id']),
                        "chunk_text": row['chunk_text'],
                        "chunk_index": row['chunk_index'],
                        "chunk_metadata": dict(row['chunk_metadata']) if row['chunk_metadata'] else {},
                        "filename": row['filename'],
                        "file_type": row['file_type'],
                        "similarity_score": float(row['similarity'])
                    })

                return {
                    "reference_chunk_id": chunk_id,
                    "reference_text": ref_chunk['chunk_text'],
                    "reference_document_id": str(ref_chunk['document_id']),
                    "similar_chunks": similar_chunks,
                    "count": len(similar_chunks)
                }

        except Exception as e:
            logger.error(f"Get similar chunks failed: {e}")
            return {"reference_chunk_id": chunk_id, "similar_chunks": [], "error": str(e)}

    async def cleanup(self):
        """Clean up resources on shutdown."""
        try:
            if self.db_pool:
                await self.db_pool.close()
            logger.info("Server cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    async def run(self):
        """Run the MCP server."""
        try:
            await self.initialize()
            async with stdio_transport(self.server) as transport:
                await transport.start()
                logger.info("Document Retrieval MCP Server started")
                # Keep server running
                await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            await self.cleanup()


def main():
    """Main entry point."""
    # Validate required environment variables
    required_vars = ["SUPABASE_URL", "SUPABASE_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        sys.exit(1)

    # Run the server
    server = DocumentRetrievalServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()