# Document Retrieval MCP Server

A Model Context Protocol (MCP) server that enables AI agents to search and retrieve relevant document content from existing embeddings stored in Supabase vector database. This server performs retrieval-only operations without generating new embeddings.

## Overview

The Document Retrieval MCP Server queries pre-generated document embeddings stored in Supabase PostgreSQL with pgvector extension. It provides semantic search capabilities for AI agents to find relevant document chunks based on similarity to query text.

## Features

- 🔍 **Semantic Search**: Query documents using natural language with OpenAI embeddings
- 📄 **Document Context Retrieval**: Get full document content or specific chunks
- 📋 **Document Listing**: Browse available documents with pagination
- 🔗 **Similarity Matching**: Find related chunks based on existing embeddings
- 🚀 **High Performance**: Connection pooling, TTL caching, and optimized vector queries
- 🔒 **Multi-tenant Security**: User/session/project-based access control
- 🎯 **MCP Protocol Compliant**: Full compatibility with Claude Desktop and other MCP clients

## Prerequisites

- Python 3.10 or higher
- Supabase project with pgvector extension enabled
- Existing document embeddings in Supabase (generated using OpenAI text-embedding-3-small)
- OpenAI API key for query embedding generation
- Supabase service role key

## Database Schema

The server expects the following tables in your Supabase database:

```sql
-- Documents table
CREATE TABLE documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL,
    session_id text NOT NULL,
    project_id text DEFAULT '-',
    filename text NOT NULL,
    file_type text NOT NULL,
    file_size integer,
    total_chunks integer,
    upload_date timestamp with time zone DEFAULT now(),
    processing_status text DEFAULT 'completed',
    metadata jsonb
);

-- Document embeddings table
CREATE TABLE document_embeddings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
    user_id uuid NOT NULL,
    session_id text NOT NULL,
    project_id text DEFAULT '-',
    chunk_text text NOT NULL,
    embedding vector(1536), -- OpenAI text-embedding-3-small
    chunk_index integer NOT NULL,
    chunk_metadata jsonb,
    created_at timestamp with time zone DEFAULT now()
);

-- Create indexes for performance
CREATE INDEX idx_embeddings_vector ON document_embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_embeddings_user_session ON document_embeddings(user_id, session_id);
CREATE INDEX idx_documents_user_session ON documents(user_id, session_id);
```

## Installation

### Option 1: Install from source

```bash
# Clone the repository
git clone https://github.com/your-org/document-retrieval-mcp
cd document-retrieval-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Install as package

```bash
pip install document-retrieval-mcp
```

## Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your configuration:
```env
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_API_KEY=your-service-role-key
OPENAI_API_KEY=your-openai-api-key

# Optional - customize as needed
EMBEDDING_MODEL=text-embedding-3-small
MAX_RESULTS_LIMIT=20
DEFAULT_SIMILARITY_THRESHOLD=0.7
```

## Usage

### Running the Server

```bash
# Direct execution
python src/server.py

# Or if installed as package
document-retrieval-mcp
```

### Claude Desktop Integration

Add the following to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcp_servers": {
    "document-retrieval": {
      "command": "python",
      "args": ["/path/to/document-retrieval-mcp/src/server.py"],
      "env": {
        "SUPABASE_URL": "https://your-project.supabase.co",
        "SUPABASE_API_KEY": "your-service-role-key",
        "OPENAI_API_KEY": "your-openai-api-key"
      }
    }
  }
}
```

## Available Tools

### 1. `search_documents`
Search for documents using semantic similarity.

**Parameters:**
- `query` (string, required): Search query text
- `user_id` (string, required): User identifier
- `session_id` (string, required): Session identifier
- `project_id` (string, optional): Project filter (default: "-")
- `top_k` (integer, optional): Number of results (default: 5, max: 20)
- `similarity_threshold` (float, optional): Minimum similarity (default: 0.7)

**Example:**
```json
{
  "tool": "search_documents",
  "params": {
    "query": "machine learning algorithms",
    "user_id": "user-123",
    "session_id": "session-456",
    "top_k": 10
  }
}
```

### 2. `get_document_context`
Retrieve full document or specific chunks.

**Parameters:**
- `document_id` (string, required): Document UUID
- `user_id` (string, required): User identifier
- `session_id` (string, required): Session identifier
- `chunk_ids` (array, optional): Specific chunk UUIDs to retrieve

**Example:**
```json
{
  "tool": "get_document_context",
  "params": {
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "user-123",
    "session_id": "session-456"
  }
}
```

### 3. `list_user_documents`
List all documents accessible to the user.

**Parameters:**
- `user_id` (string, required): User identifier
- `session_id` (string, required): Session identifier
- `project_id` (string, optional): Project filter
- `page` (integer, optional): Page number (default: 1)
- `per_page` (integer, optional): Items per page (default: 20, max: 100)

**Example:**
```json
{
  "tool": "list_user_documents",
  "params": {
    "user_id": "user-123",
    "session_id": "session-456",
    "page": 1,
    "per_page": 50
  }
}
```

### 4. `get_similar_chunks`
Find chunks similar to a reference chunk.

**Parameters:**
- `chunk_id` (string, required): Reference chunk UUID
- `user_id` (string, required): User identifier
- `session_id` (string, required): Session identifier
- `top_k` (integer, optional): Number of results (default: 3, max: 10)

**Example:**
```json
{
  "tool": "get_similar_chunks",
  "params": {
    "chunk_id": "chunk-789",
    "user_id": "user-123",
    "session_id": "session-456",
    "top_k": 5
  }
}
```

## Resources

The server provides two informational resources:

### 1. `resource://server-info`
Returns current server status and configuration.

### 2. `resource://schema-info`
Returns database schema information.

## Usage Examples with Claude

After configuring the server in Claude Desktop, you can use natural language:

```
User: "Search for documents about machine learning algorithms"
Claude: I'll search for documents about machine learning algorithms.
[Uses search_documents tool]

User: "Show me the full content of document ABC"
Claude: I'll retrieve the complete content of that document.
[Uses get_document_context tool]

User: "What documents do I have access to?"
Claude: Let me list all your available documents.
[Uses list_user_documents tool]

User: "Find similar content to this chunk"
Claude: I'll find chunks with similar content.
[Uses get_similar_chunks tool]
```

## Performance Optimization

The server implements several optimization strategies:

1. **Connection Pooling**: Maintains 5-20 database connections
2. **TTL Caching**: Caches metadata for 5 minutes
3. **Vector Indexes**: Uses IVFFlat indexes for fast similarity search
4. **Query Optimization**: Efficient SQL with proper WHERE clauses
5. **Async Operations**: Full async/await for concurrent requests

## Security

- **Service Role Key**: Bypasses RLS for performance
- **Application-Level Security**: WHERE clauses filter by user/session/project
- **Input Validation**: JSON Schema validation on all parameters
- **Error Sanitization**: No sensitive data in error messages
- **Environment Variables**: Secrets never hardcoded

## Troubleshooting

### Common Issues

1. **"Missing required environment variables"**
   - Ensure all required variables are set in `.env` or environment

2. **"Connection pool timeout"**
   - Check Supabase URL and API key
   - Verify network connectivity

3. **"No results above similarity threshold"**
   - Lower the similarity_threshold parameter
   - Ensure embeddings exist for the user/session

4. **"Document not found or access denied"**
   - Verify user_id and session_id match existing records
   - Check document_id is valid

### Logging

Enable debug logging by setting:
```bash
export LOG_LEVEL=DEBUG
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# With coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src tests

# Lint
ruff check src tests

# Type checking
mypy src
```

## Architecture

The server follows a layered architecture:

```
┌─────────────────┐
│   MCP Client    │
│ (Claude Desktop)│
└────────┬────────┘
         │ JSON-RPC
┌────────▼────────┐
│   MCP Server    │
│  (Protocol Layer)│
└────────┬────────┘
         │
┌────────▼────────┐
│  Business Logic │
│  (Tools & Cache)│
└────────┬────────┘
         │
┌────────▼────────┐
│  Data Access    │
│ (AsyncPG + OAI) │
└────────┬────────┘
         │
┌────────▼────────┐
│    Supabase     │
│  (PostgreSQL +  │
│    pgvector)    │
└─────────────────┘
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: [https://github.com/your-org/document-retrieval-mcp/issues](https://github.com/your-org/document-retrieval-mcp/issues)
- Documentation: [https://docs.your-org.com/mcp/document-retrieval](https://docs.your-org.com/mcp/document-retrieval)

## Acknowledgments

- Built with the [Model Context Protocol](https://modelcontextprotocol.io)
- Powered by [Supabase](https://supabase.com) and [pgvector](https://github.com/pgvector/pgvector)
- Embeddings by [OpenAI](https://openai.com)