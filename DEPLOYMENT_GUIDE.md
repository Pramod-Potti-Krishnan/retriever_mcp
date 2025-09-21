# Document Retrieval MCP Server - Deployment Guide

## Quick Start

### 1. Prerequisites

Ensure you have:
- Python 3.10 or higher
- Supabase project with pgvector extension
- OpenAI API key
- Existing document embeddings in Supabase

### 2. Installation

```bash
# Clone or navigate to the server directory
cd mcp-servers/document-retrieval-mcp/

# Run the setup script
./setup.sh

# Update .env with your credentials
nano .env
```

### 3. Configuration

Update `.env` with your actual credentials:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_API_KEY=your-service-role-key
OPENAI_API_KEY=sk-your-openai-key
```

### 4. Run the Server

```bash
# Using the run script
./run_server.sh

# Or manually
source venv/bin/activate
python src/server.py
```

## Claude Desktop Integration

### macOS

1. Locate Claude configuration:
```bash
~/Library/Application Support/Claude/claude_desktop_config.json
```

2. Add the MCP server configuration:
```json
{
  "mcp_servers": {
    "document-retrieval": {
      "command": "python",
      "args": ["/absolute/path/to/mcp-servers/document-retrieval-mcp/src/server.py"],
      "env": {
        "SUPABASE_URL": "your-supabase-url",
        "SUPABASE_API_KEY": "your-api-key",
        "OPENAI_API_KEY": "your-openai-key"
      }
    }
  }
}
```

3. Restart Claude Desktop

### Windows

1. Locate Claude configuration:
```
%APPDATA%\Claude\claude_desktop_config.json
```

2. Update paths for Windows:
```json
{
  "mcp_servers": {
    "document-retrieval": {
      "command": "python",
      "args": ["C:\\path\\to\\mcp-servers\\document-retrieval-mcp\\src\\server.py"],
      "env": {
        "SUPABASE_URL": "your-supabase-url",
        "SUPABASE_API_KEY": "your-api-key",
        "OPENAI_API_KEY": "your-openai-key"
      }
    }
  }
}
```

## Database Setup

If you haven't set up the database tables yet:

```sql
-- Run in Supabase SQL editor

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
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

-- Create embeddings table
CREATE TABLE IF NOT EXISTS document_embeddings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
    user_id uuid NOT NULL,
    session_id text NOT NULL,
    project_id text DEFAULT '-',
    chunk_text text NOT NULL,
    embedding vector(1536),
    chunk_index integer NOT NULL,
    chunk_metadata jsonb,
    created_at timestamp with time zone DEFAULT now()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_embeddings_user_session
ON document_embeddings(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_documents_user_session
ON documents(user_id, session_id);
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .env .

CMD ["python", "src/server.py"]
```

Build and run:
```bash
docker build -t document-retrieval-mcp .
docker run -d --name mcp-server \
  -e SUPABASE_URL=$SUPABASE_URL \
  -e SUPABASE_API_KEY=$SUPABASE_API_KEY \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  document-retrieval-mcp
```

### Systemd Service (Linux)

Create `/etc/systemd/system/document-retrieval-mcp.service`:

```ini
[Unit]
Description=Document Retrieval MCP Server
After=network.target

[Service]
Type=simple
User=mcp
WorkingDirectory=/opt/document-retrieval-mcp
Environment="PATH=/opt/document-retrieval-mcp/venv/bin"
EnvironmentFile=/opt/document-retrieval-mcp/.env
ExecStart=/opt/document-retrieval-mcp/venv/bin/python /opt/document-retrieval-mcp/src/server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable document-retrieval-mcp
sudo systemctl start document-retrieval-mcp
```

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SUPABASE_URL` | ✅ | - | Your Supabase project URL |
| `SUPABASE_API_KEY` | ✅ | - | Service role key (not anon key) |
| `OPENAI_API_KEY` | ✅ | - | OpenAI API key for embeddings |
| `EMBEDDING_MODEL` | ❌ | text-embedding-3-small | OpenAI embedding model |
| `VECTOR_DIMENSIONS` | ❌ | 1536 | Embedding vector dimensions |
| `DATABASE_URL` | ❌ | - | Direct PostgreSQL connection |
| `DB_POOL_MIN_SIZE` | ❌ | 5 | Min database connections |
| `DB_POOL_MAX_SIZE` | ❌ | 20 | Max database connections |
| `MAX_RESULTS_LIMIT` | ❌ | 20 | Max search results |
| `DEFAULT_SIMILARITY_THRESHOLD` | ❌ | 0.7 | Default similarity score |
| `CACHE_TTL` | ❌ | 300 | Cache time-to-live (seconds) |
| `CACHE_MAX_SIZE` | ❌ | 1000 | Max cache entries |
| `LOG_LEVEL` | ❌ | INFO | Logging level |

## Monitoring

### Health Check

```bash
# Check if server is running
curl http://localhost:3000/health

# Check MCP protocol
echo '{"jsonrpc":"2.0","method":"list_tools","params":{},"id":1}' | \
  python src/server.py
```

### Logs

```bash
# View logs
tail -f mcp-debug.log

# Enable debug logging
export LOG_LEVEL=DEBUG
```

### Performance Monitoring

Monitor key metrics:
- Response time < 1 second
- Cache hit rate > 80%
- Database pool utilization < 80%
- Memory usage < 500MB

## Troubleshooting

### Server won't start

1. Check Python version:
```bash
python --version  # Should be >= 3.10
```

2. Verify environment variables:
```bash
python -c "import os; print(os.getenv('SUPABASE_URL'))"
```

3. Test database connection:
```bash
python -c "from supabase import create_client; client = create_client('url', 'key')"
```

### No results returned

1. Verify embeddings exist:
```sql
SELECT COUNT(*) FROM document_embeddings WHERE user_id = 'your-user-id';
```

2. Check similarity threshold:
- Lower threshold returns more results
- Default is 0.7, try 0.5 for broader matches

3. Verify vector dimensions:
- Must be 1536 for text-embedding-3-small
- Check existing embeddings match configuration

### Claude Desktop can't connect

1. Verify server is running:
```bash
ps aux | grep server.py
```

2. Check configuration path:
- Path must be absolute
- Verify file permissions

3. Restart Claude Desktop after config changes

### Performance issues

1. Increase connection pool:
```env
DB_POOL_MIN_SIZE=10
DB_POOL_MAX_SIZE=30
```

2. Optimize vector index:
```sql
-- Recreate index with more lists
DROP INDEX idx_embeddings_vector;
CREATE INDEX idx_embeddings_vector
ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 200);
```

3. Enable caching:
```env
CACHE_TTL=600  # 10 minutes
CACHE_MAX_SIZE=2000
```

## Security Considerations

1. **Use Service Role Key**: Required for bypassing RLS
2. **Secure Environment**: Never commit .env files
3. **Network Security**: Use HTTPS in production
4. **Rate Limiting**: Implement at API gateway level
5. **Monitoring**: Log all access attempts

## Support

- GitHub Issues: [Report bugs and request features]
- Documentation: See README.md for detailed API reference
- MCP Protocol: https://modelcontextprotocol.io