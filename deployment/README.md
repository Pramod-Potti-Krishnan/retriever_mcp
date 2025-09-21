# Fly.io MCP Server Deployment Guide

## Overview

This directory contains scripts and configurations for deploying MCP (Model Context Protocol) servers to Fly.io. The architecture is designed to support multiple MCP servers for presentation generation.

## MCP Server Architecture

MCP servers use **stdio protocol** (not HTTP), which means:
- They communicate via standard input/output
- They don't expose HTTP endpoints
- They're designed for AI agent integration (like Claude Desktop)

## Available MCP Servers

### 1. Presentation Retrieval MCP (`presentation-retrieval-mcp`)
- **Purpose**: Document retrieval from Supabase vector embeddings
- **Tools**:
  - `search_documents` - Semantic search across documents
  - `get_document_context` - Retrieve document sections
  - `list_user_documents` - List available documents
  - `get_similar_chunks` - Find similar content chunks

### Future MCP Servers (Planned)
- `presentation-content-mcp` - Content generation
- `presentation-layout-mcp` - Layout and styling
- `presentation-export-mcp` - Export to various formats

## Prerequisites

1. **Install Fly.io CLI**:
   ```bash
   # macOS
   brew install flyctl

   # Linux
   curl -L https://fly.io/install.sh | sh

   # Windows
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   ```

2. **Create Fly.io account**:
   ```bash
   flyctl auth signup
   # or login if you have an account
   flyctl auth login
   ```

3. **Required Environment Variables**:
   - `DATABASE_URL` - PostgreSQL connection string
   - `OPENAI_API_KEY` - For embedding generation
   - `SUPABASE_URL` (optional)
   - `SUPABASE_API_KEY` (optional)

## Deployment Steps

### 1. First-Time Setup

```bash
# Make scripts executable
chmod +x deployment/*.sh

# Create the app on Fly.io
flyctl apps create presentation-retrieval-mcp

# Configure secrets
./deployment/secrets_setup.sh presentation-retrieval-mcp
```

### 2. Deploy the MCP Server

```bash
# Deploy using the script
./deployment/deploy_single.sh presentation-retrieval-mcp

# Or deploy manually
flyctl deploy --app presentation-retrieval-mcp
```

### 3. Verify Deployment

```bash
# Check app status
flyctl status --app presentation-retrieval-mcp

# View logs
flyctl logs --app presentation-retrieval-mcp

# SSH into the container (for debugging)
flyctl ssh console --app presentation-retrieval-mcp
```

## Environment Configuration

### Required Secrets

Set these using `flyctl secrets set` or the `secrets_setup.sh` script:

```bash
# Database connection
flyctl secrets set DATABASE_URL="postgresql://..." --app presentation-retrieval-mcp

# OpenAI API
flyctl secrets set OPENAI_API_KEY="sk-..." --app presentation-retrieval-mcp
```

### Optional Configuration

```bash
# Embedding model configuration
flyctl secrets set EMBEDDING_MODEL="text-embedding-3-small" --app presentation-retrieval-mcp
flyctl secrets set VECTOR_DIMENSIONS="1536" --app presentation-retrieval-mcp

# Query defaults
flyctl secrets set MAX_RESULTS_LIMIT="20" --app presentation-retrieval-mcp
flyctl secrets set DEFAULT_SIMILARITY_THRESHOLD="0.7" --app presentation-retrieval-mcp

# Connection pool
flyctl secrets set DB_POOL_MIN_SIZE="5" --app presentation-retrieval-mcp
flyctl secrets set DB_POOL_MAX_SIZE="20" --app presentation-retrieval-mcp

# Cache settings
flyctl secrets set CACHE_TTL="300" --app presentation-retrieval-mcp
flyctl secrets set CACHE_MAX_SIZE="1000" --app presentation-retrieval-mcp

# Logging
flyctl secrets set LOG_LEVEL="INFO" --app presentation-retrieval-mcp
```

## Using the MCP Server

### With Claude Desktop

1. Configure Claude Desktop to use the MCP server
2. The server communicates via stdio, not HTTP
3. Tools are available through the MCP protocol

### Local Testing

```bash
# Test locally
python src/mcp_server_flyio.py

# Or with environment variables
export DATABASE_URL="..."
export OPENAI_API_KEY="..."
python src/mcp_server_flyio.py
```

## Multi-MCP Deployment

To deploy multiple MCP servers:

1. **Copy the structure**:
   ```bash
   cp -r presentation-retrieval-mcp presentation-content-mcp
   ```

2. **Update fly.toml**:
   - Change `app = "presentation-content-mcp"`
   - Adjust other settings as needed

3. **Deploy**:
   ```bash
   ./deployment/deploy_single.sh presentation-content-mcp
   ```

## Monitoring & Debugging

### View Logs
```bash
flyctl logs --app presentation-retrieval-mcp
```

### Check Resource Usage
```bash
flyctl scale show --app presentation-retrieval-mcp
```

### SSH Access
```bash
flyctl ssh console --app presentation-retrieval-mcp
```

### Health Checks
The MCP server includes process-based health checks (not HTTP).

## Troubleshooting

### Common Issues

1. **"Application failed to respond"**
   - This is expected! MCP servers use stdio, not HTTP
   - The server is running correctly even if there's no web interface

2. **Database connection errors**
   - Verify DATABASE_URL is correct
   - Check network connectivity from Fly.io region
   - Ensure PostgreSQL allows connections from Fly.io

3. **OpenAI API errors**
   - Verify OPENAI_API_KEY is valid
   - Check API rate limits
   - Ensure correct model name in EMBEDDING_MODEL

### Debug Commands

```bash
# Check secrets (values are hidden)
flyctl secrets list --app presentation-retrieval-mcp

# View app configuration
flyctl config show --app presentation-retrieval-mcp

# Check deployment status
flyctl status --app presentation-retrieval-mcp

# Scale resources if needed
flyctl scale vm shared-cpu-1x --memory 512 --app presentation-retrieval-mcp
```

## Security Notes

1. **Never commit secrets** to Git
2. Use Fly.io secrets management
3. Keep `.env` files local only
4. Use service role keys for Supabase (if applicable)
5. Rotate API keys regularly

## Scripts Reference

### `deploy_single.sh`
Deploys a single MCP server to Fly.io.

Usage:
```bash
./deployment/deploy_single.sh [app-name]
```

### `secrets_setup.sh`
Interactive script to configure environment variables.

Usage:
```bash
./deployment/secrets_setup.sh presentation-retrieval-mcp
```

## Support

For issues:
1. Check Fly.io status: https://status.fly.io/
2. Review logs: `flyctl logs --app [app-name]`
3. Fly.io documentation: https://fly.io/docs/
4. MCP documentation: https://modelcontextprotocol.io/