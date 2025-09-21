# Dockerfile for Presentation Retrieval MCP Server
# Optimized for Fly.io deployment

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 mcp && \
    mkdir -p /app && \
    chown -R mcp:mcp /app

WORKDIR /app

# Copy and install Python dependencies
COPY --chown=mcp:mcp requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared configuration
COPY --chown=mcp:mcp shared/ ./shared/

# Copy source code
COPY --chown=mcp:mcp src/ ./src/

# Switch to non-root user
USER mcp

# MCP servers use stdio, not HTTP ports
# No EXPOSE directive needed

# Health check for Fly.io monitoring
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; import asyncpg; sys.exit(0)" || exit 1

# Run the MCP server
CMD ["python", "src/mcp_server_flyio.py"]