"""
Shared Base Configuration for Multi-MCP Server Architecture

This module provides common configuration and utilities for all
presentation-related MCP servers deployed on Fly.io.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MCPServerConfig:
    """Base configuration for MCP servers."""
    name: str
    version: str = "1.0.0"
    region: Optional[str] = None
    log_level: str = "INFO"


class SharedConfig:
    """Shared configuration for all presentation MCP servers."""

    # Database configuration (shared across all MCPs)
    DATABASE_CONFIG = {
        "url": os.getenv("DATABASE_URL"),
        "pool_min_size": int(os.getenv("DB_POOL_MIN_SIZE", "5")),
        "pool_max_size": int(os.getenv("DB_POOL_MAX_SIZE", "20")),
        "timeout": 10,
        "command_timeout": 10,
    }

    # OpenAI configuration (shared for embedding and generation)
    OPENAI_CONFIG = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "generation_model": os.getenv("GENERATION_MODEL", "gpt-4"),
        "vector_dimensions": int(os.getenv("VECTOR_DIMENSIONS", "1536")),
    }

    # Supabase configuration (optional, for direct Supabase client usage)
    SUPABASE_CONFIG = {
        "url": os.getenv("SUPABASE_URL"),
        "api_key": os.getenv("SUPABASE_API_KEY"),
    }

    # Cache configuration
    CACHE_CONFIG = {
        "ttl": int(os.getenv("CACHE_TTL", "300")),  # 5 minutes
        "max_size": int(os.getenv("CACHE_MAX_SIZE", "1000")),
    }

    # Query defaults
    QUERY_DEFAULTS = {
        "max_results": int(os.getenv("MAX_RESULTS_LIMIT", "20")),
        "similarity_threshold": float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.7")),
    }

    # Fly.io specific
    FLY_CONFIG = {
        "region": os.getenv("FLY_REGION", "unknown"),
        "app_name": os.getenv("FLY_APP_NAME", "unknown"),
        "machine_id": os.getenv("FLY_MACHINE_ID", "unknown"),
    }

    @classmethod
    def validate_required_env_vars(cls, required_vars: list) -> bool:
        """Validate that required environment variables are set."""
        missing = []
        for var in required_vars:
            if not os.getenv(var):
                missing.append(var)

        if missing:
            logging.error(f"Missing required environment variables: {', '.join(missing)}")
            return False
        return True

    @classmethod
    def get_database_url(cls) -> str:
        """Get database URL with validation."""
        url = cls.DATABASE_CONFIG["url"]
        if not url:
            raise ValueError("DATABASE_URL environment variable is required")
        return url

    @classmethod
    def get_openai_key(cls) -> str:
        """Get OpenAI API key with validation."""
        key = cls.OPENAI_CONFIG["api_key"]
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return key


def setup_logging(server_name: str, log_level: str = None) -> logging.Logger:
    """Set up consistent logging for MCP servers."""
    level = log_level or os.getenv("LOG_LEVEL", "INFO")

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=f'%(asctime)s - {server_name} - %(levelname)s - %(message)s',
        stream=sys.stderr  # Fly.io captures stderr for logs
    )

    logger = logging.getLogger(server_name)

    # Log Fly.io deployment info
    if SharedConfig.FLY_CONFIG["region"] != "unknown":
        logger.info(f"Running on Fly.io - Region: {SharedConfig.FLY_CONFIG['region']}, "
                   f"App: {SharedConfig.FLY_CONFIG['app_name']}, "
                   f"Machine: {SharedConfig.FLY_CONFIG['machine_id']}")

    return logger


def get_connection_string() -> str:
    """Get PostgreSQL connection string with proper formatting."""
    return SharedConfig.get_database_url()


def get_cache_config() -> Dict[str, Any]:
    """Get cache configuration."""
    return SharedConfig.CACHE_CONFIG.copy()


def get_query_defaults() -> Dict[str, Any]:
    """Get default query parameters."""
    return SharedConfig.QUERY_DEFAULTS.copy()


# MCP Server registry for multi-server coordination
MCP_SERVERS = {
    "presentation-retrieval-mcp": {
        "name": "Document Retrieval MCP",
        "description": "Retrieves and searches document embeddings",
        "port": None,  # stdio-based
        "tools": ["search_documents", "get_document_context", "list_user_documents", "get_similar_chunks"]
    },
    "presentation-content-mcp": {
        "name": "Content Generation MCP",
        "description": "Generates presentation content from retrieved documents",
        "port": None,
        "tools": ["generate_slide_content", "create_outline", "summarize_section", "generate_speaker_notes"]
    },
    "presentation-layout-mcp": {
        "name": "Layout Builder MCP",
        "description": "Creates presentation layouts and styling",
        "port": None,
        "tools": ["create_slide_layout", "apply_theme", "generate_visuals", "export_presentation"]
    },
    "presentation-export-mcp": {
        "name": "Export Service MCP",
        "description": "Exports presentations to various formats",
        "port": None,
        "tools": ["export_pptx", "export_pdf", "export_html", "export_markdown"]
    }
}