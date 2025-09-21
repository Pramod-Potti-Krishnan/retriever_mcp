#!/bin/bash

# Setup secrets for Fly.io MCP servers
# This script helps configure environment variables for deployment

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Check if app name is provided
if [ $# -eq 0 ]; then
    print_error "No app name provided"
    echo "Usage: $0 <app-name>"
    echo "Example: $0 presentation-retrieval-mcp"
    exit 1
fi

APP_NAME=$1

print_info "Setting up secrets for: $APP_NAME"

# Function to read secret value securely
read_secret() {
    local prompt=$1
    local var_name=$2
    local default_val=$3

    if [ -n "$default_val" ]; then
        read -p "$prompt [$default_val]: " value
        value=${value:-$default_val}
    else
        read -s -p "$prompt: " value
        echo
    fi

    echo "$value"
}

# Required secrets
print_info "Enter required secrets (values will be hidden):"

# Database URL
print_warning "DATABASE_URL is required for MCP server to function"
DB_URL=$(read_secret "Enter DATABASE_URL (PostgreSQL connection string)" "DATABASE_URL")

# OpenAI API Key
print_warning "OPENAI_API_KEY is required for embedding generation"
OPENAI_KEY=$(read_secret "Enter OPENAI_API_KEY" "OPENAI_API_KEY")

# Optional secrets
print_info "Enter optional secrets (press Enter to skip):"

# Supabase configuration
SUPABASE_URL=$(read_secret "Enter SUPABASE_URL (optional)" "SUPABASE_URL")
SUPABASE_KEY=$(read_secret "Enter SUPABASE_API_KEY (optional)" "SUPABASE_API_KEY")

# Configuration options
print_info "Configuration options (press Enter for defaults):"

# Model configuration
EMBEDDING_MODEL=$(read_secret "Embedding model" "EMBEDDING_MODEL" "text-embedding-3-small")
VECTOR_DIMS=$(read_secret "Vector dimensions" "VECTOR_DIMENSIONS" "1536")

# Query configuration
MAX_RESULTS=$(read_secret "Max results limit" "MAX_RESULTS_LIMIT" "20")
SIM_THRESHOLD=$(read_secret "Similarity threshold" "DEFAULT_SIMILARITY_THRESHOLD" "0.7")

# Pool configuration
POOL_MIN=$(read_secret "DB pool min size" "DB_POOL_MIN_SIZE" "5")
POOL_MAX=$(read_secret "DB pool max size" "DB_POOL_MAX_SIZE" "20")

# Cache configuration
CACHE_TTL=$(read_secret "Cache TTL (seconds)" "CACHE_TTL" "300")
CACHE_SIZE=$(read_secret "Cache max size" "CACHE_MAX_SIZE" "1000")

# Log level
LOG_LEVEL=$(read_secret "Log level (DEBUG/INFO/WARNING/ERROR)" "LOG_LEVEL" "INFO")

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    print_error "Fly CLI (flyctl) is not installed"
    print_info "Install it from: https://fly.io/docs/getting-started/installing-flyctl/"
    exit 1
fi

# Set secrets in Fly.io
print_info "Setting secrets in Fly.io..."

# Required secrets
flyctl secrets set DATABASE_URL="$DB_URL" --app "$APP_NAME"
flyctl secrets set OPENAI_API_KEY="$OPENAI_KEY" --app "$APP_NAME"

# Optional secrets
if [ -n "$SUPABASE_URL" ]; then
    flyctl secrets set SUPABASE_URL="$SUPABASE_URL" --app "$APP_NAME"
fi

if [ -n "$SUPABASE_KEY" ]; then
    flyctl secrets set SUPABASE_API_KEY="$SUPABASE_KEY" --app "$APP_NAME"
fi

# Configuration
flyctl secrets set EMBEDDING_MODEL="$EMBEDDING_MODEL" --app "$APP_NAME"
flyctl secrets set VECTOR_DIMENSIONS="$VECTOR_DIMS" --app "$APP_NAME"
flyctl secrets set MAX_RESULTS_LIMIT="$MAX_RESULTS" --app "$APP_NAME"
flyctl secrets set DEFAULT_SIMILARITY_THRESHOLD="$SIM_THRESHOLD" --app "$APP_NAME"
flyctl secrets set DB_POOL_MIN_SIZE="$POOL_MIN" --app "$APP_NAME"
flyctl secrets set DB_POOL_MAX_SIZE="$POOL_MAX" --app "$APP_NAME"
flyctl secrets set CACHE_TTL="$CACHE_TTL" --app "$APP_NAME"
flyctl secrets set CACHE_MAX_SIZE="$CACHE_SIZE" --app "$APP_NAME"
flyctl secrets set LOG_LEVEL="$LOG_LEVEL" --app "$APP_NAME"

print_success "Secrets configured successfully!"

# Optionally save to local .env file
read -p "Save to local .env file? (y/n): " save_local

if [[ "$save_local" == "y" ]]; then
    cat > .env << EOF
# MCP Server Environment Variables
# WARNING: Keep this file secure and never commit to git

# Database
DATABASE_URL=$DB_URL

# OpenAI
OPENAI_API_KEY=$OPENAI_KEY

# Supabase (optional)
SUPABASE_URL=$SUPABASE_URL
SUPABASE_API_KEY=$SUPABASE_KEY

# Embedding Configuration
EMBEDDING_MODEL=$EMBEDDING_MODEL
VECTOR_DIMENSIONS=$VECTOR_DIMS

# Query Configuration
MAX_RESULTS_LIMIT=$MAX_RESULTS
DEFAULT_SIMILARITY_THRESHOLD=$SIM_THRESHOLD

# Connection Pool
DB_POOL_MIN_SIZE=$POOL_MIN
DB_POOL_MAX_SIZE=$POOL_MAX

# Cache
CACHE_TTL=$CACHE_TTL
CACHE_MAX_SIZE=$CACHE_SIZE

# Logging
LOG_LEVEL=$LOG_LEVEL
EOF

    print_success ".env file created"
    print_warning "Remember to add .env to .gitignore!"
fi

print_info "To view current secrets: flyctl secrets list --app $APP_NAME"
print_info "To update a secret: flyctl secrets set KEY=value --app $APP_NAME"