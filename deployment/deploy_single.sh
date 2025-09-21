#!/bin/bash

# Deploy a single MCP server to Fly.io
# Usage: ./deploy_single.sh [app-name]

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Check if app name provided
APP_NAME=${1:-presentation-retrieval-mcp}

print_info "Deploying MCP server: $APP_NAME"

# Check if fly CLI is installed
if ! command -v flyctl &> /dev/null; then
    print_error "Fly CLI (flyctl) is not installed"
    print_info "Install it from: https://fly.io/docs/getting-started/installing-flyctl/"
    exit 1
fi

# Check if logged in to Fly.io
if ! flyctl auth whoami &> /dev/null; then
    print_warning "Not logged in to Fly.io"
    print_info "Running: flyctl auth login"
    flyctl auth login
fi

# Navigate to the project directory
cd "$(dirname "$0")/.."

# Check if fly.toml exists
if [ ! -f "fly.toml" ]; then
    print_error "fly.toml not found in current directory"
    exit 1
fi

# Check if app exists, create if not
if flyctl apps list | grep -q "$APP_NAME"; then
    print_info "App $APP_NAME already exists"
else
    print_info "Creating app: $APP_NAME"
    flyctl apps create "$APP_NAME"
fi

# Set secrets (environment variables)
print_info "Setting secrets for $APP_NAME..."

# Check if .env file exists for local secrets
if [ -f ".env" ]; then
    print_info "Loading secrets from .env file..."

    # Read .env file and set secrets
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [ -z "$key" ] && continue

        # Remove quotes from value
        value="${value%\"}"
        value="${value#\"}"

        # Set secret in Fly.io
        print_info "Setting secret: $key"
        echo "$value" | flyctl secrets set "$key"=- --app "$APP_NAME"
    done < .env
else
    print_warning ".env file not found"
    print_info "Make sure to set the following secrets manually:"
    echo "  - DATABASE_URL"
    echo "  - OPENAI_API_KEY"
    echo "  - SUPABASE_URL (optional)"
    echo "  - SUPABASE_API_KEY (optional)"

    print_info "Use: flyctl secrets set KEY=value --app $APP_NAME"
fi

# Deploy the application
print_info "Deploying application to Fly.io..."
flyctl deploy --app "$APP_NAME"

# Check deployment status
if [ $? -eq 0 ]; then
    print_success "Deployment successful!"

    # Show app info
    print_info "App information:"
    flyctl info --app "$APP_NAME"

    # Show recent logs
    print_info "Recent logs:"
    flyctl logs --app "$APP_NAME" --recent

    print_success "MCP server $APP_NAME deployed successfully!"
    print_info "Monitor logs: flyctl logs --app $APP_NAME"
    print_info "SSH into machine: flyctl ssh console --app $APP_NAME"
else
    print_error "Deployment failed"
    exit 1
fi