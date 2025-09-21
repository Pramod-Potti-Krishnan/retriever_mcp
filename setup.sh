#!/bin/bash
# Document Retrieval MCP Server Setup Script

set -e

echo "🚀 Setting up Document Retrieval MCP Server..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}→${NC} $1"
}

# Check Python version
print_info "Checking Python version..."
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    REQUIRED_VERSION="3.10"
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
        print_success "Python $PYTHON_VERSION is installed"
    else
        print_error "Python $PYTHON_VERSION is too old. Required: Python >= 3.10"
        exit 1
    fi
else
    print_error "Python 3 is not installed"
    exit 1
fi

# Create virtual environment
print_info "Creating virtual environment..."
if [ -d "venv" ]; then
    print_info "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip -q
print_success "pip upgraded"

# Install dependencies
print_info "Installing dependencies..."
pip install -r requirements.txt -q
print_success "Dependencies installed"

# Check for .env file
print_info "Checking configuration..."
if [ -f ".env" ]; then
    print_success "Configuration file (.env) exists"
else
    print_info "Creating .env from template..."
    cp .env.example .env
    print_success "Created .env file - Please update with your credentials"
fi

# Validate environment variables
print_info "Validating environment variables..."
if [ -f ".env" ]; then
    source .env
    MISSING_VARS=()

    if [ -z "$SUPABASE_URL" ]; then
        MISSING_VARS+=("SUPABASE_URL")
    fi

    if [ -z "$SUPABASE_API_KEY" ]; then
        MISSING_VARS+=("SUPABASE_API_KEY")
    fi

    if [ -z "$OPENAI_API_KEY" ]; then
        MISSING_VARS+=("OPENAI_API_KEY")
    fi

    if [ ${#MISSING_VARS[@]} -ne 0 ]; then
        print_error "Missing required environment variables: ${MISSING_VARS[*]}"
        print_info "Please update .env file with your credentials"
    else
        print_success "All required environment variables are set"
    fi
fi

# Create Claude Desktop config directory if not exists
CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
if [ -d "$CLAUDE_CONFIG_DIR" ]; then
    print_info "Claude Desktop configuration directory found"

    # Check if config file exists
    if [ -f "$CLAUDE_CONFIG_DIR/claude_desktop_config.json" ]; then
        print_info "Claude Desktop configuration already exists"
        print_info "To add this MCP server, manually update: $CLAUDE_CONFIG_DIR/claude_desktop_config.json"
        print_info "Example configuration is available in: claude_desktop_config.json"
    else
        print_info "Creating Claude Desktop configuration..."
        # Update paths in the config
        sed "s|\${HOME}|$HOME|g" claude_desktop_config.json > "$CLAUDE_CONFIG_DIR/claude_desktop_config.json"
        print_success "Claude Desktop configuration created"
    fi
else
    print_info "Claude Desktop not found - install Claude Desktop and run setup again"
fi

# Create run script
print_info "Creating run script..."
cat > run_server.sh << 'EOF'
#!/bin/bash
# Run Document Retrieval MCP Server

# Activate virtual environment
source venv/bin/activate

# Load environment variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Run the server
python src/server.py
EOF

chmod +x run_server.sh
print_success "Run script created (./run_server.sh)"

# Installation complete
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  📦 Document Retrieval MCP Server Setup Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "  1. Update .env with your credentials:"
echo "     - SUPABASE_URL"
echo "     - SUPABASE_API_KEY"
echo "     - OPENAI_API_KEY"
echo ""
echo "  2. Run the server:"
echo "     ./run_server.sh"
echo ""
echo "  3. Restart Claude Desktop to load the MCP server"
echo ""
echo "  For more information, see README.md"
echo "═══════════════════════════════════════════════════════════"