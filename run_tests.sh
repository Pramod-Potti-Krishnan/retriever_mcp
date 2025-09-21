#!/bin/bash
# Test runner script for Document Retrieval MCP Server

set -e  # Exit on any error

echo "🧪 Document Retrieval MCP Server - Test Runner"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt
pip install -r tests/requirements-test.txt

# Set up test environment
echo "⚙️ Setting up test environment..."
if [ ! -f ".env.test" ]; then
    cp .env.example .env.test
    echo "📝 Created .env.test - please configure test environment variables"
fi

# Run linting (optional but recommended)
echo "🔍 Running code quality checks..."
if command -v flake8 &> /dev/null; then
    flake8 src/ --max-line-length=120 --ignore=E203,W503 || echo "⚠️ Linting warnings found"
fi

# Run unit tests
echo "🏃 Running unit tests..."
pytest tests/unit/ -v --cov=src --cov-report=term-missing --cov-report=html:htmlcov/unit

# Run integration tests
echo "🔗 Running integration tests..."
pytest tests/integration/ -v --tb=short

# Run performance tests (optional - can be slow)
if [ "$1" = "--with-performance" ]; then
    echo "⚡ Running performance tests..."
    pytest tests/unit/test_performance.py -v -m performance
fi

# Generate comprehensive coverage report
echo "📊 Generating coverage report..."
pytest tests/ --cov=src --cov-report=html:htmlcov --cov-report=xml:coverage.xml --cov-fail-under=90

# Run test client if server is available
echo "🖥️ Testing MCP client functionality..."
if [ "$1" = "--with-client" ]; then
    echo "Starting test client..."
    cd test-client
    python test_client.py python ../src/server.py &
    CLIENT_PID=$!
    sleep 2
    kill $CLIENT_PID 2>/dev/null || true
    cd ..
fi

echo ""
echo "✅ All tests completed!"
echo "📊 Coverage report: file://$(pwd)/htmlcov/index.html"
echo "📋 Test results summary:"
echo "   - Unit tests: tests/unit/"
echo "   - Integration tests: tests/integration/"
echo "   - Coverage: $(pwd)/htmlcov/index.html"
echo ""
echo "🎉 Testing complete! Check the reports for detailed results."