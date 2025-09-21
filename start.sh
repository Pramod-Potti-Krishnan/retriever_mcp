#!/bin/bash
# Start script for Railway deployment
# Handles multiple scenarios for finding and running uvicorn

echo "Starting application..."
echo "PORT: ${PORT:-8000}"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Try multiple methods to start the application
if [ -f "start.py" ]; then
    echo "Using start.py entry point..."
    python start.py
elif command -v uvicorn &> /dev/null; then
    echo "Found uvicorn in PATH, using direct command..."
    uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
elif python -c "import uvicorn" 2>/dev/null; then
    echo "Found uvicorn module, using python -m..."
    python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
else
    # Last resort: try to find uvicorn in common locations
    echo "Searching for uvicorn..."
    UVICORN_PATH=$(find /opt -name uvicorn 2>/dev/null | head -1)
    if [ -n "$UVICORN_PATH" ]; then
        echo "Found uvicorn at: $UVICORN_PATH"
        python "$UVICORN_PATH" main:app --host 0.0.0.0 --port ${PORT:-8000}
    else
        echo "ERROR: Could not find uvicorn! Installation may have failed."
        echo "Installed packages:"
        pip list
        exit 1
    fi
fi