#!/usr/bin/env python
"""
Start script for Railway deployment
Programmatically runs uvicorn to avoid PATH issues
"""
import os
import sys
import uvicorn

if __name__ == "__main__":
    # Get port from environment variable, default to 8000
    port = int(os.environ.get("PORT", 8000))

    # Run the FastAPI app
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )