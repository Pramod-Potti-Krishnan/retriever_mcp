#!/usr/bin/env python3
"""
Test script to verify API structure and endpoints
"""
import requests
import json

API_BASE = "http://localhost:8080"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{API_BASE}/health")
        print("✓ Health endpoint:", response.status_code)
        if response.status_code == 200:
            print("  Response:", json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print("✗ Health endpoint failed:", e)
        return False

def test_endpoints():
    """List all available endpoints"""
    endpoints = [
        ("/health", "GET"),
        ("/search-presentation-content", "POST"),
        ("/extract-key-points", "POST"),
        ("/find-supporting-data", "POST"),
        ("/user-documents/presentation-ready", "GET"),
        ("/generate-slide-suggestions", "POST"),
        ("/find-presentation-examples", "GET"),
        ("/extract-citations-sources", "POST"),
    ]

    print("\nAvailable API Endpoints:")
    print("=" * 50)
    for endpoint, method in endpoints:
        print(f"{method:6} {endpoint}")
    print("=" * 50)

if __name__ == "__main__":
    print("Testing Presentation Retrieval API")
    print("=" * 50)

    # Test health endpoint
    if test_health():
        print("\n✅ API is running successfully!")
    else:
        print("\n⚠️ API is not accessible. Make sure it's running.")

    # List all endpoints
    test_endpoints()