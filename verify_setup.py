#!/usr/bin/env python3
"""
Setup Verification Script

Verifies that the MCP server environment is correctly configured.
"""

import os
import sys
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Terminal colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_status(status: bool, message: str):
    """Print status with color."""
    if status:
        print(f"{Colors.GREEN}✅{Colors.ENDC} {message}")
    else:
        print(f"{Colors.FAIL}❌{Colors.ENDC} {message}")

def print_header(title: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.ENDC}")
    print("-" * 40)

def check_environment_variables() -> Dict[str, bool]:
    """Check if required environment variables are set."""
    print_header("1. Environment Variables")

    variables = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_API_KEY": os.getenv("SUPABASE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    }

    results = {}
    for var_name, var_value in variables.items():
        if var_value:
            if var_name == "OPENAI_API_KEY":
                display = f"{var_value[:10]}...{var_value[-4:]}" if len(var_value) > 20 else "***"
            elif var_name == "SUPABASE_API_KEY":
                display = f"{var_value[:20]}...{var_value[-10:]}" if len(var_value) > 40 else "***"
            else:
                display = var_value

            print_status(True, f"{var_name}: {display}")
            results[var_name] = True
        else:
            print_status(False, f"{var_name}: Not set")
            results[var_name] = False

    return results

def check_python_packages() -> Dict[str, bool]:
    """Check if required Python packages are installed."""
    print_header("2. Python Packages")

    packages = [
        "mcp",
        "asyncpg",
        "openai",
        "supabase",
        "cachetools",
        "numpy",
        "python-dotenv",
        "jsonschema"
    ]

    results = {}
    for package in packages:
        try:
            __import__(package)
            print_status(True, f"{package}: Installed")
            results[package] = True
        except ImportError:
            print_status(False, f"{package}: Not installed")
            results[package] = False

    return results

def check_supabase_connection() -> bool:
    """Check connection to Supabase."""
    print_header("3. Supabase Connection")

    try:
        from supabase import create_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_API_KEY")

        if not url or not key:
            print_status(False, "Missing Supabase credentials")
            return False

        client = create_client(url, key)

        # Try to query documents table
        result = client.table("documents").select("id").limit(1).execute()
        print_status(True, "Connected to Supabase")

        # Check tables
        try:
            docs = client.table("documents").select("id", count="exact").execute()
            embeddings = client.table("document_embeddings").select("id", count="exact").execute()

            print_status(True, f"Documents table: {docs.count or 0} records")
            print_status(True, f"Embeddings table: {embeddings.count or 0} records")

            return True

        except Exception as e:
            print_status(False, f"Tables check failed: {str(e)}")
            return False

    except Exception as e:
        print_status(False, f"Connection failed: {str(e)}")
        return False

def check_openai_connection() -> bool:
    """Check connection to OpenAI API."""
    print_header("4. OpenAI API")

    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print_status(False, "Missing OpenAI API key")
            return False

        client = OpenAI(api_key=api_key)

        # Test embedding generation
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="test"
        )

        print_status(True, f"Connected to OpenAI")
        print_status(True, f"Embedding model: text-embedding-3-small")
        print_status(True, f"Vector dimensions: {len(response.data[0].embedding)}")

        return True

    except Exception as e:
        print_status(False, f"API check failed: {str(e)}")
        return False

def check_file_structure() -> Dict[str, bool]:
    """Check if required files exist."""
    print_header("5. File Structure")

    files = [
        "src/server.py",
        ".env",
        "requirements.txt",
        "pyproject.toml",
        "README.md"
    ]

    results = {}
    for file in files:
        exists = os.path.exists(file)
        print_status(exists, f"{file}: {'Found' if exists else 'Missing'}")
        results[file] = exists

    return results

def print_summary(checks: Dict[str, Any]):
    """Print verification summary."""
    print_header("Verification Summary")

    all_passed = True

    # Environment variables
    env_passed = all(checks["environment"].values())
    print_status(env_passed, f"Environment: {sum(checks['environment'].values())}/{len(checks['environment'])} variables set")
    all_passed = all_passed and env_passed

    # Packages
    pkg_passed = all(checks["packages"].values())
    print_status(pkg_passed, f"Packages: {sum(checks['packages'].values())}/{len(checks['packages'])} installed")
    all_passed = all_passed and pkg_passed

    # Connections
    print_status(checks["supabase"], "Supabase: " + ("Connected" if checks["supabase"] else "Failed"))
    print_status(checks["openai"], "OpenAI: " + ("Connected" if checks["openai"] else "Failed"))
    all_passed = all_passed and checks["supabase"] and checks["openai"]

    # Files
    files_passed = all(checks["files"].values())
    print_status(files_passed, f"Files: {sum(checks['files'].values())}/{len(checks['files'])} found")
    all_passed = all_passed and files_passed

    # Final status
    print("\n" + "=" * 50)
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ All checks passed! Your MCP server is ready to run.{Colors.ENDC}")
        print("\nNext steps:")
        print("1. Run the MCP server: python src/server.py")
        print("2. Test with Streamlit dashboard: streamlit run test_dashboard.py")
        print("3. Test MCP protocol: python test_mcp_direct.py")
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}⚠️ Some checks failed. Please fix the issues above.{Colors.ENDC}")
        print("\nTroubleshooting:")
        if not env_passed:
            print("- Copy .env.example to .env and fill in your credentials")
        if not pkg_passed:
            print("- Run: pip install -r requirements.txt")
        if not checks["supabase"]:
            print("- Check your Supabase URL and API key")
            print("- Ensure the database tables exist")
        if not checks["openai"]:
            print("- Check your OpenAI API key")
            print("- Ensure you have API credits")

def main():
    """Main verification routine."""
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 50}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}Document Retrieval MCP Server - Setup Verification{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 50}{Colors.ENDC}")

    checks = {
        "environment": check_environment_variables(),
        "packages": check_python_packages(),
        "supabase": check_supabase_connection(),
        "openai": check_openai_connection(),
        "files": check_file_structure()
    }

    print_summary(checks)

if __name__ == "__main__":
    main()