# 🧪 Testing Guide - Document Retrieval MCP Server

Complete guide for testing the MCP server with multiple testing interfaces.

## Quick Start Testing

### Step 1: Verify Setup

First, ensure everything is properly configured:

```bash
# Run the verification script
python verify_setup.py
```

This checks:
- ✅ Environment variables are set
- ✅ Python packages are installed
- ✅ Supabase connection works
- ✅ OpenAI API is accessible
- ✅ Required files exist

### Step 2: Insert Test Data

Create sample documents and embeddings:

```bash
# Insert default test data
python insert_test_data.py

# Or specify custom user/session
python insert_test_data.py --user-id "your-uuid" --session-id "session-456"

# Clean up test data when done
python insert_test_data.py --cleanup
```

This creates 5 sample documents about:
- AI Fundamentals
- Python Basics
- Web Development
- Data Science
- Cloud Computing

### Step 3: Test the MCP Server

## Testing Options

### Option 1: Streamlit Dashboard (Visual Testing) 🎨

Best for: Visual testing, debugging, and demonstrations

```bash
# Install UI dependencies
pip install -r requirements-test-ui.txt

# Launch the dashboard
streamlit run test_dashboard.py
```

Open http://localhost:8501 in your browser.

**Features:**
- Connection status indicators
- Test all 4 MCP tools interactively
- Insert test data with one click
- View results in formatted tables
- Real-time error messages

**Dashboard Tabs:**
1. **🔍 Search Documents** - Test semantic search
2. **📄 Get Document Context** - Retrieve full documents
3. **📋 List Documents** - Browse all documents
4. **🔗 Similar Chunks** - Find related content
5. **➕ Insert Test Data** - Create sample data

### Option 2: MCP Protocol Tester (Direct Testing) 🔧

Best for: Protocol compliance and automated testing

```bash
# Run direct MCP protocol tests
python test_mcp_direct.py
```

**What it tests:**
- MCP server starts correctly
- List tools returns 4 tools
- List resources returns 2 resources
- Search documents works
- List user documents works
- Server info resource accessible

**Expected output:**
```
🚀 MCP Server Direct Protocol Testing
==================================================
✅ MCP Server started

📋 Testing: List Tools
----------------------------------------
✅ Found 4 tools:
   - search_documents: Search existing document embeddings using se...
   - get_document_context: Retrieve specific document sections from ...
   - list_user_documents: List all documents with stored embeddings...
   - get_similar_chunks: Find similar document chunks to a given c...

🔍 Testing: Search Documents
----------------------------------------
✅ Search completed:
   - Query: machine learning
   - Results found: 3

📊 Test Summary
==================================================
✅ Passed: 5/5
❌ Failed: 0/5

🎉 All tests passed!
```

### Option 3: Manual Testing with CLI

Best for: Quick debugging and troubleshooting

```bash
# Start the MCP server manually
python src/server.py

# In another terminal, send JSON-RPC requests
echo '{"jsonrpc":"2.0","method":"list_tools","params":{},"id":1}' | python src/server.py
```

## Test Scenarios

### 1. Basic Connectivity Test
```python
# verify_setup.py handles this automatically
python verify_setup.py
```

### 2. Search Test
```python
# Using the dashboard
1. Go to "Search Documents" tab
2. Enter query: "machine learning"
3. Click "Search"
4. Verify results show relevant documents
```

### 3. Document Retrieval Test
```python
# Get a document ID from list_user_documents
1. Go to "List Documents" tab
2. Click "List Documents"
3. Copy a document ID
4. Go to "Get Document Context" tab
5. Paste the ID and click "Get Context"
```

### 4. Performance Test
```python
# Check response times in the dashboard
- Search should complete in < 2 seconds
- Document listing should be < 1 second
- Context retrieval should be < 1 second
```

## Troubleshooting

### Issue: "Missing environment variables"

**Solution:**
```bash
# Copy and fill the environment file
cp .env.example .env
nano .env  # Add your credentials
```

### Issue: "No results found"

**Solution:**
```bash
# Insert test data first
python insert_test_data.py

# Verify data exists
python verify_setup.py  # Check embeddings count
```

### Issue: "Connection failed"

**Solution:**
1. Check Supabase URL format: `https://[project-ref].supabase.co`
2. Verify you're using the SERVICE ROLE key (not anon key)
3. Test OpenAI API key separately:
```python
from openai import OpenAI
client = OpenAI(api_key="your-key")
client.embeddings.create(model="text-embedding-3-small", input="test")
```

### Issue: "MCP server won't start"

**Solution:**
```bash
# Check Python version
python --version  # Must be >= 3.10

# Reinstall dependencies
pip install -r requirements.txt

# Check for port conflicts
lsof -i :3000  # If using port 3000
```

## Advanced Testing

### Load Testing
```python
# Create a load test script
import asyncio
import time

async def load_test():
    tasks = []
    for i in range(100):
        # Create 100 concurrent searches
        tasks.append(search_documents(f"query {i}", user_id, session_id))

    start = time.time()
    results = await asyncio.gather(*tasks)
    end = time.time()

    print(f"100 searches in {end-start:.2f} seconds")
    print(f"Average: {(end-start)/100:.3f} seconds per search")
```

### Integration Testing
```bash
# Run the full test suite
cd tests/
pytest -v

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m performance
```

### Claude Desktop Testing
```bash
# Configure Claude Desktop
1. Copy configuration from claude_desktop_config.json
2. Update paths and credentials
3. Restart Claude Desktop
4. Test with: "Search for documents about machine learning"
```

## Testing Checklist

Before deployment, ensure:

- [ ] `verify_setup.py` - All checks pass
- [ ] `test_mcp_direct.py` - All protocol tests pass
- [ ] Streamlit dashboard - All tools work correctly
- [ ] Search returns relevant results
- [ ] Document retrieval works
- [ ] Error handling works (test with invalid IDs)
- [ ] Performance meets requirements (< 2s for searches)
- [ ] Claude Desktop integration works

## Performance Benchmarks

Expected performance with test data:

| Operation | Target | Typical |
|-----------|--------|---------|
| Search (5 results) | < 2s | 0.5-1s |
| List documents | < 1s | 0.2-0.5s |
| Get context | < 1s | 0.3-0.7s |
| Similar chunks | < 1.5s | 0.4-0.8s |

## Monitoring

While testing, monitor:

1. **Server logs:**
```bash
tail -f mcp-debug.log
```

2. **Resource usage:**
```bash
# In test dashboard, check connection status
# Shows cache size and pool connections
```

3. **Database queries:**
- Check Supabase dashboard for query performance
- Monitor embedding search times

## Next Steps

After successful testing:

1. **Production deployment** - See DEPLOYMENT_GUIDE.md
2. **Claude Desktop integration** - Configure and test
3. **Performance tuning** - Optimize based on your data
4. **Security hardening** - Review access controls

---

## Quick Commands Reference

```bash
# Setup and verify
python verify_setup.py

# Insert test data
python insert_test_data.py

# Run tests
streamlit run test_dashboard.py      # Visual testing
python test_mcp_direct.py            # Protocol testing
python src/server.py                 # Start server

# Clean up
python insert_test_data.py --cleanup # Remove test data
```

Need help? Check the README.md or file an issue on GitHub.