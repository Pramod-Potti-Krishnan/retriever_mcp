# Presentation Retrieval API - Agent-Ready REST Service

🚀 **LIVE API**: https://web-production-61829.up.railway.app

A production-ready REST API service that enables AI agents to autonomously build presentations by searching and retrieving relevant content from vector-embedded documents. Deployed on Railway with semantic search capabilities powered by OpenAI embeddings and Supabase vector database.

## 📋 Overview

The Presentation Retrieval API is designed specifically for **AI agent integration** to enable autonomous presentation creation. It provides 7 specialized endpoints for content discovery, key point extraction, slide generation, and supporting data retrieval. The service queries pre-generated document embeddings stored in Supabase PostgreSQL with pgvector extension.

### 🎯 **Perfect for AI Agents Building:**
- 📊 **Autonomous Presentations** - Complete slide deck generation
- 🔍 **Research Summaries** - Key insights from multiple documents
- 📈 **Data-Driven Reports** - Statistics and supporting evidence
- 📝 **Content Synthesis** - Multi-document analysis and compilation
- 🎨 **Presentation Templates** - Structure and format suggestions

## ✨ Key Features for AI Agents

- 🔍 **Semantic Search**: Natural language queries with OpenAI text-embedding-3-small
- 🎯 **Presentation-Focused**: 7 specialized endpoints for autonomous slide creation
- ⚡ **High Performance**: Connection pooling (5-20 connections), TTL caching (5min)
- 📊 **Content Intelligence**: Auto-extraction of key points, statistics, and citations
- 🔄 **Agent-Friendly**: RESTful design, JSON schemas, comprehensive error handling
- 🚀 **Production Ready**: Deployed on Railway, health monitoring, CORS enabled
- 🔒 **Secure Access**: User/session/project filtering, optional JWT authentication

## 🚀 Quick Start for Agents

### Test the API
```bash
# Health check
curl https://web-production-61829.up.railway.app/health

# Interactive API docs
open https://web-production-61829.up.railway.app/docs
```

### Basic Content Search
```bash
curl -X POST https://web-production-61829.up.railway.app/search-presentation-content \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "limit": 10,
    "similarity_threshold": 0.7
  }'
```

## 📡 API Endpoints Reference

**Base URL**: `https://web-production-61829.up.railway.app`

### 1. 🔍 Search Presentation Content
**Purpose**: Primary endpoint for finding relevant content for presentations

```http
POST /search-presentation-content
```

**Request:**
```json
{
  "query": "machine learning algorithms",
  "limit": 10,
  "similarity_threshold": 0.7,
  "content_types": ["key_point", "supporting_data"],
  "topic": "AI fundamentals",
  "audience_level": "intermediate",
  "user_id": "optional-user-filter",
  "metadata_filters": {"tag": "technical"}
}
```

**Response:**
```json
{
  "success": true,
  "query": "machine learning algorithms",
  "results": [
    {
      "chunk_id": "chunk_123",
      "document_id": "doc_456",
      "content": "Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning...",
      "similarity_score": 0.89,
      "metadata": {
        "page_number": 15,
        "section_title": "ML Fundamentals"
      }
    }
  ],
  "total_results": 8,
  "execution_time_ms": 145
}
```

### 2. 🎯 Extract Key Points
**Purpose**: Extract structured key points from specific documents

```http
POST /extract-key-points
```

**Request:**
```json
{
  "document_ids": ["doc_123", "doc_456"],
  "topic": "neural networks",
  "max_points": 10,
  "summarize": true
}
```

**Response:**
```json
{
  "success": true,
  "topic": "neural networks",
  "key_points": [
    {
      "point": "Neural networks consist of interconnected nodes called neurons",
      "supporting_text": "Each neuron receives inputs, processes them, and produces an output...",
      "source_document_id": "doc_123",
      "importance_score": 0.95,
      "page_number": 8,
      "tags": ["architecture", "fundamentals"]
    }
  ],
  "total_documents_processed": 2,
  "execution_time_ms": 320
}
```

### 3. 🎨 Generate Slide Suggestions
**Purpose**: AI-powered slide content generation for presentations

```http
POST /generate-slide-suggestions
```

**Request:**
```json
{
  "topic": "Introduction to Machine Learning",
  "outline": ["Definition", "Types", "Applications", "Challenges"],
  "duration_minutes": 30,
  "slide_count": 15,
  "style": "professional"
}
```

**Response:**
```json
{
  "success": true,
  "slides": [
    {
      "slide_number": 1,
      "title": "What is Machine Learning?",
      "content_type": "introduction",
      "main_content": "Machine learning is a subset of artificial intelligence...",
      "bullet_points": [
        "Subset of artificial intelligence",
        "Learns from data without explicit programming",
        "Improves performance with experience"
      ],
      "speaker_notes": "Start with a relatable example...",
      "suggested_visuals": "Brain network diagram",
      "citations": ["doc_123"]
    }
  ],
  "total_slides": 15,
  "estimated_duration_minutes": 30,
  "presentation_outline": ["What is Machine Learning?", "Types of ML", "..."],
  "execution_time_ms": 890
}
```

### 4. 📊 Find Supporting Data
**Purpose**: Discover statistics, data points, and visualization suggestions

```http
POST /find-supporting-data
```

**Request:**
```bash
curl -X POST "https://web-production-61829.up.railway.app/find-supporting-data?query=AI market growth&limit=5"
```

**Response:**
```json
{
  "success": true,
  "query": "AI market growth",
  "data_points": [
    {
      "chunk_id": "stat_789",
      "content": "The global AI market is expected to reach $1.8 trillion by 2030",
      "similarity_score": 0.91,
      "metadata": {"source": "McKinsey Report 2023"}
    }
  ],
  "statistics": [
    {"metric": "Market Size 2030", "value": "$1.8T", "source": "McKinsey"}
  ],
  "visualizations": [
    {"type": "line_chart", "title": "AI Market Growth 2020-2030"}
  ],
  "total_results": 5,
  "execution_time_ms": 210
}
```

### 5. 📚 Get User Documents
**Purpose**: List available documents for presentation use

```http
GET /user-documents/presentation-ready
```

**Response:**
```json
{
  "success": true,
  "documents": [
    {
      "document_id": "doc_123",
      "filename": "AI_Research_2023.pdf",
      "title": "Artificial Intelligence Research Trends",
      "chunk_count": 45,
      "upload_date": "2023-12-01T10:00:00Z",
      "metadata": {"tags": ["AI", "research", "2023"]}
    }
  ],
  "total_documents": 12,
  "presentation_ready_count": 12,
  "execution_time_ms": 25
}
```

### 6. 🔍 Find Presentation Examples
**Purpose**: Discover existing presentation templates and structures

```http
GET /find-presentation-examples?topic=machine learning&limit=5
```

**Response:**
```json
{
  "success": true,
  "examples": [
    {
      "title": "ML 101: A Beginner's Guide",
      "structure": ["Introduction", "Core Concepts", "Algorithms", "Applications"],
      "slide_count": 20,
      "estimated_duration": 25,
      "audience_level": "beginner",
      "key_topics": ["supervised learning", "neural networks", "applications"]
    }
  ],
  "total_examples": 5,
  "execution_time_ms": 180
}
```

### 7. 📝 Extract Citations & Sources
**Purpose**: Generate proper citations for presentation sources

```http
POST /extract-citations-sources
```

**Request:**
```json
{
  "document_ids": ["doc_123", "doc_456"],
  "citation_style": "APA",
  "include_urls": true
}
```

**Response:**
```json
{
  "success": true,
  "citations": [
    {
      "text": "Machine learning algorithms have shown remarkable progress...",
      "source": "Smith, J. (2023). AI Advances in Healthcare",
      "document_id": "doc_123",
      "page_number": 15,
      "url": "https://example.com/paper",
      "author": "Dr. Jane Smith",
      "date": "2023-03-15T00:00:00Z"
    }
  ],
  "total_citations": 8,
  "citation_style": "APA",
  "execution_time_ms": 120
}
```

### 8. ❤️ Health Check
**Purpose**: Monitor API status and dependencies

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.1",
  "database_connected": true,
  "cache_enabled": true,
  "services": {
    "openai": true,
    "database": true,
    "cache": true
  }
}
```

## 🤖 Agent Integration Examples

### Python with requests
```python
import requests
import json

class PresentationAPI:
    def __init__(self, base_url="https://web-production-61829.up.railway.app"):
        self.base_url = base_url

    def search_content(self, query, limit=10, similarity_threshold=0.7):
        """Search for presentation content"""
        response = requests.post(
            f"{self.base_url}/search-presentation-content",
            json={
                "query": query,
                "limit": limit,
                "similarity_threshold": similarity_threshold
            }
        )
        return response.json()

    def generate_slides(self, topic, outline=None, slide_count=10):
        """Generate slide suggestions"""
        response = requests.post(
            f"{self.base_url}/generate-slide-suggestions",
            json={
                "topic": topic,
                "outline": outline,
                "slide_count": slide_count,
                "style": "professional"
            }
        )
        return response.json()

    def extract_key_points(self, document_ids, topic, max_points=10):
        """Extract key points from documents"""
        response = requests.post(
            f"{self.base_url}/extract-key-points",
            json={
                "document_ids": document_ids,
                "topic": topic,
                "max_points": max_points,
                "summarize": True
            }
        )
        return response.json()

# Usage example
api = PresentationAPI()

# Step 1: Search for content
results = api.search_content("machine learning algorithms", limit=5)
print(f"Found {len(results['results'])} relevant chunks")

# Step 2: Generate slides
slides = api.generate_slides("Introduction to Machine Learning", slide_count=8)
print(f"Generated {slides['total_slides']} slides")

# Step 3: Extract key points
if results['results']:
    doc_ids = [chunk['document_id'] for chunk in results['results'][:3]]
    key_points = api.extract_key_points(doc_ids, "machine learning", max_points=5)
    print(f"Extracted {len(key_points['key_points'])} key points")
```

### LangChain Integration
```python
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
import requests

class PresentationSearchTool(BaseTool):
    name = "presentation_search"
    description = "Search for presentation content using semantic similarity"

    def _run(self, query: str, limit: int = 10) -> str:
        response = requests.post(
            "https://web-production-61829.up.railway.app/search-presentation-content",
            json={"query": query, "limit": limit}
        )
        data = response.json()

        if data["success"]:
            results = []
            for chunk in data["results"]:
                results.append(f"Content: {chunk['content'][:200]}... (Score: {chunk['similarity_score']:.2f})")
            return "\n\n".join(results)
        return "No results found"

class SlideGeneratorTool(BaseTool):
    name = "generate_slides"
    description = "Generate slide suggestions for a presentation topic"

    def _run(self, topic: str, slide_count: int = 10) -> str:
        response = requests.post(
            "https://web-production-61829.up.railway.app/generate-slide-suggestions",
            json={
                "topic": topic,
                "slide_count": slide_count,
                "style": "professional"
            }
        )
        data = response.json()

        if data["success"]:
            slides_summary = []
            for slide in data["slides"]:
                slides_summary.append(f"Slide {slide['slide_number']}: {slide['title']}")
            return f"Generated {len(data['slides'])} slides:\n" + "\n".join(slides_summary)
        return "Failed to generate slides"

# Create agent with presentation tools
tools = [PresentationSearchTool(), SlideGeneratorTool()]

prompt = PromptTemplate.from_template("""
You are a presentation creation assistant. Use the available tools to help create presentations.

Tools available:
{tools}

Question: {input}
{agent_scratchpad}
""")

# agent = create_react_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### JavaScript/Node.js Integration
```javascript
class PresentationAPI {
    constructor(baseUrl = 'https://web-production-61829.up.railway.app') {
        this.baseUrl = baseUrl;
    }

    async searchContent(query, limit = 10, similarityThreshold = 0.7) {
        const response = await fetch(`${this.baseUrl}/search-presentation-content`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                limit,
                similarity_threshold: similarityThreshold
            })
        });
        return response.json();
    }

    async generateSlides(topic, outline = null, slideCount = 10) {
        const response = await fetch(`${this.baseUrl}/generate-slide-suggestions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                topic,
                outline,
                slide_count: slideCount,
                style: 'professional'
            })
        });
        return response.json();
    }

    async findSupportingData(query, limit = 5) {
        const response = await fetch(
            `${this.baseUrl}/find-supporting-data?query=${encodeURIComponent(query)}&limit=${limit}`,
            { method: 'POST' }
        );
        return response.json();
    }
}

// Usage example
const api = new PresentationAPI();

async function createPresentation(topic) {
    try {
        // Step 1: Search for content
        const contentResults = await api.searchContent(topic, 10);
        console.log(`Found ${contentResults.total_results} relevant pieces of content`);

        // Step 2: Generate slide structure
        const slides = await api.generateSlides(topic, null, 12);
        console.log(`Generated ${slides.total_slides} slides`);

        // Step 3: Find supporting data
        const supportingData = await api.findSupportingData(topic);
        console.log(`Found ${supportingData.total_results} supporting data points`);

        return {
            content: contentResults.results,
            slides: slides.slides,
            data: supportingData.data_points
        };
    } catch (error) {
        console.error('Error creating presentation:', error);
    }
}

// Create a presentation about AI
createPresentation('artificial intelligence trends').then(presentation => {
    console.log('Presentation created successfully!');
});
```

## 🔄 Autonomous Presentation Builder Workflow

Here's a complete workflow for an AI agent to autonomously build a presentation:

### Step 1: Topic Analysis & Content Discovery
```python
def discover_content(topic, subtopics=None):
    """Discover relevant content for the presentation"""
    api = PresentationAPI()

    # Main topic search
    main_results = api.search_content(topic, limit=15, similarity_threshold=0.75)

    # Subtopic searches if provided
    subtopic_results = []
    if subtopics:
        for subtopic in subtopics:
            results = api.search_content(f"{topic} {subtopic}", limit=8)
            subtopic_results.extend(results['results'])

    return {
        'main_content': main_results['results'],
        'subtopic_content': subtopic_results,
        'total_chunks': len(main_results['results']) + len(subtopic_results)
    }
```

### Step 2: Structure Planning
```python
def plan_presentation_structure(topic, duration_minutes=30):
    """Plan the presentation structure and slide count"""
    api = PresentationAPI()

    # Estimate slide count (2-3 minutes per slide)
    slide_count = max(8, min(20, duration_minutes // 2))

    # Generate initial structure
    slides = api.generate_slides(
        topic=topic,
        slide_count=slide_count,
        duration_minutes=duration_minutes
    )

    # Get presentation examples for inspiration
    examples_response = requests.get(
        f"https://web-production-61829.up.railway.app/find-presentation-examples",
        params={"topic": topic, "limit": 3}
    )
    examples = examples_response.json()

    return {
        'slides': slides['slides'],
        'outline': slides['presentation_outline'],
        'estimated_duration': slides['estimated_duration_minutes'],
        'examples': examples.get('examples', [])
    }
```

### Step 3: Content Population
```python
def populate_slide_content(slides, discovered_content):
    """Populate slides with discovered content"""
    api = PresentationAPI()
    enriched_slides = []

    for slide in slides:
        # Find most relevant content for this slide
        slide_query = f"{slide['title']} {slide['main_content'][:100]}"
        relevant_content = api.search_content(slide_query, limit=3)

        # Get supporting data if needed
        if slide['content_type'] in ['supporting_data', 'key_point']:
            supporting_data = requests.post(
                "https://web-production-61829.up.railway.app/find-supporting-data",
                params={"query": slide['title'], "limit": 2}
            ).json()
            slide['supporting_data'] = supporting_data.get('data_points', [])

        # Add relevant content to slide
        slide['source_content'] = relevant_content['results']
        enriched_slides.append(slide)

    return enriched_slides
```

### Step 4: Citation & Source Management
```python
def add_citations(slides, source_documents):
    """Add proper citations to all slides"""
    api = PresentationAPI()

    # Extract citations from all source documents
    doc_ids = list(set([doc['document_id'] for doc in source_documents]))
    citations_response = requests.post(
        "https://web-production-61829.up.railway.app/extract-citations-sources",
        json={
            "document_ids": doc_ids,
            "citation_style": "APA",
            "include_urls": True
        }
    )
    citations = citations_response.json()

    # Map citations to slides
    for slide in slides:
        slide_citations = []
        for source in slide.get('source_content', []):
            doc_id = source['document_id']
            # Find matching citations
            matching_citations = [
                c for c in citations['citations']
                if c['document_id'] == doc_id
            ]
            slide_citations.extend(matching_citations)
        slide['citations'] = slide_citations

    return slides
```

### Step 5: Complete Autonomous Workflow
```python
def create_autonomous_presentation(topic, duration_minutes=30, subtopics=None):
    """Complete autonomous presentation creation workflow"""

    print(f"🎯 Creating presentation: {topic}")

    # Step 1: Discover content
    print("🔍 Discovering content...")
    content = discover_content(topic, subtopics)
    print(f"   Found {content['total_chunks']} relevant content pieces")

    # Step 2: Plan structure
    print("📋 Planning structure...")
    structure = plan_presentation_structure(topic, duration_minutes)
    print(f"   Created {len(structure['slides'])} slides")

    # Step 3: Populate content
    print("📝 Populating slide content...")
    populated_slides = populate_slide_content(
        structure['slides'],
        content['main_content'] + content['subtopic_content']
    )

    # Step 4: Add citations
    print("📚 Adding citations...")
    final_slides = add_citations(
        populated_slides,
        content['main_content'] + content['subtopic_content']
    )

    print("✅ Presentation created successfully!")

    return {
        'title': topic,
        'slides': final_slides,
        'outline': structure['outline'],
        'duration': structure['estimated_duration'],
        'source_count': content['total_chunks'],
        'citation_count': sum(len(s.get('citations', [])) for s in final_slides)
    }

# Example usage
presentation = create_autonomous_presentation(
    topic="The Future of Artificial Intelligence",
    duration_minutes=25,
    subtopics=["machine learning", "neural networks", "AI ethics", "automation"]
)

print(f"Created presentation with {len(presentation['slides'])} slides")
print(f"Estimated duration: {presentation['duration']} minutes")
print(f"Sources cited: {presentation['citation_count']}")
```

## 🧪 Testing & Development

### Interactive API Testing
```bash
# 1. Health Check
curl https://web-production-61829.up.railway.app/health

# 2. Test Content Search
curl -X POST https://web-production-61829.up.railway.app/search-presentation-content \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "limit": 5,
    "similarity_threshold": 0.7
  }'

# 3. Test Slide Generation
curl -X POST https://web-production-61829.up.railway.app/generate-slide-suggestions \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Introduction to AI",
    "slide_count": 8,
    "duration_minutes": 20
  }'
```

### API Documentation & Explorer
- **Swagger UI**: https://web-production-61829.up.railway.app/docs
- **ReDoc**: https://web-production-61829.up.railway.app/redoc
- **OpenAPI JSON**: https://web-production-61829.up.railway.app/openapi.json

### Response Times & Performance
- **Search queries**: 100-300ms (depending on result count)
- **Slide generation**: 500-1000ms (depending on complexity)
- **Key points extraction**: 200-500ms
- **Health check**: <50ms

### Error Handling Examples
```python
import requests

def handle_api_errors(response):
    """Example error handling for the API"""
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            return data
        else:
            print(f"API Error: {data.get('error', 'Unknown error')}")
    elif response.status_code == 422:
        print("Validation Error:", response.json())
    elif response.status_code == 500:
        print("Server Error:", response.text)
    else:
        print(f"HTTP {response.status_code}: {response.text}")
    return None

# Example usage
response = requests.post(
    "https://web-production-61829.up.railway.app/search-presentation-content",
    json={"query": "test", "limit": 5}
)
result = handle_api_errors(response)
```

### Rate Limiting & Best Practices
```python
import time
import requests
from functools import wraps

def rate_limit(calls_per_second=5):
    """Simple rate limiting decorator"""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@rate_limit(calls_per_second=3)  # 3 calls per second
def search_content(query):
    response = requests.post(
        "https://web-production-61829.up.railway.app/search-presentation-content",
        json={"query": query, "limit": 10}
    )
    return response.json()
```

### Client Libraries & SDKs

#### Python SDK Example
```python
# presentation_api_client.py
import requests
from typing import List, Dict, Optional
import logging

class PresentationAPIClient:
    def __init__(self, base_url: str = "https://web-production-61829.up.railway.app"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise

    def search_content(self, query: str, **kwargs) -> Dict:
        return self._make_request(
            'POST', '/search-presentation-content',
            json={'query': query, **kwargs}
        )

    def generate_slides(self, topic: str, **kwargs) -> Dict:
        return self._make_request(
            'POST', '/generate-slide-suggestions',
            json={'topic': topic, **kwargs}
        )

    def extract_key_points(self, document_ids: List[str], topic: str, **kwargs) -> Dict:
        return self._make_request(
            'POST', '/extract-key-points',
            json={'document_ids': document_ids, 'topic': topic, **kwargs}
        )

    def health_check(self) -> Dict:
        return self._make_request('GET', '/health')
```

#### TypeScript SDK Example
```typescript
// presentation-api-client.ts
interface SearchRequest {
  query: string;
  limit?: number;
  similarity_threshold?: number;
  content_types?: string[];
  user_id?: string;
}

interface SlideRequest {
  topic: string;
  outline?: string[];
  slide_count?: number;
  duration_minutes?: number;
}

class PresentationAPIClient {
  private baseUrl: string;

  constructor(baseUrl = 'https://web-production-61829.up.railway.app') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  private async makeRequest<T>(
    method: string,
    endpoint: string,
    data?: any
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const config: RequestInit = {
      method,
      headers: { 'Content-Type': 'application/json' },
    };

    if (data) {
      config.body = JSON.stringify(data);
    }

    const response = await fetch(url, config);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    return response.json();
  }

  async searchContent(request: SearchRequest) {
    return this.makeRequest('POST', '/search-presentation-content', request);
  }

  async generateSlides(request: SlideRequest) {
    return this.makeRequest('POST', '/generate-slide-suggestions', request);
  }

  async healthCheck() {
    return this.makeRequest('GET', '/health');
  }
}
```

## 🔧 Configuration & Environment Variables

### Required Environment Variables
```env
# Database Connection (Required)
DATABASE_URL=postgresql://postgres.[project]:[password]@[host]:6543/postgres

# OpenAI Configuration (Required)
OPENAI_API_KEY=sk-proj-...

# Supabase Configuration (Optional)
SUPABASE_URL=https://[project].supabase.co
SUPABASE_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Optional Configuration
```env
# Model Settings
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DIMENSIONS=1536

# Performance Settings
DB_POOL_MIN_SIZE=5
DB_POOL_MAX_SIZE=20
CACHE_TTL=300
CACHE_MAX_SIZE=1000

# Query Settings
MAX_RESULTS_LIMIT=20
DEFAULT_SIMILARITY_THRESHOLD=0.7

# Logging
LOG_LEVEL=INFO
```

### Authentication Configuration
```env
# JWT Settings (Optional)
JWT_SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
REQUIRE_AUTH=false  # Set to true to enforce authentication
```

## Prerequisites

- Python 3.10 or higher
- Supabase project with pgvector extension enabled
- Existing document embeddings in Supabase (generated using OpenAI text-embedding-3-small)
- OpenAI API key for query embedding generation
- Supabase service role key

## Database Schema

The server expects the following tables in your Supabase database:

```sql
-- Documents table
CREATE TABLE documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL,
    session_id text NOT NULL,
    project_id text DEFAULT '-',
    filename text NOT NULL,
    file_type text NOT NULL,
    file_size integer,
    total_chunks integer,
    upload_date timestamp with time zone DEFAULT now(),
    processing_status text DEFAULT 'completed',
    metadata jsonb
);

-- Document embeddings table
CREATE TABLE document_embeddings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
    user_id uuid NOT NULL,
    session_id text NOT NULL,
    project_id text DEFAULT '-',
    chunk_text text NOT NULL,
    embedding vector(1536), -- OpenAI text-embedding-3-small
    chunk_index integer NOT NULL,
    chunk_metadata jsonb,
    created_at timestamp with time zone DEFAULT now()
);

-- Create indexes for performance
CREATE INDEX idx_embeddings_vector ON document_embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_embeddings_user_session ON document_embeddings(user_id, session_id);
CREATE INDEX idx_documents_user_session ON documents(user_id, session_id);
```

## Installation

### Option 1: Install from source

```bash
# Clone the repository
git clone https://github.com/your-org/document-retrieval-mcp
cd document-retrieval-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Install as package

```bash
pip install document-retrieval-mcp
```

## Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your configuration:
```env
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_API_KEY=your-service-role-key
OPENAI_API_KEY=your-openai-api-key

# Optional - customize as needed
EMBEDDING_MODEL=text-embedding-3-small
MAX_RESULTS_LIMIT=20
DEFAULT_SIMILARITY_THRESHOLD=0.7
```

## Usage

### Running the Server

```bash
# Direct execution
python src/server.py

# Or if installed as package
document-retrieval-mcp
```

### Claude Desktop Integration

Add the following to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcp_servers": {
    "document-retrieval": {
      "command": "python",
      "args": ["/path/to/document-retrieval-mcp/src/server.py"],
      "env": {
        "SUPABASE_URL": "https://your-project.supabase.co",
        "SUPABASE_API_KEY": "your-service-role-key",
        "OPENAI_API_KEY": "your-openai-api-key"
      }
    }
  }
}
```

## Available Tools

### 1. `search_documents`
Search for documents using semantic similarity.

**Parameters:**
- `query` (string, required): Search query text
- `user_id` (string, required): User identifier
- `session_id` (string, required): Session identifier
- `project_id` (string, optional): Project filter (default: "-")
- `top_k` (integer, optional): Number of results (default: 5, max: 20)
- `similarity_threshold` (float, optional): Minimum similarity (default: 0.7)

**Example:**
```json
{
  "tool": "search_documents",
  "params": {
    "query": "machine learning algorithms",
    "user_id": "user-123",
    "session_id": "session-456",
    "top_k": 10
  }
}
```

### 2. `get_document_context`
Retrieve full document or specific chunks.

**Parameters:**
- `document_id` (string, required): Document UUID
- `user_id` (string, required): User identifier
- `session_id` (string, required): Session identifier
- `chunk_ids` (array, optional): Specific chunk UUIDs to retrieve

**Example:**
```json
{
  "tool": "get_document_context",
  "params": {
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "user-123",
    "session_id": "session-456"
  }
}
```

### 3. `list_user_documents`
List all documents accessible to the user.

**Parameters:**
- `user_id` (string, required): User identifier
- `session_id` (string, required): Session identifier
- `project_id` (string, optional): Project filter
- `page` (integer, optional): Page number (default: 1)
- `per_page` (integer, optional): Items per page (default: 20, max: 100)

**Example:**
```json
{
  "tool": "list_user_documents",
  "params": {
    "user_id": "user-123",
    "session_id": "session-456",
    "page": 1,
    "per_page": 50
  }
}
```

### 4. `get_similar_chunks`
Find chunks similar to a reference chunk.

**Parameters:**
- `chunk_id` (string, required): Reference chunk UUID
- `user_id` (string, required): User identifier
- `session_id` (string, required): Session identifier
- `top_k` (integer, optional): Number of results (default: 3, max: 10)

**Example:**
```json
{
  "tool": "get_similar_chunks",
  "params": {
    "chunk_id": "chunk-789",
    "user_id": "user-123",
    "session_id": "session-456",
    "top_k": 5
  }
}
```

## Resources

The server provides two informational resources:

### 1. `resource://server-info`
Returns current server status and configuration.

### 2. `resource://schema-info`
Returns database schema information.

## Usage Examples with Claude

After configuring the server in Claude Desktop, you can use natural language:

```
User: "Search for documents about machine learning algorithms"
Claude: I'll search for documents about machine learning algorithms.
[Uses search_documents tool]

User: "Show me the full content of document ABC"
Claude: I'll retrieve the complete content of that document.
[Uses get_document_context tool]

User: "What documents do I have access to?"
Claude: Let me list all your available documents.
[Uses list_user_documents tool]

User: "Find similar content to this chunk"
Claude: I'll find chunks with similar content.
[Uses get_similar_chunks tool]
```

## Performance Optimization

The server implements several optimization strategies:

1. **Connection Pooling**: Maintains 5-20 database connections
2. **TTL Caching**: Caches metadata for 5 minutes
3. **Vector Indexes**: Uses IVFFlat indexes for fast similarity search
4. **Query Optimization**: Efficient SQL with proper WHERE clauses
5. **Async Operations**: Full async/await for concurrent requests

## Security

- **Service Role Key**: Bypasses RLS for performance
- **Application-Level Security**: WHERE clauses filter by user/session/project
- **Input Validation**: JSON Schema validation on all parameters
- **Error Sanitization**: No sensitive data in error messages
- **Environment Variables**: Secrets never hardcoded

## Troubleshooting

### Common Issues

1. **"Missing required environment variables"**
   - Ensure all required variables are set in `.env` or environment

2. **"Connection pool timeout"**
   - Check Supabase URL and API key
   - Verify network connectivity

3. **"No results above similarity threshold"**
   - Lower the similarity_threshold parameter
   - Ensure embeddings exist for the user/session

4. **"Document not found or access denied"**
   - Verify user_id and session_id match existing records
   - Check document_id is valid

### Logging

Enable debug logging by setting:
```bash
export LOG_LEVEL=DEBUG
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# With coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src tests

# Lint
ruff check src tests

# Type checking
mypy src
```

## Architecture

The server follows a layered architecture:

```
┌─────────────────┐
│   MCP Client    │
│ (Claude Desktop)│
└────────┬────────┘
         │ JSON-RPC
┌────────▼────────┐
│   MCP Server    │
│  (Protocol Layer)│
└────────┬────────┘
         │
┌────────▼────────┐
│  Business Logic │
│  (Tools & Cache)│
└────────┬────────┘
         │
┌────────▼────────┐
│  Data Access    │
│ (AsyncPG + OAI) │
└────────┬────────┘
         │
┌────────▼────────┐
│    Supabase     │
│  (PostgreSQL +  │
│    pgvector)    │
└─────────────────┘
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: [https://github.com/your-org/document-retrieval-mcp/issues](https://github.com/your-org/document-retrieval-mcp/issues)
- Documentation: [https://docs.your-org.com/mcp/document-retrieval](https://docs.your-org.com/mcp/document-retrieval)

## Acknowledgments

- Built with the [Model Context Protocol](https://modelcontextprotocol.io)
- Powered by [Supabase](https://supabase.com) and [pgvector](https://github.com/pgvector/pgvector)
- Embeddings by [OpenAI](https://openai.com)