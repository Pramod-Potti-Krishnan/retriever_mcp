"""
Service layer for Presentation Retrieval API
"""
import os
import json
import asyncio
import asyncpg
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import openai
from openai import AsyncOpenAI
import numpy as np
from cachetools import TTLCache
import hashlib

from models import (
    DocumentChunk, KeyPoint, SlideContent, Citation,
    ContentType, PresentationExample
)

logger = logging.getLogger(__name__)


class PresentationRetrievalService:
    """Service for retrieving and processing presentation content"""

    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.vector_dimensions = int(os.getenv("VECTOR_DIMENSIONS", "1536"))

        # Connection pool
        self.pool = None
        self.pool_min_size = int(os.getenv("DB_POOL_MIN_SIZE", "5"))
        self.pool_max_size = int(os.getenv("DB_POOL_MAX_SIZE", "20"))

        # Cache configuration
        cache_ttl = int(os.getenv("CACHE_TTL", "300"))
        cache_max_size = int(os.getenv("CACHE_MAX_SIZE", "1000"))
        self.cache = TTLCache(maxsize=cache_max_size, ttl=cache_ttl)

        # Query settings
        self.max_results_limit = int(os.getenv("MAX_RESULTS_LIMIT", "20"))
        self.default_similarity_threshold = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.7"))

    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.db_url,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                command_timeout=60
            )
            logger.info(f"Database pool created: min={self.pool_min_size}, max={self.pool_max_size}")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

    def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters"""
        key_data = json.dumps(kwargs, sort_keys=True)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        cache_key = self._get_cache_key("embedding", text=text)

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            self.cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def search_presentation_content(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        content_types: Optional[List[ContentType]] = None,
        user_id: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Search for presentation-relevant content using vector similarity"""

        # Generate query embedding
        query_embedding = await self._get_embedding(query)

        # Build SQL query
        sql = """
        SELECT
            id,
            document_id,
            content,
            metadata,
            1 - (embedding <=> $1::vector) as similarity_score
        FROM document_chunks
        WHERE 1 - (embedding <=> $1::vector) > $2
        """

        params = [query_embedding, similarity_threshold]
        param_count = 2

        # Add optional filters
        if user_id:
            param_count += 1
            sql += f" AND metadata->>'user_id' = ${param_count}"
            params.append(user_id)

        if content_types:
            param_count += 1
            types_list = [ct.value for ct in content_types]
            sql += f" AND metadata->>'content_type' = ANY(${param_count})"
            params.append(types_list)

        if metadata_filters:
            for key, value in metadata_filters.items():
                param_count += 1
                sql += f" AND metadata->>'{key}' = ${param_count}"
                params.append(str(value))

        sql += f" ORDER BY similarity_score DESC LIMIT ${param_count + 1}"
        params.append(limit)

        # Execute query
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        # Convert to DocumentChunk objects
        chunks = []
        for row in rows:
            chunk = DocumentChunk(
                chunk_id=str(row['id']),
                document_id=row['document_id'],
                content=row['content'],
                metadata=json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata'],
                similarity_score=row['similarity_score'],
                page_number=row['metadata'].get('page_number'),
                section_title=row['metadata'].get('section_title')
            )
            chunks.append(chunk)

        return chunks

    async def extract_key_points(
        self,
        document_ids: List[str],
        topic: str,
        max_points: int = 10,
        summarize: bool = True
    ) -> List[KeyPoint]:
        """Extract key points from specified documents"""

        # Fetch document chunks
        sql = """
        SELECT id, document_id, content, metadata
        FROM document_chunks
        WHERE document_id = ANY($1)
        ORDER BY document_id, COALESCE((metadata->>'page_number')::int, 0)
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, document_ids)

        if not rows:
            return []

        # Group content by document
        documents = {}
        for row in rows:
            doc_id = row['document_id']
            if doc_id not in documents:
                documents[doc_id] = []
            documents[doc_id].append(row['content'])

        # Extract key points using LLM
        key_points = []
        for doc_id, contents in documents.items():
            combined_content = "\n\n".join(contents[:10])  # Limit content length

            prompt = f"""
            Extract the {max_points} most important key points about "{topic}" from this document.
            For each key point:
            1. Provide a clear, concise statement
            2. Include supporting text from the document
            3. Rate its importance (0.0-1.0)

            Document content:
            {combined_content[:4000]}

            Return as JSON array with fields: point, supporting_text, importance_score
            """

            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )

                points_data = json.loads(response.choices[0].message.content)
                for point_data in points_data.get('points', [])[:max_points]:
                    key_point = KeyPoint(
                        point=point_data['point'],
                        supporting_text=point_data['supporting_text'],
                        source_document_id=doc_id,
                        importance_score=point_data['importance_score'],
                        tags=[topic]
                    )
                    key_points.append(key_point)
            except Exception as e:
                logger.error(f"Failed to extract key points: {e}")

        # Sort by importance
        key_points.sort(key=lambda x: x.importance_score, reverse=True)

        return key_points[:max_points]

    async def find_supporting_data(
        self,
        query: str,
        limit: int = 10
    ) -> tuple[List[DocumentChunk], Optional[List[Dict]], Optional[List[Dict]]]:
        """Find supporting data, statistics, and visualization suggestions"""

        # Search for data-rich content
        chunks = await self.search_presentation_content(
            query=query,
            limit=limit,
            content_types=[ContentType.SUPPORTING_DATA, ContentType.EXAMPLE]
        )

        # Extract statistics and data points
        statistics = []
        visualizations = []

        for chunk in chunks:
            # Look for numeric data in content
            import re
            numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', chunk.content)
            if numbers:
                statistics.append({
                    "source": chunk.document_id,
                    "data": numbers,
                    "context": chunk.content[:200]
                })

            # Suggest visualizations based on content
            if any(word in chunk.content.lower() for word in ['trend', 'growth', 'increase', 'decrease']):
                visualizations.append({
                    "type": "line_chart",
                    "description": "Time series showing trends",
                    "source": chunk.document_id
                })
            elif any(word in chunk.content.lower() for word in ['comparison', 'versus', 'compared']):
                visualizations.append({
                    "type": "bar_chart",
                    "description": "Comparison between categories",
                    "source": chunk.document_id
                })

        return chunks, statistics[:5], visualizations[:3]

    async def get_user_documents(
        self,
        user_id: str,
        presentation_ready: bool = False
    ) -> List[Dict[str, Any]]:
        """Get user's documents, optionally filtered for presentation readiness"""

        sql = """
        SELECT DISTINCT
            document_id,
            metadata->>'title' as title,
            metadata->>'description' as description,
            metadata->>'document_type' as document_type,
            metadata->>'created_at' as created_at,
            COUNT(*) as chunk_count
        FROM document_chunks
        WHERE metadata->>'user_id' = $1
        """

        if presentation_ready:
            sql += " AND metadata->>'presentation_ready' = 'true'"

        sql += " GROUP BY document_id, metadata ORDER BY created_at DESC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, user_id)

        documents = []
        for row in rows:
            documents.append({
                "document_id": row['document_id'],
                "title": row['title'] or "Untitled",
                "description": row['description'],
                "document_type": row['document_type'],
                "created_at": row['created_at'],
                "chunk_count": row['chunk_count'],
                "presentation_ready": presentation_ready
            })

        return documents

    async def generate_slide_suggestions(
        self,
        topic: str,
        outline: Optional[List[str]] = None,
        duration_minutes: Optional[int] = None,
        slide_count: Optional[int] = None
    ) -> List[SlideContent]:
        """Generate slide content suggestions for a presentation"""

        # Determine slide count based on duration
        if not slide_count and duration_minutes:
            slide_count = max(5, min(duration_minutes // 2, 30))  # 2 minutes per slide average
        elif not slide_count:
            slide_count = 10  # Default

        # Search for relevant content
        chunks = await self.search_presentation_content(query=topic, limit=20)

        # Generate presentation structure
        slides = []

        # Title slide
        slides.append(SlideContent(
            slide_number=1,
            title=topic,
            content_type=ContentType.INTRODUCTION,
            main_content=f"Presentation on {topic}",
            bullet_points=[],
            speaker_notes=f"Welcome everyone. Today we'll be discussing {topic}."
        ))

        # Generate content slides based on outline or discovered content
        if outline:
            for i, section in enumerate(outline[:slide_count-2], start=2):
                # Find relevant content for this section
                section_chunks = [c for c in chunks if section.lower() in c.content.lower()]

                slide = SlideContent(
                    slide_number=i,
                    title=section,
                    content_type=ContentType.KEY_POINT,
                    main_content=section_chunks[0].content[:200] if section_chunks else section,
                    bullet_points=self._extract_bullet_points(section_chunks[:3]) if section_chunks else [],
                    speaker_notes=f"Discuss {section} in detail.",
                    citations=[c.document_id for c in section_chunks[:2]]
                )
                slides.append(slide)
        else:
            # Auto-generate slides from discovered content
            key_topics = await self._discover_key_topics(chunks, slide_count - 2)
            for i, topic_data in enumerate(key_topics, start=2):
                slide = SlideContent(
                    slide_number=i,
                    title=topic_data['title'],
                    content_type=ContentType.KEY_POINT,
                    main_content=topic_data['content'],
                    bullet_points=topic_data['bullets'],
                    speaker_notes=topic_data['notes'],
                    suggested_visuals=topic_data.get('visual'),
                    citations=topic_data.get('citations', [])
                )
                slides.append(slide)

        # Conclusion slide
        slides.append(SlideContent(
            slide_number=len(slides) + 1,
            title="Conclusion",
            content_type=ContentType.CONCLUSION,
            main_content=f"Key takeaways from our discussion on {topic}",
            bullet_points=self._generate_summary_points(slides[1:-1]),
            speaker_notes="Summarize the main points and open for questions."
        ))

        return slides

    async def find_presentation_examples(
        self,
        topic: str,
        limit: int = 5
    ) -> List[PresentationExample]:
        """Find example presentation structures for a topic"""

        # Search for presentation templates and examples
        chunks = await self.search_presentation_content(
            query=f"{topic} presentation example template structure",
            limit=20
        )

        # Generate example presentations
        examples = []

        # Create a basic example
        basic_slides = await self.generate_slide_suggestions(
            topic=topic,
            slide_count=8
        )

        examples.append(PresentationExample(
            title=f"Basic {topic} Presentation",
            description="A simple, straightforward presentation structure",
            slides=basic_slides,
            total_slides=len(basic_slides),
            estimated_duration_minutes=len(basic_slides) * 2,
            tags=["basic", "simple", topic.lower()]
        ))

        # Create a detailed example
        detailed_slides = await self.generate_slide_suggestions(
            topic=topic,
            slide_count=15
        )

        examples.append(PresentationExample(
            title=f"Comprehensive {topic} Analysis",
            description="An in-depth exploration with detailed analysis",
            slides=detailed_slides,
            total_slides=len(detailed_slides),
            estimated_duration_minutes=len(detailed_slides) * 3,
            tags=["detailed", "comprehensive", topic.lower()]
        ))

        return examples[:limit]

    async def extract_citations_sources(
        self,
        document_ids: List[str]
    ) -> List[Citation]:
        """Extract citations and sources from documents"""

        sql = """
        SELECT document_id, content, metadata
        FROM document_chunks
        WHERE document_id = ANY($1)
        AND (
            content ~* r'\(\d{4}\)' OR
            content ~* r'et al\.' OR
            content ~* r'\[\d+\]' OR
            metadata->>'has_citations' = 'true'
        )
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, document_ids)

        citations = []
        for row in rows:
            # Extract citation patterns
            import re

            # Pattern for academic citations (Author, Year)
            academic_pattern = r'([A-Z][a-z]+ (?:et al\.|and [A-Z][a-z]+), \d{4})'
            matches = re.findall(academic_pattern, row['content'])

            for match in matches:
                citations.append(Citation(
                    text=match,
                    source=row['metadata'].get('source', 'Unknown'),
                    document_id=row['document_id'],
                    page_number=row['metadata'].get('page_number')
                ))

            # Pattern for numbered references [1], [2], etc.
            ref_pattern = r'\[(\d+)\]'
            ref_matches = re.findall(ref_pattern, row['content'])

            for ref_num in ref_matches:
                citations.append(Citation(
                    text=f"Reference [{ref_num}]",
                    source=row['metadata'].get('source', 'Document'),
                    document_id=row['document_id'],
                    page_number=row['metadata'].get('page_number')
                ))

        # Deduplicate citations
        unique_citations = []
        seen = set()
        for citation in citations:
            key = (citation.text, citation.document_id)
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)

        return unique_citations

    def _extract_bullet_points(self, chunks: List[DocumentChunk]) -> List[str]:
        """Extract bullet points from document chunks"""
        bullets = []
        for chunk in chunks:
            # Look for list items in content
            lines = chunk.content.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    bullets.append(line.lstrip('•-* '))
                elif len(bullets) < 3 and len(line) < 100 and line:
                    bullets.append(line)

        return bullets[:5]

    async def _discover_key_topics(self, chunks: List[DocumentChunk], count: int) -> List[Dict]:
        """Discover key topics from document chunks"""
        topics = []

        # Simple clustering by content similarity
        for i, chunk in enumerate(chunks[:count]):
            topic = {
                'title': self._extract_title_from_content(chunk.content),
                'content': chunk.content[:200],
                'bullets': self._extract_bullet_points([chunk]),
                'notes': f"Expand on this topic using the provided content.",
                'citations': [chunk.document_id]
            }

            # Suggest visuals based on content
            if any(word in chunk.content.lower() for word in ['data', 'statistics', 'numbers']):
                topic['visual'] = "Chart or graph recommended"
            elif any(word in chunk.content.lower() for word in ['process', 'flow', 'steps']):
                topic['visual'] = "Flowchart or diagram recommended"

            topics.append(topic)

        return topics

    def _extract_title_from_content(self, content: str) -> str:
        """Extract a suitable title from content"""
        # Take first sentence or first 50 characters
        sentences = content.split('.')
        if sentences:
            title = sentences[0].strip()
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        return content[:50] + "..."

    def _generate_summary_points(self, slides: List[SlideContent]) -> List[str]:
        """Generate summary points from slides"""
        points = []
        for slide in slides:
            if slide.content_type == ContentType.KEY_POINT:
                # Create a summary point from the slide title
                points.append(f"Discussed {slide.title}")

        return points[:5]