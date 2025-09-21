"""
Pydantic models for Presentation Retrieval API
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ContentType(str, Enum):
    """Types of content for presentations"""
    INTRODUCTION = "introduction"
    KEY_POINT = "key_point"
    SUPPORTING_DATA = "supporting_data"
    EXAMPLE = "example"
    CONCLUSION = "conclusion"
    TRANSITION = "transition"
    CITATION = "citation"


class SearchRequest(BaseModel):
    """Base search request model"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Additional metadata filters")


class PresentationSearchRequest(SearchRequest):
    """Search specifically for presentation content"""
    content_types: Optional[List[ContentType]] = Field(None, description="Filter by content types")
    topic: Optional[str] = Field(None, description="Specific topic to focus on")
    audience_level: Optional[str] = Field(None, description="Target audience level (beginner, intermediate, expert)")


class KeyPointsRequest(BaseModel):
    """Request for extracting key points"""
    document_ids: List[str] = Field(..., min_items=1, description="Document IDs to extract from")
    topic: str = Field(..., description="Topic to focus on")
    max_points: int = Field(10, ge=1, le=20, description="Maximum key points to extract")
    summarize: bool = Field(True, description="Whether to summarize the key points")


class SlideGenerationRequest(BaseModel):
    """Request for generating slide suggestions"""
    topic: str = Field(..., description="Main topic of the presentation")
    outline: Optional[List[str]] = Field(None, description="Presentation outline")
    duration_minutes: Optional[int] = Field(None, ge=5, le=120, description="Presentation duration")
    slide_count: Optional[int] = Field(None, ge=1, le=100, description="Target number of slides")
    style: Optional[str] = Field("professional", description="Presentation style")


class DocumentChunk(BaseModel):
    """A chunk of document content"""
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None


class KeyPoint(BaseModel):
    """A key point extracted from documents"""
    point: str
    supporting_text: str
    source_document_id: str
    importance_score: float = Field(..., ge=0.0, le=1.0)
    page_number: Optional[int] = None
    tags: List[str] = []


class SlideContent(BaseModel):
    """Content for a single slide"""
    slide_number: int
    title: str
    content_type: ContentType
    main_content: str
    bullet_points: List[str] = []
    speaker_notes: Optional[str] = None
    suggested_visuals: Optional[str] = None
    citations: List[str] = []


class Citation(BaseModel):
    """Citation information"""
    text: str
    source: str
    document_id: str
    page_number: Optional[int] = None
    url: Optional[str] = None
    author: Optional[str] = None
    date: Optional[datetime] = None


class PresentationExample(BaseModel):
    """Example presentation structure"""
    title: str
    description: str
    slides: List[SlideContent]
    total_slides: int
    estimated_duration_minutes: int
    tags: List[str] = []


class PresentationSearchResponse(BaseModel):
    """Response for presentation content search"""
    success: bool
    query: str
    results: List[DocumentChunk]
    total_results: int
    execution_time_ms: float


class KeyPointsResponse(BaseModel):
    """Response for key points extraction"""
    success: bool
    topic: str
    key_points: List[KeyPoint]
    total_documents_processed: int
    execution_time_ms: float


class SlideGenerationResponse(BaseModel):
    """Response for slide generation"""
    success: bool
    slides: List[SlideContent]
    total_slides: int
    estimated_duration_minutes: int
    presentation_outline: List[str]
    execution_time_ms: float


class CitationsResponse(BaseModel):
    """Response for citations extraction"""
    success: bool
    citations: List[Citation]
    total_citations: int
    execution_time_ms: float


class UserDocumentsResponse(BaseModel):
    """Response for user documents listing"""
    success: bool
    documents: List[Dict[str, Any]]
    total_documents: int
    presentation_ready_count: int
    execution_time_ms: float


class PresentationExamplesResponse(BaseModel):
    """Response for presentation examples"""
    success: bool
    examples: List[PresentationExample]
    total_examples: int
    execution_time_ms: float


class SupportingDataResponse(BaseModel):
    """Response for finding supporting data"""
    success: bool
    query: str
    data_points: List[DocumentChunk]
    statistics: Optional[List[Dict[str, Any]]] = None
    visualizations: Optional[List[Dict[str, Any]]] = None
    total_results: int
    execution_time_ms: float


class ErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = False
    error: str
    error_code: str
    details: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    database_connected: bool
    cache_enabled: bool
    services: Dict[str, bool]