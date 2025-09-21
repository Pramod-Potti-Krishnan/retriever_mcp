"""
Presentation Retrieval API - FastAPI Application
Optimized for autonomous presentation building
"""
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import jwt
from jwt import PyJWTError

from models import (
    PresentationSearchRequest, PresentationSearchResponse,
    KeyPointsRequest, KeyPointsResponse,
    SlideGenerationRequest, SlideGenerationResponse,
    CitationsResponse, UserDocumentsResponse,
    PresentationExamplesResponse, SupportingDataResponse,
    ErrorResponse, HealthCheckResponse, ContentType
)
from services import PresentationRetrievalService

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Service instance
service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global service

    # Startup
    logger.info("Starting Presentation Retrieval API...")
    service = PresentationRetrievalService()
    await service.initialize()
    logger.info("Service initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Presentation Retrieval API...")
    if service:
        await service.close()
    logger.info("Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Presentation Retrieval API",
    description="AI-powered presentation content retrieval and generation API",
    version="1.0.1",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Authentication dependency
async def get_api_key(authorization: Optional[str] = Header(None)):
    """Simple API key authentication"""
    if not authorization:
        # For now, allow unauthenticated access in development
        # In production, enforce authentication
        if os.getenv("REQUIRE_AUTH", "false").lower() == "true":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header required"
            )
        return None

    # Extract token from "Bearer <token>" format
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme"
            )

        # Verify JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")  # Return user ID
    except (ValueError, PyJWTError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check API health status"""
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        database_connected=service.pool is not None if service else False,
        cache_enabled=True,
        services={
            "openai": True,
            "database": service.pool is not None if service else False,
            "cache": True
        }
    )


# 1. Search Presentation Content
@app.post("/search-presentation-content", response_model=PresentationSearchResponse)
async def search_presentation_content(
    request: PresentationSearchRequest,
    user_id: Optional[str] = Depends(get_api_key)
):
    """
    Search for presentation-relevant content using semantic similarity.
    This is the primary endpoint for finding content to include in presentations.
    """
    try:
        start_time = time.time()

        results = await service.search_presentation_content(
            query=request.query,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
            content_types=request.content_types,
            user_id=request.user_id or user_id,
            metadata_filters=request.metadata_filters
        )

        execution_time = (time.time() - start_time) * 1000

        return PresentationSearchResponse(
            success=True,
            query=request.query,
            results=results,
            total_results=len(results),
            execution_time_ms=execution_time
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 2. Extract Key Points
@app.post("/extract-key-points", response_model=KeyPointsResponse)
async def extract_key_points(
    request: KeyPointsRequest,
    user_id: Optional[str] = Depends(get_api_key)
):
    """
    Extract key points from specified documents.
    Useful for creating presentation outlines and summaries.
    """
    try:
        start_time = time.time()

        key_points = await service.extract_key_points(
            document_ids=request.document_ids,
            topic=request.topic,
            max_points=request.max_points,
            summarize=request.summarize
        )

        execution_time = (time.time() - start_time) * 1000

        return KeyPointsResponse(
            success=True,
            topic=request.topic,
            key_points=key_points,
            total_documents_processed=len(request.document_ids),
            execution_time_ms=execution_time
        )
    except Exception as e:
        logger.error(f"Key points extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 3. Find Supporting Data
@app.post("/find-supporting-data", response_model=SupportingDataResponse)
async def find_supporting_data(
    query: str,
    limit: int = 10,
    user_id: Optional[str] = Depends(get_api_key)
):
    """
    Find supporting data, statistics, and visualization suggestions for presentations.
    """
    try:
        start_time = time.time()

        chunks, statistics, visualizations = await service.find_supporting_data(
            query=query,
            limit=limit
        )

        execution_time = (time.time() - start_time) * 1000

        return SupportingDataResponse(
            success=True,
            query=query,
            data_points=chunks,
            statistics=statistics,
            visualizations=visualizations,
            total_results=len(chunks),
            execution_time_ms=execution_time
        )
    except Exception as e:
        logger.error(f"Finding supporting data failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 4. Get User Documents (Presentation-Ready)
@app.get("/user-documents/presentation-ready", response_model=UserDocumentsResponse)
async def get_presentation_ready_documents(
    user_id: Optional[str] = Depends(get_api_key)
):
    """
    Get user's documents that are ready for presentation use.
    """
    try:
        start_time = time.time()

        if not user_id:
            # Return empty list if no user ID
            return UserDocumentsResponse(
                success=True,
                documents=[],
                total_documents=0,
                presentation_ready_count=0,
                execution_time_ms=0
            )

        documents = await service.get_user_documents(
            user_id=user_id,
            presentation_ready=True
        )

        execution_time = (time.time() - start_time) * 1000

        return UserDocumentsResponse(
            success=True,
            documents=documents,
            total_documents=len(documents),
            presentation_ready_count=len(documents),
            execution_time_ms=execution_time
        )
    except Exception as e:
        logger.error(f"Getting user documents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 5. Generate Slide Suggestions
@app.post("/generate-slide-suggestions", response_model=SlideGenerationResponse)
async def generate_slide_suggestions(
    request: SlideGenerationRequest,
    user_id: Optional[str] = Depends(get_api_key)
):
    """
    Generate slide content suggestions based on topic and parameters.
    This is the main endpoint for autonomous slide creation.
    """
    try:
        start_time = time.time()

        slides = await service.generate_slide_suggestions(
            topic=request.topic,
            outline=request.outline,
            duration_minutes=request.duration_minutes,
            slide_count=request.slide_count
        )

        execution_time = (time.time() - start_time) * 1000

        # Create presentation outline from slides
        outline = [slide.title for slide in slides]

        # Calculate estimated duration
        estimated_duration = len(slides) * 2  # 2 minutes per slide average

        return SlideGenerationResponse(
            success=True,
            slides=slides,
            total_slides=len(slides),
            estimated_duration_minutes=estimated_duration,
            presentation_outline=outline,
            execution_time_ms=execution_time
        )
    except Exception as e:
        logger.error(f"Slide generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 6. Find Presentation Examples
@app.get("/find-presentation-examples", response_model=PresentationExamplesResponse)
async def find_presentation_examples(
    topic: str,
    limit: int = 5,
    user_id: Optional[str] = Depends(get_api_key)
):
    """
    Find example presentation structures and templates for a given topic.
    """
    try:
        start_time = time.time()

        examples = await service.find_presentation_examples(
            topic=topic,
            limit=limit
        )

        execution_time = (time.time() - start_time) * 1000

        return PresentationExamplesResponse(
            success=True,
            examples=examples,
            total_examples=len(examples),
            execution_time_ms=execution_time
        )
    except Exception as e:
        logger.error(f"Finding presentation examples failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 7. Extract Citations and Sources
@app.post("/extract-citations-sources", response_model=CitationsResponse)
async def extract_citations_sources(
    document_ids: list[str],
    user_id: Optional[str] = Depends(get_api_key)
):
    """
    Extract citations and sources from documents for proper attribution.
    """
    try:
        start_time = time.time()

        citations = await service.extract_citations_sources(
            document_ids=document_ids
        )

        execution_time = (time.time() - start_time) * 1000

        return CitationsResponse(
            success=True,
            citations=citations,
            total_citations=len(citations),
            execution_time_ms=execution_time
        )
    except Exception as e:
        logger.error(f"Citation extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "details": None
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "details": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )