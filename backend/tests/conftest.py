"""
Test configuration and fixtures for the RAG system tests.
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, MagicMock, patch
from fastapi.testclient import TestClient
from typing import Generator

# Add backend directory to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from rag_system import RAGSystem


@pytest.fixture
def test_config() -> Config:
    """Create a test configuration with mocked values."""
    config = Config()
    config.ANTHROPIC_API_KEY = "test_key"
    config.ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    with patch('anthropic.Anthropic') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Mocked response")]
        mock_instance.messages.create.return_value = mock_response
        
        yield mock_instance


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for testing."""
    with patch('vector_store.VectorStore') as mock_store:
        mock_instance = MagicMock()
        mock_store.return_value = mock_instance
        
        # Mock search results
        mock_instance.search.return_value = [
            ("Sample content 1", {"course_title": "Test Course", "lesson_number": 1}),
            ("Sample content 2", {"course_title": "Test Course", "lesson_number": 2})
        ]
        
        mock_instance.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Test Course 1", "Test Course 2"]
        }
        
        yield mock_instance


@pytest.fixture
def mock_rag_system(test_config, mock_anthropic_client, mock_vector_store):
    """Create a mocked RAG system for testing."""
    with patch.multiple(
        'rag_system',
        VectorStore=lambda config: mock_vector_store,
        AIGenerator=lambda config: MagicMock(generate_response=lambda query, context, history: ("Test response", [])),
        SessionManager=lambda: MagicMock(create_session=lambda: "test_session_123")
    ):
        rag_system = RAGSystem(test_config)
        rag_system.query = MagicMock(return_value=("Test response", ["Test source 1", "Test source 2"]))
        rag_system.get_course_analytics = MagicMock(return_value={
            "total_courses": 2,
            "course_titles": ["Test Course 1", "Test Course 2"]
        })
        yield rag_system


@pytest.fixture
def test_app():
    """Create a test FastAPI application without static file mounting."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Union
    
    # Create a minimal test app that mirrors the main app structure
    app = FastAPI(title="Test Course Materials RAG System")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class SourceItem(BaseModel):
        text: str
        link: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, SourceItem]]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Mock RAG system for the app
    mock_rag = MagicMock()
    mock_rag.query.return_value = ("Test response", ["Test source"])
    mock_rag.session_manager.create_session.return_value = "test_session_123"
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course 1", "Test Course 2"]
    }
    
    # API endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag.session_manager.create_session()
            answer, sources = mock_rag.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def read_root():
        return {"message": "Course Materials RAG System"}
    
    return app


@pytest.fixture
def test_client(test_app) -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    with TestClient(test_app) as client:
        yield client


@pytest.fixture
def sample_query_data():
    """Sample data for testing query endpoints."""
    return {
        "valid_query": {
            "query": "What is machine learning?",
            "session_id": "test_session_123"
        },
        "query_without_session": {
            "query": "Explain deep learning"
        },
        "empty_query": {
            "query": ""
        },
        "expected_response": {
            "answer": "Test response",
            "sources": ["Test source"],
            "session_id": "test_session_123"
        }
    }


@pytest.fixture
def sample_course_documents():
    """Sample course documents for testing."""
    return [
        {
            "filename": "course1.txt",
            "content": """Course Title: Machine Learning Basics
Course Link: https://example.com/ml-basics
Course Instructor: Dr. Jane Smith

Lesson 1: Introduction to ML
This lesson covers the fundamentals of machine learning including supervised and unsupervised learning approaches.

Lesson 2: Linear Regression
We explore linear regression models and their applications in predictive analytics.
"""
        },
        {
            "filename": "course2.txt", 
            "content": """Course Title: Deep Learning Fundamentals
Course Link: https://example.com/dl-fundamentals
Course Instructor: Prof. John Doe

Lesson 1: Neural Networks
Introduction to artificial neural networks and their basic architecture.

Lesson 2: Backpropagation
Understanding the backpropagation algorithm for training neural networks.
"""
        }
    ]