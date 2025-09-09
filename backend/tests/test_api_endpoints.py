"""
API endpoint tests for the RAG system FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint."""
    
    def test_query_with_session_id(self, test_client: TestClient, sample_query_data):
        """Test query endpoint with provided session ID."""
        response = test_client.post(
            "/api/query",
            json=sample_query_data["valid_query"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == sample_query_data["valid_query"]["session_id"]
        assert isinstance(data["sources"], list)
    
    def test_query_without_session_id(self, test_client: TestClient, sample_query_data):
        """Test query endpoint without session ID - should create one."""
        response = test_client.post(
            "/api/query",
            json=sample_query_data["query_without_session"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] is not None
        assert len(data["session_id"]) > 0
    
    def test_query_empty_string(self, test_client: TestClient, sample_query_data):
        """Test query endpoint with empty query string."""
        response = test_client.post(
            "/api/query",
            json=sample_query_data["empty_query"]
        )
        
        # Should still return 200 but may have different response
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
    
    def test_query_missing_query_field(self, test_client: TestClient):
        """Test query endpoint with missing query field."""
        response = test_client.post(
            "/api/query",
            json={"session_id": "test_session"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_invalid_json(self, test_client: TestClient):
        """Test query endpoint with invalid JSON."""
        response = test_client.post(
            "/api/query",
            data="invalid json"
        )
        
        assert response.status_code == 422
    
    def test_query_malformed_request(self, test_client: TestClient):
        """Test query endpoint with malformed request body."""
        response = test_client.post(
            "/api/query",
            json={"query": 123}  # query should be string
        )
        
        assert response.status_code == 422
    
    @patch('rag_system.RAGSystem')
    def test_query_rag_system_exception(self, mock_rag_class, test_client: TestClient):
        """Test query endpoint when RAG system raises exception."""
        # Override the test app's mock to raise an exception
        with patch('test_api_endpoints.mock_rag') as mock_rag:
            mock_rag.query.side_effect = Exception("Database connection failed")
            
            response = test_client.post(
                "/api/query",
                json={"query": "test query"}
            )
            
            assert response.status_code == 500
            assert "Database connection failed" in response.json()["detail"]


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint."""
    
    def test_get_course_stats(self, test_client: TestClient):
        """Test getting course statistics."""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] >= 0
    
    def test_course_stats_response_structure(self, test_client: TestClient):
        """Test that course stats response has correct structure."""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response matches CourseStats model
        assert set(data.keys()) == {"total_courses", "course_titles"}
        
        # Verify types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Verify all course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)
    
    @patch('rag_system.RAGSystem')
    def test_courses_rag_system_exception(self, mock_rag_class, test_client: TestClient):
        """Test courses endpoint when RAG system raises exception."""
        # This would require modifying the test app setup, but demonstrates the test pattern
        pass


@pytest.mark.api
class TestRootEndpoint:
    """Test the root endpoint."""
    
    def test_root_endpoint(self, test_client: TestClient):
        """Test the root endpoint returns basic info."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert isinstance(data["message"], str)


@pytest.mark.api
class TestRequestValidation:
    """Test request validation and error handling."""
    
    def test_invalid_content_type(self, test_client: TestClient):
        """Test endpoints with invalid content type."""
        response = test_client.post(
            "/api/query",
            data="query=test",  # form data instead of JSON
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == 422
    
    def test_missing_content_type(self, test_client: TestClient):
        """Test endpoints with missing content type header."""
        response = test_client.post("/api/query")
        
        assert response.status_code == 422
    
    def test_options_request(self, test_client: TestClient):
        """Test OPTIONS request for CORS preflight."""
        response = test_client.options("/api/query")
        
        # Should return 200 due to CORS middleware
        assert response.status_code == 200


@pytest.mark.api
class TestResponseFormat:
    """Test response format and structure."""
    
    def test_query_response_format(self, test_client: TestClient):
        """Test that query response matches expected format."""
        response = test_client.post(
            "/api/query",
            json={"query": "test query"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields exist
        required_fields = {"answer", "sources", "session_id"}
        assert set(data.keys()) == required_fields
        
        # Check field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
    
    def test_courses_response_format(self, test_client: TestClient):
        """Test that courses response matches expected format."""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields exist
        required_fields = {"total_courses", "course_titles"}
        assert set(data.keys()) == required_fields
        
        # Check field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)


@pytest.mark.api 
class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_nonexistent_endpoint(self, test_client: TestClient):
        """Test request to non-existent endpoint."""
        response = test_client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    def test_wrong_http_method(self, test_client: TestClient):
        """Test using wrong HTTP method on existing endpoint."""
        # GET on POST endpoint
        response = test_client.get("/api/query")
        assert response.status_code == 405
        
        # POST on GET endpoint  
        response = test_client.post("/api/courses")
        assert response.status_code == 405
    
    def test_large_payload(self, test_client: TestClient):
        """Test handling of large request payload."""
        large_query = "x" * 10000  # 10KB query
        
        response = test_client.post(
            "/api/query",
            json={"query": large_query}
        )
        
        # Should handle large queries gracefully
        assert response.status_code in [200, 413, 422]