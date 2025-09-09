import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool(unittest.TestCase):
    """Test cases for CourseSearchTool.execute() method"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_successful_search_with_results(self):
        """Test execute() with successful search returning results"""
        # Setup mock results
        mock_results = SearchResults(
            documents=["Course content about machine learning", "More ML content"],
            metadata=[
                {"course_title": "ML Course", "lesson_number": 1},
                {"course_title": "ML Course", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"
        
        # Execute
        result = self.search_tool.execute("machine learning")
        
        # Verify
        self.assertIn("ML Course", result)
        self.assertIn("Course content about machine learning", result)
        self.assertIn("Lesson 1", result)
        self.assertIn("Lesson 2", result)
        self.mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=None
        )
    
    def test_search_with_course_filter(self):
        """Test execute() with course name filter"""
        mock_results = SearchResults(
            documents=["Anthropic course content"],
            metadata=[{"course_title": "Anthropic Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None
        
        result = self.search_tool.execute("AI models", course_name="Anthropic")
        
        self.assertIn("Anthropic Course", result)
        self.mock_vector_store.search.assert_called_once_with(
            query="AI models",
            course_name="Anthropic",
            lesson_number=None
        )
    
    def test_search_with_lesson_filter(self):
        """Test execute() with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://lesson3.com"
        
        result = self.search_tool.execute("lesson content", lesson_number=3)
        
        self.assertIn("Test Course - Lesson 3", result)
        self.mock_vector_store.search.assert_called_once_with(
            query="lesson content",
            course_name=None,
            lesson_number=3
        )
    
    def test_search_with_both_filters(self):
        """Test execute() with both course name and lesson number filters"""
        mock_results = SearchResults(
            documents=["Specific content"],
            metadata=[{"course_title": "Specific Course", "lesson_number": 2}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://specific.com"
        
        result = self.search_tool.execute("content", course_name="Specific", lesson_number=2)
        
        self.assertIn("Specific Course - Lesson 2", result)
        self.mock_vector_store.search.assert_called_once_with(
            query="content",
            course_name="Specific",
            lesson_number=2
        )
    
    def test_search_with_error_from_vector_store(self):
        """Test execute() when vector store returns an error"""
        mock_results = SearchResults.empty("ChromaDB connection failed")
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        self.assertEqual(result, "ChromaDB connection failed")
        self.assertEqual(self.search_tool.last_sources, [])
    
    def test_search_with_no_results(self):
        """Test execute() when search returns no results"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("non-existent content")
        
        self.assertEqual(result, "No relevant content found.")
    
    def test_search_with_no_results_and_course_filter(self):
        """Test execute() when search with course filter returns no results"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("content", course_name="Non-existent Course")
        
        self.assertEqual(result, "No relevant content found in course 'Non-existent Course'.")
    
    def test_search_with_no_results_and_lesson_filter(self):
        """Test execute() when search with lesson filter returns no results"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("content", lesson_number=99)
        
        self.assertEqual(result, "No relevant content found in lesson 99.")
    
    def test_search_with_no_results_and_both_filters(self):
        """Test execute() when search with both filters returns no results"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("content", course_name="Course", lesson_number=5)
        
        self.assertEqual(result, "No relevant content found in course 'Course' in lesson 5.")
    
    def test_source_tracking(self):
        """Test that sources are properly tracked for the UI"""
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": None}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.side_effect = ["https://link1.com", None]
        
        self.search_tool.execute("test query")
        
        # Check sources are tracked correctly
        expected_sources = [
            {"text": "Course A - Lesson 1", "link": "https://link1.com"},
            {"text": "Course B", "link": None}
        ]
        self.assertEqual(self.search_tool.last_sources, expected_sources)
    
    def test_get_tool_definition(self):
        """Test that tool definition is correctly formatted"""
        definition = self.search_tool.get_tool_definition()
        
        self.assertEqual(definition["name"], "search_course_content")
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)
        self.assertEqual(definition["input_schema"]["required"], ["query"])
        
        properties = definition["input_schema"]["properties"]
        self.assertIn("query", properties)
        self.assertIn("course_name", properties)
        self.assertIn("lesson_number", properties)


class TestCourseOutlineTool(unittest.TestCase):
    """Test cases for CourseOutlineTool.execute() method"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.outline_tool = CourseOutlineTool(self.mock_vector_store)
    
    def test_successful_course_outline_retrieval(self):
        """Test execute() with successful course outline retrieval"""
        mock_outline = {
            "title": "Machine Learning Course",
            "course_link": "https://ml-course.com",
            "instructor": "Dr. Smith",
            "lesson_count": 3,
            "lessons": [
                {"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "https://ml-course.com/1"},
                {"lesson_number": 2, "lesson_title": "Algorithms", "lesson_link": "https://ml-course.com/2"},
                {"lesson_number": 3, "lesson_title": "Applications", "lesson_link": "https://ml-course.com/3"}
            ]
        }
        self.mock_vector_store.get_course_outline.return_value = mock_outline
        
        result = self.outline_tool.execute("Machine Learning")
        
        self.assertIn("**Machine Learning Course**", result)
        self.assertIn("Course Link: https://ml-course.com", result)
        self.assertIn("Instructor: Dr. Smith", result)
        self.assertIn("This course has 3 lessons:", result)
        self.assertIn("Lesson 1: Introduction - https://ml-course.com/1", result)
        self.assertIn("Lesson 2: Algorithms - https://ml-course.com/2", result)
        self.assertIn("Lesson 3: Applications - https://ml-course.com/3", result)
    
    def test_course_not_found(self):
        """Test execute() when course is not found"""
        self.mock_vector_store.get_course_outline.return_value = None
        
        result = self.outline_tool.execute("Non-existent Course")
        
        self.assertEqual(result, "No course found matching 'Non-existent Course'.")
    
    def test_get_tool_definition(self):
        """Test that tool definition is correctly formatted"""
        definition = self.outline_tool.get_tool_definition()
        
        self.assertEqual(definition["name"], "get_course_outline")
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)
        self.assertEqual(definition["input_schema"]["required"], ["course_title"])


class TestToolManager(unittest.TestCase):
    """Test cases for ToolManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tool_manager = ToolManager()
        self.mock_vector_store = Mock()
    
    def test_register_and_execute_search_tool(self):
        """Test registering and executing a search tool"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        
        # Mock the search results
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None
        
        self.tool_manager.register_tool(search_tool)
        
        # Test tool registration
        definitions = self.tool_manager.get_tool_definitions()
        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0]["name"], "search_course_content")
        
        # Test tool execution
        result = self.tool_manager.execute_tool("search_course_content", query="test")
        self.assertIn("Test Course", result)
    
    def test_get_last_sources(self):
        """Test getting last sources from tools"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        search_tool.last_sources = [{"text": "Test Source", "link": "https://test.com"}]
        
        self.tool_manager.register_tool(search_tool)
        
        sources = self.tool_manager.get_last_sources()
        self.assertEqual(sources, [{"text": "Test Source", "link": "https://test.com"}])
    
    def test_reset_sources(self):
        """Test resetting sources from all tools"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        search_tool.last_sources = [{"text": "Test Source", "link": "https://test.com"}]
        
        self.tool_manager.register_tool(search_tool)
        self.tool_manager.reset_sources()
        
        self.assertEqual(search_tool.last_sources, [])
    
    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        result = self.tool_manager.execute_tool("nonexistent_tool", query="test")
        self.assertEqual(result, "Tool 'nonexistent_tool' not found")


if __name__ == '__main__':
    unittest.main()