import unittest
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os
import tempfile
import shutil

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import Config


class TestRAGSystem(unittest.TestCase):
    """Integration tests for RAGSystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a test config
        self.test_config = Config()
        self.test_config.ANTHROPIC_API_KEY = "test_key"
        self.test_config.ANTHROPIC_MODEL = "test_model"
        self.test_config.EMBEDDING_MODEL = "test_embedding"
        self.test_config.CHUNK_SIZE = 800
        self.test_config.CHUNK_OVERLAP = 100
        self.test_config.MAX_RESULTS = 5
        self.test_config.MAX_HISTORY = 2
        self.test_config.CHROMA_PATH = "./test_chroma_db"
        
        # Mock all the dependencies
        with patch('rag_system.DocumentProcessor') as mock_doc_processor, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_generator, \
             patch('rag_system.SessionManager') as mock_session_manager, \
             patch('rag_system.ToolManager') as mock_tool_manager, \
             patch('rag_system.CourseSearchTool') as mock_search_tool, \
             patch('rag_system.CourseOutlineTool') as mock_outline_tool:
            
            self.mock_doc_processor = mock_doc_processor.return_value
            self.mock_vector_store = mock_vector_store.return_value
            self.mock_ai_generator = mock_ai_generator.return_value
            self.mock_session_manager = mock_session_manager.return_value
            self.mock_tool_manager = mock_tool_manager.return_value
            self.mock_search_tool = mock_search_tool.return_value
            self.mock_outline_tool = mock_outline_tool.return_value
            
            self.rag_system = RAGSystem(self.test_config)
    
    def test_initialization(self):
        """Test RAGSystem initialization"""
        self.assertIsNotNone(self.rag_system.document_processor)
        self.assertIsNotNone(self.rag_system.vector_store)
        self.assertIsNotNone(self.rag_system.ai_generator)
        self.assertIsNotNone(self.rag_system.session_manager)
        self.assertIsNotNone(self.rag_system.tool_manager)
        
        # Verify tools were registered
        self.mock_tool_manager.register_tool.assert_called()
        self.assertEqual(self.mock_tool_manager.register_tool.call_count, 2)  # search + outline tools
    
    def test_query_without_session(self):
        """Test query processing without session ID"""
        # Mock AI generator response
        self.mock_ai_generator.generate_response.return_value = "AI response about course content"
        self.mock_tool_manager.get_tool_definitions.return_value = [{"name": "test_tool"}]
        self.mock_tool_manager.get_last_sources.return_value = [{"text": "Source 1", "link": "http://source1.com"}]
        
        response, sources = self.rag_system.query("What is machine learning?")
        
        self.assertEqual(response, "AI response about course content")
        self.assertEqual(sources, [{"text": "Source 1", "link": "http://source1.com"}])
        
        # Verify AI generator was called correctly
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        self.assertIn("Answer this question about course materials: What is machine learning?", call_args['query'])
        self.assertIsNone(call_args['conversation_history'])
        self.assertEqual(call_args['tools'], [{"name": "test_tool"}])
        self.assertEqual(call_args['tool_manager'], self.mock_tool_manager)
        
        # Verify sources were retrieved and reset
        self.mock_tool_manager.get_last_sources.assert_called_once()
        self.mock_tool_manager.reset_sources.assert_called_once()
    
    def test_query_with_session(self):
        """Test query processing with session ID"""
        session_id = "test_session_123"
        history = "Previous conversation history"
        
        self.mock_session_manager.get_conversation_history.return_value = history
        self.mock_ai_generator.generate_response.return_value = "Contextual AI response"
        self.mock_tool_manager.get_tool_definitions.return_value = []
        self.mock_tool_manager.get_last_sources.return_value = []
        
        response, sources = self.rag_system.query("Follow-up question", session_id=session_id)
        
        self.assertEqual(response, "Contextual AI response")
        self.assertEqual(sources, [])
        
        # Verify session history was retrieved
        self.mock_session_manager.get_conversation_history.assert_called_once_with(session_id)
        
        # Verify AI generator received history
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        self.assertEqual(call_args['conversation_history'], history)
        
        # Verify session was updated
        self.mock_session_manager.add_exchange.assert_called_once_with(
            session_id, "Follow-up question", "Contextual AI response"
        )
    
    def test_add_course_document_success(self):
        """Test successful addition of a course document"""
        file_path = "/path/to/course.txt"
        mock_course = Mock()
        mock_course.title = "Test Course"
        mock_chunks = [Mock(), Mock(), Mock()]  # 3 chunks
        
        self.mock_doc_processor.process_course_document.return_value = (mock_course, mock_chunks)
        
        course, chunk_count = self.rag_system.add_course_document(file_path)
        
        self.assertEqual(course, mock_course)
        self.assertEqual(chunk_count, 3)
        
        # Verify document processing
        self.mock_doc_processor.process_course_document.assert_called_once_with(file_path)
        
        # Verify vector store operations
        self.mock_vector_store.add_course_metadata.assert_called_once_with(mock_course)
        self.mock_vector_store.add_course_content.assert_called_once_with(mock_chunks)
    
    def test_add_course_document_error(self):
        """Test handling of errors during document addition"""
        file_path = "/path/to/invalid_course.txt"
        
        self.mock_doc_processor.process_course_document.side_effect = Exception("File not found")
        
        with patch('builtins.print') as mock_print:
            course, chunk_count = self.rag_system.add_course_document(file_path)
        
        self.assertIsNone(course)
        self.assertEqual(chunk_count, 0)
        
        # Verify error was printed
        mock_print.assert_called_once()
        self.assertIn("Error processing course document", mock_print.call_args[0][0])
    
    def test_add_course_folder_with_clear(self):
        """Test adding course folder with existing data clear"""
        folder_path = "/path/to/courses"
        
        # Mock file system
        with patch('os.path.exists') as mock_exists, \
             patch('os.listdir') as mock_listdir, \
             patch('os.path.isfile') as mock_isfile, \
             patch('os.path.join') as mock_join, \
             patch('builtins.print') as mock_print:
            
            mock_exists.return_value = True
            mock_listdir.return_value = ["course1.txt", "course2.pdf", "not_a_course.jpg"]
            mock_isfile.side_effect = lambda path: path.endswith(('.txt', '.pdf'))
            mock_join.side_effect = lambda folder, file: f"{folder}/{file}"
            
            # Mock existing course titles
            self.mock_vector_store.get_existing_course_titles.return_value = []
            
            # Mock document processing
            mock_course1 = Mock()
            mock_course1.title = "Course 1"
            mock_course2 = Mock() 
            mock_course2.title = "Course 2"
            
            self.mock_doc_processor.process_course_document.side_effect = [
                (mock_course1, [Mock(), Mock()]),  # 2 chunks
                (mock_course2, [Mock()])           # 1 chunk
            ]
            
            courses, chunks = self.rag_system.add_course_folder(folder_path, clear_existing=True)
            
            self.assertEqual(courses, 2)
            self.assertEqual(chunks, 3)
            
            # Verify data was cleared
            self.mock_vector_store.clear_all_data.assert_called_once()
            
            # Verify both courses were processed
            self.assertEqual(self.mock_doc_processor.process_course_document.call_count, 2)
            
            # Verify metadata and content were added for both courses
            self.assertEqual(self.mock_vector_store.add_course_metadata.call_count, 2)
            self.assertEqual(self.mock_vector_store.add_course_content.call_count, 2)
    
    def test_add_course_folder_skip_existing(self):
        """Test adding course folder skips existing courses"""
        folder_path = "/path/to/courses"
        
        with patch('os.path.exists') as mock_exists, \
             patch('os.listdir') as mock_listdir, \
             patch('os.path.isfile') as mock_isfile, \
             patch('os.path.join') as mock_join, \
             patch('builtins.print') as mock_print:
            
            mock_exists.return_value = True
            mock_listdir.return_value = ["course1.txt"]
            mock_isfile.return_value = True
            mock_join.return_value = "/path/to/courses/course1.txt"
            
            # Mock existing course titles (course already exists)
            self.mock_vector_store.get_existing_course_titles.return_value = ["Course 1"]
            
            # Mock document processing
            mock_course = Mock()
            mock_course.title = "Course 1"  # Same title as existing
            
            self.mock_doc_processor.process_course_document.return_value = (mock_course, [Mock()])
            
            courses, chunks = self.rag_system.add_course_folder(folder_path, clear_existing=False)
            
            self.assertEqual(courses, 0)  # No new courses added
            self.assertEqual(chunks, 0)   # No new chunks added
            
            # Verify data was not cleared
            self.mock_vector_store.clear_all_data.assert_not_called()
            
            # Verify course was processed but not added
            self.mock_doc_processor.process_course_document.assert_called_once()
            self.mock_vector_store.add_course_metadata.assert_not_called()
            self.mock_vector_store.add_course_content.assert_not_called()
    
    def test_add_course_folder_nonexistent_path(self):
        """Test handling of nonexistent folder path"""
        folder_path = "/nonexistent/path"
        
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.print') as mock_print:
            
            mock_exists.return_value = False
            
            courses, chunks = self.rag_system.add_course_folder(folder_path)
            
            self.assertEqual(courses, 0)
            self.assertEqual(chunks, 0)
            
            mock_print.assert_called_once_with(f"Folder {folder_path} does not exist")
    
    def test_get_course_analytics(self):
        """Test getting course analytics"""
        self.mock_vector_store.get_course_count.return_value = 3
        self.mock_vector_store.get_existing_course_titles.return_value = ["Course A", "Course B", "Course C"]
        
        analytics = self.rag_system.get_course_analytics()
        
        expected = {
            "total_courses": 3,
            "course_titles": ["Course A", "Course B", "Course C"]
        }
        self.assertEqual(analytics, expected)
    
    def test_query_error_propagation(self):
        """Test that errors in query processing are properly handled"""
        self.mock_ai_generator.generate_response.side_effect = Exception("API Error")
        
        with self.assertRaises(Exception) as context:
            self.rag_system.query("Test question")
        
        self.assertEqual(str(context.exception), "API Error")
    
    def test_tool_integration(self):
        """Test that tools are properly integrated with AI generator"""
        query = "What courses are available?"
        
        # Mock tool definitions
        expected_tools = [
            {"name": "search_course_content", "description": "Search courses"},
            {"name": "get_course_outline", "description": "Get course outline"}
        ]
        self.mock_tool_manager.get_tool_definitions.return_value = expected_tools
        
        # Mock AI response
        self.mock_ai_generator.generate_response.return_value = "Here are the available courses..."
        self.mock_tool_manager.get_last_sources.return_value = []
        
        response, sources = self.rag_system.query(query)
        
        # Verify tools were passed to AI generator
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        self.assertEqual(call_args['tools'], expected_tools)
        self.assertEqual(call_args['tool_manager'], self.mock_tool_manager)
    
    def test_end_to_end_query_flow(self):
        """Test complete end-to-end query flow"""
        session_id = "test_session"
        query = "What is covered in the ML course?"
        
        # Setup mocks for complete flow
        self.mock_session_manager.get_conversation_history.return_value = None
        self.mock_tool_manager.get_tool_definitions.return_value = [{"name": "search_tool"}]
        self.mock_ai_generator.generate_response.return_value = "The ML course covers..."
        self.mock_tool_manager.get_last_sources.return_value = [{"text": "ML Course", "link": "http://ml.com"}]
        
        # Execute query
        response, sources = self.rag_system.query(query, session_id)
        
        # Verify complete flow
        self.assertEqual(response, "The ML course covers...")
        self.assertEqual(sources, [{"text": "ML Course", "link": "http://ml.com"}])
        
        # Verify all components were called in correct order
        self.mock_session_manager.get_conversation_history.assert_called_once_with(session_id)
        self.mock_tool_manager.get_tool_definitions.assert_called_once()
        self.mock_ai_generator.generate_response.assert_called_once()
        self.mock_tool_manager.get_last_sources.assert_called_once()
        self.mock_tool_manager.reset_sources.assert_called_once()
        self.mock_session_manager.add_exchange.assert_called_once_with(session_id, query, response)


if __name__ == '__main__':
    unittest.main()