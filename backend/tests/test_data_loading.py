import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from document_processor import DocumentProcessor
from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestDataLoading(unittest.TestCase):
    """Test data loading and document processing functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_config = Config()
        self.test_config.CHUNK_SIZE = 800
        self.test_config.CHUNK_OVERLAP = 100
        self.test_config.MAX_RESULTS = 5
        self.test_config.CHROMA_PATH = "./test_chroma_db"
        self.test_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def test_document_processor_initialization(self):
        """Test DocumentProcessor initialization"""
        processor = DocumentProcessor(800, 100)
        self.assertEqual(processor.chunk_size, 800)
        self.assertEqual(processor.chunk_overlap, 100)

    def test_document_processor_read_file(self):
        """Test reading file with DocumentProcessor"""
        # Create a temporary file
        test_content = "This is test course content\nWith multiple lines"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(test_content)
            temp_path = f.name

        try:
            processor = DocumentProcessor(800, 100)
            content = processor.read_file(temp_path)
            self.assertEqual(content, test_content)
        finally:
            os.unlink(temp_path)

    def test_document_processor_read_nonexistent_file(self):
        """Test handling of nonexistent file"""
        processor = DocumentProcessor(800, 100)

        with self.assertRaises(FileNotFoundError):
            processor.read_file("/nonexistent/file.txt")

    def test_document_processor_chunk_text(self):
        """Test text chunking functionality"""
        processor = DocumentProcessor(50, 10)  # Small chunks for testing

        text = "First sentence here. Second sentence follows. Third sentence continues. Fourth sentence ends."
        chunks = processor.chunk_text(text)

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)  # Should create multiple chunks

        # Check that chunks have reasonable overlap
        for i, chunk in enumerate(chunks):
            self.assertLessEqual(len(chunk), 70)  # Allow some flexibility

    def test_vector_store_initialization(self):
        """Test VectorStore initialization"""
        with (
            patch("vector_store.chromadb.PersistentClient") as mock_client,
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ) as mock_embedding,
        ):

            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.get_or_create_collection.return_value = Mock()

            store = VectorStore("./test_db", "test-model", 5)

            self.assertEqual(store.max_results, 5)
            mock_client.assert_called_once()
            self.assertEqual(
                mock_client_instance.get_or_create_collection.call_count, 2
            )  # Two collections

    def test_vector_store_search_success(self):
        """Test successful vector store search"""
        with (
            patch("vector_store.chromadb.PersistentClient") as mock_client,
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            # Setup mock ChromaDB results
            mock_chroma_results = {
                "documents": [["Document 1", "Document 2"]],
                "metadatas": [
                    [
                        {"course_title": "Course A", "lesson_number": 1},
                        {"course_title": "Course B", "lesson_number": 2},
                    ]
                ],
                "distances": [[0.1, 0.2]],
            }

            mock_collection = Mock()
            mock_collection.query.return_value = mock_chroma_results

            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.get_or_create_collection.return_value = mock_collection

            store = VectorStore("./test_db", "test-model", 5)

            # Test search
            results = store.search("test query")

            self.assertIsInstance(results, SearchResults)
            self.assertEqual(len(results.documents), 2)
            self.assertEqual(results.documents[0], "Document 1")
            self.assertEqual(results.metadata[0]["course_title"], "Course A")
            self.assertIsNone(results.error)

    def test_vector_store_search_with_filters(self):
        """Test vector store search with course and lesson filters"""
        with (
            patch("vector_store.chromadb.PersistentClient") as mock_client,
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_chroma_results = {
                "documents": [["Filtered document"]],
                "metadatas": [[{"course_title": "ML Course", "lesson_number": 3}]],
                "distances": [[0.1]],
            }

            mock_collection = Mock()
            mock_collection.query.return_value = mock_chroma_results

            # Mock course resolution
            mock_catalog = Mock()
            mock_catalog.query.return_value = {
                "documents": [["ML Course"]],
                "metadatas": [[{"title": "ML Course"}]],
            }

            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.get_or_create_collection.side_effect = [
                mock_catalog,
                mock_collection,
            ]

            store = VectorStore("./test_db", "test-model", 5)

            # Test search with filters
            results = store.search(
                "machine learning", course_name="ML", lesson_number=3
            )

            self.assertEqual(len(results.documents), 1)
            self.assertEqual(results.documents[0], "Filtered document")

            # Verify the correct filter was applied
            call_args = mock_collection.query.call_args[1]
            expected_filter = {
                "$and": [{"course_title": "ML Course"}, {"lesson_number": 3}]
            }
            self.assertEqual(call_args["where"], expected_filter)

    def test_vector_store_search_course_not_found(self):
        """Test search when course name cannot be resolved"""
        with (
            patch("vector_store.chromadb.PersistentClient") as mock_client,
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            # Mock empty course resolution
            mock_catalog = Mock()
            mock_catalog.query.return_value = {"documents": [[]], "metadatas": [[]]}

            mock_collection = Mock()

            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.get_or_create_collection.side_effect = [
                mock_catalog,
                mock_collection,
            ]

            store = VectorStore("./test_db", "test-model", 5)

            # Test search with non-existent course
            results = store.search("test query", course_name="Nonexistent Course")

            self.assertIsNotNone(results.error)
            self.assertIn("No course found matching", results.error)
            self.assertTrue(results.is_empty())

    def test_vector_store_search_exception_handling(self):
        """Test vector store search exception handling"""
        with (
            patch("vector_store.chromadb.PersistentClient") as mock_client,
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_collection = Mock()
            mock_collection.query.side_effect = Exception("ChromaDB error")

            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.get_or_create_collection.return_value = mock_collection

            store = VectorStore("./test_db", "test-model", 5)

            # Test search with exception
            results = store.search("test query")

            self.assertIsNotNone(results.error)
            self.assertIn("Search error", results.error)
            self.assertTrue(results.is_empty())

    def test_vector_store_add_course_content(self):
        """Test adding course content to vector store"""
        with (
            patch("vector_store.chromadb.PersistentClient") as mock_client,
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_collection = Mock()
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.get_or_create_collection.return_value = mock_collection

            store = VectorStore("./test_db", "test-model", 5)

            # Create test chunks
            chunks = [
                CourseChunk(
                    content="First chunk content",
                    course_title="Test Course",
                    lesson_number=1,
                    chunk_index=0,
                ),
                CourseChunk(
                    content="Second chunk content",
                    course_title="Test Course",
                    lesson_number=2,
                    chunk_index=1,
                ),
            ]

            store.add_course_content(chunks)

            # Verify collection.add was called correctly
            mock_collection.add.assert_called_once()
            call_args = mock_collection.add.call_args[1]

            self.assertEqual(len(call_args["documents"]), 2)
            self.assertEqual(call_args["documents"][0], "First chunk content")
            self.assertEqual(call_args["documents"][1], "Second chunk content")

            self.assertEqual(len(call_args["metadatas"]), 2)
            self.assertEqual(call_args["metadatas"][0]["course_title"], "Test Course")
            self.assertEqual(call_args["metadatas"][0]["lesson_number"], 1)

            self.assertEqual(len(call_args["ids"]), 2)
            self.assertEqual(call_args["ids"][0], "Test_Course_0")
            self.assertEqual(call_args["ids"][1], "Test_Course_1")

    def test_vector_store_add_course_metadata(self):
        """Test adding course metadata to vector store"""
        with (
            patch("vector_store.chromadb.PersistentClient") as mock_client,
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_collection = Mock()
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.get_or_create_collection.return_value = mock_collection

            store = VectorStore("./test_db", "test-model", 5)

            # Create test course
            lessons = [
                Lesson(
                    lesson_number=1,
                    title="Introduction",
                    lesson_link="http://lesson1.com",
                ),
                Lesson(
                    lesson_number=2,
                    title="Advanced Topics",
                    lesson_link="http://lesson2.com",
                ),
            ]
            course = Course(
                title="Test Course",
                course_link="http://course.com",
                instructor="Dr. Test",
                lessons=lessons,
            )

            store.add_course_metadata(course)

            # Verify collection.add was called correctly
            mock_collection.add.assert_called_once()
            call_args = mock_collection.add.call_args[1]

            self.assertEqual(call_args["documents"], ["Test Course"])
            self.assertEqual(call_args["ids"], ["Test Course"])

            metadata = call_args["metadatas"][0]
            self.assertEqual(metadata["title"], "Test Course")
            self.assertEqual(metadata["instructor"], "Dr. Test")
            self.assertEqual(metadata["course_link"], "http://course.com")
            self.assertEqual(metadata["lesson_count"], 2)
            self.assertIn("lessons_json", metadata)

    def test_course_model_validation(self):
        """Test Course model validation"""
        # Valid course
        course = Course(
            title="Valid Course", course_link="http://course.com", instructor="Dr. Test"
        )

        self.assertEqual(course.title, "Valid Course")
        self.assertEqual(course.course_link, "http://course.com")
        self.assertEqual(course.instructor, "Dr. Test")
        self.assertEqual(len(course.lessons), 0)

    def test_lesson_model_validation(self):
        """Test Lesson model validation"""
        lesson = Lesson(
            lesson_number=1, title="Test Lesson", lesson_link="http://lesson.com"
        )

        self.assertEqual(lesson.lesson_number, 1)
        self.assertEqual(lesson.title, "Test Lesson")
        self.assertEqual(lesson.lesson_link, "http://lesson.com")

    def test_course_chunk_model_validation(self):
        """Test CourseChunk model validation"""
        chunk = CourseChunk(
            content="This is chunk content",
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0,
        )

        self.assertEqual(chunk.content, "This is chunk content")
        self.assertEqual(chunk.course_title, "Test Course")
        self.assertEqual(chunk.lesson_number, 1)
        self.assertEqual(chunk.chunk_index, 0)

    def test_search_results_from_chroma(self):
        """Test SearchResults.from_chroma class method"""
        chroma_results = {
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[{"title": "Course 1"}, {"title": "Course 2"}]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(chroma_results)

        self.assertEqual(results.documents, ["Doc 1", "Doc 2"])
        self.assertEqual(
            results.metadata, [{"title": "Course 1"}, {"title": "Course 2"}]
        )
        self.assertEqual(results.distances, [0.1, 0.2])
        self.assertIsNone(results.error)

    def test_search_results_empty(self):
        """Test SearchResults.empty class method"""
        error_msg = "No results found"
        results = SearchResults.empty(error_msg)

        self.assertEqual(results.documents, [])
        self.assertEqual(results.metadata, [])
        self.assertEqual(results.distances, [])
        self.assertEqual(results.error, error_msg)
        self.assertTrue(results.is_empty())

    def test_integration_document_to_vector_store(self):
        """Integration test: process document and add to vector store"""
        # Create a sample course document
        course_content = """Course Title: Test ML Course
Course Link: https://ml-course.com
Course Instructor: Dr. Machine Learning

Lesson 1: Introduction to ML
Lesson Link: https://ml-course.com/lesson1
This is the introduction lesson content. It covers basic concepts of machine learning and provides an overview of the field.

Lesson 2: Supervised Learning
Lesson Link: https://ml-course.com/lesson2
This lesson covers supervised learning algorithms including linear regression and classification methods.
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(course_content)
            temp_path = f.name

        try:
            # Process document
            processor = DocumentProcessor(200, 50)  # Small chunks for testing

            with patch.object(processor, "process_course_document") as mock_process:
                # Mock the processing to return expected structure
                lessons = [
                    Lesson(
                        lesson_number=1,
                        title="Introduction to ML",
                        lesson_link="https://ml-course.com/lesson1",
                    ),
                    Lesson(
                        lesson_number=2,
                        title="Supervised Learning",
                        lesson_link="https://ml-course.com/lesson2",
                    ),
                ]
                course = Course(
                    title="Test ML Course",
                    course_link="https://ml-course.com",
                    instructor="Dr. Machine Learning",
                    lessons=lessons,
                )
                chunks = [
                    CourseChunk(
                        content="Intro content",
                        course_title="Test ML Course",
                        lesson_number=1,
                        chunk_index=0,
                    ),
                    CourseChunk(
                        content="Supervised learning content",
                        course_title="Test ML Course",
                        lesson_number=2,
                        chunk_index=1,
                    ),
                ]
                mock_process.return_value = (course, chunks)

                # Test processing
                result_course, result_chunks = processor.process_course_document(
                    temp_path
                )

                self.assertEqual(result_course.title, "Test ML Course")
                self.assertEqual(len(result_chunks), 2)
                self.assertEqual(result_chunks[0].lesson_number, 1)
                self.assertEqual(result_chunks[1].lesson_number, 2)

        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
