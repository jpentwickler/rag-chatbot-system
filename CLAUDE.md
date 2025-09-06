# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Create .env file with API key
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Running the Application
```bash
# Start development server (must run from backend directory)
cd backend
uv run uvicorn app:app --reload --port 8000

# Alternative quick start (from root directory)
./run.sh
```

Access points:
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Development Environment
- Python 3.13+ required
- Uses uv package manager instead of pip
- Windows users should use Git Bash for commands
- ChromaDB data stored in `./chroma_db/` (auto-created)

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials with a modular, tool-based architecture.

### Core Architecture Patterns

**1. Tool-Based RAG**: Uses Claude's tool-calling capabilities where the AI decides when to search knowledge base vs. use general knowledge.

**2. Modular Component Design**: Each major function is separated into focused modules:
- `rag_system.py` - Central orchestrator that coordinates all components
- `ai_generator.py` - Claude API integration with tool execution pipeline
- `search_tools.py` - Anthropic tool definitions and vector search interface
- `vector_store.py` - ChromaDB operations and search logic
- `document_processor.py` - Course document parsing and intelligent text chunking
- `session_manager.py` - Conversation history management

**3. Structured Document Processing**: Course materials follow strict format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson X: [lesson_title]
Lesson Link: [optional_url]
[lesson content...]
```

Documents are parsed into hierarchical Course → Lesson → Chunk structure, with context-aware chunking that preserves educational relationships.

### Data Flow Architecture

**Query Processing Pipeline**:
1. Frontend → FastAPI (`app.py`) → RAG System (`rag_system.py`)
2. RAG System → AI Generator → Claude API with tool definitions
3. Claude decides: Use tools OR general knowledge
4. If tools: Search Tool → Vector Store → ChromaDB semantic search
5. Results returned with source attribution for UI

**Knowledge Base Construction**:
1. Course documents (`docs/*.txt`) → Document Processor
2. Hierarchical parsing: Course metadata + Lesson structure
3. Intelligent sentence-based chunking with overlap
4. Context enhancement: Chunks prefixed with course/lesson info
5. Vector embeddings → ChromaDB storage

### Key Configuration (config.py)

- **Model**: `claude-sonnet-4-20250514` (latest Claude model)
- **Embeddings**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Chunking**: 800 chars with 100 char overlap
- **Search**: Max 5 results, 2 message conversation history
- **ChromaDB**: Local storage in `./chroma_db`

### Frontend Integration

- Single-page web app with vanilla JS
- Session-based conversations with persistent chat
- Real-time source attribution display
- Course analytics sidebar showing loaded materials

### Tool System

The search tool (`CourseSearchTool`) provides Claude with:
- `search_course_content(query, course_name?, lesson_number?)` 
- Smart course name partial matching
- Lesson-specific filtering capabilities
- Automatic source tracking for UI display

This architecture enables Claude to intelligently decide when course-specific searches are needed while maintaining conversation context and providing transparent source attribution.
- always use uv to run the server do not use pip directly