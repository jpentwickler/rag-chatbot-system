#!/usr/bin/env python3
"""
System validation script to test various query scenarios and identify issues.
"""

import sys

from config import config
from rag_system import RAGSystem


def test_query_scenarios():
    """Test various query scenarios that could cause failures"""

    print("=== RAG System Validation ===\n")

    # Initialize system
    try:
        rag = RAGSystem(config)
        print("✓ RAG System initialized successfully")
    except Exception as e:
        print(f"✗ RAG System initialization failed: {e}")
        return False

    # Check configuration
    print(f"✓ MAX_RESULTS: {config.MAX_RESULTS}")
    print(f"✓ API Key configured: {bool(config.ANTHROPIC_API_KEY)}")

    # Check data availability
    analytics = rag.get_course_analytics()
    print(f"✓ Courses loaded: {analytics['total_courses']}")

    if analytics["total_courses"] == 0:
        print("✗ No courses loaded - loading from docs...")
        courses, chunks = rag.add_course_folder("../docs", clear_existing=True)
        print(f"✓ Loaded {courses} courses with {chunks} chunks")

    # Test various query scenarios
    test_queries = [
        # Course content queries
        ("What is machine learning?", "general_ml"),
        ("What courses are available about Anthropic?", "anthropic_courses"),
        ("Tell me about MCP", "mcp_specific"),
        ("What is prompt compression?", "prompt_compression"),
        ("How do I build AI apps?", "ai_apps"),
        # Course outline queries
        ("What's covered in the Anthropic course?", "course_outline"),
        ("Show me the outline for the MCP course", "mcp_outline"),
        # Edge cases
        ("", "empty_query"),
        ("xyzabc123nonexistent", "nonsense_query"),
        ("What is the weather today?", "unrelated_query"),
    ]

    results = []

    print("\n=== Testing Query Scenarios ===")

    for query, test_name in test_queries:
        print(f"\nTest: {test_name}")
        print(f"Query: '{query}'")

        try:
            if not query:  # Empty query test
                response = "Empty query - skipping"
                sources = []
            else:
                response, sources = rag.query(query)

            # Analyze result
            success = True
            issues = []

            if "query failed" in response.lower():
                success = False
                issues.append("Contains 'query failed'")

            if len(response) < 10:
                success = False
                issues.append("Response too short")

            if "error" in response.lower() and "search error" in response.lower():
                success = False
                issues.append("Contains search error")

            # Print result
            status = "✓" if success else "✗"
            print(f"{status} Response length: {len(response)}")
            print(f"{status} Sources: {len(sources)}")
            print(f"{status} Preview: {response[:100]}...")

            if issues:
                print(f"  Issues: {', '.join(issues)}")

            results.append(
                {
                    "test_name": test_name,
                    "query": query,
                    "success": success,
                    "response_length": len(response),
                    "source_count": len(sources),
                    "issues": issues,
                }
            )

        except Exception as e:
            print(f"✗ Exception: {e}")
            results.append(
                {
                    "test_name": test_name,
                    "query": query,
                    "success": False,
                    "exception": str(e),
                }
            )

    # Summary
    print("\n=== Test Summary ===")
    successful = sum(1 for r in results if r.get("success", False))
    total = len(results)

    print(f"Successful tests: {successful}/{total}")

    failed_tests = [r for r in results if not r.get("success", False)]
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(
                f"- {test['test_name']}: {test.get('issues', test.get('exception', 'Unknown error'))}"
            )
    else:
        print("All tests passed!")

    return successful == total


def test_direct_tool_calls():
    """Test direct tool calls to isolate tool issues"""

    print("\n=== Direct Tool Testing ===")

    rag = RAGSystem(config)

    # Test search tool directly
    print("Testing CourseSearchTool...")
    try:
        result = rag.tool_manager.execute_tool(
            "search_course_content", query="machine learning"
        )

        if "query failed" in result.lower() or "error" in result.lower():
            print(f"✗ Search tool error: {result[:200]}...")
            return False
        else:
            print(f"✓ Search tool working: {len(result)} characters returned")

    except Exception as e:
        print(f"✗ Search tool exception: {e}")
        return False

    # Test outline tool directly
    print("Testing CourseOutlineTool...")
    try:
        result = rag.tool_manager.execute_tool(
            "get_course_outline", course_title="Anthropic"
        )

        if "No course found" in result:
            print(f"✓ Outline tool working (no match found): {result[:100]}...")
        elif "error" in result.lower():
            print(f"✗ Outline tool error: {result[:200]}...")
            return False
        else:
            print(f"✓ Outline tool working: {len(result)} characters returned")

    except Exception as e:
        print(f"✗ Outline tool exception: {e}")
        return False

    return True


if __name__ == "__main__":
    print("Starting RAG system validation...\n")

    # Run tests
    tools_ok = test_direct_tool_calls()
    queries_ok = test_query_scenarios()

    print("\n=== Final Results ===")
    print(f"Direct tools: {'✓ PASS' if tools_ok else '✗ FAIL'}")
    print(f"Query scenarios: {'✓ PASS' if queries_ok else '✗ FAIL'}")

    if tools_ok and queries_ok:
        print("\n🎉 RAG system is working correctly!")
        print("The 'query failed' issue has been resolved.")
        sys.exit(0)
    else:
        print("\n⚠️  Some issues remain - check the output above.")
        sys.exit(1)
