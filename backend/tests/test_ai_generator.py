import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator


class MockAnthropicResponse:
    """Mock response from Anthropic API"""
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content if isinstance(content, list) else [Mock(text=content)]
        self.stop_reason = stop_reason


class MockToolUseContent:
    """Mock tool use content block"""
    def __init__(self, name, input_params, tool_id="test_id"):
        self.type = "tool_use"
        self.name = name
        self.input = input_params
        self.id = tool_id


class TestAIGenerator(unittest.TestCase):
    """Test cases for AIGenerator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        self.model = "claude-sonnet-4-20250514"
        
        # Mock the anthropic client - reset for each test
        self.mock_client = Mock()
        self.mock_client.reset_mock()  # Ensure clean state
        
        # Create AIGenerator with mocked client
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator(self.api_key, self.model)
    
    def test_initialization(self):
        """Test AIGenerator initialization"""
        self.assertEqual(self.ai_generator.model, self.model)
        self.assertIn("model", self.ai_generator.base_params)
        self.assertEqual(self.ai_generator.base_params["model"], self.model)
        self.assertEqual(self.ai_generator.base_params["temperature"], 0)
        self.assertEqual(self.ai_generator.base_params["max_tokens"], 800)
    
    def test_generate_response_without_tools(self):
        """Test generate_response without tools (direct text response)"""
        mock_response = MockAnthropicResponse("This is a direct response.")
        self.mock_client.messages.create.return_value = mock_response
        
        result = self.ai_generator.generate_response("What is AI?")
        
        self.assertEqual(result, "This is a direct response.")
        
        # Verify API call
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertEqual(call_args["model"], self.model)
        self.assertEqual(call_args["messages"], [{"role": "user", "content": "What is AI?"}])
        self.assertNotIn("tools", call_args)
    
    def test_generate_response_with_tools_no_tool_use(self):
        """Test generate_response with tools available but not used"""
        mock_response = MockAnthropicResponse("This is a direct response without tool use.")
        self.mock_client.messages.create.return_value = mock_response
        
        tools = [{"name": "test_tool", "description": "A test tool"}]
        
        result = self.ai_generator.generate_response("What is AI?", tools=tools)
        
        self.assertEqual(result, "This is a direct response without tool use.")
        
        # Verify API call includes tools
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertEqual(call_args["tools"], tools)
        self.assertEqual(call_args["tool_choice"], {"type": "auto"})
    
    def test_generate_response_with_conversation_history(self):
        """Test generate_response with conversation history"""
        mock_response = MockAnthropicResponse("Response with context.")
        self.mock_client.messages.create.return_value = mock_response
        
        history = "User: Previous question\nAssistant: Previous answer"
        
        result = self.ai_generator.generate_response("Follow-up question", conversation_history=history)
        
        self.assertEqual(result, "Response with context.")
        
        # Verify system prompt includes history
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertIn("Previous conversation:", call_args["system"])
        self.assertIn(history, call_args["system"])
    
    def test_generate_response_with_tool_use(self):
        """Test generate_response when AI decides to use a tool"""
        # Mock tool use response
        tool_use_content = MockToolUseContent("search_course_content", {"query": "machine learning"})
        mock_initial_response = MockAnthropicResponse([tool_use_content], stop_reason="tool_use")
        
        # Mock final response after tool execution
        mock_final_response = MockAnthropicResponse("Based on the search results, machine learning is...")
        
        self.mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Machine learning course content found."
        
        tools = [{"name": "search_course_content", "description": "Search course content"}]
        
        result = self.ai_generator.generate_response(
            "What is machine learning?", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        self.assertEqual(result, "Based on the search results, machine learning is...")
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", 
            query="machine learning"
        )
        
        # Verify two API calls were made
        self.assertEqual(self.mock_client.messages.create.call_count, 2)
    
    def test_handle_tool_execution_single_tool(self):
        """Test _handle_tool_execution with single tool call"""
        # Setup initial response with tool use
        tool_use_content = MockToolUseContent("test_tool", {"param": "value"}, "tool_123")
        mock_initial_response = MockAnthropicResponse([tool_use_content], stop_reason="tool_use")
        
        # Setup base params
        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system prompt",
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        # Mock final response
        mock_final_response = MockAnthropicResponse("Final response with tool results")
        self.mock_client.messages.create.return_value = mock_final_response
        
        # Execute
        result = self.ai_generator._handle_tool_execution(
            mock_initial_response, 
            base_params, 
            mock_tool_manager
        )
        
        self.assertEqual(result, "Final response with tool results")
        
        # Verify tool execution
        mock_tool_manager.execute_tool.assert_called_once_with("test_tool", param="value")
        
        # Verify final API call
        final_call_args = self.mock_client.messages.create.call_args[1]
        self.assertEqual(len(final_call_args["messages"]), 3)  # original + assistant + tool_result
        
        # Check tool result message format
        tool_result_msg = final_call_args["messages"][2]
        self.assertEqual(tool_result_msg["role"], "user")
        self.assertEqual(len(tool_result_msg["content"]), 1)
        self.assertEqual(tool_result_msg["content"][0]["type"], "tool_result")
        self.assertEqual(tool_result_msg["content"][0]["tool_use_id"], "tool_123")
        self.assertEqual(tool_result_msg["content"][0]["content"], "Tool execution result")
    
    def test_handle_tool_execution_multiple_tools(self):
        """Test _handle_tool_execution with multiple tool calls"""
        # Setup initial response with multiple tool uses
        tool_use_1 = MockToolUseContent("tool_1", {"param1": "value1"}, "tool_123")
        tool_use_2 = MockToolUseContent("tool_2", {"param2": "value2"}, "tool_456")
        mock_initial_response = MockAnthropicResponse([tool_use_1, tool_use_2], stop_reason="tool_use")
        
        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system prompt",
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        # Mock final response
        mock_final_response = MockAnthropicResponse("Final response with multiple tool results")
        self.mock_client.messages.create.return_value = mock_final_response
        
        # Execute
        result = self.ai_generator._handle_tool_execution(
            mock_initial_response, 
            base_params, 
            mock_tool_manager
        )
        
        self.assertEqual(result, "Final response with multiple tool results")
        
        # Verify both tools were executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call("tool_1", param1="value1")
        mock_tool_manager.execute_tool.assert_any_call("tool_2", param2="value2")
        
        # Verify final API call has multiple tool results
        final_call_args = self.mock_client.messages.create.call_args[1]
        tool_result_msg = final_call_args["messages"][2]
        self.assertEqual(len(tool_result_msg["content"]), 2)  # Two tool results
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        system_prompt = AIGenerator.SYSTEM_PROMPT
        
        # Check for key elements
        self.assertIn("search_course_content", system_prompt)
        self.assertIn("get_course_outline", system_prompt)
        self.assertIn("Tool Usage Guidelines", system_prompt)
        self.assertIn("Response Protocol", system_prompt)
        self.assertIn("Brief, Concise and focused", system_prompt)
    
    def test_api_error_handling(self):
        """Test handling of API errors"""
        self.mock_client.messages.create.side_effect = Exception("API Error")
        
        with self.assertRaises(Exception) as context:
            self.ai_generator.generate_response("Test query")
        
        self.assertEqual(str(context.exception), "API Error")
    
    def test_empty_tool_results_handling(self):
        """Test handling when no tool results are generated"""
        # Mock response with tool use but no actual tool use content
        mock_response = MockAnthropicResponse([], stop_reason="tool_use")
        
        base_params = {
            "messages": [{"role": "user", "content": "Test"}],
            "system": "Test system",
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
        
        mock_tool_manager = Mock()
        
        # Mock final response
        mock_final_response = MockAnthropicResponse("Response with no tools")
        self.mock_client.messages.create.return_value = mock_final_response
        
        result = self.ai_generator._handle_tool_execution(
            mock_response, 
            base_params, 
            mock_tool_manager
        )
        
        self.assertEqual(result, "Response with no tools")
        
        # Verify no tools were executed
        mock_tool_manager.execute_tool.assert_not_called()
    
    def test_sequential_tool_calling_two_rounds(self):
        """Test sequential tool calling over two rounds"""
        # Mock first round - tool use response
        tool_use_1 = MockToolUseContent("get_course_outline", {"course_name": "course_x"}, "tool_1")
        mock_response_1 = MockAnthropicResponse([tool_use_1], stop_reason="tool_use")
        
        # Mock second round - tool use response
        tool_use_2 = MockToolUseContent("search_course_content", {"query": "machine learning"}, "tool_2")
        mock_response_2 = MockAnthropicResponse([tool_use_2], stop_reason="tool_use")
        
        # Mock final response after max rounds
        mock_final_response = MockAnthropicResponse("Based on both searches, here is the comprehensive answer.")
        
        self.mock_client.messages.create.side_effect = [mock_response_1, mock_response_2, mock_final_response]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Course X outline", "ML content found"]
        
        tools = [
            {"name": "get_course_outline", "description": "Get course outline"},
            {"name": "search_course_content", "description": "Search course content"}
        ]
        
        result = self.ai_generator.generate_response(
            "Compare lesson 4 of course X with ML topics",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        self.assertEqual(result, "Based on both searches, here is the comprehensive answer.")
        
        # Verify both tools were executed in sequence
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="course_x")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="machine learning")
        
        # Verify three API calls were made (2 rounds + final)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
    
    def test_sequential_termination_max_rounds(self):
        """Test that sequential calling stops after 2 rounds"""
        # Mock responses for both rounds with tool use
        tool_use_1 = MockToolUseContent("search_course_content", {"query": "topic1"}, "tool_1")
        tool_use_2 = MockToolUseContent("search_course_content", {"query": "topic2"}, "tool_2")
        
        mock_response_1 = MockAnthropicResponse([tool_use_1], stop_reason="tool_use")
        mock_response_2 = MockAnthropicResponse([tool_use_2], stop_reason="tool_use")
        mock_final_response = MockAnthropicResponse("Final answer after max rounds.")
        
        self.mock_client.messages.create.side_effect = [mock_response_1, mock_response_2, mock_final_response]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        tools = [{"name": "search_course_content", "description": "Search content"}]
        
        result = self.ai_generator.generate_response(
            "Complex query requiring multiple searches",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        self.assertEqual(result, "Final answer after max rounds.")
        
        # Verify exactly 2 tool executions
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        
        # Verify exactly 3 API calls (2 rounds + final without tools)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
        
        # Verify final call has no tools
        final_call_args = self.mock_client.messages.create.call_args_list[2][1]
        self.assertNotIn("tools", final_call_args)
    
    def test_sequential_termination_no_tools(self):
        """Test that sequential calling stops when Claude doesn't use tools"""
        # Mock first round response without tool use
        mock_response = MockAnthropicResponse("Direct answer without tools needed.")
        self.mock_client.messages.create.return_value = mock_response
        
        mock_tool_manager = Mock()
        tools = [{"name": "search_course_content", "description": "Search content"}]
        
        result = self.ai_generator.generate_response(
            "Simple question",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        self.assertEqual(result, "Direct answer without tools needed.")
        
        # Verify no tools were executed
        mock_tool_manager.execute_tool.assert_not_called()
        
        # Verify only one API call was made
        self.assertEqual(self.mock_client.messages.create.call_count, 1)
    
    def test_sequential_tool_failure_handling(self):
        """Test handling of tool execution failures in sequential calling"""
        # Mock tool use response
        tool_use_content = MockToolUseContent("search_course_content", {"query": "test"}, "tool_1")
        mock_response = MockAnthropicResponse([tool_use_content], stop_reason="tool_use")
        
        # Mock tool manager that returns None (simulates _execute_tools_for_round returning None)
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # Mock the _execute_tools_for_round to return None on failure
        with patch.object(self.ai_generator, '_execute_tools_for_round', return_value=None):
            self.mock_client.messages.create.return_value = mock_response
            
            tools = [{"name": "search_course_content", "description": "Search content"}]
            
            result = self.ai_generator.generate_response(
                "Query that causes tool failure",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            self.assertEqual(result, "I encountered an error while processing your request.")
            
            # Verify only one API call was made (before failure)
            self.assertEqual(self.mock_client.messages.create.call_count, 1)
    
    def test_conversation_context_preservation(self):
        """Test that conversation context is preserved between rounds"""
        # Create a fresh mock client to avoid state leakage
        fresh_mock_client = Mock()
        
        # Mock two rounds of tool calling
        tool_use_1 = MockToolUseContent("get_course_outline", {"course_name": "test"}, "tool_1")
        tool_use_2 = MockToolUseContent("search_course_content", {"query": "ml"}, "tool_2")
        
        mock_response_1 = MockAnthropicResponse([tool_use_1], stop_reason="tool_use")
        mock_response_2 = MockAnthropicResponse([tool_use_2], stop_reason="tool_use")
        mock_final = MockAnthropicResponse("Final response with context.")
        
        fresh_mock_client.messages.create.side_effect = [mock_response_1, mock_response_2, mock_final]
        
        # Create a fresh AIGenerator instance
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = fresh_mock_client
            fresh_ai_generator = AIGenerator(self.api_key, self.model)
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Outline result", "Search result"]
        
        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        
        result = fresh_ai_generator.generate_response(
            "Test query",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify the messages array grows with each round
        api_calls = fresh_mock_client.messages.create.call_args_list
        
        # Verify we made exactly 3 API calls (2 rounds + final)
        self.assertEqual(len(api_calls), 3)
        
        # First call: just user message
        first_messages = api_calls[0][1]["messages"]
        self.assertEqual(len(first_messages), 1, f"Expected 1 message, got {len(first_messages)}: {first_messages}")
        self.assertEqual(first_messages[0]["role"], "user")
        
        # Second call: user + assistant + tool_result (from first round)
        second_messages = api_calls[1][1]["messages"]
        self.assertEqual(len(second_messages), 3)
        self.assertEqual(second_messages[0]["role"], "user")  # Original query
        self.assertEqual(second_messages[1]["role"], "assistant")  # Tool use from round 1
        self.assertEqual(second_messages[2]["role"], "user")  # Tool results from round 1
        
        # Final call: full conversation context (user + assistant + user + assistant + user)
        final_messages = api_calls[2][1]["messages"]
        self.assertEqual(len(final_messages), 5)
        roles = [msg["role"] for msg in final_messages]
        expected_roles = ["user", "assistant", "user", "assistant", "user"]
        self.assertEqual(roles, expected_roles)
    
    def test_execute_tools_for_round_single_tool(self):
        """Test _execute_tools_for_round method with single tool"""
        tool_use_content = MockToolUseContent("test_tool", {"param": "value"}, "tool_123")
        mock_response = MockAnthropicResponse([tool_use_content], stop_reason="tool_use")
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        result = self.ai_generator._execute_tools_for_round(mock_response, mock_tool_manager, 1)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "tool_result")
        self.assertEqual(result[0]["tool_use_id"], "tool_123")
        self.assertEqual(result[0]["content"], "Tool execution result")
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with("test_tool", param="value")
    
    def test_execute_tools_for_round_error_handling(self):
        """Test _execute_tools_for_round error handling"""
        tool_use_content = MockToolUseContent("failing_tool", {"param": "value"}, "tool_123")
        mock_response = MockAnthropicResponse([tool_use_content], stop_reason="tool_use")
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")
        
        result = self.ai_generator._execute_tools_for_round(mock_response, mock_tool_manager, 1)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "tool_result")
        self.assertEqual(result[0]["tool_use_id"], "tool_123")
        self.assertIn("Error executing failing_tool", result[0]["content"])


if __name__ == '__main__':
    unittest.main()