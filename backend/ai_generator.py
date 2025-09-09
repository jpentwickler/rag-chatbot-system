from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Tool Usage Guidelines:
- **search_course_content**: Use for questions about specific course content, lesson details, or educational materials
- **get_course_outline**: Use for questions about course structure, outlines, lesson lists, or "what's covered" type queries
- **Sequential tool usage**: You can make up to 2 sequential tool calls to gather comprehensive information
- Use multiple rounds for comparisons, complex queries requiring different searches, or when building upon previous results
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Tool Selection:
- **Course outline questions** (e.g. "What's covered in...", "Course outline", "What lessons are included"): Use get_course_outline tool
- **Content/material questions** (e.g. specific concepts, detailed explanations): Use search_course_content tool
- **Complex comparisons**: First gather information about one topic, then search for related topics
- **Multi-part questions**: Break down into separate searches to gather complete information
- **General knowledge questions**: Answer using existing knowledge without using tools

Sequential Tool Examples:
- "Compare lesson 4 of course X with similar topics": First get course outline for course X, then search for similar topics
- "Find courses discussing the same concept as lesson Y": First get lesson Y content, then search for courses with that concept
- "What's the difference between approach A and B across courses": Search for approach A, then search for approach B

Response Protocol:
- **Course outline responses**: Include course title, course link (if available), and complete lesson list with numbers and titles
- **Content responses**: Provide detailed answers based on search results
- **Sequential responses**: Build upon previous tool results, referencing information gathered in earlier searches
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with up to 2 rounds of sequential tool usage.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string

        Termination conditions:
        - Maximum 2 rounds completed
        - No tool_use blocks in response
        - Tool execution fails
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize conversation state for sequential tool calling
        messages = [{"role": "user", "content": query}]
        round_count = 0
        max_rounds = 2

        # Sequential tool calling loop
        while round_count < max_rounds:
            # Prepare API call parameters with tools available (make a copy of messages)
            api_params = {
                **self.base_params,
                "messages": messages.copy(),
                "system": system_content,
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # Termination condition: no tool use
            if response.stop_reason != "tool_use":
                return response.content[0].text

            # Handle tool execution for this round
            tool_execution_result = self._execute_tools_for_round(
                response, tool_manager, round_count + 1
            )

            # Termination condition: tool execution failed
            if tool_execution_result is None:
                return "I encountered an error while processing your request."

            # Add AI response and tool results to conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_execution_result})

            round_count += 1

        # Final API call after max rounds (without tools to force completion)
        final_params = {
            **self.base_params,
            "messages": messages.copy(),
            "system": system_content,
            # No tools - force Claude to provide final answer
        }

        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _execute_tools_for_round(self, response, tool_manager, round_number: int):
        """
        Execute tools for a single round with comprehensive error handling.

        Args:
            response: The response containing tool use requests
            tool_manager: Manager to execute tools
            round_number: Current round number (for debugging)

        Returns:
            List of tool results, or None if execution failed
        """
        if not tool_manager:
            return None

        tool_results = []

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    # Execute the tool
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    # Validate result
                    if not tool_result or not isinstance(tool_result, str):
                        tool_result = f"Tool {content_block.name} returned no results"

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )

                except Exception as e:
                    # Create error result but continue with other tools
                    error_message = f"Error executing {content_block.name}: {str(e)}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": error_message,
                        }
                    )

        return tool_results if tool_results else None

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
