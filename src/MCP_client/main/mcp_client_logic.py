from contextlib import AsyncExitStack
from typing import Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from MCP_client.personality.prompt import SYSTEM_PROMPT
import json
import os

load_dotenv()
class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.conversation_history = []
    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server and list available tools."""
        # Setup SSE connection
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize session
        await self.session.initialize()

        print("Connected to SSE server.")
        print("Listing tools...")

        response = await self.session.list_tools()
        tools = response.tools

        print("Available tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")

    async def cleanup(self):
        """Cleanup session and stream contexts."""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and call tools as needed."""

        # List tools from MCP
        tool_response = await self.session.list_tools()
        tool_declarations = []

        for tool in tool_response.tools:
            tool_declarations.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            })

        tools = types.Tool(function_declarations=tool_declarations)
        config = types.GenerateContentConfig(tools=[tools])

        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=query)]
            )
        ]

        response = self.genai_client.models.generate_content(
            model="gemini-2.0-flash",
            config=config,
            contents=contents,
        )

        part = response.candidates[0].content.parts[0]
        final_text = []
        tool_results = []

        if hasattr(part, "function_call"):
            tool_call = part.function_call
            tool_name = tool_call.name
            tool_args = tool_call.args

            # Call MCP tool
            result = await self.session.call_tool(tool_name, tool_args)
            tool_results.append({"call": tool_name, "result": result})
            final_text.append(f"[Called tool {tool_name} with args {tool_args}]")

            # Return result to Gemini
            contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(name=tool_name, response={"result": result.content})]
            ))

            followup = self.genai_client.models.generate_content(
                model="gemini-2.0-flash",
                config=config,
                contents=contents,
            )

            final_text.append(followup.text)
        else:
            final_text.append(part.text)

        return "\n".join(final_text)

    async def chat(self, query: str) -> dict:
        """
        Enhanced chat method that maintains conversation history and handles tools gracefully.
        """
        result = await self.process_chat_query(query, self.conversation_history)

        # Update internal conversation history
        self.conversation_history = result["conversation_history"]

        return result

    async def process_chat_query(self, query: str, conversation_history: list = None) -> dict:
        """
        Process a chat query with maintained conversation history and optional tool invocation.
        """
        if conversation_history is None:
            conversation_history = []

        # List available tools from MCP
        tool_response = await self.session.list_tools()
        tool_declarations = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
            for tool in tool_response.tools
        ]

        # Configure tools
        config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=[types.Tool(
            function_declarations=tool_declarations)]) if tool_declarations else types.GenerateContentConfig()

        # Build conversation contents
        contents = conversation_history.copy()
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=query)]
        ))

        # Generate initial response
        response = self.genai_client.models.generate_content(
            model="gemini-2.0-flash",
            config=config,
            contents=contents,
        )

        tool_calls_made = []
        final_text = []
        result = []

        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call:
                tool_call = part.function_call
                tool_name = tool_call.name
                tool_args = dict(tool_call.args)

                try:
                    # First tool call
                    result = await self.session.call_tool(tool_name, tool_args)
                    tool_result_data = result.content
                    tool_data = json.loads(tool_result_data[0].text) if isinstance(tool_result_data,
                                                                                   list) else json.loads(
                        tool_result_data)

                    # Check for fallback if tool is retrieve_documents
                    fallback_needed = False
                    if tool_name == "retrieve_documents":
                        try:
                            top_score = tool_data["results"][0][1][0]["score"]
                            fallback_needed = top_score < 0.7
                        except (KeyError, IndexError, TypeError):
                            fallback_needed = True

                    if fallback_needed:
                        # Fallback to web_search
                        fallback_tool_name = "web_search"
                        fallback_args = {"query": query}
                        fallback_result = await self.session.call_tool(fallback_tool_name, fallback_args)

                        tool_calls_made.extend([
                            {
                                "tool_name": tool_name,
                                "args": tool_args,
                                "result": tool_result_data
                            },
                            {
                                "tool_name": fallback_tool_name,
                                "args": fallback_args,
                                "result": fallback_result.content
                            }
                        ])

                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[types.Part.from_function_response(
                            name=tool_name,
                            response={"result": tool_result_data}
                        )]))
                        contents.append(types.Content(role="model", parts=[
                            types.Part(function_call=types.FunctionCall(name=fallback_tool_name, args=fallback_args))]))
                        contents.append(types.Content(role="user", parts=[types.Part.from_function_response(
                            name=fallback_tool_name,
                            response={"result": fallback_result.content}
                        )]))

                    else:
                        tool_calls_made.append({
                            "tool_name": tool_name,
                            "args": tool_args,
                            "result": tool_result_data
                        })

                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[types.Part.from_function_response(
                            name=tool_name,
                            response={"result": tool_result_data}
                        )]))

                except Exception as e:
                    error_msg = f"Error calling tool {tool_name}: {str(e)}"
                    tool_calls_made.append({
                        "tool_name": tool_name,
                        "args": tool_args,
                        "error": str(e)
                    })

                    contents.append(types.Content(role="user", parts=[types.Part(text=error_msg)]))

            elif hasattr(part, "text") and part.text:
                final_text.append(part.text)

        # If any tool calls were made, follow up with additional model response
        if tool_calls_made:
            try:
                followup = self.genai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    config=config,
                    contents=contents,
                )

                final_response = ""
                for part in followup.candidates[0].content.parts:
                    if hasattr(part, "text"):
                        final_response += part.text

                final_text.append(final_response)
                contents.append(types.Content(role="model", parts=[types.Part(text=final_response)]))

            except Exception as e:
                error_msg = "I encountered an issue processing the tool results. Please try again."
                final_text.append(error_msg)
                contents.append(types.Content(role="model", parts=[types.Part(text=error_msg)]))
        else:
            # No tool call - just return generated text
            response_text = "\n".join(final_text)
            contents.append(types.Content(role="model", parts=[types.Part(text=response_text)]))

        return {
            "documents": json.loads(result.content[0].text) if result else [],
            "response": "\n".join(final_text),
            "tool_calls": tool_calls_made,
            "conversation_history": contents,
            "has_tool_calls": len(tool_calls_made) > 0
        }

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []

    def get_conversation_info(self) -> dict:
        """Get information about the current conversation."""
        return {
            "message_count": len(self.conversation_history),
            "has_messages": len(self.conversation_history) > 0,
            "connected_to_server": self.session is not None
        }

    async def get_available_tools(self) -> list:
        """Get list of available tools with their descriptions."""
        if not self.session:
            return []

        tool_response = await self.session.list_tools()
        tools_info = []

        for tool in tool_response.tools:
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            })

        return tools_info
