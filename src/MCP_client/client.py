import os
import uvicorn
import logging
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from main.mcp_client_logic import MCPClient

load_dotenv()
MCP_SERVER_URL = "http://localhost:8080/mcp"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 🚀 Initialize FastAPI
app = FastAPI(title="MCP Client for ResearchSoup")

class ToolName(str, Enum):
    retrieve_documents = "retrieve_documents"
    retrieve_vector_stores = "retrieve_vector_stores"
    search_web = "search_web"

class QueryRequest(BaseModel):
    query: str = Field()
    tool_name: Optional[ToolName] = Field(
        None, description="Select the tool to use for this query"
    )
@app.get("/list_tools")
async def list_tools():
    client = MCPClient()  # 🔥 Create a fresh MCPClient for this request
    try:
        logger.info("Connecting to SSE server...")
        await client.connect_to_sse_server(server_url=MCP_SERVER_URL)
        tools = await client.get_available_tools()
        return tools
    except Exception as e:
        logger.exception("Error in /list_tools endpoint")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/query")
# # async def query_mcp(request: QueryRequest):
# #     query = request.query.strip()
# #     if query.lower() in {"quit", "exit"}:
# #         raise HTTPException(status_code=400, detail="Quit/exit command not allowed via API.")
# #
# #     client = MCPClient()  # 🔥 Create a fresh MCPClient for this request
# #
# #     try:
# #         logger.info("Connecting to SSE server...")
# #         await client.connect_to_sse_server(server_url=MCP_SERVER_URL)
# #
# #         response = await client.(query)
# #
# #         await client.cleanup()  # 🔄 Cleanup after request
# #         return {"response": response}
# #     except Exception as e:
# #         logger.exception("Error in /query endpoint")
# #         raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_mcp(
    query: str = Query(..., description="Your question or prompt", example="Summarize this document."),
    tool_name: Optional[ToolName] = Query(
        None,
        description="Select a tool to process the query"
    )
):
    query = query.strip()
    print("query:", query)
    print("tool_name:", tool_name)

    if query.lower() in {"quit", "exit"}:
        raise HTTPException(status_code=400, detail="Quit/exit command not allowed via API.")

    client = MCPClient()

    try:
        logger.info("Connecting to SSE server...")
        await client.connect_to_sse_server(server_url=MCP_SERVER_URL)

        if tool_name:
            response = await client.chat_with_tool(query, tool_name)
        else:
            response = await client.chat_with_all_tools(query)

        await client.cleanup()
        logger.info(f"Chat response: {response}")
        return {"response": response}

    except Exception as e:
        logger.exception("Error in /chat endpoint")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
