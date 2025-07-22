import os
import uvicorn
import logging
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from main.mcp_client_logic import MCPClient

load_dotenv()
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 🚀 Initialize FastAPI
app = FastAPI(title="MCP Client for ResearchSoup")

# Request body model
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_mcp(request: QueryRequest):
    query = request.query.strip()
    if query.lower() in {"quit", "exit"}:
        raise HTTPException(status_code=400, detail="Quit/exit command not allowed via API.")

    client = MCPClient()  # 🔥 Create a fresh MCPClient for this request

    try:
        logger.info("Connecting to SSE server...")
        await client.connect_to_sse_server(server_url=MCP_SERVER_URL)

        response = await client.process_query(query)

        await client.cleanup()  # 🔄 Cleanup after request
        return {"response": response}
    except Exception as e:
        logger.exception("Error in /query endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_mcp(request: QueryRequest):
    query = request.query.strip()
    if query.lower() in {"quit", "exit"}:
        raise HTTPException(status_code=400, detail="Quit/exit command not allowed via API.")

    client = MCPClient()  # 🔥 Create a fresh MCPClient for this request

    try:
        logger.info("Connecting to SSE server...")
        await client.connect_to_sse_server(server_url=MCP_SERVER_URL)

        response = await client.chat(query)

        await client.cleanup()  # 🔄 Cleanup after request
        logger.info(f"Chat response: {response}")
        return {"response": response}
    except Exception as e:
        logger.exception("Error in /chat endpoint")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
