import io
import os
import uuid
import uvicorn
import logging
import requests
import base64

from dotenv import load_dotenv
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse

from fastapi_mcp import FastApiMCP
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from models.models import Document
from embeddings.vectorizer import FastEmbedder
from parsers.pdf_parser import parse_pdf_chunks
from utils.input_handler import prepare_documents
from ingestion.vector_store import VectorStore
from search.store_search import (
    vector_search_with_filter,
    is_vector_similar,
    list_vector_stores,
    is_collection_exist,
    extract_from_all_collections,
)

# ----------------------------------------------------
# Load environment and initialize logger
# ----------------------------------------------------
load_dotenv()
BRAVE_SEARCH_API = os.getenv("BRAVE_SEARCH_API")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------
# FastAPI App and Lifespan Setup
# ----------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI app lifespan...")
    embedder = FastEmbedder()
    print(embedder)
    app.state.embedder = embedder
    yield
    logger.info("Shutting down FastAPI app lifespan...")


app = FastAPI(
    title="MCP Server for ResearchSoup",
    lifespan=lifespan,
    swagger_ui_parameters={"operationsSorter": "method"},
)


# ----------------------------------------------------
# Document Ingestion Endpoint
# ----------------------------------------------------
@app.post("/ingest_documents")
async def ingest_documents(
    researcher: str = Form(...),
    findings: str = Form(...),
    collection_name: str = Form("AI_store"),
    embedding_model: Optional[str] = Form(None),
    pdfs: List[UploadFile] = File(...),
):
    try:
        embedder = app.state.embedder
        vector_size = embedder.vector_size

        if embedding_model:
            logger.info(f"Using custom embedding model: {embedding_model}")
            embedder = FastEmbedder(embedding_model)
            vector_size = embedder.vector_size

        file_buffers, gcs_paths, all_docs, all_vectors = [], [], [], []

        # 1️⃣ Process PDFs
        for pdf in pdfs:
            pdf_bytes = await pdf.read()
            file_buffers.append((pdf.filename, pdf_bytes))
            gcs_filename = f"uploads/{uuid.uuid4()}_{pdf.filename}"
            gcs_paths.append(gcs_filename)

            chunks = parse_pdf_chunks(io.BytesIO(pdf_bytes))
            logger.info(f"Extracted {len(chunks)} chunks from {pdf.filename}")

            chunk_vectors = embedder.embed(chunks)

            for i, (chunk, vector) in enumerate(zip(chunks, chunk_vectors)):
                doc = Document(
                    text=chunk,
                    metadata={
                        "file_path": gcs_filename,
                        "filename": pdf.filename,
                        "researcher": researcher,
                        "chunk_index": i,
                        "num_chunks": len(chunks),
                    },
                )
                all_docs.append(doc)
                all_vectors.append(vector)

        # 2️⃣ Prepare documents
        docs = prepare_documents(researcher, findings, all_docs)
        store = VectorStore(collection_name=collection_name, vector_size=vector_size)

        if not is_collection_exist(collection_name):
            store.init_collection()

        # 3️⃣ Check for duplicates using similarity
        search_queries = [
            models.QueryRequest(query=vector, score_threshold=0.95)
            for vector in all_vectors
        ]
        search_results = store.client.query_batch_points(
            collection_name=collection_name,
            requests=search_queries,
        )

        insertable_docs, insertable_vectors, skipped_docs = [], [], []
        for doc, vector, result in zip(docs, all_vectors, search_results):
            if result.points and len(result.points) > 0:
                skipped_docs.append(doc)
            else:
                insertable_docs.append(doc)
                insertable_vectors.append(vector)

        # 4️⃣ Upload selected files
        used_indices = {i for i, d in enumerate(all_docs) if d in insertable_docs}
        uploaded_files = []

        for i in used_indices:
            original_name, content = file_buffers[i]
            gcs_filename = gcs_paths[i]
            # upload_to_gcs(content, gcs_filename)
            # logger.info(f"Uploaded file to GCS: {gcs_filename}")
            uploaded_files.append(gcs_filename)

        # 5️⃣ Ingest documents into the vector store
        if insertable_docs and insertable_vectors:
            store.ingest_batch(insertable_docs, insertable_vectors)

        return {
            "message": f"Ingested {len(insertable_docs)} chunks into '{collection_name}' successfully.",
            "docs_ingested": len(insertable_docs),
            "docs_skipped": len(skipped_docs),
            "inserted_documents": [Path(d.pdf_path).name for d in insertable_docs],
            "skipped_documents": [Path(d.pdf_path).name for d in skipped_docs],
            "saved_files": uploaded_files,
        }

    except Exception as e:
        logger.exception("Failed to ingest documents")
        return {
            "error": str(e),
            "message": "Document ingestion failed. See server logs for more details.",
        }


# ----------------------------------------------------
# File Download Endpoint
# ----------------------------------------------------
@app.get("/download/")
def download_file(file_path: str):
    """Download a PDF file by providing its full path on the server."""
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    if not file_path.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=os.path.basename(file_path),
    )


# ----------------------------------------------------
# Document Retrieval Endpoint
# ----------------------------------------------------
@app.post(
    "/retrieve_documents",
    operation_id="retrieve_documents",
    description="Retrieve and summarize relevant documents from vector store.",
)
async def retrieve_documents(
    query_text: str, collection_name: str = "AI_store", limit: int = 5
):
    try:
        embedder = app.state.embedder
        query_vector = embedder.embed([query_text])[0]

        print(query_vector)
        results = vector_search_with_filter(
            query_vector=query_vector, collection_name=collection_name, limit=limit
        )
        return {"results": [res for res in results]}

    except UnexpectedResponse as e:
        return JSONResponse(
            status_code=500, content={"error": f"Qdrant query failed: {str(e)}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Unexpected error: {str(e)}"}
        )


# ----------------------------------------------------
# Web Search Endpoint (Brave API)
# ----------------------------------------------------
@app.post(
    "/search_web",
    operation_id="search_web",
    description="Search queries over the internet and provide research links with summaries.",
)
async def search_web(query: str):
    try:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "x-subscription-token": BRAVE_SEARCH_API,
            },
            params={"q": query},
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, detail="Brave API Error"
            )

        json_data = response.json()
        web_results = json_data.get("web", {}).get("results", [])
        results = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "description": item.get("description"),
            }
            for item in web_results
        ]

        return {"results": results}

    except requests.exceptions.RequestException as e:
        return JSONResponse(
            status_code=500, content={"error": f"Request failed: {str(e)}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Unexpected error: {str(e)}"}
        )


# ----------------------------------------------------
# Retrieve Vector Stores
# ----------------------------------------------------
@app.get(
    "/retrieve_vector_stores",
    operation_id="retrieve_vector_stores",
    description="Retrieve the names of all vector stores.",
)
async def retrieve_vector_stores():
    return list_vector_stores()


# ----------------------------------------------------
# Collection Details Endpoint
# ----------------------------------------------------
@app.get(
    "/collection_details",
    operation_id="get_collection_details",
    description="Get details of all vector stores including researchers, files, and counts.",
)
async def get_collection_details():
    try:
        collection_names = list_vector_stores()
        all_collections_data = {}

        for collection_name in collection_names:
            try:
                store = VectorStore(
                    collection_name=collection_name,
                    vector_size=app.state.embedder.vector_size,
                )
                unique_researchers = set()
                file_objects = {}
                total_chunks = 0
                next_offset = None

                while True:
                    scroll_result, next_offset = store.client.scroll(
                        collection_name=collection_name,
                        scroll_filter=None,
                        limit=1000,
                        with_payload=True,
                        with_vectors=False,
                        offset=next_offset,
                    )
                    if not scroll_result:
                        break

                    total_chunks += len(scroll_result)

                    for point in scroll_result:
                        metadata = point.payload
                        researcher = metadata.get("researcher")
                        pdf_path = metadata.get("pdf_path")

                        if pdf_path:
                            filename = os.path.basename(pdf_path)
                            if researcher:
                                unique_researchers.add(researcher)
                            if pdf_path not in file_objects:
                                file_objects[pdf_path] = {
                                    "filename": filename,
                                    "researcher": researcher,
                                }

                    if next_offset is None:
                        break

                total_documents = len(file_objects)
                files_list = list(file_objects.values())

                all_collections_data[collection_name] = {
                    "researchers": sorted(list(unique_researchers)),
                    "files": files_list,
                    "total_documents": total_documents,
                    "total_chunks": total_chunks,
                }

            except UnexpectedResponse as e:
                logger.warning(
                    f"Collection '{collection_name}' failed to query: {str(e)}"
                )
                all_collections_data[collection_name] = {
                    "error": f"Failed to query collection: {str(e)}",
                    "researchers": [],
                    "files": [],
                    "total_documents": 0,
                    "total_chunks": 0,
                }

        return all_collections_data

    except Exception as e:
        logger.exception("Failed to retrieve collection details")
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"},
        )


@app.post(
    "/chat_with_researcher",
    operation_id="chat_with_researcher",
    description="Chat specifically with documents uploaded by a given researcher (supports JSON or form input).",
)
async def chat_with_researcher(
    request: Request,
    researcher: str = Form(None, description="Name of the researcher."),
    query_text: str = Form(None, description="User's question or search query."),
    collection_name: str = Form("AI_store", description="Vector collection name."),
    limit: int = Form(5, description="Number of documents to retrieve."),
):
    try:
        # Try JSON body first
        try:
            json_data = await request.json()
            if json_data:
                researcher = json_data.get("researcher", researcher)
                query_text = json_data.get("query_text", query_text)
                collection_name = json_data.get("collection_name", collection_name)
                limit = int(json_data.get("limit", limit))
        except Exception:
            # Ignore if not JSON — it might be form-data
            pass

        # Validate required fields
        if not researcher or not query_text:
            return JSONResponse(
                status_code=400,
                content={"error": "Both 'researcher' and 'query_text' are required."},
            )

        # Embed query text
        embedder = app.state.embedder
        query_vector = embedder.embed([query_text])[0]

        # Perform vector search
        results = vector_search_with_filter(
            query_vector=query_vector,
            collection_name=collection_name,
            limit=limit,
            keyword=researcher,
        )

        # Handle no results
        if not results.points:
            return JSONResponse(
                status_code=404,
                content={
                    "message": f"No documents found for researcher '{researcher}'."
                },
            )

        # Build response
        docs = [
            {
                "id": p.id,
                "score": getattr(p, "score", None),
                "text": p.payload.get("text", ""),
                "researcher": p.payload.get("researcher", "unknown"),
                "filename": p.payload.get("filename", "unknown"),
            }
            for p in results.points
        ]

        combined_text = " ".join([d["text"] for d in docs if d["text"].strip()])

        simulated_response = (
            f"Based on {researcher}'s research documents, here’s what I found regarding '{query_text}':\n\n"
            f"{combined_text[:600]}\n\n"
            f"(Summary generated from {len(results.points)} retrieved documents.)"
        )

        return {
            "researcher": researcher,
            "query": query_text,
            "results_found": len(results.points),
            "documents": docs,
            "context_snippet": combined_text[:1500],
            "response": simulated_response,
        }

    except UnexpectedResponse as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Qdrant query failed: {str(e)}"},
        )

    except Exception as e:
        logger.exception("Chat with researcher failed")
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}"},
        )


# ----------------------------------------------------
# Health Check
# ----------------------------------------------------
@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for readiness probes"""
    return JSONResponse(content={"status": "ok"})


# ----------------------------------------------------
# Main MCP Mount and Server Start
# ----------------------------------------------------
if __name__ == "__main__":
    mcp = FastApiMCP(
        app,
        include_operations=[
            "retrieve_documents",
            "retrieve_vector_stores",
            "search_web",
            "chat_with_researcher",
        ],
    )
    mcp.mount()

    port = int(os.environ.get("PORT", 8082))
    uvicorn.run(app, host="0.0.0.0", port=port)
