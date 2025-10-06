import io
import os
import uuid
import uvicorn
import logging
import requests
from dotenv import load_dotenv
from typing import List, Optional
from fastapi_mcp import FastApiMCP
from models.models import Document
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from embeddings.vectorizer import FastEmbedder
from parsers.pdf_parser import parse_pdf_chunks
from utils.input_handler import prepare_documents
from ingestion.vector_store import VectorStore
from qdrant_client.http.exceptions import UnexpectedResponse
from utils.gcs_utils import upload_to_gcs, generate_signed_url
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from search.store_search import vector_search_with_filter, is_vector_similar, list_vector_stores, is_collection_exist, \
    extract_from_all_collections
from pathlib import Path
from qdrant_client import models

load_dotenv()
BRAVE_SEARCH_API = os.getenv("BRAVE_SEARCH_API")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI app lifespan...")
    embedder = FastEmbedder()
    app.state.embedder = embedder
    yield
    logger.info("Shutting down FastAPI app lifespan...")

app = FastAPI(title="MCP Server for ResearchSoup", lifespan=lifespan, swagger_ui_parameters={"operationsSorter": "method"})

@app.post("/ingest_documents")
async def ingest_documents(
    researcher: str = Form(...),
    findings: str = Form(...),
    collection_name: str = Form("AI_store"),
    embedding_model: Optional[str] = Form(None),
    pdfs: List[UploadFile] = File(...)
):
    try:
        embedder = app.state.embedder
        vector_size = embedder.vector_size

        if embedding_model:
            logger.info(f"Using custom embedding model: {embedding_model}")
            embedder = FastEmbedder(embedding_model)
            vector_size = embedder.vector_size

        file_buffers = []
        gcs_paths = []
        all_docs = []
        all_vectors = []

        for pdf in pdfs:
            pdf_bytes = await pdf.read()
            file_buffers.append((pdf.filename, pdf_bytes))

            gcs_filename = f"uploads/{uuid.uuid4()}_{pdf.filename}"
            gcs_paths.append(gcs_filename)

            chunks = parse_pdf_chunks(io.BytesIO(pdf_bytes))
            logger.info(f"Extracted {len(chunks)} chunks from {pdf.filename}")

            # === 2. Embed all chunks at once ===
            chunk_vectors = embedder.embed(chunks)

            # === 3. Create Document objects per chunk ===
            for i, (chunk, vector) in enumerate(zip(chunks, chunk_vectors)):
                doc = Document(
                    text=chunk,
                    metadata={
                        "file_path": gcs_filename,
                        "filename": pdf.filename,
                        "chunk_index": i,
                        "num_chunks": len(chunks),
                    },
                )
                all_docs.append(doc)
                all_vectors.append(vector)

        docs = prepare_documents(researcher, findings, all_docs)

        store = VectorStore(collection_name=collection_name, vector_size=vector_size)

        if not is_collection_exist(collection_name):
            store.init_collection()

        search_queries = [
            models.QueryRequest(
                query=vector,
                score_threshold=0.95,
            )
            for vector in all_vectors
        ]

        search_results = store.client.query_batch_points(
            collection_name=collection_name,
            requests=search_queries,
        )

        print(search_results)
        insertable_docs = []
        insertable_vectors = []
        skipped_docs = []

        for doc, vector, result in zip(docs, all_vectors, search_results):
            if result.points and len(result.points) > 0:
                skipped_docs.append(doc)
            else:
                insertable_docs.append(doc)
                insertable_vectors.append(vector)

        used_indices = {
            i for i, d in enumerate(all_docs)
            if d in insertable_docs
        }
        uploaded_files = []
        for i in used_indices:
            original_name, content = file_buffers[i]
            gcs_filename = gcs_paths[i]
            upload_to_gcs(content, gcs_filename)
            logger.info(f"Uploaded file to GCS: {gcs_filename}")
            uploaded_files.append(gcs_filename)

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

# @app.post("/ingest_documents")
# async def ingest_documents(
#     researcher: str = Form(...),
#     findings: str = Form(...),
#     collection_name: str = Form("AI_store"),
#     embedding_model: Optional[str] = Form(None),
#     pdfs: List[UploadFile] = File(...)
# ):
#     try:
#         embedder = app.state.embedder
#         vector_size = embedder.vector_size
#
#         if embedding_model:
#             logger.info(f"Using custom embedding model: {embedding_model}")
#             embedder = FastEmbedder(embedding_model)
#             vector_size = embedder.vector_size
#
#         file_buffers = []
#         doc_sources = []
#         gcs_paths = []
#         all_docs = []
#
#         # 1. Read and parse PDFs
#         for idx, pdf in enumerate(pdfs):
#             pdf_bytes = await pdf.read()
#             file_buffers.append((pdf.filename, pdf_bytes))
#
#             gcs_filename = f"uploads/{uuid.uuid4()}_{pdf.filename}"
#             gcs_paths.append(gcs_filename)
#
#             chunks = parse_pdf_chunks(io.BytesIO(pdf_bytes))
#             logger.info(f"Extracted {len(chunks)} chunks from {pdf.filename}")
#
#             for chunk in chunks:
#                 doc = Document(text=chunk, metadata={"file_path": gcs_filename})
#                 doc_sources.append(idx)
#                 all_docs.append(doc)
#
#         # 2. Embed & prepare documents
#         docs = prepare_documents(researcher, findings, all_docs)
#         vectors = embedder.embed([doc.text for doc in all_docs])
#
#         insertable_docs = []
#         insertable_vectors = []
#         skipped_docs = []
#
#         for doc, vector in zip(docs, vectors):
#             if is_collection_exist(collection_name) and is_vector_similar(vector, collection_name):
#                 skipped_docs.append(doc)
#             else:
#                 insertable_docs.append(doc)
#                 insertable_vectors.append(vector)
#
#         used_pdf_indices = set(doc_sources[docs.index(doc)] for doc in insertable_docs)
#         uploaded_files = []
#
#         # 3. Upload selected PDFs to GCS
#         for i in used_pdf_indices:
#             original_name, content = file_buffers[i]
#             gcs_filename = gcs_paths[i]
#             upload_to_gcs(content, gcs_filename)
#             logger.info(f"Uploaded file to GCS: {gcs_filename}")
#             uploaded_files.append(gcs_filename)
#
#         # 4. Ingest into vector store
#         store = VectorStore(collection_name=collection_name, vector_size=vector_size)
#         store.init_collection()
#         store.ingest(insertable_docs, insertable_vectors)
#
#         return {
#             "message": f"Ingested {len(insertable_docs)} chunks into collection '{collection_name}' successfully.",
#             "docs_ingested": len(insertable_docs),
#             "docs_skipped": len(skipped_docs),
#             "inserted_documents": insertable_docs,
#             "skipped_documents": skipped_docs,
#             "saved_files": uploaded_files
#         }
#
#     except Exception as e:
#         logger.exception("Failed to ingest documents")
#         return {
#             "error": str(e),
#             "message": "Document ingestion failed. See server logs for more details."
#         }

@app.get("/download/")
def download_file(file_path: str):
    try:
        signed_url = generate_signed_url(file_path)
        return {"download_url": signed_url}
    except Exception as e:
        logger.exception("Failed to generate signed URL")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/")
def download_file(file_path: str):
    """
    Download a PDF file by providing its full path on the server.
    """
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    if not file_path.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    return FileResponse(
        path=file_path,
        media_type='application/pdf',
        filename=os.path.basename(file_path)
    )

@app.post("/retrieve_documents", operation_id="retrieve_documents", description="This tool is used to retrieve "
                                                                            "documents from a vector store.")
async def retrieve_documents(
    query_text: str,
    collection_name: str = "AI_store",
    limit: int = 5
):
    try:
        embedder = app.state.embedder
        query_vector = embedder.embed([query_text])[0]
        results = vector_search_with_filter(
            query_vector=query_vector,

            collection_name=collection_name,
            limit=limit
        )

        return {"results": [res for res in results]}

    except UnexpectedResponse as e:
        return JSONResponse(status_code=500, content={"error": f"Qdrant query failed: {str(e)}"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {str(e)}"})

# @app.post("/quick_search", operation_id="quick_search", description= "Quickly iterate through multiple vector stores"
#                                                                      "to find the relevant findings and collection_names.")
# async def quick_search(query:str):
#     try:
#         # Search all collections for the query
#         results = extract_from_all_collections(query)
#
#         # Format response
#         response = {
#             "query": query,
#             "total_collections": len(results),
#             "collections": []
#         }
#
#         for collection_name, metadata_list in results.items():
#             collection_data = {
#                 "collection_name": collection_name,
#                 "match_count": len(metadata_list),
#                 "findings": metadata_list
#             }
#             response["collections"].append(collection_data)
#
#         return response
#
#     except Exception as e:
#         return {
#             "error": str(e),
#             "query": query,
#             "collections": []
#         }


@app.post("/search_web", operation_id="search_web", description="Searches your queries over the internet and provide"
                                                            "links to the research along with the summary of them.")
async def search_web(query: str):
    try:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "x-subscription-token": BRAVE_SEARCH_API
            },
            params={
                "q": query
            },
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Brave API Error")

        json_data = response.json()

        # Extract the top N results
        web_results = json_data.get("web", {}).get("results", [])

        # Clean results: title, url, description
        results = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "description": item.get("description")
            }
            for item in web_results
        ]

        return {"results": results}

    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=500, content={"error": f"Request failed: {str(e)}"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {str(e)}"})

@app.get("/retrieve_vector_stores", operation_id="retrieve_vector_stores", description="retrieves the names of vector"
                                                                                       "stores.")
async def retrieve_vector_stores():
    return list_vector_stores()

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for readiness probes"""
    return JSONResponse(content={"status": "ok"})


if __name__ == "__main__":
    mcp = FastApiMCP(app, include_operations=["retrieve_documents", "retrieve_vector_stores", "search_web"])
    mcp.mount()
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)