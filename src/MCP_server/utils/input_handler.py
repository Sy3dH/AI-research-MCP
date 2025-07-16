from typing import List
from MCP_server.models.models import DocumentModel, Document

def prepare_documents(researcher: str, findings: str, chunks: List[Document]) -> List[DocumentModel]:
    return [
        DocumentModel(
            text=chunk.text,
            researcher=researcher,
            findings=findings,
            pdf_path= chunk.metadata["file_path"],
        )
        for chunk in chunks
    ]
