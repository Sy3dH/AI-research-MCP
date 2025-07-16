from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DocumentModel(BaseModel):
    text: str
    researcher: str
    findings: Optional[str] = ""
    pdf_path: str
    time: Optional[str] = Field(default_factory=lambda: datetime.now().isoformat())

class Document:
    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata