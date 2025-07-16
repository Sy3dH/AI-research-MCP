from pdfminer.high_level import extract_text
from typing import List
def parse_pdf_chunks(file_path: str, max_chunk_size: int = 500) -> List[str]:
    """
    Extract text from a PDF using pdfminer and chunk it into parts of approximately max_chunk_size characters.
    """
    full_text = extract_text(file_path).replace("\n", " ")
    sentences = full_text.split(". ")

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 2 <= max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
