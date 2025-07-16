from typing import List
from fastembed import TextEmbedding
from numpy import ndarray

class FastEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
            self.model = TextEmbedding(model_name)
            print(f"Model '{model_name}' loaded.")
            self.vector_size = TextEmbedding.get_embedding_size(model_name)

    def embed(self, documents: List[str]) -> List[ndarray]:
        return list(self.model.embed(documents))