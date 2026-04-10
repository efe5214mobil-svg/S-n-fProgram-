import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self):
        self.index = faiss.read_index("vector_db.index")

        with open("metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text):
        return self.model.encode([text])[0].astype("float32")

    def search(self, query: str, sinif: str = None, k: int = 5):
        query_vec = self.embed(query).reshape(1, -1)

        distances, indices = self.index.search(query_vec, k * 3)

        results = []

        for i in indices[0]:
            if i < len(self.metadata):
                item = self.metadata[i]

                if sinif and item["sinif"] != sinif:
                    continue

                results.append(item)

            if len(results) >= k:
                break

        return results
