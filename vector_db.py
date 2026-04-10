import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self):
        if not os.path.exists("vector_db.index"):
            raise FileNotFoundError(
                "vector_db.index bulunamadı. Önce builder.py çalıştır."
            )

        self.index = faiss.read_index("vector_db.index")

        with open("metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
