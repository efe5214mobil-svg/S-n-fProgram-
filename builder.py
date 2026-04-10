import pandas as pd
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorBuilder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def build_text(self, row):
        return f"""
Sınıf: {row['sinif']}
Gün: {row['gun']}
Girilen Ders Saati : {row['saat']}
Ders: {row['ders']}
Öğretmen: {row['ogretmen']}
"""

    def build(self, csv_path):
        df = pd.read_csv(csv_path)

        texts = []
        metadata = []

        for _, row in df.iterrows():
            texts.append(self.build_text(row))
            metadata.append(row.to_dict())

        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, "vector_db.index")

        with open("metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        print("✔ Vector DB oluşturuldu")
