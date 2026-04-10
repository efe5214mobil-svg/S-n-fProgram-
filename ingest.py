import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

def create_vector_db(csv_path):
    print("🔄 Veriler okunuyor ve işleniyor...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # CSV Oku
    df = pd.read_csv(csv_path, sep=None, engine='python', encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Tüm sütunları birleştirerek anlamlı bir metin oluştur
    df["full_text"] = df.apply(lambda row: " ".join(row.values.astype(str)), axis=1)

    # Embedding oluştur
    print("🧠 Vektörler oluşturuluyor (Embedding)...")
    embeddings = model.encode(df["full_text"].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # FAISS İndeksi oluştur
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Kaydet
    faiss.write_index(index, "vektor_db.index")
    df.to_pickle("metadata.pkl")
    print(f"✅ Başarılı! {len(df)} kayıt 'vektor_db.index' ve 'metadata.pkl' olarak kaydedildi.")

if __name__ == "__main__":
    create_vector_db("SinifProgramiYeniDüzenlendi2.csv")
