import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

def create_db():
    # 1. Veriyi oku
    df = pd.read_csv("SinifProgramiYeniDüzenlendi.csv", sep=None, engine='python', encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # 2. Her satırı bir dökümana dönüştür
    documents = []
    for _, row in df.iterrows():
        # LLM'in en iyi anlayacağı metin formatı
        content = f"Sınıf: {row['Sinif']}, Gün: {row['Gun']}, Saat: {row['Girilen Ders Saati']}, Ders: {row['Ders']}, Öğretmen: {row['Ogretmen']}, Yer: {row['Yer']}"
        doc = Document(page_content=content, metadata=row.to_dict())
        documents.append(doc)

    # 3. Embedding modelini seç ve DB oluştur
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documents, embeddings)

    # 4. Kaydet
    vector_db.save_local("faiss_index")
    print("✅ Vektör veritabanı 'faiss_index' klasörüne kaydedildi.")

if __name__ == "__main__":
    create_db()
