import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def create_db():
    # CSV Oku
    df = pd.read_csv("SinifProgramiYeniDüzenlendi.csv", sep=None, engine='python', encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Döküman oluştur
    documents = []
    for _, row in df.iterrows():
        content = f"Sınıf: {row['Sinif']}, Gün: {row['Gun']}, Saat: {row['Girilen Ders Saati']}, Ders: {row['Ders']}, Öğretmen: {row['Ogretmen']}, Yer: {row['Yer']}"
        documents.append(Document(page_content=content, metadata=row.to_dict()))

    # Vektörleştir ve Kaydet
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local("faiss_index")
    print("✅ 'faiss_index' klasörü oluşturuldu.")

if __name__ == "__main__":
    create_db()
