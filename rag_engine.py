import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS # Veya Chroma, hangisini istersen
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class OkulAsistani:
    def __init__(self, groq_api_key, db_path="vektor_db.index"):
        # 1. Embedding Modelini Yükle
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 2. Vektör DB'yi Yükle (Daha önce oluşturduğun FAISS indeksi)
        # Not: Eğer Chroma kullanıyorsan Chroma.from_persist_directory kullanmalısın
        self.vector_db = FAISS.load_local(
            db_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

        # 3. Groq LLM Ayarı (Llama 3 - En hızlısı)
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-8b-8192",
            temperature=0.1
        )

        # 4. Prompt Tasarımı (Kişilik kazandırdığımız kısım)
        prompt_template = """
        Sen bir okul ders programı asistanısın. Aşağıdaki ders programı verilerini kullanarak soruyu yanıtla.
        Bilmediğin bir şey olursa "Bu konuda bilgim yok" de, asla uydurma.
        
        PROGRAM VERİLERİ:
        {context}
        
        SORU: {question}
        
        CEVAP:"""
        
        self.QA_PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )

    def cevapla(self, soru):
        # 5. RAG Zincirini Kur ve Çalıştır
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": self.QA_PROMPT}
        )
        
        sonuc = qa_chain.invoke(soru)
        return sonuc["result"]

# --- KONTROL ETMEK İÇİN (OPSİYONEL) ---
if __name__ == "__main__":
    # Test amaçlı buraya key yazabilirsin, ama arayüzde secrets kullanacağız.
    KEY = "gsk_xxxx..." 
    asistan = OkulAsistani(groq_api_key=KEY)
    print(asistan.cevapla("Matematik dersi ne zaman?"))
