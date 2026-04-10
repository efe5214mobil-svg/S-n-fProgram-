from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

class OkulAsistani:
    def __init__(self, api_key):
        # Kaynakları Yükle
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = FAISS.load_local(
            "faiss_index", 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Groq Yapılandırması
        self.llm = ChatGroq(
            groq_api_key=api_key, 
            model_name="llama3-8b-8192", 
            temperature=0.1
        )

        # Prompt Şablonu
        template = """Sen bir okul asistanısın. Aşağıdaki verilere dayanarak soruyu yanıtla.
        Bilgi yoksa 'Bu konuda bilgim yok' de.
        
        VERİLER: {context}
        SORU: {question}
        CEVAP:"""
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def sorgula(self, soru):
        # RetrievalQA kullanımı
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": self.prompt}
        )
        return chain.invoke(soru)["result"]
