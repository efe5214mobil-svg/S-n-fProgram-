import streamlit as st
import pandas as pd
import faiss
import os

# LangChain bileşenlerini güvenli yollardan import ediyoruz
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="AI Okul Asistanı", page_icon="🎓")

# --- ASİSTAN SINIFI ---
class OkulAsistaniEngine:
    def __init__(self, api_key):
        # Embedding modeli
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Vektör DB Yükleme (Klasör isminin 'vektor_db.index' olduğundan emin ol)
        self.db = FAISS.load_local(
            "vektor_db.index", 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

        # Groq LLM Yapılandırması
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-8b-8192",
            temperature=0.2
        )

        # Prompt Şablonu
        template = """Sen bir okul asistanısın. Aşağıdaki ders programı verilerine göre soruyu yanıtla.
        Verilerde bilgi yoksa "Bilmiyorum" de.
        
        VERİLER: {context}
        SORU: {question}
        CEVAP:"""
        
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def sor(self, soru):
        # RAG Zinciri
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": self.prompt}
        )
        return chain.invoke(soru)["result"]

# --- ARAYÜZ ---
st.title("🎓 Akıllı Ders Programı Asistanı")

@st.cache_resource
def asistan_yukle():
    if "GROQ_API_KEY" in st.secrets:
        return OkulAsistaniEngine(st.secrets["GROQ_API_KEY"])
    else:
        st.error("GROQ_API_KEY Secrets içinde bulunamadı!")
        return None

asistan = asistan_yukle()

# Sohbet Geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Soru Girişi
if prompt := st.chat_input("Ders programınız hakkında bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if asistan:
            with st.spinner("Program taranıyor..."):
                response = asistan.sor(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
