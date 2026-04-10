import streamlit as st
import pandas as pd
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="AI Okul Asistanı",
    page_icon="🎓",
    layout="centered"
)

# --- CSS TASARIM ---
st.markdown("""
    <style>
    .stChatInputContainer { padding-bottom: 20px; }
    .stChatMessage { border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- RAG MOTORU (Tek Dosya İçinde) ---
class OkulAsistani:
    def __init__(self, groq_api_key):
        # En uyumlu embedding yolu
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Vektör DB Yükle
        self.vector_db = FAISS.load_local(
            "vektor_db.index", 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

        # Groq LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-8b-8192",
            temperature=0.1
        )

        # Prompt Tasarımı
        template = """
        Sen bir okul ders programı asistanısın. Aşağıdaki verileri kullanarak soruyu yanıtla.
        Verilerde yoksa "Bilmiyorum" de.
        
        VERİLER: {context}
        SORU: {question}
        CEVAP:"""
        
        self.QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    def cevapla(self, soru):
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": self.QA_PROMPT}
        )
        return qa_chain.invoke(soru)["result"]

# --- APP ARAYÜZÜ ---
st.title("🎓 Akıllı Ders Programı Asistanı")

@st.cache_resource
def get_asistan():
    if "GROQ_API_KEY" in st.secrets:
        return OkulAsistani(st.secrets["GROQ_API_KEY"])
    else:
        st.error("Secrets içerisinde GROQ_API_KEY bulunamadı!")
        return None

asistan = get_asistan()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Sorunuzu yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if asistan:
            with st.spinner("Düşünüyorum..."):
                response = asistan.cevapla(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
