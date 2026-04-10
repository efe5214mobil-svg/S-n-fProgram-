import os
import streamlit as st
import faiss
import pickle
import numpy as np
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer

# =========================
# 🔥 AUTO BUILD VECTOR DB
# =========================
class VectorBuilder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def build_text(self, row):
        return f"""
Sınıf: {row['sinif']}
Gün: {row['gun']}
Saat: {row['saat']}
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


# =========================
# 🔥 VECTOR DB
# =========================
class VectorDB:
    def __init__(self):
        self.index = faiss.read_index("vector_db.index")

        with open("metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text):
        return self.model.encode([text])[0].astype("float32")

    def search(self, query, sinif=None, k=5):
        q = self.embed(query).reshape(1, -1)

        _, indices = self.index.search(q, k * 3)

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


# =========================
# 🔥 RAG SYSTEM
# =========================
class RAGSystem:
    def __init__(self):
        self.db = VectorDB()

        self.client = Groq(
            api_key=st.secrets["GROQ_API_KEY"]
        )

    def extract(self, query):
        siniflar = ["9-A","9-B","10-A","10-B","11-A","11-B","12-A","12-B"]

        sinif = None
        for s in siniflar:
            if s.lower() in query.lower():
                sinif = s

        gunler = ["pazartesi","salı","çarşamba","perşembe","cuma"]

        gun = None
        for g in gunler:
            if g in query.lower():
                gun = g.capitalize()

        return sinif, gun

    def generate(self, query):
        sinif, gun = self.extract(query)

        results = self.db.search(query, sinif=sinif)

        if gun:
            results = [r for r in results if r["gun"].lower() == gun.lower()]

        context = "\n\n".join([
            f"{r['sinif']} | {r['gun']} | {r['saat']} | {r['ders']} | {r['ogretmen']}"
            for r in results
        ])

        prompt = f"""
Sen ders programı asistanısın.

SADECE VERİYİ KULLAN.

BAĞLAM:
{context}

SORU:
{query}
"""

        res = self.client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "Ders programı asistanısın."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return res.choices[0].message.content


# =========================
# 🔥 STREAMLIT APP
# =========================
st.set_page_config(page_title="Ders Programı AI", layout="wide")

st.title("📚 Akıllı Ders Programı Sistemi")

# =========================
# AUTO INIT DB (CRITICAL FIX)
# =========================
if not os.path.exists("vector_db.index"):
    st.warning("Vector DB oluşturuluyor...")
    builder = VectorBuilder()
    builder.build("SinifProgramiYeniDüzenlendi.csv")
    st.success("Vector DB hazır!")

# =========================
# RAG INIT
# =========================
rag = RAGSystem()

query = st.text_input("Sorunu yaz:")

if "chat" not in st.session_state:
    st.session_state.chat = []

if st.button("Gönder"):
    if query:
        answer = rag.generate(query)
        st.session_state.chat.append((query, answer))

for q, a in reversed(st.session_state.chat):
    st.markdown(f"**🧑 Sen:** {q}")
    st.markdown(f"**🤖 AI:** {a}")
    st.divider()
