import streamlit as st
from groq import Groq
from vector_db import VectorDB

class RAGSystem:
    def __init__(self):
        self.db = VectorDB()

        self.client = Groq(
            api_key=st.secrets["GROQ_API_KEY"]
        )

    def extract_entities(self, query: str):
        siniflar = ["9-A/BL","9-B/BL","9-C/BL","9-D/BL","9-E/EL","9-F/EL","9-G/EL","9-H/EL","9-I/EL","9/ATP","10-A/BL","10-B/BL","10-D/HB","10-E/HB","10/ATP","11-A/BL","11-B/BL","11-C/EN","11-D/HB","11/ATP","12-A/BL","12-B/BL","12-F/EN","12-G/HB","12/ATP"]

        sinif = None
        for s in siniflar:
            if s.lower() in query.lower():
                sinif = s
                break

        gunler = ["pazartesi","salı","çarşamba","perşembe","cuma"]

        gun = None
        for g in gunler:
            if g in query.lower():
                gun = g.capitalize()
                break

        return sinif, gun

    def generate_answer(self, query: str):
        sinif, gun = self.extract_entities(query)

        results = self.db.search(query, sinif=sinif, k=5)

        if gun:
            results = [r for r in results if r["gun"].lower() == gun.lower()]

        context = "\n\n".join([
            f"{r['sinif']} | {r['gun']} | {r['saat']} | {r['ders']} | {r['ogretmen']}"
            for r in results
        ])

        prompt = f"""
Sen bir ders programı asistanısın.

Sadece verilen veriyi kullan.
Tahmin yapma.

BAĞLAM:
{context}

SORU:
{query}

Cevap:
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "Ders programı asistanısın."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content
