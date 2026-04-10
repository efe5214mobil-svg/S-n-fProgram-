import streamlit as st
from rag import RAGSystem

st.set_page_config(page_title="Ders Programı AI", layout="wide")

st.title("📚 Ders Programı Asistanı")

rag = RAGSystem()

query = st.text_input("Sorunu yaz:")

if "chat" not in st.session_state:
    st.session_state.chat = []

if st.button("Gönder"):
    if query:
        answer = rag.generate_answer(query)
        st.session_state.chat.append((query, answer))

for q, a in reversed(st.session_state.chat):
    st.markdown(f"**🧑 Sen:** {q}")
    st.markdown(f"**🤖 AI:** {a}")
    st.divider()
