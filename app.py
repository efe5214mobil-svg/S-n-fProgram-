import streamlit as st
from rag_engine import OkulAsistani

st.set_page_config(page_title="AI Ders Programı", page_icon="🎓")
st.title("🎓 Akıllı Ders Programı Asistanı")

@st.cache_resource
def init_engine():
    if "GROQ_API_KEY" in st.secrets:
        return OkulAsistani(st.secrets["GROQ_API_KEY"])
    return None

asistan = init_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if asistan:
            with st.spinner("Düşünüyorum..."):
                cevap = asistan.sorgula(prompt)
                st.markdown(cevap)
                st.session_state.messages.append({"role": "assistant", "content": cevap})
        else:
            st.error("API Anahtarı eksik!")
