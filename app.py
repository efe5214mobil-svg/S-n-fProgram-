import streamlit as st
from rag_engine import OkulAsistani

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="AI Okul Asistanı",
    page_icon="🎓",
    layout="centered"
)

# --- TASARIM (CSS) ---
st.markdown("""
    <style>
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    .stChatMessage {
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Akıllı Ders Programı Asistanı")

# --- MOTORU BAŞLATMA ---
# API Key doğrudan secrets içinden alınıyor
@st.cache_resource
def get_asistan():
    try:
        # Secrets içinde 'GROQ_API_KEY' tanımlı olmalı
        api_key = st.secrets["GROQ_API_KEY"]
        return OkulAsistani(groq_api_key=api_key, db_path="vektor_db.index")
    except Exception as e:
        st.error("Sistem başlatılamadı. Lütfen yöneticiye başvurun veya API anahtarını kontrol edin.")
        return None

asistan = get_asistan()

# --- SOHBET SİSTEMİ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mesaj geçmişini görüntüle
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Soru alma alanı
if prompt := st.chat_input("Ders programı hakkında bir soru sorun..."):
    # Kullanıcı mesajı
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistan yanıtı
    with st.chat_message("assistant"):
        if asistan:
            with st.spinner("Program kontrol ediliyor..."):
                response = asistan.cevapla(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Asistan şu an hizmet veremiyor.")
