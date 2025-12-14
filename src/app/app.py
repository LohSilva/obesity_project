import streamlit as st
from modulo import sistema_preditivo
# --- Configuração da Página ---
st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🩺",
    layout="wide"
)

# --- BARRA LATERAL ---
st.sidebar.title("Navegação")
st.sidebar.markdown("Selecione a Seção:")

selecao = st.sidebar.radio(
    "Selecione a Seção:",
    options=["Sistema Preditivo"],
    label_visibility="collapsed"
)

# --- ROTEAMENTO ---
if selecao == "Sistema Preditivo":
    sistema_preditivo.run()
    
