import streamlit as st
from modulo import storytelling, sistema_preditivo, performance_modelo 

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
    options=[
        "Sistema Preditivo", 
        "Interpretação (Performance do Modelo)",
        "Visão Analítica (Data Storytelling)"
    ],
    label_visibility="collapsed"
)

# --- ROTEAMENTO ---
if selecao == "Sistema Preditivo":
    sistema_preditivo.run()
    
elif selecao == "Interpretação (Performance do Modelo)":
    performance_modelo.run()
elif selecao == "Visão Analítica (Data Storytelling)":
    storytelling.run()