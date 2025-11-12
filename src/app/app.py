import streamlit as st
from modulo import storytelling, sistema_preditivo  #1. Importe os módulos

#Configuração da página (opcional, mas recomendado)
st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="📊",
    layout="wide"
)

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.title("Navegação")
st.sidebar.markdown("Selecione a Seção:")

selecao = st.sidebar.radio(
    "Selecione a Seção:",  #O st.radio precisa de um label
    options=["Sistema Preditivo", "Visão Analítica"],
    label_visibility="collapsed" #Esconde o label para ficar mais limpo
)

# --- ROTEAMENTO (Decidindo qual página mostrar) ---
if selecao == "Sistema Preditivo":
    sistema_preditivo.run() #2. Chame a função run() do sistema

elif selecao == "Visão Analítica":
    storytelling.run()  #2. Chame a função run() do storytelling