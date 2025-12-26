import streamlit as st
from modulo import sistema_preditivo, painel_analitico
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
    "Selecione a Seção:",  #O st.radio precisa de um label
    options=["Sistema Preditivo", "Painel Analítico"],
    label_visibility="collapsed" #Esconde o label para ficar mais limpo
)

# --- ROTEAMENTO (Decidindo qual página mostrar) ---
if selecao == "Sistema Preditivo":
    sistema_preditivo.run() #2. Chame a função run() do sistema

elif selecao == "Painel Analítico":
    painel_analitico.run()  #2. Chame a função run() do painel_analitico