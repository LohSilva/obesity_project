import streamlit as st

def run():
    st.title("Sistema Preditivo")
    
    #1. Um aviso sutil no canto da tela.
    st.toast("Esta seção está em desenvolvimento.", icon="🚧") 
    
    #2. Uma mensagem "fantasma" na página principal
    st.markdown(
        "## Em Desenvolvimento\n\n"
        "A ferramenta de simulação preditiva será implementada aqui "
        "após a finalização e validação do modelo de Machine Learning."
    )
    
    #3. Pare a execução do script aqui
    st.stop()