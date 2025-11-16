import streamlit as st
import os

# --- 1. SETUP: DEFINIÇÃO DE CAMINHOS ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
REPORTS_DIR = os.path.join(project_root, "reports", "figures")

# Imagens de Validação
CM_PATH = os.path.join(REPORTS_DIR, "matriz_confusao_final.png")
REPORT_PATH = os.path.join(REPORTS_DIR, "classification_report_final.png")
SHAP_PATH = os.path.join(REPORTS_DIR, "shap_summary_bar.png")


# --- 2. FRONTEND: A PÁGINA DE PERFORMANCE ---
def run():
    st.title("Interpretação e Performance do Modelo")
    
    st.header("Transparência e Performance do Modelo")
    st.markdown(
        "Esta ferramenta é alimentada por um modelo de **Random Forest** treinado na "
        "**base de dados corrigida (padrão OMS)**, conforme justificado na Visão Analítica. "
        "As métricas abaixo refletem o desempenho final do modelo no conjunto de teste "
        "(20% dos dados que ele nunca havia visto)."
    )

    st.subheader("Métricas de Performance (Acurácia de 78.25%)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia Final", "78.25%")
    col2.metric("F1-Score (Ponderado)", "0.78")
    col3.metric("F1-Score (Macro Avg)", "0.79")
    
    # --- O BLOCO UNIFICADO DE DIAGNÓSTICO ---
    st.subheader("Diagnóstico do Modelo: Uma Tradução")
    st.info(
        "**1. Onde podemos confiar no modelo (Alta Confiança):**\n"
        "O modelo é **extremamente confiável** para identificar os casos mais críticos. "
        "Como o 'Relatório de Classificação' [cite: 21-30] mostra, quando o modelo prevê "
        "`Obesidade Grau III`, ele está **correto 90% das vezes** (Precision de 0.90) [cite: 777-778]. "
        "Isso nos dá alta confiança nos alertas de alto risco."
        "\n\n"
        "**2. Onde o modelo tem mais dificuldade (O Ponto de Atenção):**\n"
        "O principal desafio do modelo é na **fronteira entre 'Peso Normal' e 'Sobrepeso'**. "
        "A 'Matriz de Confusão' [cite: 11-20] mostra que, de 60 pacientes de 'Peso Normal', "
        "o modelo 'confundiu' 13 deles como 'Sobrepeso'. "
        "Isso é confirmado pelo 'Recall' de 0.68 para 'Peso Normal' [cite: 780-796]."
        "\n\n"
        "**3. Conclusão para o Médico:**\n"
        "Use esta ferramenta como um **forte apoio à triagem** para casos graves. "
        "Tenha atenção redobrada ao avaliar pacientes na fronteira "
        "entre 'Peso Normal' e 'Sobrepeso', pois é onde as nuances "
        "comportamentais mais impactam a previsão."
    )
    
    # --- Os Gráficos ---
    st.subheader("Evidências Visuais do Desempenho")
    
    try:
        st.image(CM_PATH, caption="Matriz de Confusão (Desempenho no Teste Final)")
    except FileNotFoundError:
        st.warning(f"Imagem da Matriz de Confusão não encontrada em {CM_PATH}")

    try:
        st.image(REPORT_PATH, caption="Relatório de Classificação (Heatmap)")
    except FileNotFoundError:
        st.warning(f"Imagem do Relatório de Classificação não encontrado em {REPORT_PATH}")
        
    # --- Seção do SHAP ---
    st.subheader("O que o modelo considera mais importante?")
    st.markdown(
        "Para confiar em um modelo, precisamos saber *como* ele toma "
        "as decisões. Este gráfico SHAP [cite: 802-818] mostra quais fatores mais 'pesaram' "
        "na previsão final."
    )
    
    try:
        st.image(SHAP_PATH, caption="Importância Global das Features (SHAP)")
        
        st.success(
            "**Tradução para o Médico (Conclusão de Confiança):**"
            "\n\n"
            "1. **O modelo pensa como um clínico:** O gráfico [cite: 802-818] prova que o modelo "
            "baseia suas decisões nos fatores de maior impacto clínico: `Idade`, "
            "`Consumo de Vegetais`, `Gênero` e `Histórico Familiar`."
            "\n\n"
            "2. **Os Índices funcionam:** Os índices que criamos (`indice_risco_alimentar` [cite: 108-121] "
            "e `indice_estilo_vida` [cite: 104]) são altamente preditivos e confirmam que o *conjunto* "
            "de hábitos é mais importante que fatores isolados."
            "\n\n"
            "**Diagnóstico Final:** O modelo é confiável. Ele não está 'trapaceando' "
            "ou usando correlações espúrias. Suas previsões são baseadas em "
            "fatores que fazem sentido clínico."
        )
    except FileNotFoundError:
        st.warning(f"Gráfico SHAP não encontrado em {SHAP_PATH}. Execute generate_shap.py primeiro.")