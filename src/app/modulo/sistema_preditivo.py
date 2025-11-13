import streamlit as st
import pandas as pd
import joblib
import os
import time #Para um efeito de "loading"

# --- 1. SETUP: CARREGAMENTO DOS MODELOS E CAMINHOS ---
#Usando a mesma lógica de caminho robusto 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODELS_DIR = os.path.join(project_root, "models")
REPORTS_DIR = os.path.join(project_root, "reports", "figures")

#Caminhos para os artefatos treinados
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_pipeline.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

#Imagens de Validação
CM_PATH = os.path.join(REPORTS_DIR, "matriz_confusao_final.png")
REPORT_PATH = os.path.join(REPORTS_DIR, "classification_report_final.png")
SHAP_PATH = os.path.join(REPORTS_DIR, "shap_summary_bar.png")


# --- Cache de Recursos ---
#@st.cache_resource é o comando para carregar
@st.cache_resource
def carregar_modelo():
    """Carrega o pipeline do modelo e o label encoder."""
    try:
        modelo = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return modelo, label_encoder
    except FileNotFoundError:
        st.error(
            "Erro Crítico: Arquivos de modelo não encontrados."
            f"Verifique se 'random_forest_pipeline.joblib' e 'label_encoder.joblib' "
            f"existem na pasta: {MODELS_DIR}"
        )
        return None, None

#Carrega os modelos uma vez
modelo_pipeline, label_encoder = carregar_modelo()

# --- 2. BACKEND: A FUNÇÃO DE ENGENHARIA DE FEATURES ---
def preparar_dados_para_previsao(inputs_humanos):
    """
    Pega o dicionário de inputs do médico, replica a engenharia de features
    e retorna um DataFrame de 1 linha pronto para o modelo.
    """    
    # --- Mapeamentos (Baseado no dicionario_obesity_fiap.pdf) ---    
    #Mapeamentos Simples
    map_genero = {'Feminino': 'Female', 'Masculino': 'Male'}
    map_sim_nao = {'Sim': 'yes', 'Não': 'no'}
    map_transporte = {
        'Automóvel': 'Automobile', 'Motocicleta': 'Motorbike', 
        'Bicicleta': 'Bike', 'Transporte Público': 'Public_Transportation', 
        'Caminhada': 'Walking'
    }     
    #Mapeamentos de Risco (para o índice de risco)
    map_risco_calc = {
        'Não': 'no', 'Às vezes': 'Sometimes', 
        'Frequente': 'Frequently', 'Sempre': 'Always'
    }
    map_risco_caec = {
        'Não': 'no', 'Às vezes': 'Sometimes', 
        'Frequente': 'Frequently', 'Sempre': 'Always'
    }
    
    #Mapeamentos Numéricos (para os índices)
    map_fcvc = {'Raramente (ou nunca)': 1, 'Às vezes': 2, 'Sempre': 3}
    map_ch2o = {'Menos de 1L': 1, 'Entre 1L e 2L': 2, 'Mais de 2L': 3}
    map_faf = {
        'Sedentário (0 dias)': 0, '1-2 dias': 1, 
        '3-4 dias': 2, '4-5 dias ou mais': 3
    }
    map_tue = {
        '0-2 horas/dia': 0, '3-5 horas/dia': 1, 'Mais de 5 horas/dia': 2
    }
    map_ncp = {
        '1 refeição': 1, 
        '2 refeições': 2, 
        '3 refeições': 3, 
        '4 ou mais': 4
    }
    # --- Início da Engenharia de Features (replicando seu notebook) ---
    
    #1. Criar Índices (baseado na sua lógica)
    
    #Índice de Estilo de Vida = (positivos) - (negativo)
    val_fcvc = map_fcvc[inputs_humanos['consumo_frequente_vegetais']]
    val_ch2o = map_ch2o[inputs_humanos['consumo_diario_agua']]
    val_faf = map_faf[inputs_humanos['frequencia_semanal_atividade_fisica']]
    val_tue = map_tue[inputs_humanos['tempo_uso_dispositivo']]
    
    indice_estilo_vida = (val_fcvc + val_ch2o + val_faf) - val_tue
    
    #Índice de Risco Alimentar (usando o mapeamento de risco do seu notebook)
    map_risco_numerico = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'yes': 2, 'Always': 3}
    
    val_favc_txt = map_sim_nao[inputs_humanos['consumo_frequente_alimentos_caloricos']]
    val_caec_txt = map_risco_caec[inputs_humanos['consumo_lanches_entre_refeicoes']]
    val_calc_txt = map_risco_calc[inputs_humanos['consumo_bebida_alcoolica']]
    
    indice_risco_alimentar = (
        map_risco_numerico[val_favc_txt] + 
        map_risco_numerico[val_caec_txt] + 
        map_risco_numerico[val_calc_txt]
    )

    #2. Criar o DataFrame de 1 linha
    dados_para_modelo = {
        'genero': map_genero[inputs_humanos['genero']],
        'idade': inputs_humanos['idade'],
        'historico_familiar': map_sim_nao[inputs_humanos['historico_familiar']],
        'consumo_frequente_alimentos_caloricos': val_favc_txt,
        'consumo_frequente_vegetais': val_fcvc,
        'numero_refeicoes_principais_dia': map_ncp[inputs_humanos['numero_refeicoes_principais_dia']],
        'consumo_lanches_entre_refeicoes': map_risco_caec[inputs_humanos['consumo_lanches_entre_refeicoes']],
        'habito_fumar': map_sim_nao[inputs_humanos['habito_fumar']],
        'consumo_diario_agua': val_ch2o,
        'monitora_caloria_diaria': map_sim_nao[inputs_humanos['monitora_caloria_diaria']],
        'frequencia_semanal_atividade_fisica': val_faf,
        'tempo_uso_dispositivo': val_tue,
        'consumo_bebida_alcoolica': val_calc_txt,
        'transporte_habitual': map_transporte[inputs_humanos['transporte_habitual']],
        'indice_estilo_vida': indice_estilo_vida,
        'indice_risco_alimentar': indice_risco_alimentar,        
    }
    
    #Pega as colunas na ordem correta que o pipeline espera
    ordem_colunas = [
        'genero', 'idade', 'historico_familiar', 
        'consumo_frequente_alimentos_caloricos', 'consumo_frequente_vegetais',
        'numero_refeicoes_principais_dia', 'consumo_lanches_entre_refeicoes', 
        'habito_fumar', 'consumo_diario_agua', 'monitora_caloria_diaria',
        'frequencia_semanal_atividade_fisica', 'tempo_uso_dispositivo',
        'consumo_bebida_alcoolica', 'transporte_habitual', 'indice_estilo_vida',
        'indice_risco_alimentar' 
    ]

    #Cria o DataFrame final com a ordem correta
    df_predicao = pd.DataFrame([dados_para_modelo], columns=ordem_colunas)
    
    return df_predicao

# --- 3. FRONTEND: A APLICAÇÃO STREAMLIT ---
def run():
    
    if not modelo_pipeline or not label_encoder:
        #Se os modelos não carregaram, pare aqui.
        st.stop()
        
    st.title("Sistema Preditivo de Nível de Obesidade")

    # --- CRIA AS ABAS ---
    tab1, tab2 = st.tabs(
        ["Simulador de Risco", "Interpretação do Resultado"]
    )

    # --- CONTEÚDO DA ABA 1: A FERRAMENTA ---
    with tab1:
        st.header("Ferramenta de Apoio à Decisão")
        st.markdown(
            "Insira os dados do paciente para estimar o nível de risco de obesidade "
            "com base no padrão da OMS."
        )

        # Usamos um formulário para que todos os inputs sejam enviados de uma vez
        with st.form(key="prediction_form"):
            
            # Dicionário para armazenar todos os inputs
            inputs = {}

            # --- Grupo 1: Perfil do Paciente ---
            st.subheader("Perfil do Paciente")
            col1, col2 = st.columns(2) # Usar colunas SÓ para perfil básico
            with col1:
                inputs['idade'] = st.number_input("Idade (anos)", min_value=14, max_value=100, value=25)
            with col2:
                inputs['genero'] = st.radio("Gênero", ['Feminino', 'Masculino'], horizontal=True)

            # --- Grupo 2: Histórico Familiar ---
            st.subheader("Histórico Familiar")
            inputs['historico_familiar'] = st.radio("Histórico familiar de obesidade?", ['Não', 'Sim'], horizontal=True)

            # --- Grupo 3: Modo de Locomoção ---
            st.subheader("Modo de Locomoção")
            inputs['transporte_habitual'] = st.selectbox(
                "Transporte habitual?", 
                ['Automóvel', 'Motocicleta', 'Bicicleta', 'Transporte Público', 'Caminhada']
            )

            # --- Grupo 4: Hábitos ---
            st.subheader("Hábitos Diários")
            inputs['habito_fumar'] = st.radio("O paciente fuma?", ['Não', 'Sim'], horizontal=True)
            inputs['monitora_caloria_diaria'] = st.radio("Monitora calorias diárias?", ['Não', 'Sim'], horizontal=True)
            
            # (TROCAMOS SLIDER POR SELECTBOX)
            inputs['numero_refeicoes_principais_dia'] = st.selectbox(
                "Número de refeições principais/dia (NCP):", 
                ['1 refeição', '2 refeições', '3 refeições', '4 ou mais']
            )
            
            # (TROCAMOS SLIDER POR SELECTBOX)
            inputs['consumo_diario_agua'] = st.selectbox(
                "Consumo Diário de Água (CH2O):",
                options=['Menos de 1L', 'Entre 1L e 2L', 'Mais de 2L']
            )

            # --- Grupo 5: Comportamentos ---
            st.subheader("Comportamentos (Alimentação e Atividade)")
            
            # (TROCAMOS SLIDER POR SELECTBOX)
            inputs['consumo_frequente_vegetais'] = st.selectbox(
                "Consumo de Vegetais (FCVC):",
                options=['Raramente (ou nunca)', 'Às vezes', 'Sempre']
            )
            
            # (TROCAMOS SLIDER POR SELECTBOX)
            inputs['frequencia_semanal_atividade_fisica'] = st.selectbox(
                "Atividade Física Semanal (FAF):",
                options=['Sedentário (0 dias)', '1-2 dias', '3-4 dias', '4-5 dias ou mais']
            )
            
            # (TROCAMOS SLIDER POR SELECTBOX)
            inputs['tempo_uso_dispositivo'] = st.selectbox(
                "Tempo em dispositivos (TUE):",
                options=['0-2 horas/dia', '3-5 horas/dia', 'Mais de 5 horas/dia']
            )
            
            inputs['consumo_frequente_alimentos_caloricos'] = st.radio(
                "Consumo Frequente de Alimentos Calóricos (FAVC)?",
                ['Não', 'Sim'], index=1, horizontal=True
            )
            inputs['consumo_lanches_entre_refeicoes'] = st.selectbox(
                "Consumo de Lanches entre Refeições (CAEC):",
                ['Não', 'Às vezes', 'Frequente', 'Sempre'], index=1
            )
            inputs['consumo_bebida_alcoolica'] = st.selectbox(
                "Consumo de Bebida Alcoólica (CALC):",
                ['Não', 'Às vezes', 'Frequente', 'Sempre'], index=0
            )

            st.markdown("---")
            
            #Botão de Previsão
            submit_button = st.form_submit_button(label='Analisar Risco', type="primary")

        # --- Lógica de Previsão ---
        if submit_button:
            with st.spinner("Analisando perfil e executando modelo..."):
                
                #1. Preparar os dados
                df_predicao = preparar_dados_para_previsao(inputs)
                
                #2. Prever (retorna um número, ex: [4])
                previsao_numerica = modelo_pipeline.predict(df_predicao)
                
                #3. Decodificar (retorna o texto, ex: ['Peso Normal'])
                previsao_texto = label_encoder.inverse_transform(previsao_numerica)
                
                time.sleep(1) #Simula o processamento

            #4. Exibir o resultado
            st.subheader("Resultado da Análise:")
            resultado = previsao_texto[0] #Pega o texto (ex: 'Sobrepeso')
            
            if resultado in ['Obesidade Grau I', 'Obesidade Grau II', 'Obesidade Grau III']:
                st.error(f"Nível de Risco Estimado: {resultado}")
                st.markdown(
                    "**Recomendação:** O modelo indica um risco elevado, "
                    "classificado em um nível de obesidade. Recomenda-se "
                    "avaliação clínica detalhada."
                )
            elif resultado == 'Sobrepeso':
                st.warning(f"Nível de Risco Estimado: {resultado}")
                st.markdown(
                    "**Recomendação:** O modelo indica que o perfil do paciente o coloca na **faixa de alerta (Sobrepeso)**. "
                    "Este é o **momento-chave** para intervenção e mudança de hábitos, "
                    "antes que o risco progrida para um nível de obesidade."
                )
            else: #Peso Normal ou Insuficiente
                st.success(f"Nível de Risco Estimado: {resultado}")
                st.markdown(
                    "**Recomendação:** O modelo indica um perfil de peso "
                    "dentro da faixa normal ou insuficiente."
                )

    # --- CONTEÚDO DA ABA 2: A JUSTIFICATIVA ---
    with tab2:
        st.header("Transparência e Performance do Modelo")
        st.markdown(
            "Esta ferramenta é alimentada por um modelo de **Random Forest** treinado na "
            "**base de dados corrigida (padrão OMS)**, conforme justificado na Visão Analítica. "
            "As métricas abaixo refletem o desempenho final do modelo no conjunto de teste "
            "(20% dos dados que ele nunca havia visto)."
        )

        # 1. As Métricas Obrigatórias (Atualizadas para 78.25%)
        st.subheader("Métricas de Performance (Acurácia de 78.25%)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Acurácia Final", "78.25%")
        col2.metric("F1-Score (Ponderado)", "0.78")
        col3.metric("F1-Score (Macro Avg)", "0.79")
        
        st.info(
            "**O que isso significa?** A acurácia de 78.25% indica que o modelo acerta "
            "quase 4 em cada 5 previsões. É uma ferramenta de *apoio* robusta, "
            "mas não substitui o diagnóstico clínico."
        )
        
        # 2. A Matriz de Confusão (Traduzida)
        st.subheader("Diagnóstico do Modelo: Uma Tradução")
        st.info(
            "**1. Onde podemos confiar no modelo (Alta Confiança):**\n"
            "O modelo é **extremamente confiável** para identificar os casos mais críticos. "
            "Como o 'Relatório de Classificação'  mostra, quando o modelo prevê "
            "**Obesidade Grau III**, ele está **correto 90% das vezes** (Precision de 0.90) . "
            "Isso nos dá alta confiança nos alertas de alto risco."
            "\n\n"
            "**2. Onde o modelo tem mais dificuldade (O Ponto de Atenção):**\n"
            "O principal desafio do modelo é na **fronteira entre 'Peso Normal' e 'Sobrepeso'**. "
            "A 'Matriz de Confusão'  mostra que, de 60 pacientes de 'Peso Normal', "
            "o modelo 'confundiu' 11 deles como 'Sobrepeso'. "
            "Isso é confirmado pelo 'Recall' de 0.68 para 'Peso Normal' ."
            "\n\n"
            "**3. Conclusão:**\n"
            "Use esta ferramenta como um **forte apoio à triagem** para casos graves. "
            "Tenha atenção redobrada ao avaliar pacientes na fronteira "
            "entre 'Peso Normal' e 'Sobrepeso', pois é onde as nuances "
            "comportamentais mais impactam a previsão."
        )        

        # 3. O Relatório de Classificação (Traduzido)
        st.subheader("Evidências Visuais do Desempenho")
        try:
            st.image(CM_PATH, caption="Matriz de Confusão (Desempenho no Teste Final)")            
        except FileNotFoundError:
            st.warning(f"Imagem da Matriz de Confusão não encontrada em {CM_PATH}")

        try:            
            st.image(REPORT_PATH, caption="Relatório de Classificação (Heatmap)")       
        except FileNotFoundError:
            st.warning(f"Imagem do Relatório de Classificação não encontrado em {REPORT_PATH}")

        st.subheader("O que o modelo considera mais importante?")
        st.markdown(
            "Finalmente, para confiar em um modelo, precisamos saber *como* ele toma "
            "as decisões. Este gráfico SHAP mostra quais fatores mais 'pesaram' "
            "na previsão final."
        )
        
        try:
            SHAP_PATH = os.path.join(REPORTS_DIR, "shap_summary_bar.png")
            st.image(SHAP_PATH, caption="Importância Global das Features (SHAP)")
            
            st.success(
                "**Conclusão de Confiança:**"
                "\n\n"
                "1. **O modelo pensa como um clínico:** O gráfico prova que o modelo "
                "baseia suas decisões nos fatores de maior impacto clínico: `Idade`, "
                "`Consumo de Vegetais`, `Gênero` e `Histórico Familiar`."
                "\n\n"
                "2. **Os Índices funcionam:** Os índices que criamos (`indice_risco_alimentar` "
                "e `indice_estilo_vida`) são altamente preditivos e confirmam que o *conjunto* "
                "de hábitos é mais importante que fatores isolados."
                "\n\n"
                "**Diagnóstico Final:** O modelo é confiável. Ele não está 'trapaceando' "
                "ou usando correlações espúrias. Suas previsões são baseadas em "
                "fatores que fazem sentido médico."
            )
        except FileNotFoundError:
            st.warning("Gráfico SHAP não encontrado. Execute generate_shap.py primeiro.")