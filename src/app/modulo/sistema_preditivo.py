import streamlit as st
import pandas as pd
import joblib
import os
import time #Para um efeito de "loading"

# --- 1. SETUP: CARREGAMENTO DOS MODELOS E CAMINHOS ---

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODELS_DIR = os.path.join(project_root, "models")
REPORTS_DIR = os.path.join(project_root, "reports", "figures")

#Caminhos para os artefatos treinados
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_pipeline.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")



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
        st.stop()
        
    st.title("Sistema Preditivo de Nível de Obesidade")

    st.header("Ferramenta de Apoio à Decisão")
    st.markdown(
        "Insira os dados do paciente para estimar o nível de risco de obesidade "
        "com base nos hábitos informados."
    )

    with st.form(key="prediction_form"):
        inputs = {}

        # --- Novo grupo: Dados Antropométricos ---
        st.subheader("Dados Antropométricos")

        col_peso, col_altura = st.columns(2)
        with col_peso:
            inputs['peso'] = st.number_input("Peso (kg)", min_value=30.0, max_value=300.0, value=70.0)

        with col_altura:
            inputs['altura'] = st.number_input("Altura (m)", min_value=1.20, max_value=2.20, value=1.70)
            
        # --- Grupo 1: Perfil do Paciente ---
        st.subheader("Perfil do Paciente")
        col1, col2 = st.columns(2)
        with col1:
            inputs['idade'] = st.number_input("Idade (anos)", min_value=14, max_value=100, value=25)
        with col2:
            inputs['genero'] = st.radio("Gênero do Paciente", ['Feminino', 'Masculino'], horizontal=True)

        # --- Grupo 2: Histórico Familiar ---
        st.subheader("Histórico Familiar")
        inputs['historico_familiar'] = st.radio("Histórico familiar de obesidade?", ['Não', 'Sim'], horizontal=True)

        # --- Grupo 3: Modo de Locomoção ---
        st.subheader("Modo de Locomoção")
        inputs['transporte_habitual'] = st.selectbox(
            "Transporte habitual?", 
            ['Automóvel', 'Motocicleta', 'Bicicleta', 'Transporte Público', 'Caminhada']
        )

        # --- Grupo 4: Hábitos Diários ---
        st.subheader("Hábitos Diários")
        inputs['habito_fumar'] = st.radio("O paciente fuma?", ['Não', 'Sim'], horizontal=True)
        inputs['monitora_caloria_diaria'] = st.radio("Monitora calorias diárias?", ['Não', 'Sim'], horizontal=True)
        inputs['numero_refeicoes_principais_dia'] = st.selectbox(
            "Número de refeições principais/dia (NCP):", 
            ['1 refeição', '2 refeições', '3 refeições', '4 ou mais']
        )
        inputs['consumo_diario_agua'] = st.selectbox(
            "Consumo Diário de Água (CH2O):",
            options=['Menos de 1L', 'Entre 1L e 2L', 'Mais de 2L']
        )

        # --- Grupo 5: Comportamentos ---
        st.subheader("Comportamentos (Alimentação e Atividade)")
        inputs['consumo_frequente_vegetais'] = st.selectbox(
            "Consumo de Vegetais (FCVC):",
            options=['Raramente (ou nunca)', 'Às vezes', 'Sempre']
        )
        inputs['frequencia_semanal_atividade_fisica'] = st.selectbox(
            "Atividade Física Semanal (FAF):",
            options=['Sedentário (0 dias)', '1-2 dias', '3-4 dias', '4-5 dias ou mais']
        )
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
        submit_button = st.form_submit_button(label='Analisar Risco', type="primary")

    # -------- CÁLCULO DO IMC SÓ APÓS O CLIQUE --------
    if submit_button:

        # Calcular IMC
        imc = inputs['peso'] / (inputs['altura'] ** 2)

        def classificar_imc(imc):
            if imc < 18.5:
                return "Abaixo do Peso"
            elif imc < 25:
                return "Peso Normal"
            elif imc < 30:
                return "Sobrepeso"
            elif imc < 35:
                return "Obesidade Grau I"
            elif imc < 40:
                return "Obesidade Grau II"
            else:
                return "Obesidade Grau III"
                
        classificacao_oms = classificar_imc(imc)

        with st.spinner("Analisando perfil e executando modelo..."):
            df_predicao = preparar_dados_para_previsao(inputs)
            previsao_numerica = modelo_pipeline.predict(df_predicao)
            previsao_texto = label_encoder.inverse_transform(previsao_numerica)
            time.sleep(1)

        # ----- EXIBIR IMC -----
        st.subheader("Informações Antropométricas")
        st.write(f"**Peso informado:** {inputs['peso']} kg")
        st.write(f"**Altura informada:** {inputs['altura']} m")
        st.write(f"**IMC calculado:** {imc:.1f}")
        st.write(f"**Classificação OMS:** {classificacao_oms}")

        # ----- RESULTADO -----
        resultado = previsao_texto[0]

        st.subheader("Resultado da Análise:")

        if resultado in ['Obesidade Grau I', 'Obesidade Grau II', 'Obesidade Grau III']:
            st.error("⚠ Risco Elevado")
            st.markdown(
                "**Recomendação:** O conjunto de hábitos informados indica um alto risco "
                "de evolução ou manutenção de quadros relacionados à obesidade. "
                "É recomendada uma avaliação clínica profissional e ações imediatas."
            )

        elif resultado == 'Sobrepeso':
            st.warning("⚠ Risco Moderado")
            st.markdown(
                "**Recomendação:** Os hábitos informados sugerem que o paciente está em "
                "uma zona de atenção. Pequenas mudanças de estilo de vida podem reduzir "
                "significativamente o risco de progressão."
            )

        else:
            st.info("ℹ Risco Baixo")
            st.markdown(
                "**Recomendação:** O modelo indica baixo risco com base nos hábitos "
                "informados, mas reforça-se a importância de manter rotinas saudáveis "
                "para prevenção futura."
            )