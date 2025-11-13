import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- 1. SETUP: CARREGAMENTO DOS MODELOS E CAMINHOS ---
print("--- [1/6] Iniciando script de geração SHAP... ---")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(project_root, "data", "processed", "obesity_gold.csv")
MODELS_DIR = os.path.join(project_root, "models")
REPORTS_DIR = os.path.join(project_root, "reports", "figures")

MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_pipeline.joblib")
LE_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
SHAP_BAR_PATH = os.path.join(REPORTS_DIR, 'shap_summary_bar.png')

# --- 2. CARREGAR MODELO E LABEL ENCODER ---
print("--- [2/6] Carregando artefatos (modelo e encoder)... ---")
try:
    modelo_pipeline = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LE_PATH)
except FileNotFoundError:
    print("ERRO: Modelo ou Label Encoder não encontrados. Rode train_model.py primeiro.")
    exit()

# --- 3. CARREGAR E PREPARAR OS DADOS ---
print("--- [3/6] Carregando e preparando dados... ---")
df = pd.read_csv(DATA_PATH)

TARGET = 'classe_peso_oms'
y_encoded = label_encoder.transform(df[TARGET])

features_de_leakage_e_intermediarias = [
    'classe_peso_corporal', 'IMC', 'peso_kg', 'altura_m',
    'risco_alimentos_caloricos_num', 'risco_lanches_num', 
    'risco_alcool_num', 'comportamento_saudavel'
]

X = df.drop(columns=[TARGET] + features_de_leakage_e_intermediarias)

#Recria EXATAMENTE os mesmos splits de treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- 4. CRIAR O EXPLICADOR SHAP ---
#O SHAP (especialmente para Random Forest) funciona com o modelo
#e os dados de *treino* (para aprender a "linha de base").
print("--- [4/6] Desmontando pipeline e pré-processando dados... ---")
try:
    #Extrai as duas etapas do pipeline
    preprocessor = modelo_pipeline.named_steps['preprocessor']
    model = modelo_pipeline.named_steps['model']
    
    #Transforma os dados brutos (com texto) em dados numéricos
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    #Pega os nomes das features FINAIS (ex: 'cat__genero_Female')
    #Isso é crucial para o gráfico SHAP
    feature_names = preprocessor.get_feature_names_out()
    
    # Converte os arrays numéricos de volta para DataFrames com os nomes corretos
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    print("Dados pré-processados com sucesso para o SHAP.")

except Exception as e:
    print(f"ERRO ao desmontar o pipeline: {e}")
    print("Verifique se os passos do seu pipeline se chamam 'preprocessor' e 'model'.")
    exit()

# --- 5. CALCULAR OS VALORES SHAP ---
#Agora calculamos as "explicações" para o conjunto de *teste*
print("--- [5/6] Criando o explicador SHAP (TreeExplainer)... ---")
explainer = shap.TreeExplainer(model)

# --- 6. CALCULAR OS VALORES SHAP ---
#Calculamos os valores SHAP nos dados de teste PRÉ-PROCESSADOS
print("--- [6/6] Calculando valores SHAP (Isso pode ser demorado)... ---")
shap_values = explainer.shap_values(X_test_processed_df)

# --- 7. GERAR E SALVAR O GRÁFICO DE BARRAS ---
print(f"--- [7/7] Salvando gráfico de barras SHAP em {SHAP_BAR_PATH}... ---")

plt.figure()
shap.summary_plot(
    shap_values, 
    X_test_processed_df, 
    plot_type="bar", 
    class_names=label_encoder.classes_,
    show=False
)
plt.title("Importância Global das Features (Impacto Médio no Modelo)")
plt.xlabel("Contribuição Média (Valor SHAP)")

plt.tight_layout() 


#Adição de 'bbox_inches' e 'pad_inches'
plt.savefig(
    SHAP_BAR_PATH, 
    bbox_inches='tight', #Garante que os labels (ex: num_idade) não sejam cortados
    pad_inches=0.3       #Adiciona uma margem de 0.3 polegadas
)

plt.close()

print("--- Script SHAP finalizado com sucesso! ---")