# --- Bloco 1: Setup (Importações e Caminhos) ---
import pandas as pd
import numpy as np
import os
import joblib  #Para salvar o modelo final (ex: .joblib)
import matplotlib.pyplot as plt
import seaborn as sns

#Ferramentas de Pipeline e Pré-processamento
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#Ferramentas de Modelagem e Avaliação
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Modelos que Vamos Comparar
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("--- [1/9] Script de treinamento iniciado. ---")

# --- Definição de Caminhos ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(project_root, "data", "processed", "obesity_gold.csv")
MODELS_DIR = os.path.join(project_root, "models")
REPORTS_DIR = os.path.join(project_root, "reports", "figures")

#Garante que as pastas de saída existam
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Bloco 2: Carga e Definição de Features ---
print(f"--- [2/9] Carregando dados de {DATA_PATH}... ---")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"ERRO CRÍTICO: Arquivo de dados não encontrado em {DATA_PATH}")
    print("Por favor, execute o notebook create_gold.ipynb primeiro.")
    exit()

TARGET = 'classe_peso_oms'

# --- Codificando o alvo (y) ---
print(f"Codificando o alvo (y): {TARGET}...")

#1. Preparar o LabelEncoder
le = LabelEncoder()

#2. Treinar o encoder no alvo (y) e transformá-lo em números
y_encoded = le.fit_transform(df[TARGET])

#3. Salvar o LabelEncoder (o "mapa de tradução") na pasta models/
#Isso será útil para usar no Streamlit depois
le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
joblib.dump(le, le_path)
print(f"LabelEncoder (mapa de tradução) salvo em: {le_path}")

#Definição das colunas que "vazam" a resposta
features_de_leakage = [
    'classe_peso_corporal', #Alvo original falho
    'IMC',
    'peso_kg',
    'altura_m',
    'risco_alimentos_caloricos_num',
    'risco_lanches_num',
    'risco_alcool_num',
    'comportamento_saudavel'
]

#X são as features
X = df.drop(columns=[TARGET] + features_de_leakage)

print(f"Alvo (y) definido como: {TARGET}")
print(f"Features de treinamento (X): {list(X.columns)}")

# --- Bloco 3: Divisão Treino/Teste ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f"--- [3/9] Dados divididos: {len(y_train)} para treino/validação, {len(y_test)} para teste final. ---")

# --- Bloco 4: Pipeline de Pré-processamento ---
print("--- [4/9] Definindo o pipeline de pré-processamento... ---")

#1. Identifica quais colunas são numéricas e quais são categóricas
colunas_numericas = X_train.select_dtypes(include=np.number).columns
colunas_categoricas = X_train.select_dtypes(include='object').columns

print(f"Colunas numéricas: {list(colunas_numericas)}")
print(f"Colunas categóricas: {list(colunas_categoricas)}")

#2. Cria o transformador para colunas NUMÉRICAS
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

#3. Cria o transformador para colunas CATEGÓRICAS
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    # handle_unknown='ignore' evita erros se o modelo vir um valor novo
])

#4. Cria o "Gerente" (ColumnTransformer)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, colunas_numericas),
        ('cat', categorical_transformer, colunas_categoricas)
    ],
    remainder='passthrough' 
)

# --- Bloco 5: Validação Cruzada (Comparando Modelos) ---
print("--- [5/9] Iniciando Validação Cruzada (K-Fold)... ---")

#Criação dos pipelines completos
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', RandomForestClassifier(random_state=42))])

pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', XGBClassifier(use_label_encoder=False, 
                                                        eval_metric='mlogloss', 
                                                        random_state=42))])

#Dicionário de modelos para testar
modelos = {
    "Random Forest": pipeline_rf,
    "XGBoost": pipeline_xgb
}

#Executação da validação cruzada
resultados = {}
for nome, modelo in modelos.items():
    print(f"Validando {nome}...")
    # cv=5 significa 5 Folds (5 "simulados")
    scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
    resultados[nome] = scores.mean()

print("\n--- Resultados da Validação Cruzada (Acurácia Média) ---")
for nome, media in resultados.items():
    print(f"{nome}: {media * 100:.2f}%")

#Visualizando o modelo que retornou o melhor resultado
modelo_vencedor_nome = max(resultados, key=resultados.get)
modelo_vencedor_pipeline = modelos[modelo_vencedor_nome]

print(f"\nModelo com melhor desempenho é: {modelo_vencedor_nome}")

# --- Bloco 6: Treinamento Final do Modelo escolhido ---
print(f"\n--- [6/9] Treinando o modelo com melhor desempenho ({modelo_vencedor_nome}) nos 80% de dados... ---")

#Nota: modelo_vencedor_pipeline já é o pipeline_rf, 
modelo_vencedor_pipeline.fit(X_train, y_train)

print("Modelo final treinado com sucesso.")
print()

# --- Bloco 7: Avaliação Final ---
print(f"--- [7/9] Avaliando o modelo final nos 20% de dados de teste... ---")

#Fazer previsões nos dados de teste
y_pred_final = modelo_vencedor_pipeline.predict(X_test)

#Gerando as métricas (Precision, Recall, F1-Score)
acuracia_final = accuracy_score(y_test, y_pred_final)
reporte_final = classification_report(y_test, y_pred_final)

print("\n--- RESULTADO FINAL (NA BASE DE TESTE) ---")
print(f"Acurácia Final: {acuracia_final * 100:.2f}%")
print(f"\nRelatório de Classificação Final:\n{reporte_final}")

#Verificar se atingiu o requisito
if acuracia_final >= 0.75:
    print("STATUS: SUCESSO! A acurácia final atende aos requisitos do projeto (>= 75%).")
else:
    print("STATUS: ATENÇÃO! A acurácia final ficou abaixo dos 75%.")

# --- Bloco 8: Salvar Artefatos Finais ---
print(f"--- [8/9] Salvando artefatos... ---")

#1. Salvando a Matriz de Confusão
print("Salvando gráfico da Matriz de Confusão...")
cm = confusion_matrix(y_test, y_pred_final)

#Mapeando os nomes das classes (ex: 'Sobrepeso') e não os números (ex: 5)
#Através do LabelEncoder
try:
    le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    le = joblib.load(le_path)
    nomes_classes = le.classes_
except FileNotFoundError:
    print("Aviso: label_encoder.joblib não encontrado. Usando números nas classes.")
    nomes_classes = modelo_vencedor_pipeline.classes_

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=nomes_classes, 
            yticklabels=nomes_classes)
plt.title('Matriz de Confusão - Desempenho no Teste Final')
plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Prevista')
plt.tight_layout() #Ajusta o layout para não cortar os nomes

cm_path = os.path.join(REPORTS_DIR, 'matriz_confusao_final.png')
plt.savefig(cm_path)
print(f"Matriz de Confusão salva em: {cm_path}")

#2. Salvar o Modelo (Pipeline Completo)
modelo_final_path = os.path.join(MODELS_DIR, f"{modelo_vencedor_nome.lower().replace(' ', '_')}_pipeline.joblib")
joblib.dump(modelo_vencedor_pipeline, modelo_final_path)
print(f"Modelo (pipeline completo) salvo em: {modelo_final_path}")

print("--- Script de treinamento finalizado com sucesso! ---")
print()
# --- Bloco 9: Salvar o Relatório de Classificação como Imagem ---
print("--- [9/9] Salvando Relatório de Classificação como imagem... ---")

#1. Gera o relatório como um DICIONÁRIO
report_dict = classification_report(
    y_test, 
    y_pred_final, 
    target_names=nomes_classes, #Para usar os nomes (ex: 'Sobrepeso')
    output_dict=True
)

#2. Carrega o dicionário em um DataFrame do Pandas
report_df = pd.DataFrame(report_dict).transpose()

#3. Removendo as linhas de 'accuracy' e 'support' para focar apenas nas métricas por classe.
report_df_plot = report_df.drop(['accuracy']) #Remove a linha de acurácia total
report_df_plot = report_df_plot.drop(columns=['support']) #Remove a coluna de contagem

#4. Criar o Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    report_df_plot, 
    annot=True,     #Escreve os números (ex: 0.77)
    fmt='.2f',      #Formata com 2 casas decimais
    cmap='Blues',   #Mesma paleta da Matriz de Confusão
    cbar=False      #Opcional: remove a barra de cor lateral
)
plt.title('Relatório de Classificação Final (Heatmap)')
plt.xlabel('Métricas')
plt.ylabel('Classes')
plt.tight_layout()

#5. Salvar a imagem
report_img_path = os.path.join(REPORTS_DIR, 'classification_report_final.png')
plt.savefig(report_img_path)

print(f"Relatório de Classificação salvo em: {report_img_path}")
print("--- Script de treinamento finalizado com sucesso! ---")

# --- Bloco 9: Debug - Verificando o Mapeamento de Classes ---
print("\n--- [Debug] Mapeamento de Classes (Tradução) ---")

# Carrega o "mapa" que salvamos
le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
le = joblib.load(le_path)

# Pega os nomes das classes (ex: 'Sobrepeso') e os números (ex: 0, 1)
# O .classes_ nos dá o "mapa" na ordem correta
mapeamento_classes = dict(zip(le.transform(le.classes_), le.classes_))

print("O modelo usa os seguintes números para cada classe:")
for numero, nome in mapeamento_classes.items():
    print(f"Classe {numero}: {nome}")