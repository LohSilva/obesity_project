# check_model_features.py
import joblib
import os

print("--- Verificador de Features do Modelo ---")

try:
    # 1. Monta o caminho para o modelo
    project_root = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(project_root, "models", "random_forest_pipeline.joblib")
    
    # 2. Carrega o pipeline treinado
    modelo_carregado = joblib.load(model_path)
    print(f"Modelo carregado de: {model_path}")

    # 3. Acessa o "Pré-processador" (a primeira etapa do pipeline)
    # (Isso só funciona se você nomeou o passo de 'preprocessor' no seu train_model.py)
    preprocessor = modelo_carregado.named_steps['preprocessor']
    
    # 4. Pergunta ao pré-processador: "Quais features você foi treinado para esperar?"
    features_esperadas = preprocessor.feature_names_in_
    
    print(f"\n[SUCESSO] O modelo foi treinado e espera EXATAMENTE estas {len(features_esperadas)} colunas:")
    print("--------------------------------------------------")
    
    # Imprime a lista
    for i, feature in enumerate(features_esperadas):
        print(f"{i+1:02d}: {feature}")
        
    print("--------------------------------------------------")
    print("O app Streamlit DEVE enviar um DataFrame com estas colunas.")

except Exception as e:
    print(f"\n[ERRO] Não foi possível carregar ou inspecionar o modelo:")
    print(e)
    print("\nVerifique se o nome 'random_forest_pipeline.joblib' está correto e se o")
    print("primeiro passo do seu pipeline se chama 'preprocessor'.")