# Projeto de Previsão de Nível de Obesidade 🧠

Este projeto tem como objetivo prever o **nível de obesidade** com base em variáveis de estilo de vida e características físicas.  
Ele cobre todas as etapas de um pipeline de *Data Analytics e Machine Learning*, desde a **extração e tratamento de dados**, **engenharia de atributos**, **treinamento e avaliação de modelos**, até a **apresentação dos resultados via dashboard interativo (Streamlit)**.

## 🧩 Estrutura do Projeto:

- **data/** → Armazena os dados em diferentes estágios (`raw`, `interim`, `processed`)
- **notebooks/** → Análises exploratórias e testes de hipóteses
- **src/** → Código modular (ETL, features, modelagem, app)
- **models/** → Modelos treinados (`.pkl`)
- **reports/** → Relatórios, figuras e storytelling final

## 🚀 Como Executar Localmente (VS Code):

1. **Criar ambiente virtual e instalar dependências**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate        # Windows
   #source .venv/bin/activate     # Mac/Linux
   pip install -r requirements.txt


2. **Colocar o dataset**
   - Adicione o arquivo `obesity.csv` na pasta `data/raw/`.

3. **Rodar os módulos**
   ```bash
   python -m src.data.ingest
   python -m src.data.preprocess
   python -m src.models.train_model
   streamlit run src/app/streamlit_app.py

## 🎯 Objetivos Técnicos:

Criar um pipeline completo de Ciência de Dados, cobrindo:

- Extração e tratamento dos dados
- Engenharia de atributos (feature engineering)
- Escolha, treinamento e avaliação de modelo de Machine Learning
- Apresentação dos resultados via dashboard (Streamlit)

## 📊 Métricas:

- **Principal:** Acurácia mínima de 75% 
- **Adicionais:** F1-Score, Precision, Recall e Matriz de Confusão

## 🧱 Boas Práticas Adotadas

- Estrutura modular de diretórios (padrão de projetos de dados)
- Separação entre dados `raw`, `interim` e `processed` (garante reprodutibilidade)
- Uso de arquivo `.env` para variáveis de ambiente (segurança e flexibilidade)
- Controle de dependências via `requirements.txt`
- Versionamento limpo com `.gitignore` (protege dados sensíveis)
- Preparado para expansão futura com CI/CD e Docker