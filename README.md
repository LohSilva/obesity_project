# 🏥 Projeto de Previsão de Nível de Obesidade

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow?style=for-the-badge)

Este projeto desenvolve um pipeline completo de *Data Analytics* e *Machine Learning* para prever o nível de obesidade de um indivíduo com base em hábitos alimentares, físicos e dados demográficos. A solução final é um **dashboard interativo em Streamlit** projetado para auxiliar equipes médicas na rápida identificação de perfis de risco.

---

## 🎯 1. O Desafio (Contexto de Negócio)

O objetivo deste projeto, parte do Tech Challenge da Pós-Graduação em Data Analytics, era atuar como Cientista de Dados em um hospital. O desafio era claro: desenvolver um modelo de *Machine Learning* capaz de auxiliar a equipe médica a diagnosticar a obesidade, uma condição de saúde global crescente e multifatorial.

A solução deveria ir além de um modelo: era preciso entregar uma **aplicação preditiva (Streamlit)** e uma **visão analítica** com *insights* acionáveis para a equipe médica.

## 💡 2. A Solução: Dashboard Interativo

Para atender a esse desafio, foi construído um sistema preditivo completo:

* **Pipeline de Dados Robusto:** Utilizando a Arquitetura Medalhão (Bronze, Silver, Gold) para garantir a qualidade, governança e reprodutibilidade dos dados, desde a ingestão crua até a camada analítica.
* **Modelo Preditivo:** Após testes com algoritmos como *Random Forest* e *XGBoost* foi selecionado um modelo com **acurácia superior a 75%**, focado em métricas de precisão e *recall*.
* **Dashboard de Insights (Streamlit):** Uma interface interativa onde a equipe médica pode:
    * Realizar previsões individuais em tempo real.
    * Visualizar métricas de desempenho do modelo (Matriz de Confusão).
    * **Entender o "Porquê":** Gráficos de interpretabilidade (SHAP) explicam quais fatores (ex: "consumo de fast food", "atividade física") mais influenciam o risco de obesidade para um paciente.

## 🚀 3. Como Executar Localmente (VS Code)

1.  **Criar ambiente virtual e instalar dependências:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # ou 
    .\.venv\Scripts\activate no Windows
    pip install -r requirements.txt
    ```
2.  **Executar o Pipeline de Dados:**
    * (Opcional, se os dados processados não estiverem no Git)
    ```bash
    python src/data/preprocess.py
    python src/models/train_model.py
    ```
3.  **Iniciar o Dashboard Streamlit:**
    ```bash
    streamlit run src/app/streamlit_app.py
    ```

## 🛠️ 4. Estrutura do Projeto e Boas Práticas

Este projeto foi construído seguindo padrões profissionais de Engenharia de Dados e MLOps para garantir excelência e reprodutibilidade.

* **Arquitetura Medalhão:** Separação clara dos dados em camadas `data/raw` (Bronze), `data/interim` (Silver) e `data/processed` (Gold).
* **Código Modular:** O código-fonte reside em `src/`, com responsabilidades separadas para processamento de dados (`src/data`), engenharia de features (`src/features`), modelagem (`src/models`) e a aplicação (`src/app`).
* **Gestão de Dependências:** O arquivo `requirements.txt` garante um ambiente de execução consistente.
* **Versionamento (Git):** Uso de `.gitignore` para proteger dados sensíveis e artefatos de modelo, mantendo o repositório limpo.

## 🧩 5. Metodologia e Estratégia Analítica

A solução foi desenvolvida com base em boas práticas de engenharia e ciência de dados, seguindo a Arquitetura Medalhão (Bronze, Silver e Gold).
Essa abordagem garante organização, escalabilidade e rastreabilidade em todas as etapas do ciclo de vida dos dados — desde a coleta até a modelagem e visualização.

O pipeline segue a filosofia ELT (Extract, Load, Transform), permitindo maior flexibilidade na limpeza e transformação dos dados.
O modelo de aprendizado de máquina será escolhido com base em testes comparativos, priorizando desempenho e interpretabilidade.

## 📘 6. Documentação Completa

A justificativa técnica detalha todas as etapas do pipeline, incluindo arquitetura de dados, modelagem, métricas e storytelling analítico.

📄 **Acesse aqui:** [Justificativa Técnica (PDF)](docs/justificativa_tecnica.pdf)

## 🧾 7. Conclusão

Este projeto consolida o aprendizado prático em Data Analytics e Machine Learning, implementando um pipeline completo e reproduzível — da ingestão à comunicação visual — com aplicabilidade direta em contextos de saúde pública e bem-estar.


## 👩‍💻 Equipe de Desenvolvimento

| Nome | Contato |
|------|----------|
| **Lo-Ruama Silva** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lo-ruama-silva/)|
| **Ruan Lucas** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ruanlucas12) |
| **Lucas Dantas** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](#) |
| **Guilherme Silva** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](#) |


💬 Desenvolvido como parte do Tech Challenge da Pós-Tech em Data Analytics – FIAP.