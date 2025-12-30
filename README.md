# Projeto de Previsão de Nível de Obesidade - Tech Challenge FIAP

Este projeto consiste em um pipeline de Data Science para prever o nível de obesidade com base em hábitos de vida e dados demográficos. O foco principal foi a correção de um viés crítico (data leakage) identificado no dataset original, resultando em um modelo validado com acurácia de 78.25% seguindo os critérios da OMS.

## 🚀 Como Executar o Projeto (Reprodutibilidade)

Existem duas formas de rodar a aplicação localmente. A via Docker é a recomendada por garantir que o ambiente seja idêntico ao de desenvolvimento.

### Opção 1: Via Docker (Recomendado)
Certifique-se de ter o Docker instalado e executando.
1. **Construir a imagem:** `docker build -t obesity_project .`
2. **Executar o container:** `docker run -p 8501:8501 obesity_project`
3. **Acessar:** `http://localhost:8501`

### Opção 2: Via Python Local
1. Instale as dependências: `pip install -r requirements.txt`
2. Inicie o dashboard: `streamlit run src/app/app.py`

---

## 🏗️ Arquitetura e Metodologia

* **Arquitetura Medalhão:** Organização dos dados em camadas Bronze (Raw), Silver (Processed) e Gold (Final/Model).
* **Engenharia de Features:** Criação de índices comportamentais (estilo de vida e risco alimentar) para aumentar o poder preditivo.
* **Modelagem:** Comparação entre Random Forest e XGBoost com validação cruzada K-Fold. O modelo final foi serializado em `.joblib`.
* **Containerização:** Uso de Docker para isolamento de dependências e portabilidade total do pipeline.

## 📘 Documentação Completa

Toda a jornada, desde a arquitetura de dados, a prova do data leakage e a análise de performance do modelo (Acurácia, F1-Score, Matriz de Confusão e SHAP) estão documentados no relatório técnico.

📄**Acesse aqui:** [Relatório Técnico (PDF)](docs/relatorio_tecnico/relatorio_tecnico.pdf)



## 👩‍💻 Equipe de Desenvolvimento

| Nome | Contato |
|------|----------|
| **Lo-Ruama Silva** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lo-ruama-silva/)|
| **Ruan Lucas** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ruanlucas12) |
| **Lucas Dantas** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lucas-ninomiya-dantas-78428820a) |
| **Guilherme Silva** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](#) |

💬 Desenvolvido como parte do Tech Challenge da Pós-Tech em Data Analytics – FIAP.
