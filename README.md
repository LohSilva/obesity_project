## 🏥**Projeto de Previsão de Nível de Obesidade**

Este projeto desenvolve um pipeline completo de Data Analytics e Machine Learning para prever o nível de obesidade de um indivíduo com base em hábitos alimentares, físicos e dados demográficos. A solução final é um dashboard interativo em Streamlit projetado para auxiliar equipes médicas na rápida identificação de perfis de risco.

### 🌟**Destaque do Projeto: A Investigação do "Data Leakage"**

Mais do que apenas treinar um modelo, o núcleo deste projeto foi uma investigação analítica que descobriu uma falha crítica no dataset original.

Nossa análise provou que esses resultados eram falsamente inflados devido a um vazamento de dados (data leakage), onde a variável-alvo original (classe_peso_corporal) era criada usando regras de IMC diferentes para cada gênero.

Este projeto documenta a descoberta, a prova e a correção dessa falha, culminando em um modelo robusto, com acurácia honesta de **78.25%**, treinado em um alvo cientificamente válido (classe_peso_oms) e pronto para uso clínico.

## 🚀**1. Acesse a Aplicação (Deploy)**

A aplicação interativa está hospedada no Streamlit Community Cloud e pode ser acessada publicamente.

Link: https://projeto-obesidade.streamlit.app//

O dashboard é dividido em duas seções:

- **Visão Analítica:** O storytelling completo que documenta a investigação do data leakage e a análise dos fatores de risco.

- **Sistema Preditivo:** A ferramenta interativa para o médico inserir dados do paciente e receber a previsão de risco.

## 💡 **2. A Solução: Duas Ferramentas em Uma**

Para atender ao desafio, foram construídas duas soluções integradas:

**Visão Analítica (O "Porquê"):** Um storytelling de dados que prova a falha no dataset original (o leakage) e valida a criação de uma nova variável-alvo (classe_peso_oms) baseada nos padrões da OMS.

**Sistema Preditivo (O "O Quê"):**

- Ferramenta (Aba 1): Uma interface limpa para o médico inserir os dados do paciente e receber uma previsão.

- Interpretação (Aba 2): Uma "tradução para o médico" da performance do modelo, usando a Matriz de Confusão e gráficos SHAP para provar que o modelo é confiável e "pensa" de forma clínica.

## 🛠️**3. Arquitetura e Metodologia**

O projeto segue padrões profissionais de Engenharia de Dados para garantir qualidade e reprodutibilidade.

**Arquitetura Medalhão:** Os dados foram processados seguindo as camadas Bronze (ingestão), Silver (limpeza e engenharia de features) e Gold (camada final, pronta para modelagem).

**Engenharia de Features:** O insight mais importante foi obtido através da criação de features de engenharia, como indice_estilo_vida e indice_risco_alimentar, que se provaram preditores mais fortes do que os dados brutos.

**Pipeline de Modelagem (src/models/):** O treinamento foi feito de forma robusta, usando Pipelines do Scikit-learn para pré-processamento, LabelEncoder para o alvo, e Validação Cruzada (K-Fold) para comparar Random Forest e XGBoost.

**Interpretabilidade (XAI):** O modelo final foi validado com SHAP para garantir que suas decisões são baseadas em fatores clinicamente relevantes.

## 📘**4. Documentação Completa**

Toda a jornada, desde a arquitetura de dados, a prova do data leakage e a análise de performance do modelo (Acurácia, F1-Score, Matriz de Confusão e SHAP) estão documentados no relatório técnico.

📄**Acesse aqui:** [Relatório Técnico (PDF)](docs/relatorio_tecnico/relatorio_tecnico.pdf)

## 🚀**5. Como Executar o Dashboard Localmente**

Este repositório está configurado para o deploy (lendo o CSV e o modelo .joblib). Não é necessário rodar os scripts de treinamento para executar o app.

1. Clone o repositório:

git clone [https://github.com/seu-usuario/obesity_project.git](https://github.com/seu-usuario/obesity_project.git)

cd obesity_project

2. Crie o ambiente virtual e instale as dependências:

python -m venv .venv

.\.venv\Scripts\activate 

pip install -r requirements.txt

3. Inicie o Dashboard Streamlit:

streamlit run src/app/app.py

### Como Recriar o Modelo (Avançado)

Se você deseja rodar o pipeline de treinamento do zero:

- Treinar o Modelo:

    python src/models/train_model.py

- Gerar os Gráficos SHAP:

python src/models/generate_shap.py

## 👩‍💻 Equipe de Desenvolvimento

| Nome | Contato |
|------|----------|
| **Lo-Ruama Silva** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lo-ruama-silva/)|
| **Ruan Lucas** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ruanlucas12) |
| **Lucas Dantas** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lucas-ninomiya-dantas-78428820a) |
| **Guilherme Silva** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](#) |

💬 Desenvolvido como parte do Tech Challenge da Pós-Tech em Data Analytics – FIAP.
