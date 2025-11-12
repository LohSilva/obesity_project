import streamlit as st
import plotly.express as px
import pandas as pd
import os

#Caminho da base do projeto (3 níveis acima deste arquivo)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


#Caminho do arquivo CSV da camada Gold
csv_path = os.path.join(project_root, "data", "processed", "obesity_gold.csv")


@st.cache_data
def carregar_dados():
    try:
        df_gold = pd.read_csv(csv_path)
        #Garante que a tabela contém as colunas essenciais
        colunas_esperadas = ['IMC', 'classe_peso_oms']
        for col in colunas_esperadas:
            if col not in df_gold.columns:
                st.warning(f"⚠️ Coluna ausente na base: {col}")
        return df_gold
    except FileNotFoundError:
        st.error("❌ Arquivo 'obesity_gold.csv' não encontrado em data/processed/.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ Erro ao carregar dados: {e}")
        return pd.DataFrame()

#Carrega globalmente
df_gold = carregar_dados()

def run():  

    st.title("Visão Analítica - Nível Obesidade")

    st.markdown(
        "Esta análise mergulha nos dados de 2.111 indivíduos para entender os fatores da obesidade. A primeira vista, os dados mostram um perfil jovem (média de 24 anos) e com IMC médio na faixa normal (29.7). No entanto, uma baixa taxa de hábitos saudáveis (59.1%) sugere que o risco está mais ligado ao comportamento do que ao perfil demográfico."
    )

    #Méttricas gerais em cards
    st.markdown("## Resumo Estatístico")

    # --- CSS para estilizar os cards nativos ---
    st.markdown(
        """
    <style>
        [data-testid="stMetric"] {
        background-color: #1F2937;
        border: 1px solid #4B5563;        
        text-align: center;
                
        border-radius: 14px;
        padding: 18px 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08); 
        height: 130px !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
        }
        
        [data-testid="stMetricLabel"] {
        color: #9CA3AF !important;
        text-align: center;
        font-size: 16px !important;         
        justify-content: center;         
        white-space: normal !important; 
        text-overflow: unset !important; 
        line-height: 1.3 !important; 
        }
    
        [data-testid="stMetricValue"] {
        color: #F3F4F6 !important;
        font-size: 26px !important;
        font-weight: bold;
        margin-bottom: 5px;
        justify-content: center; 
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
    
    # --- Cria e preenche as colunas ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
     st.metric(
     label="Total de indivíduos", 
     value=f"{len(df_gold):,}".replace(",", ".")
    )

    with col2:
     st.metric(
     label="Idade média", 
     value=f"{df_gold['idade'].mean():.1f} anos"
    )

    with col3:
     st.metric(
     label="IMC médio", 
     value=f"{df_gold['IMC'].mean():.1f}"
    )

    with col4:
     st.metric(
     label="Taxa de hábitos saudáveis", 
     value=f"{df_gold['comportamento_saudavel'].mean()*100:.1f}%"
    )
    st.info("**Interpretação:** A base apresenta indivíduos predominantemente jovens e adultos, com IMC médio dentro da faixa de peso normal, " \
    "mas uma taxa de hábitos saudáveis moderada, sugerindo oportunidades de melhoria comportamental.")
    
    st.markdown("---") 
    # --- 1. DISTRIBUIÇÃO DE GÊNERO ---
    st.header("Distribuição por Gênero")
    st.markdown(
        "A amostra é composta por indivíduos de ambos os gêneros, "
        "permitindo analisar diferenças no comportamento e nos padrões de peso corporal."
    )

    # 1. Calcular os valores
    contagem_genero = df_gold['genero'].value_counts()
    total_individuos = len(df_gold)
    cont_female = contagem_genero.get('Female', 0)
    cont_male = contagem_genero.get('Male', 0)
    pct_female = (cont_female / total_individuos) * 100
    pct_male = (cont_male / total_individuos) * 100

    # 2. Criar duas colunas para os cards
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Contagem Gênero Feminino",
            value=f"{cont_female:,.0f}",
            delta=f"{pct_female:.1f}% da Amostra",
            delta_color="off" # Desliga a cor verde/vermelha
        )

    with col2:
        st.metric(
            label="Contagem Gênero Masculino",
            value=f"{cont_male:,.0f}",
            delta=f"{pct_male:.1f}% da Amostra",
            delta_color="off"
        )

    # 3. O Insight (o mesmo, mas agora apoiado pelos cards)
    st.info(
        "**Interpretação:** A base de dados é perfeitamente equilibrada entre homens "
        "e mulheres (quase 50% para cada). Isso é crucial, pois nos permite investigar, "
        "sem viés de amostragem, se os fatores de risco impactam os gêneros "
        "de forma diferente."
    )

    # --- 2. DISTRIBUIÇÃO DOS NÍVEIS DE PESO ---
    st.markdown("---") 
    st.header("Distribuição dos Níveis de Peso Corporal")
    st.markdown("A análise da variável `classe_peso_corporal` revela a proporção de indivíduos em cada categoria de peso.")

    #1. Criar o gráfico
    fig_peso = px.histogram(
    df_gold, 
    x='classe_peso_corporal',
    title='Contagem por Classe de Peso Corporal',
    color='classe_peso_corporal', #Uma cor para cada barra
    text_auto=True #Adiciona os rótulos
    )

    #2. Ajustar layout
    fig_peso.update_layout(
    template='plotly_dark',
    yaxis_title='Contagem',
    xaxis_title='Classe de Peso' #Ajusta o nome do eixo X
    )

    #3. Exibir o gráfico
    st.plotly_chart(fig_peso, use_container_width=True)

    #4. Adicionar o Insight
    st.info(
        "**Interpretação:** O gráfico revela um ponto crucial sobre a base de dados: ela **não é** "
    "uma amostra da população geral. As categorias com maior volume de registros são, "
    "de longe, **Obesidade Tipo I (351)** e **Obesidade Tipo III (324)**."
    "\n\n"
    "Isso indica que o dataset está fortemente concentrado em indivíduos que já se "
    "encontram em níveis de obesidade, tornando-o ideal para entender as nuances *entre* "
    "os diferentes graus de obesidade, mas menos representativo das fases iniciais (como Peso Normal)."
    )


    st.header("Investigação da Variável-Alvo: O 'Data Leakage'")
    st.markdown(
        "Durante a análise, foi identificada uma anomalia crítica na variável-alvo "
        "original ('classe_peso_corporal'). Esta seção documenta a descoberta "
        "e a validação desse problema."
    )

    # --- 3. NÍVEL DE OBESIDADE POR GÊNERO ---
    st.markdown("---") 
    st.header("Nível de Obesidade por Gênero")
    st.markdown("Compara-se a distribuição dos níveis de obesidade entre homens e mulheres.")

    #1. Criar o Gráfico Agrupado
    fig_obesidade_genero = px.histogram(
        df_gold,
        x='classe_peso_corporal',
        color='genero',        #Cria as barras agrupadas por gênero
        barmode='group',     #'group' coloca as barras lado a lado
        text_auto=True,      #Adiciona os rótulos
        title='Contagem de Gênero por Classe de Peso',
        color_discrete_map={ 
            'Female': '#FF69B4', 
            'Male': '#1E90FF'
        }
    )

    #2. Ajustar layout
    fig_obesidade_genero.update_layout(
        template='plotly_dark',
        yaxis_title='Contagem',
        xaxis_title='Classe de Peso'
    )

    #3. Exibir o gráfico
    st.plotly_chart(fig_obesidade_genero, use_container_width=True) 

    #4. Adicionar o Insight
    st.info(
        "**Descoberta Inesperada:** O gráfico revela um padrão que não é natural: "
    "as classes de obesidade mais altas parecem ser mutuamente exclusivas por gênero. \n\n"
    
    "* **'Obesidade Tipo II'** é quase exclusiva do gênero masculino.\n"
    "* **'Obesidade Tipo III'** é quase exclusiva do gênero feminino.\n\n"
    
    "Isso não sugere um padrão comportamental, mas sim um **possível problema na "
    "definição da variável** original. Vamos investigar isso no próximo gráfico."
    )

    # --- 4. VALIDAÇÃO: IMC vs CLASSE DE PESO por GÊNERO ---
    st.markdown("---") 
    st.header("Validação: IMC por Classe de Peso e Gênero")
    st.markdown(
        "Após a descoberta da separação de gênero nos tipos de obesidade, "
        "investigamos se a causa está nos critérios de IMC usados para "
        "definir cada classe."
    )

    #1. Criar o Boxplot
    fig_val = px.box(
        df_gold,
        x='classe_peso_corporal',
        y='IMC',
        color='genero',  
        title='Distribuição do IMC por Classe de Peso e Gênero',
        color_discrete_map={
            'Female': '#FF69B4', 
            'Male': '#1E90FF'
        }
    )

    #2. Ajustar layout
    fig_val.update_layout(
        template='plotly_dark',
        yaxis_title='IMC',
        xaxis_title='Classe de Peso'
    )

    #3. Exibir o gráfico
    st.plotly_chart(fig_val, use_container_width=True)

    #4. Adicionar o Insight
    st.warning(
        "**Interpretação:** O gráfico acima deve confirmar que os critérios de "
        "IMC para 'Obesidade Tipo II' e 'Tipo III' são diferentes para homens e mulheres. "
        "\n\n"
        "Observe se a faixa de IMC (a 'caixa') para 'Obesidade Tipo II' em homens "
        "é diferente da faixa para mulheres. O mesmo vale para 'Obesidade Tipo III'. "
        "Isso confirma o 'vazamento de dados' no dataset original."
    )
    # --- 5. TABELA DE PROVA: FAIXAS DE IMC POR GÊNERO ---
    st.markdown("---")
    st.header("Tabela de Prova: Faixas de IMC por Gênero")
    st.markdown(
        "A tabela abaixo resume as faixas de IMC (mínimo, médio, máximo) "
        "para cada classe de peso original, separada por gênero. "
        "Note como as faixas de 'Obesidade Tipo II' e 'Tipo III' "
        "não se sobrepõem entre os gêneros."
    )

    #1. Agrupar os dados com Pandas
    tabela_prova = df_gold.groupby(['classe_peso_corporal', 'genero'])['IMC'].agg(['count', 'min', 'mean', 'max']
        ).reset_index()

    #2. Arredondar os valores para melhor leitura
    tabela_prova = tabela_prova.round(1)

    #3. Exibir a tabela no Streamlit
    st.dataframe(tabela_prova) 
    #st.table(tabela_prova)
    
    st.markdown("---")

    
    # --- INÍCIO DA ANÁLISE CORRIGIDA ---
    st.header("Análise Corrigida: Usando o Padrão OMS")
    st.markdown(
        "Após a comprovação do 'data leakage', a variável-alvo original (`classe_peso_corporal`) "
        "é **descartada** para fins analíticos. \n\n"
        "Para prosseguir com uma análise clinicamente válida, usaremos a "
        "nova variável-alvo **`classe_peso_oms`**, que foi criada seguindo as faixas de IMC "
        "padrão da Organização Mundial da Saúde (OMS)."
    )

    #Gráfico 1: A Nova Distribuição (OMS)
    st.subheader("Distribuição da Nova Variável-Alvo (OMS)")
    st.markdown("Este gráfico mostra a distribuição de dados usando a classificação OMS. "
                "Note como ela é mais orgânica, com a maioria dos indivíduos "
                "na categoria 'Sobrepeso', ao contrário da distribuição artificial original.")

    #1. Criar o gráfico
    fig_peso_oms = px.histogram(
        df_gold, 
        x='classe_peso_oms', #USANDO A NOVA COLUNA
        title='Contagem por Classe de Peso (Padrão OMS)',
        color='classe_peso_oms',
        text_auto=True
    )

    #2. Ordenar as barras
    categorias_oms = [
        'Peso Insuficiente', 
        'Peso Normal', 
        'Sobrepeso', 
        'Obesidade Grau I', 
        'Obesidade Grau II', 
        'Obesidade Grau III'
    ]
    fig_peso_oms.update_xaxes(categoryorder='array', categoryarray=categorias_oms)

    #3. Ajustar layout
    fig_peso_oms.update_layout(
        template='plotly_dark',
        yaxis_title='Contagem',
        xaxis_title='Classe de Peso (OMS)',
        showlegend=False
    )

    #4. Exibir
    st.plotly_chart(fig_peso_oms, use_container_width=True)

    st.info(
        "**Interpretação:** A nova distribuição revela a verdadeira natureza do dataset: "
        "a grande maioria dos indivíduos se concentra na faixa de 'Sobrepeso' (~27%), "
        "seguida por 'Obesidade Grau I' (~17%). Isso forma uma "
        "curva muito mais natural e confiável para a modelagem."
    )

    st.markdown("---")

    #Gráfico 2: A Confirmação Final     
    st.subheader("Confirmação Final: Gênero vs. Nível OMS")
    st.markdown(
        "Este é o gráfico-chave. Ao cruzar `genero` com a *nova* variável-alvo, "
        "podemos confirmar se o viés desapareceu. "
    )

    #1. Criar um Gráfico Agrupado
    fig_genero_oms = px.histogram(
        df_gold,
        x='classe_peso_oms', #USANDO A NOVA COLUNA
        color='genero',        
        barmode='group',     
        text_auto=True,      
        title='Contagem de Gênero por Classe de Peso (Padrão OMS)',
        color_discrete_map={ #Mantendo a consistência das cores
            'Female': '#FF69B4', 
            'Male': '#1E90FF'
        }
    )

    #2. Ordenar as barras
    fig_genero_oms.update_xaxes(categoryorder='array', categoryarray=categorias_oms)

    #3. Ajustar layout
    fig_genero_oms.update_layout(
        template='plotly_dark',
        yaxis_title='Contagem',
        xaxis_title='Classe de Peso (OMS)'
    )

    #4. Exibir o gráfico
    st.plotly_chart(fig_genero_oms, use_container_width=True)

    st.success( 
        "**Conclusão da Investigação:** Sucesso. O viés desapareceu. \n\n"
        "Ao contrário do gráfico original, aqui vemos que homens e mulheres estão "
        "**distribuídos em *todas* as categorias** de obesidade. A separação artificial "
        "de 'Tipo II' (só homens) e 'Tipo III' (só mulheres) não existe mais. \n\n"
        "**Os dados agora são válidos** para continuar a análise."
    )

    st.markdown("---")
  
    st.header("Análise do Índice de Estilo de Vida (OMS)")
    st.markdown(
        "Aqui analisamos o 'Índice de Estilo de Vida' (criado pela soma de "
        "hábitos positivos) contra cada classe de peso corrigida."
    )

    #Criar o Gráfico de Boxplot
    categorias_oms = [
        'Peso Insuficiente', 'Peso Normal', 'Sobrepeso', 
        'Obesidade Grau I', 'Obesidade Grau II', 'Obesidade Grau III'
    ]

    #1. Criar o Boxplot
    fig_estilo_vida = px.box(
        df_gold,
        x='classe_peso_oms',
        y='indice_estilo_vida',
        color='classe_peso_oms',
        title='Distribuição do Índice de Estilo de Vida por Classe de Peso (OMS)',
        category_orders={'classe_peso_oms': categorias_oms} #Ordena o eixo X
    )

    fig_estilo_vida.update_layout(
        template='plotly_dark',
        yaxis_title='Pontuação do Estilo de Vida (Quanto maior, melhor)', 
        xaxis_title='Classe de Peso (OMS)'
    )
    st.plotly_chart(fig_estilo_vida, use_container_width=True)

    st.info(
        "**Interpretação:** Devemos ver uma **tendência de queda** clara. "
        "Indivíduos em 'Peso Normal' devem ter as pontuações mais altas "
        "(medianas mais altas no boxplot), e essa pontuação deve diminuir "
        "conforme o nível de obesidade aumenta."
    )

    st.markdown("---")
    st.header("Análise do Índice de Risco Alimentar (OMS)")
    st.markdown(
        "Aqui analisamos o 'Índice de Risco Alimentar' (criado pela soma de "
        "hábitos negativos) contra cada classe de peso corrigida."
    )

    # 1. Criar o Boxplot
    fig_risco_alim = px.box(
        df_gold,
        x='classe_peso_oms',
        y='indice_risco_alimentar', # <- Sua outra nova feature no eixo Y
        color='classe_peso_oms',
        title='Distribuição do Risco Alimentar por Classe de Peso (OMS)',
        category_orders={'classe_peso_oms': categorias_oms} #Ordena o eixo X
    )

    fig_risco_alim.update_layout(
        template='plotly_dark',
        yaxis_title='Pontuação de Risco Alimentar (Quanto maior, pior)', 
        xaxis_title='Classe de Peso (OMS)'
    )
    st.plotly_chart(fig_risco_alim, use_container_width=True)

    st.info(
        "**Interpretação:** Aqui, esperamos a **tendência oposta (de subida)**. "
        "Indivíduos em 'Peso Normal' devem ter as pontuações de risco mais baixas, "
        "e essa pontuação deve aumentar conforme o nível de obesidade aumenta."
    )

    st.markdown("---")
    st.header("Conclusões da Análise Exploratória")
    st.markdown(
        "Esta visão analítica completou uma jornada crucial de investigação de dados. "
        "O que começou como uma simples exploração nos levou a uma **descoberta crítica "
        "de 'data leakage'** na variável-alvo original, que invalidava qualquer "
        "modelagem preditiva."
        "\n\n"
        "Através da validação, prova e correção deste problema (com a criação da "
        "coluna `classe_peso_oms`), fomos capazes de:"
        "\n1. **Provar** por que os dados originais eram falhos."
        "\n2. **Construir** uma base de dados analítica válida e cientificamente robusta."
        "\n3. **Identificar** os verdadeiros fatores de risco (como os Índices de Estilo de "
        "Vida e Risco Alimentar) que de fato se correlacionam com os níveis de obesidade "
        "corretos."
        "\n\n"
        "Com esta base de dados validada e os insights claros, estamos prontos "
        "para prosseguir para a próxima seção: o **Sistema Preditivo**."
    )