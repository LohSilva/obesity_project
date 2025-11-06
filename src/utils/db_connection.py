import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd


#Carrega as variaveis de ambiente do arquivo .env
load_dotenv()

#Lê as variáveis de ambiente necessárias para a conexão com o banco de dados
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

#Cria a string de conexão com o banco de dados PostgreSQL
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

#Cria o engine de conexão com o banco de dados
engine = create_engine(DATABASE_URL)

def get_data(query: str) -> pd.DataFrame:
    """
    Executa uma consulta SQL e retorna os resultados em um DataFrame do Pandas.
    """
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
    return df

def test_connection():    
    """
    Testa a conexão com o banco de dados.
    """
    try:
        with engine.connect() as connection:
            print("Conexão bem-sucedida ao banco de dados!")   
    except Exception as e:
            print(f"Erro ao conectar ao banco de dados: {e}")
        