"""
save_silver_version.py
------------------------------------
Módulo responsável por versionar automaticamente a camada Silver no PostgreSQL.
Cada execução cria uma nova tabela com timestamp no nome, garantindo rastreabilidade
e persistência das transformações realizadas durante a análise exploratória.
"""

import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# === Carrega variáveis de ambiente ===
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

# === Cria engine de conexão ===
def get_engine():
    connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_string)


# === Função principal de salvamento ===
def save_silver(df: pd.DataFrame, base_name: str = "obesity_silver"):
    """
    Salva o DataFrame como uma nova versão da camada Silver no PostgreSQL.
    
    Parâmetros:
    - df: pd.DataFrame -> DataFrame a ser salvo.
    - base_name: str -> nome base da tabela (default = "obesity_silver").
    """

    # Cria timestamp para versionamento
    version = datetime.now().strftime("%Y%m%d_%H%M")
    table_name = f"{base_name}_{version}"

    engine = get_engine()

    # Salva no banco
    try:
        df.to_sql(table_name, engine, index=False, if_exists='replace')
        print(f"✅ Tabela '{table_name}' criada com sucesso no banco de dados PostgreSQL.")
    except Exception as e:
        print(f"❌ Erro ao salvar tabela '{table_name}': {e}")
    finally:
        engine.dispose()


# === Função opcional: listar versões salvas ===
def list_silver_versions(base_name: str = "obesity_silver"):
    """
    Lista todas as tabelas versionadas existentes no banco.
    """
    engine = get_engine()
    with engine.connect() as conn:
        query = text(
            f"SELECT table_name FROM information_schema.tables WHERE table_name LIKE '{base_name}_%';"
        )
        results = conn.execute(query).fetchall()
        tables = [r[0] for r in results]
        return sorted(tables) if tables else []
    engine.dispose()
