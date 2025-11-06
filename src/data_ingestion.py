# transdevs_techexperience/src/data_ingestion.py

import pandas as pd
from src.config import RAW_DATA_PATH # Importa o caminho do arquivo de configuração
import logging

# Configura o logger para exibir mensagens de informação e erros no console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_raw_data(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Carrega o arquivo CSV com os dados brutos da pasta local 'data/raw/'.

    Args:
        file_path (str): Caminho para o arquivo CSV. Por padrão, usa RAW_DATA_PATH do config.

    Returns:
        pd.DataFrame: DataFrame contendo os dados brutos.
                      Retorna um DataFrame vazio se o arquivo não for encontrado ou houver erro.
    """
    try:
        logging.info(f"Tentando carregar dados brutos de: {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Dados brutos carregados com sucesso. Total de {len(df)} registros.")
        return df
    except FileNotFoundError:
        logging.error(f"Erro: Arquivo não encontrado em {file_path}. Verifique o caminho em src/config.py e a existência do arquivo.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado ao carregar o CSV: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Este bloco é executado apenas quando o script é chamado diretamente (para testes)
    print("--- Testando src/data_ingestion.py ---")
    raw_df = load_raw_data()
    if not raw_df.empty:
        print("\nPrimeiras 5 linhas dos dados brutos carregados:")
        print(raw_df.head())
        print(f"\nColunas originais: {raw_df.columns.tolist()}")
    else:
        print("\nNão foi possível carregar os dados brutos para teste.")