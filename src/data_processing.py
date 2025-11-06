# transdevs_techexperience/src/data_processing.py

import pandas as pd
import numpy as np
import logging
import os # Necessário para o ANONYMIZED_PII_PATH

from src.config import ORIGINAL_COL_NAMES, EXCLUSION_CRITERIA, CONSCIENCIA_OPTIONS, ANONYMIZED_PII_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia as colunas do DataFrame para nomes mais amigáveis e padronizados,
    com base no mapeamento ORIGINAL_COL_NAMES do config.
    """
    logging.info("Renomeando colunas...")
    
    # Cria uma cópia para evitar SettingWithCopyWarning
    df_renamed = df.copy() 
    
    # Filtra apenas as colunas que existem no DataFrame para renomear
    columns_to_rename = {k: v for k, v in ORIGINAL_COL_NAMES.items() if k in df_renamed.columns}
    
    if not columns_to_rename:
        logging.warning("Nenhuma coluna para renomear encontrada no DataFrame de acordo com ORIGINAL_COL_NAMES.")
        return df_renamed

    df_renamed = df_renamed.rename(columns=columns_to_rename)
    logging.info("Colunas renomeadas com sucesso.")
    return df_renamed

def handle_pii(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lida com PII: pseudonimiza o nome e remove o telefone.
    Cria um DataFrame separado para mapeamento de PII.

    Args:
        df (pd.DataFrame): DataFrame com dados brutos, incluindo PII.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame processado sem PII sensíveis (nome e telefone).
            - DataFrame contendo Participant_ID e Nome Original para fins de auditoria/referência (NÃO DEVE SER COMPARTILHADO).
    """
    logging.info("Iniciando tratamento de PII (Pseudonimização e Remoção)...")

    df_copy = df.copy() # Trabalhe com uma cópia para não alterar o DF original diretamente

    # Geração de ID único para pseudonimização
    if 'nome_completo' in df_copy.columns:
        df_copy['participant_id'] = range(1, len(df_copy) + 1)
        
        # Cria um DataFrame de mapeamento de PII (para uso *restrito* e seguro)
        pii_mapping_df = df_copy[['participant_id', 'nome_completo']].copy()
        
        # Remove a coluna de nome completo do DF principal de análise
        df_copy = df_copy.drop(columns=['nome_completo'])
    else:
        logging.warning("Coluna 'nome_completo' não encontrada. Pulando pseudonimização de nome.")
        # Cria um DataFrame vazio se a coluna não existe para manter o tipo de retorno
        pii_mapping_df = pd.DataFrame(columns=['participant_id', 'nome_completo']) 
    
    # Remoção da coluna de telefone
    if 'telefone_whatsapp' in df_copy.columns:
        df_copy = df_copy.drop(columns=['telefone_whatsapp'])
    else:
        logging.warning("Coluna 'telefone_whatsapp' não encontrada. Pulando remoção de telefone.")

    logging.info("Tratamento de PII concluído. Nome pseudonimizado, telefone removido.")
    return df_copy, pii_mapping_df

def process_conscience_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza a coluna de consciência do escopo do projeto, usando CONSCIENCIA_OPTIONS do config.
    """
    logging.info("Processando coluna 'consciencia_escopo'...")
    df_copy = df.copy()
    if 'consciencia_escopo' in df_copy.columns:
        df_copy['consciencia_escopo_padronizada'] = df_copy['consciencia_escopo'].map(CONSCIENCIA_OPTIONS)
        # Preenche valores NaN se alguma opção não foi mapeada (ex: um valor novo no formulário)
        df_copy['consciencia_escopo_padronizada'] = df_copy['consciencia_escopo_padronizada'].fillna('Outros/Não Mapeado')
        logging.info("Coluna 'consciencia_escopo' padronizada.")
    else:
        logging.warning("Coluna 'consciencia_escopo' não encontrada para padronização. Criei uma coluna placeholder.")
        df_copy['consciencia_escopo_padronizada'] = 'Não Informado'
    return df_copy

def filter_active_participants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra os participantes que não desejam continuar no projeto.
    Assume que 'consciencia_escopo_padronizada' já foi criada.
    """
    logging.info("Filtrando participantes que não querem continuar no projeto...")
    df_copy = df.copy()
    if 'consciencia_escopo_padronizada' in df_copy.columns:
        initial_count = len(df_copy)
        # Filtra quem NÃO é 'Não Quer Continuar'
        df_active = df_copy[df_copy['consciencia_escopo_padronizada'] != CONSCIENCIA_OPTIONS[EXCLUSION_CRITERIA]].copy()
        removed_count = initial_count - len(df_active)
        logging.info(f"Removidos {removed_count} participantes que não desejam continuar. Restam {len(df_active)} participantes ativos.")
        return df_active
    else:
        logging.warning("Coluna 'consciencia_escopo_padronizada' não encontrada. Nenhum participante será filtrado.")
        return df_copy

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executa o pipeline completo de pré-processamento de dados:
    1. Renomeia colunas.
    2. Lida com PII (pseudonimiza nome, remove telefone).
    3. Processa e padroniza a coluna de consciência do escopo.
    4. Salva o mapeamento de PII.
    5. Retorna o DataFrame para análise de consciência, e o DataFrame de participantes ativos.

    Args:
        df (pd.DataFrame): DataFrame bruto.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - df_processed_for_conscience: DataFrame com colunas renomeadas e consciência padronizada (para análise de consciência).
            - df_active_participants: DataFrame com participantes ativos, PII tratadas e colunas limpas (para análises posteriores).
            - pii_mapping_df: DataFrame contendo o mapeamento de Participant_ID para Nome Original.
    """
    if df.empty:
        logging.error("DataFrame de entrada está vazio. Abortando pré-processamento.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_renamed = rename_columns(df)
    
    df_with_pii_treated, pii_mapping_df = handle_pii(df_renamed)

    # Processa a coluna de consciência ANTES de filtrar, para que possamos analisá-la
    df_with_pii_treated = process_conscience_column(df_with_pii_treated)

    # DataFrame para análise de consciência (inclui quem não quer continuar)
    df_processed_for_conscience = df_with_pii_treated.copy()
    
    # Agora filtra os participantes ativos para análises de perfil e match
    df_active_participants = filter_active_participants(df_with_pii_treated)

    # Salvando o mapeamento de PII (manter MUITO SEGURO)
    if not pii_mapping_df.empty:
        os.makedirs(os.path.dirname(ANONYMIZED_PII_PATH), exist_ok=True)
        pii_mapping_df.to_csv(ANONYMIZED_PII_PATH, index=False)
        logging.info(f"Mapeamento de PII salvo em: {ANONYMIZED_PII_PATH}. **MANTENHA ESTE ARQUIVO EXTREMAMENTE SEGURO!**")
    else:
        logging.warning("Mapeamento de PII está vazio, não será salvo.")
    
    logging.info("Pré-processamento de dados concluído.")
    return df_processed_for_conscience, df_active_participants, pii_mapping_df

if __name__ == '__main__':
    # Este bloco é executado apenas quando o script é chamado diretamente (para testes)
    from src.data_ingestion import load_raw_data

    print("--- Testando src/data_processing.py ---")
    raw_df = load_raw_data()
    if not raw_df.empty:
        df_conscience, df_active, pii_map = preprocess_data(raw_df)
        
        print("\n--- Análise da Consciência do Escopo (antes do filtro de ativos) ---")
        if 'consciencia_escopo_padronizada' in df_conscience.columns:
            print(df_conscience['consciencia_escopo_padronizada'].value_counts())
            print(f"\nTotal de registros para análise de consciência: {len(df_conscience)}")

        print("\n--- Primeiras 5 linhas dos Participantes Ativos (após PII e filtro) ---")
        print(df_active.head())
        print(f"\nColunas dos participantes ativos: {df_active.columns.tolist()}")
        print(f"Total de participantes ativos: {len(df_active)}")

        print("\n--- Mapeamento de PII (Exemplo - NÃO COMPARTILHE!) ---")
        print(pii_map.head())
    else:
        print("\nNão foi possível carregar os dados brutos para pré-processamento.")