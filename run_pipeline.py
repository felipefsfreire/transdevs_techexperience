# transdevs_techexperience/run_pipeline.py

import pandas as pd
import logging
import os
from src.data_ingestion import load_raw_data
from src.data_processing import preprocess_data
from src.config import PROCESSED_DATA_PATH # Importa o caminho para salvar os dados processados

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Função principal para executar o pipeline de processamento de dados inicial.
    """
    logging.info("Iniciando o pipeline de processamento de dados do TransDevs TechExperience.")

    # 1. Carregar dados brutos
    raw_df = load_raw_data()
    if raw_df.empty:
        logging.error("Não foi possível carregar os dados brutos. Encerrando o pipeline.")
        return

    # 2. Pré-processar dados (renomear, PII, consciência, filtrar ativos)
    # df_for_conscience_analysis: inclui todos, para análise da coluna 'consciencia_escopo'
    # df_active_participants: apenas quem quer continuar, com PII tratadas
    # pii_mapping: mapeamento de ID para nome (manter seguro!)
    df_for_conscience_analysis, df_active_participants, pii_mapping = preprocess_data(raw_df)
    
    if df_active_participants.empty:
        logging.warning("Nenhum participante ativo encontrado após o pré-processamento. Verifique os dados e critérios.")
    
    # 3. Salvar dados processados (participantes ativos)
    if not df_active_participants.empty:
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True) # Garante que a pasta existe
        df_active_participants.to_csv(PROCESSED_DATA_PATH, index=False)
        logging.info(f"Dados dos participantes ativos salvos em: {PROCESSED_DATA_PATH}")
    else:
        logging.info("Nenhum dado de participante ativo para salvar.")

    # 4. Exemplo de análise da consciência (primeira análise quantitativa)
    logging.info("\n--- Resumo Quantitativo da Consciência do Escopo do Projeto ---")
    if 'consciencia_escopo_padronizada' in df_for_conscience_analysis.columns:
        conscience_counts = df_for_conscience_analysis['consciencia_escopo_padronizada'].value_counts()
        logging.info(f"\n{conscience_counts}")
        
        # Salvar essa informação em um CSV separado para o dashboard
        conscience_summary_path = os.path.join(os.path.dirname(PROCESSED_DATA_PATH), 'conscience_summary.csv')
        os.makedirs(os.path.dirname(conscience_summary_path), exist_ok=True) # Garante que a pasta existe
        conscience_counts.to_csv(conscience_summary_path)
        logging.info(f"Resumo da consciência salvo em: {conscience_summary_path}")
    else:
        logging.warning("Coluna 'consciencia_escopo_padronizada' não encontrada para análise de consciência.")

    logging.info("Pipeline de processamento de dados inicial concluído.")

if __name__ == "__main__":
    main()