# transdevs_techexperience/run_eda.py

import logging
from src.analysis.eda import load_processed_data, analyze_categorical_distributions, process_and_analyze_text_columns
from src.analysis.leadership_analysis import analyze_leadership_potential, load_pii_mapping
import os
import pandas as pd
from src.config import PROCESSED_DATA_PATH, EDA_FINAL_PATH, LEADERSHIP_ANALYSIS_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Função principal para executar a Análise Exploratória de Dados (EDA) avançada.
    """
    logging.info("Iniciando a fase de Análise Exploratória de Dados (EDA) avançada do TransDevs TechExperience.")

    # 1. Carregar dados processados
    df_active_participants = load_processed_data()
    if df_active_participants.empty:
        logging.error("Não foi possível carregar os dados processados. Encerrando a EDA.")
        return

    # 2. Análise de colunas categóricas
    logging.info("\n--- Análise das Distribuições Categóricas ---")
    categorical_cols = ['grupo_principal', 'grupo_alternativo', 'interesse_lideranca']
    analyze_categorical_distributions(df_active_participants, categorical_cols)

    # 3. Processamento e análise das colunas de texto livre (NLP, Tópicos, Sentimento)
    logging.info("\n--- Processamento e Análise de Campos de Texto Livre (NLP, Tópicos, Sentimento) ---")
    df_final_eda = process_and_analyze_text_columns(df_active_participants, ['objetivo_proposito', 'expectativas_experiencia', 'bagagem_contribuicao', 'contribuicao_grupo', 'compromisso_pessoal', 'expectativas_pos_projeto'])
    
    # Salvar o DataFrame final da EDA para uso no Streamlit
    os.makedirs(os.path.dirname(EDA_FINAL_PATH), exist_ok=True) # Garante que a pasta existe
    df_final_eda.to_csv(EDA_FINAL_PATH, index=False)
    logging.info(f"Dados finais da EDA (com NLP, tópicos, sentimento) salvos em: {EDA_FINAL_PATH}")

    # 4. Análise de Potencial de Liderança
    logging.info("\n--- Análise de Potencial de Liderança ---")
    df_pii = load_pii_mapping() # Carrega PII mapping para referência interna segura
    df_leadership_insights = analyze_leadership_potential(df_final_eda, df_pii)
    
    if not df_leadership_insights.empty:
        os.makedirs(os.path.dirname(LEADERSHIP_ANALYSIS_PATH), exist_ok=True) # Garante que a pasta existe
        df_leadership_insights.to_csv(LEADERSHIP_ANALYSIS_PATH, index=False)
        logging.info(f"Insights de liderança salvos em: {LEADERSHIP_ANALYSIS_PATH}")
    else:
        logging.warning("Nenhum insight de liderança gerado.")

    logging.info("Análise Exploratória de Dados avançada concluída.")

if __name__ == "__main__":
    main()