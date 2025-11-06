# transdevs_techexperience/src/analysis/leadership_analysis.py

import pandas as pd
import logging
import os
from collections import defaultdict
from src.config import EDA_FINAL_PATH, ANONYMIZED_PII_PATH, LEADERSHIP_TYPES, GROUP_NAMES, LEADERSHIP_SENTIMENT_COLS, TOPIC_TO_GROUP_APTITUDE_MAP
from src.analysis.nlp_processing import tokenize_and_lemmatize
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_eda_data(file_path: str = EDA_FINAL_PATH) -> pd.DataFrame:
    """
    Carrega o DataFrame final da EDA, que inclui os insights de NLP.
    """
    try:
        logging.info(f"Carregando dados finais da EDA de: {file_path}")
        df = pd.read_csv(file_path)
        # Garantir que main_topic é int (se foi lido como float por NaN)
        if 'main_topic' in df.columns:
            df['main_topic'] = df['main_topic'].fillna(0).astype(int) # Preenche NaN com 0 antes de converter para int
        logging.info(f"Dados finais da EDA carregados com sucesso. Total de {len(df)} registros.")
        return df
    except FileNotFoundError:
        logging.error(f"Erro: Arquivo EDA final não encontrado em {file_path}.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Ocorreu um erro ao carregar os dados finais da EDA: {e}")
        return pd.DataFrame()

def load_pii_mapping(file_path: str = ANONYMIZED_PII_PATH) -> pd.DataFrame:
    """
    Carrega o mapeamento de PII (participant_id para nome original) de forma segura.
    Este arquivo DEVE SER TRATADO COM EXTREMA CAUTELA e NUNCA exposto publicamente.
    """
    try:
        logging.info(f"Carregando mapeamento de PII de: {file_path}")
        df_pii = pd.read_csv(file_path)
        logging.info("Mapeamento de PII carregado com sucesso.")
        return df_pii
    except FileNotFoundError:
        logging.warning(f"Aviso: Arquivo de mapeamento de PII não encontrado em {file_path}. Nomes originais não estarão disponíveis para referência.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Ocorreu um erro ao carregar o mapeamento de PII: {e}")
        return pd.DataFrame()

def analyze_leadership_potential(df_eda: pd.DataFrame, df_pii: pd.DataFrame = None) -> pd.DataFrame:
    """
    Analisa o potencial de liderança dos participantes com base nas preferências de grupo,
    interesse em liderança, bagagem, sentimento e alinhamento de tópicos LDA.
    """
    if df_eda.empty:
        logging.error("DataFrame EDA está vazio. Não é possível analisar o potencial de liderança.")
        return pd.DataFrame()

    logging.info("Iniciando análise de potencial de liderança aprimorada...")

    df_leadership_processed = df_eda.copy() 

    id_to_name = {}
    if df_pii is not None and not df_pii.empty:
        id_to_name = df_pii.set_index('participant_id')['nome_completo'].to_dict()
    
    # Inicializa as colunas de status e sugestão para TODOS os participantes
    df_leadership_processed['sugestao_lideranca_grupo'] = 'N/A'
    df_leadership_processed['tipo_sugestao'] = 'N/A'
    df_leadership_processed['status_lideranca_final'] = 'Participante Comum' # Default
    df_leadership_processed['aptidao_score_geral'] = 0.0
    df_leadership_processed['justificativa_topico_lda'] = df_leadership_processed['main_topic'].apply(lambda x: f"Tópico {int(x)}" if not pd.isna(x) else "N/A")
    # Gerar justificativa de sentimento desde o início
    obj_sentiment = df_leadership_processed['objetivo_proposito_sentiment'].fillna('N/A')
    bag_sentiment = df_leadership_processed['bagagem_contribuicao_sentiment'].fillna('N/A')
    df_leadership_processed['justificativa_sentimento'] = obj_sentiment + "/" + bag_sentiment


    # 1. Atribuir líderes diretos e marcar seus status
    direct_leaders_mask = df_leadership_processed['interesse_lideranca'] == LEADERSHIP_TYPES['DIRETA']
    direct_leaders_to_process = df_leadership_processed[direct_leaders_mask].copy() # Trabalha apenas com quem quer liderar diretamente

    group_leadership_status = {group: {'leader_id': None, 'leader_name': None, 'type': None, 'preferred_by_direct': []} for group in GROUP_NAMES}

    for idx, row in direct_leaders_to_process.iterrows():
        participant_id = row['participant_id']
        name = id_to_name.get(participant_id, f"ID_{participant_id}")
        pref_group = row['grupo_principal']
        alt_group = row['grupo_alternativo']

        assigned_group = None
        if pref_group in GROUP_NAMES and group_leadership_status[pref_group]['leader_id'] is None:
            group_leadership_status[pref_group]['leader_id'] = participant_id
            group_leadership_status[pref_group]['leader_name'] = name
            group_leadership_status[pref_group]['type'] = 'Direta (Principal)'
            assigned_group = pref_group
            logging.info(f"Líder Direto '{name}' (ID: {participant_id}) atribuído ao grupo '{pref_group}' (preferência principal).")
        elif alt_group != 'Não tenho interesse por nenhuma outra opção' and alt_group in GROUP_NAMES and group_leadership_status[alt_group]['leader_id'] is None:
            group_leadership_status[alt_group]['leader_id'] = participant_id
            group_leadership_status[alt_group]['leader_name'] = name
            group_leadership_status[alt_group]['type'] = 'Direta (Alternativa)'
            assigned_group = alt_group
            logging.info(f"Líder Direto '{name}' (ID: {participant_id}) atribuído ao grupo alternativo '{alt_group}'.")
        else:
            if pref_group in GROUP_NAMES:
                group_leadership_status[pref_group]['preferred_by_direct'].append({'id': participant_id, 'name': name})
            logging.warning(f"Líder Direto '{name}' (ID: {participant_id}) não pôde ser atribuído a um grupo. Preferências: Principal='{pref_group}', Alternativa='{alt_group}'.")
        
        # ATUALIZA O DF PRINCIPAL COM O STATUS DO LÍDER DIRETO
        df_leadership_processed.loc[idx, 'status_lideranca_final'] = f"Líder Direto Atribuído ({group_leadership_status[assigned_group]['type']})" if assigned_group else "Líder Direto (sem atribuição)"
        df_leadership_processed.loc[idx, 'sugestao_lideranca_grupo'] = assigned_group if assigned_group else 'N/A'


    # 2. Identificar grupos que *realmente* precisam de líderes agora (após a atribuição direta)
    groups_needing_leaders = [group for group, status in group_leadership_status.items() if status['leader_id'] is None]
    logging.info(f"Grupos ainda sem liderança direta (após 1ª rodada): {groups_needing_leaders}")

    # 3. Processar Potenciais Líderes de Suporte
    # Apenas para participantes que querem SUPORTE e ainda não foram marcados como líderes diretos
    support_leaders_mask = (df_leadership_processed['interesse_lideranca'] == LEADERSHIP_TYPES['SUPORTE']) & \
                           (df_leadership_processed['status_lideranca_final'] == 'Participante Comum') # Não é um líder direto

    support_leaders_candidates_df = df_leadership_processed[support_leaders_mask].copy()

    if not support_leaders_candidates_df.empty and groups_needing_leaders:
        logging.info(f"Avaliando {len(support_leaders_candidates_df)} candidatos a líder de suporte para os grupos {groups_needing_leaders}.")

        for idx, row in support_leaders_candidates_df.iterrows():
            participant_id = row['participant_id']
            
            best_aptitude_score_for_needing_group = -1
            best_suggested_group = 'N/A'

            for group_needed in groups_needing_leaders:
                aptitude_score_for_group = 0.0

                # 1. Alinhamento de Preferência Direta
                if row['grupo_principal'] == group_needed:
                    aptitude_score_for_group += 0.5
                elif row['grupo_alternativo'] == group_needed:
                    aptitude_score_for_group += 0.3

                # 2. Alinhamento de Tópicos (usando TOPIC_TO_GROUP_APTITUDE_MAP)
                main_topic_id = row['main_topic'] if not pd.isna(row['main_topic']) else None
                if main_topic_id is not None and int(main_topic_id) in TOPIC_TO_GROUP_APTITUDE_MAP: # Garante que a chave é int
                    topic_aptitude = TOPIC_TO_GROUP_APTITUDE_MAP[int(main_topic_id)].get(group_needed, 0.0)
                    aptitude_score_for_group += topic_aptitude * 0.7

                # 3. Score de Sentimento Ponderado (Proatividade, Engajamento)
                sentiment_score_val = 0
                for col in LEADERSHIP_SENTIMENT_COLS:
                    if col in row and row[col] == 'Positivo':
                        sentiment_score_val += 1
                    elif col in row and row[col] == 'Negativo':
                        sentiment_score_val -= 1
                aptitude_score_for_group += sentiment_score_val * 0.1
                
                if not np.isfinite(aptitude_score_for_group):
                    aptitude_score_for_group = 0.0


                if aptitude_score_for_group > best_aptitude_score_for_needing_group:
                    best_aptitude_score_for_needing_group = aptitude_score_for_group
                    best_suggested_group = group_needed
            
            if best_aptitude_score_for_needing_group > 0:
                df_leadership_processed.loc[idx, 'sugestao_lideranca_grupo'] = best_suggested_group
                df_leadership_processed.loc[idx, 'tipo_sugestao'] = 'Potencial Líder (Sugestão Algorítmica)'
                df_leadership_processed.loc[idx, 'status_lideranca_final'] = 'Potencial Líder para Suporte'
                df_leadership_processed.loc[idx, 'aptidao_score_geral'] = round(best_aptitude_score_for_needing_group, 2)
            else:
                df_leadership_processed.loc[idx, 'status_lideranca_final'] = 'Participante com Interesse em Suporte (sem match forte)'
                df_leadership_processed.loc[idx, 'aptidao_score_geral'] = 0.0

    
    logging.info("Análise de potencial de liderança aprimorada concluída.")
    
    # Colunas que queremos retornar do df_leadership_processed para o CSV final
    # Renomeando as colunas no DataFrame de saída para o formato final desejado no dashboard
    df_final_insights = df_leadership_processed.rename(columns={
        'interesse_lideranca': 'lideranca_interesse_declarado',
        'grupo_principal': 'grupo_principal_preferido',
        'grupo_alternativo': 'grupo_alternativo_preferido',
        'bagagem_contribuicao_cleaned': 'justificativa_bagagem'
    })
    
    # Selecionar apenas as colunas que queremos no CSV final
    final_cols_to_return_after_rename = [
        'participant_id', 
        'lideranca_interesse_declarado',
        'grupo_principal_preferido',
        'grupo_alternativo_preferido',
        'sugestao_lideranca_grupo', 
        'tipo_sugestao', 
        'status_lideranca_final', 
        'aptidao_score_geral', 
        'justificativa_bagagem', 
        'justificativa_topico_lda', 
        'justificativa_sentimento'
    ]

    return df_final_insights[final_cols_to_return_after_rename]


if __name__ == '__main__':
    logging.info("Executando leadership_analysis.py para teste com lógica aprimorada.")
    df_eda_test = load_eda_data()
    df_pii_test = load_pii_mapping()

    if not df_eda_test.empty:
        df_leadership_insights = analyze_leadership_potential(df_eda_test, df_pii_test)
        print("\n--- Insights de Liderança Gerados (Lógica Aprimorada) ---")
        print(df_leadership_insights[['participant_id', 'status_lideranca_final', 'sugestao_lideranca_grupo', 'aptidao_score_geral', 'justificativa_bagagem']].head(10))
        
        logging.info("\n--- Resumo de Líderes Atribuídos/Potenciais ---")
        logging.info(df_leadership_insights['status_lideranca_final'].value_counts())

        os.makedirs(os.path.dirname(LEADERSHIP_ANALYSIS_PATH), exist_ok=True)
        df_leadership_insights.to_csv(LEADERSHIP_ANALYSIS_PATH, index=False)
        logging.info(f"Insights de liderança salvos em: {LEADERSHIP_ANALYSIS_PATH}")