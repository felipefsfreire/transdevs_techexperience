# transdevs_techexperience/src/analysis/eda.py

import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import logging
from collections import Counter
from src.config import (
    PROCESSED_DATA_PATH, TEXT_COLUMNS_FOR_NLP, TYPO_CORRECTION_MAP,
    OVERALL_SENTIMENT_COL # IMPORTADO CORRETAMENTE AQUI
)
from src.analysis.nlp_processing import (
    correct_typos_and_standardize,
    clean_text,
    tokenize_and_lemmatize,
    extract_ngrams,
    vectorize_text_tfidf,
    apply_topic_modeling_lda,
    analyze_sentiment,
    get_ngram_text_for_wordcloud
)
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

stop_words_pt = set(stopwords.words('portuguese'))


def load_processed_data(file_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Carrega o arquivo CSV com os dados já processados (participantes ativos).
    """
    try:
        logging.info(f"Tentando carregar dados processados de: {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Dados processados carregados com sucesso. Total de {len(df)} registros.")
        return df
    except FileNotFoundError:
        logging.error(f"Erro: Arquivo não encontrado em {file_path}. Verifique o caminho.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado ao carregar o CSV de dados processados: {e}")
        return pd.DataFrame()

def analyze_categorical_distributions(df: pd.DataFrame, columns: list) -> dict:
    """
    Realiza a análise de distribuição de colunas categóricas e as apresenta de forma humanizada.
    """
    logging.info("Analisando distribuições de colunas categóricas...")
    results = {}
    for col in columns:
        if col in df.columns:
            logging.info(f"\n--- Distribuição da coluna: '{col}' ---")
            distribution = df[col].value_counts(normalize=True) * 100
            count = df[col].value_counts()
            results[col] = {"percentual": distribution, "quantidade": count}
            
            logging.info(f"\nA distribuição de '{col}' revela:")
            for item, perc in distribution.items():
                qty = count[item]
                logging.info(f"- '{item}': {qty} pessoas ({perc:.2f}%)")
            
            if not distribution.empty:
                most_common = distribution.index[0]
                percentage = distribution.iloc[0]
                qty = count.iloc[0]
                logging.info(f"A maioria das pessoas ({qty} ou aproximadamente {percentage:.2f}%) se identifica principalmente com '{most_common}'.")
        else:
            logging.warning(f"Coluna '{col}' não encontrada no DataFrame para análise de distribuição.")
    return results

def process_and_analyze_text_columns(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    """
    Aplica o pipeline de processamento de texto (limpeza, tokenização, lematização,
    extração de n-grams, vetorização, modelagem de tópicos e análise de sentimento)
    e retorna o DataFrame com as novas colunas de texto processado e insights.
    """
    df_processed_text = df.copy()

    for col in text_columns:
        if col in df_processed_text.columns:
            df_processed_text[f'{col}_cleaned'] = df_processed_text[col].apply(correct_typos_and_standardize).apply(clean_text)
        else:
            logging.warning(f"Coluna de texto '{col}' não encontrada para limpeza.")

    for col in text_columns:
        if f'{col}_cleaned' in df_processed_text.columns:
            df_processed_text[f'{col}_lemmas'] = df_processed_text[f'{col}_cleaned'].apply(tokenize_and_lemmatize)
        
    logging.info("\n--- Análise de Palavras e N-grams Mais Comuns (Geral) ---")
    combined_cleaned_text = df_processed_text[[f'{col}_cleaned' for col in text_columns if f'{col}_cleaned' in df_processed_text.columns]].fillna('').agg(' '.join, axis=1)
    # combined_lemmas_list_of_lists agora é uma Series onde cada elemento é uma lista de lemmas de um participante
    combined_lemmas_list_of_lists = df_processed_text[[f'{col}_lemmas' for col in text_columns if f'{col}_lemmas' in df_processed_text.columns]].apply(
        lambda row: [lemma for sublist in row.values if isinstance(sublist, list) for lemma in sublist], axis=1
    )


    all_lemmas = [word for sublist in combined_lemmas_list_of_lists for word in sublist]
    df_processed_text['all_lemmas_combined'] = combined_lemmas_list_of_lists # Guarda a lista de listas para uso futuro (nuvem de palavras)
    
    word_counts = Counter(all_lemmas)
    logging.info(f"As 20 palavras mais comuns combinadas (lemmatized) são:\n{word_counts.most_common(20)}")

    logging.info("\n--- Bigrams mais comuns (Geral) ---")
    bigrams = extract_ngrams(combined_lemmas_list_of_lists.tolist(), n=2, top_n=15)
    logging.info(f"Os 15 bigrams mais comuns combinados (lemmatized) são:\n{bigrams}")

    logging.info("\n--- Trigrams mais comuns (Geral) ---")
    trigrams = extract_ngrams(combined_lemmas_list_of_lists.tolist(), n=3, top_n=15)
    logging.info(f"Os 15 trigrams mais comuns combinados (lemmatized) são:\n{trigrams}")


    logging.info("\n--- Preparando para Modelagem de Tópicos ---")
    text_for_topic_modeling = df_processed_text[[f'{col}_cleaned' for col in text_columns if f'{col}_cleaned' in df_processed_text.columns]].fillna('').agg(' '.join, axis=1)
    
    tfidf_vectorizer, tfidf_df = vectorize_text_tfidf(text_for_topic_modeling)
    
    num_topics = min(5, len(df_processed_text) - 1)
    if num_topics < 2:
        logging.warning("Número insuficiente de documentos para modelagem de tópicos significativa. Definindo para 1 tópico para evitar erros.")
        num_topics = 1 
        lda_model = None
        topics = []
    
    if not tfidf_df.empty and tfidf_df.shape[1] > 0:
        lda_model, topics = apply_topic_modeling_lda(tfidf_df, num_topics=num_topics)
        if lda_model:
            topic_distribution = lda_model.transform(tfidf_df)
            for i in range(num_topics):
                df_processed_text[f'topic_{i+1}_score'] = topic_distribution[:, i]
            df_processed_text['main_topic'] = topic_distribution.argmax(axis=1) + 1
            logging.info("\nDistribuição dos principais tópicos pelos participantes:")
            logging.info(df_processed_text['main_topic'].value_counts())
    else:
        logging.warning("Matriz TF-IDF vazia ou sem features. Pulando Modelagem de Tópicos (LDA).")
        for i in range(num_topics):
            df_processed_text[f'topic_{i+1}_score'] = np.nan
        df_processed_text['main_topic'] = np.nan

    logging.info("\n--- Análise de Sentimento por Coluna ---")
    for col in text_columns:
        if f'{col}_cleaned' in df_processed_text.columns:
            df_processed_text[f'{col}_sentiment'] = analyze_sentiment(df_processed_text[f'{col}_cleaned'])
            logging.info(f"Sentimento da coluna '{col}':\n{df_processed_text[f'{col}_sentiment'].value_counts()}")

    # NOVO: Calcular Sentimento Geral POR PARTICIPANTE, com prioridade para Negativo/Positivo
    logging.info("\n--- Calculando Sentimento Geral por Participante (Prioridade Negativa/Positiva) ---")
    
    # Obter os nomes das colunas de sentimento geradas individualmente
    individual_sentiment_cols = [f'{col}_sentiment' for col in text_columns if f'{col}_sentiment' in df_processed_text.columns]
    
    def calculate_overall_sentiment_priority(row):
        # Se algum sentimento individual é Negativo, o geral é Negativo
        if any(row[col] == "Negativo" for col in individual_sentiment_cols):
            return "Negativo"
        # Se não há negativos, mas há algum Positivo, o geral é Positivo
        elif any(row[col] == "Positivo" for col in individual_sentiment_cols):
            return "Positivo"
        # Se não há negativos nem positivos (apenas neutros), é Neutro
        else:
            return "Neutro"

    # Aplicar a função para cada linha (participante)
    df_processed_text[OVERALL_SENTIMENT_COL] = df_processed_text.apply(calculate_overall_sentiment_priority, axis=1)
    logging.info(f"Sentimento Geral (por participante com prioridade negativa/positiva):\n{df_processed_text[OVERALL_SENTIMENT_COL].value_counts()}")

    return df_processed_text


if __name__ == '__main__':
    logging.info("Executando módulo EDA para teste com integração NLP.")
    df_active = load_processed_data()

    if not df_active.empty:
        categorical_cols = ['grupo_principal', 'grupo_alternativo', 'interesse_lideranca']
        analyze_categorical_distributions(df_active, categorical_cols)

        df_final_eda = process_and_analyze_text_columns(df_active, TEXT_COLUMNS_FOR_NLP)

        print("\n--- Exemplo de Dados com Novas Colunas de Sentimento e Tópicos ---")
        print(df_final_eda[[
            'participant_id', 
            'grupo_principal', 
            'objetivo_proposito_sentiment', 
            'expectativas_experiencia_sentiment', 
            OVERALL_SENTIMENT_COL,
            'main_topic'
        ]].head())