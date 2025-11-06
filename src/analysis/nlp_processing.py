# transdevs_techexperience/src/analysis/nlp_processing.py

import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize # Mantido para fallback se spaCy falhar
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import logging
import sys # Necessário para adicionar project_root ao sys.path, especialmente em notebooks
import os
import numpy as np

from src.config import TYPO_CORRECTION_MAP, TEXT_COLUMNS_FOR_NLP, POSITIVE_WORDS, NEGATIVE_WORDS, TOPIC_MODEL_PATH, TFIDF_VECTORIZER_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Adiciona o diretório raiz do projeto ao sys.path para que 'src' seja reconhecido como um pacote
# Usar __file__ é OK aqui porque este script será importado, não executado diretamente como o main.py de um Streamlit.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Carrega o modelo de português do spaCy
try:
    nlp = spacy.load("pt_core_news_sm")
    logging.info("Modelo spaCy 'pt_core_news_sm' carregado com sucesso.")
except OSError:
    logging.error("Modelo spaCy 'pt_core_news_sm' não encontrado. Por favor, execute: python -m spacy download pt_core_news_sm")
    logging.error("A lematização e tokenização podem não funcionar corretamente.")
    nlp = None # Define nlp como None para evitar erros posteriores

# Inicializa as stopwords do NLTK (assumimos que já foram baixadas pelo script)
# Se falhar aqui, o problema é de download/path do NLTK.
try:
    stop_words_pt = set(stopwords.words('portuguese'))
    logging.info("Stopwords do NLTK carregadas com sucesso.")
except LookupError:
    logging.error("Recurso 'stopwords' do NLTK não encontrado. Verifique a execução de nltk_download_script.py.")
    stop_words_pt = set() # Fallback para set vazio para não quebrar


def correct_typos_and_standardize(text: str) -> str:
    """
    Corrige erros de digitação e padroniza termos usando o mapa definido em config.
    """
    if not isinstance(text, str):
        return ""
    words = text.split()
    corrected_words = [TYPO_CORRECTION_MAP.get(word, word) for word in words]
    return ' '.join(corrected_words)

def clean_text(text: str) -> str:
    """
    Realiza a limpeza básica de um texto: minúsculas, remove pontuação e números.
    Aplicada APÓS a correção de typos.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-záàâãéêíóôõúüç\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text: str) -> list:
    """
    Tokeniza o texto usando spaCy, remove stopwords (do spaCy) e aplica lematização.
    Prioriza spaCy. Se spaCy não estiver carregado, faz um fallback para NLTK para tokenização simples.
    """
    if not isinstance(text, str) or nlp is None:
        if nlp is None:
            logging.warning("spaCy NLP model não carregado. Usando tokenização básica do NLTK sem lematização ou remoção de stopwords.")
            tokens = [word for word in word_tokenize(text, language='portuguese') if len(word) > 1]
            return tokens
        return []
    
    doc = nlp(text)
    # Lematiza e filtra stopwords (do spaCy), pontuação e tokens de uma única letra
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space and len(token.lemma_) > 1]
    return lemmas

def extract_ngrams(token_list_of_lists: list, n: int = 2, top_n: int = 10) -> Counter:
    """
    Extrai n-grams (sequências de palavras) de uma lista de listas de tokens.
    """
    if not token_list_of_lists:
        return Counter()
    
    all_tokens = [token for sublist in token_list_of_lists for token in sublist]
    ngrams_counts = Counter(nltk.ngrams(all_tokens, n))
    
    return ngrams_counts.most_common(top_n)

def get_ngram_text_for_wordcloud(lemmas_list: list, n: int = 1) -> str:
    """
    Converte uma lista de lemmas em uma string formatada para WordCloud.
    Usa os lemmas já processados pelo spaCy (com stopwords removidas) e filtra por comprimento.
    """
    if not lemmas_list:
        return ""

    # Os lemmas já vêm sem stopwords do spaSy e com len > 1, então apenas um filtro básico de len
    filtered_lemmas = [lemma for lemma in lemmas_list if len(lemma) > 1]

    if n == 1:
        text = " ".join(filtered_lemmas)
    elif n > 1:
        ngrams = []
        for i in range(len(filtered_lemmas) - n + 1):
            ngram_tuple = filtered_lemmas[i:i+n]
            if all(len(lemma) > 1 for lemma in ngram_tuple):
                ngrams.append("_".join(ngram_tuple))
        text = " ".join(ngrams)
    else:
        text = ""
    
    return text

def vectorize_text_tfidf(texts: pd.Series, max_features: int = 1000) -> tuple[TfidfVectorizer, pd.DataFrame]:
    """
    Vetoriza uma série de textos usando TF-IDF.
    Retorna o vetorizador treinado e o DataFrame TF-IDF.
    """
    logging.info("Vetorizando textos com TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    texts_clean = texts.fillna("")
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts_clean)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    logging.info(f"Textos vetorizados. Matriz TF-IDF com {tfidf_df.shape[0]} documentos e {tfidf_df.shape[1]} features.")
    
    os.makedirs(os.path.dirname(TOPIC_MODEL_PATH), exist_ok=True)
    with open(TOPIC_MODEL_PATH, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    logging.info(f"Vetorizador TF-IDF salvo em: {TFIDF_VECTORIZER_PATH}")

    return tfidf_vectorizer, tfidf_df

def apply_topic_modeling_lda(tfidf_matrix: pd.DataFrame, num_topics: int = 5, n_top_words: int = 10) -> tuple[LatentDirichletAllocation, list]:
    """
    Aplica o modelo LDA para descobrir tópicos nos textos.
    Retorna o modelo LDA treinado e os tópicos com suas palavras-chave.
    """
    logging.info(f"Aplicando Modelagem de Tópicos (LDA) com {num_topics} tópicos...")
    if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
        logging.warning("Matriz TF-IDF vazia ou sem features para modelagem de tópicos.")
        return None, []

    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42, learning_method='batch')
    lda_output = lda_model.fit_transform(tfidf_matrix)
    
    os.makedirs(os.path.dirname(TOPIC_MODEL_PATH), exist_ok=True)
    with open(TOPIC_MODEL_PATH, 'wb') as f:
        pickle.dump(lda_model, f)
    logging.info(f"Modelo LDA salvo em: {TOPIC_MODEL_PATH}")

    feature_names = tfidf_matrix.columns
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(f"Tópico {topic_idx + 1}: {' '.join(top_words)}")
    
    logging.info("Tópicos identificados:")
    for t in topics:
        logging.info(t)
    
    return lda_model, topics

def get_sentiment_score_lexicon(text: str) -> str:
    """
    Classifica o sentimento de um texto baseado em léxicos de palavras positivas e negativas.
    """
    if not isinstance(text, str) or text.strip() == "":
        return "Neutro"
    
    lemmas = tokenize_and_lemmatize(text) # Usa a lematização do spaCy (ou fallback simples)
    
    pos_score = sum(1 for word in lemmas if word in POSITIVE_WORDS)
    neg_score = sum(1 for word in lemmas if word in NEGATIVE_WORDS)
    
    if pos_score > neg_score:
        return "Positivo"
    elif neg_score > pos_score:
        return "Negativo"
    else:
        return "Neutro"

def analyze_sentiment(texts: pd.Series) -> pd.Series:
    """
    Realiza a análise de sentimento em uma série de textos usando o léxico customizado.
    Retorna uma série com a categoria de sentimento.
    """
    logging.info("Realizando análise de sentimento com léxico customizado...")
    sentiments = texts.fillna("").apply(get_sentiment_score_lexicon)
    sentiment_series = pd.Series(sentiments, index=texts.index)
    logging.info("Análise de sentimento concluída com léxico customizado.")
    return sentiment_series

if __name__ == '__main__':
    logging.info("Executando nlp_processing.py para teste com spaCy e Léxico Customizado.")
    sample_data = {
        'text_col_1': [
            'Eu quero aprender programação e ter experiencia.',
            'Minha espectativa é de fazer novas conexões, mas tenho dúvidaddes e medo.',
            'Trago muita vontade de crescer profisionalmente.',
            'O grupo pode ajudar com um aprendizado pratico, é uma ótima oportunidade.',
            'Preciso me dedicar mais aos estudos, é um compromisso difícil e sinto incerteza.'
        ],
        'text_col_2': [
            'Desenvolvimeno backend é meu foco, adoro programar.',
            'Não gosto muito de projetos em grupo, mas estou aqui.',
            'Isso é muito legal!',
            np.nan,
            'Estou muito feliz com o projeto.'
        ]
    }
    test_df = pd.DataFrame(sample_data)

    for col in test_df.columns:
        test_df[col + '_cleaned'] = test_df[col].apply(correct_typos_and_standardize).apply(clean_text)
    
    print("\n--- Textos Limpos e Corrigidos ---")
    print(test_df[['text_col_1_cleaned', 'text_col_2_cleaned']].head())

    test_df['text_col_1_lemmas'] = test_df['text_col_1_cleaned'].apply(tokenize_and_lemmatize)
    print("\n--- Lemmas (spaCy) ---")
    print(test_df['text_col_1_lemmas'].head())

    lemmas_for_wc = test_df['text_col_1_lemmas'].explode().tolist()
    bigram_wc_text = get_ngram_text_for_wordcloud(lemmas_for_wc, n=2)
    print(f"\n--- Bigram Text for WordCloud (sample): {bigram_wc_text[:100]} ---")


    bigrams = extract_ngrams(test_df['text_col_1_lemmas'].tolist(), n=2)
    print("\n--- Bigrams mais comuns (com lemmas) ---")
    print(bigrams)

    full_cleaned_text = test_df['text_col_1_cleaned'].fillna("") + " " + test_df['text_col_2_cleaned'].fillna("")
    if not full_cleaned_text.empty:
        tfidf_vectorizer, tfidf_df = vectorize_text_tfidf(full_cleaned_text)
        if not tfidf_df.empty:
            lda_model, topics = apply_topic_modeling_lda(tfidf_df, num_topics=2)

    test_df['sentiment_text_col_1'] = analyze_sentiment(test_df['text_col_1_cleaned'])
    print("\n--- Análise de Sentimento (Léxico Customizado) ---")
    print(test_df[['text_col_1', 'sentiment_text_col_1']].head())