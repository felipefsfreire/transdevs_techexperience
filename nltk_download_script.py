# transdevs_techexperience/nltk_download_script.py

import nltk
import logging
import os # Adicionado para garantir a pasta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_nltk_resources():
    """
    Baixa os recursos essenciais do NLTK (stopwords e punkt).
    O NLTK é inteligente o suficiente para não baixar novamente se já existirem.
    """
    logging.info("Verificando e baixando recursos do NLTK (stopwords, punkt)...")
    
    # Define um diretório para os dados do NLTK dentro do projeto
    # Isso ajuda a garantir que o Python os encontre no ambiente virtual.
    project_root = os.path.abspath(os.path.dirname(__file__)) # Raiz do projeto
    nltk_data_dir = os.path.join(project_root, 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    
    # Tenta baixar stopwords
    logging.info("Tentando baixar 'stopwords'...")
    try:
        # Tenta baixar para o diretório específico
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True) 
        # Se o download funcionou, o NLTK agora deve encontrar.
        nltk.data.find('corpora/stopwords', path=nltk_data_dir) # Tenta encontrar explicitamente
        logging.info("Recurso 'stopwords' verificado/baixado com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao baixar 'stopwords': {e}. Verifique sua conexão ou permissões.")
    
    # Tenta baixar punkt
    logging.info("Tentando baixar 'punkt'...")
    try:
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        nltk.data.find('tokenizers/punkt', path=nltk_data_dir)
        logging.info("Recurso 'punkt' verificado/baixado com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao baixar 'punkt': {e}. Verifique sua conexão ou permissões.")
    
    logging.info("Processo de download de recursos do NLTK concluído.")

if __name__ == '__main__':
    print("--- Executando nltk_download_script.py ---")
    download_nltk_resources()
    print("--- Finalizado ---")