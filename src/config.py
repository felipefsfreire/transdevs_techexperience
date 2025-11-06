# transdevs_techexperience/src/config.py

import os

# Caminhos de arquivos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'Checkin TransDevs TechExperience (respostas) - Respostas ao formulário 1.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_participants.csv')
ANONYMIZED_PII_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'anonymized_pii_mapping.csv')
EDA_FINAL_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'eda_final_data.csv')
LEADERSHIP_ANALYSIS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'leadership_insights.csv')
TOPIC_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lda_model.pkl')
TFIDF_VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')


# Nomes originais das colunas do CSV
ORIGINAL_COL_NAMES = {
    'Carimbo de data/hora': 'timestamp',
    '1. Como podemos te chamar?': 'nome_completo', # Será pseudonimizado
    '2. Qual seu telefone whatsapp?': 'telefone_whatsapp', # Será removido
    '3. Você está ciente do escopo do projeto e de como seus dados inseridos nesse formulário serão usados?\n\nCaso tenha dúvidas, procure a equipe do TransDevs ou pergunte no grupo.': 'consciencia_escopo',
    '4.a. Opção principal (gostaria de atuar nesse grupo)': 'grupo_principal',
    '4.b. Opção alternativa (posso atuar nesse grupo caso minha opção principal esteja indisponível)': 'grupo_alternativo',
    '5. Você gostaria de exercer algum tipo de liderança no grupo que escolheu acima?': 'interesse_lideranca',
    '6. Qual seu grande objetivo ou propósito no TechExperience?': 'objetivo_proposito',
    '7. O que você gostaria de viver no TechExperience?\n\nPode ser algum encontro temático pra aprender algo, pode ser uma reunião de interação, pode ser uma dinâmica de quebra-gelo, enfim, o que vier a cabeça, pode mandar que a gente avalia a possibilidade e realiza.': 'expectativas_experiencia',
    '8. O que você trás na sua bagagem que vai ser útil nessa jornada para você e para o grupo?': 'bagagem_contribuicao',
    '9. Como o grupo pode contribuir com o seu objetivo ou propósito?': 'contribuicao_grupo',
    '10. Qual compromisso você precisa fazer consigo mesmo para viver o seu propósito no TransDevs?': 'compromisso_pessoal',
    '11. O quê você espera levar consigo após o final do projeto?': 'expectativas_pos_projeto',
}

# Resposta que indica que a pessoa não quer continuar no projeto
EXCLUSION_CRITERIA = "Não quero continuar no projeto."

# Opções para a coluna 'consciencia_escopo'
CONSCIENCIA_OPTIONS = {
    "Estou ciente do escopo do projeto e quero continuar.": "Ciente e Quer Continuar",
    "Ainda tenho dúvidas, mas quero continuar no projeto.": "Dúvidas, mas Quer Continuar",
    EXCLUSION_CRITERIA: "Não Quer Continuar"
}

# Colunas de texto livre para análise de NLP
TEXT_COLUMNS_FOR_NLP = [
    'objetivo_proposito',
    'expectativas_experiencia',
    'bagagem_contribuicao',
    'contribuicao_grupo',
    'compromisso_pessoal',
    'expectativas_pos_projeto'
]

# Dicionário para correção de erros de digitação (typos) e padronização.
TYPO_CORRECTION_MAP = {
    'progamação': 'programação', 'experiecia': 'experiência', 'desenvolvimeno': 'desenvolvimento',
    'tecnoogia': 'tecnologia', 'profisional': 'profissional', 'conectx': 'conexão',
    'gostaria': 'gostar', 'vivência': 'experiência', 'conheçer': 'conhecer',
    'disposta': 'disposto', 'trago': 'trazer', 'sei': 'saber', 'faço': 'fazer',
    'área': 'area', 'dúvidaddes': 'dúvidas', 'pratico': 'prático',
}

# Léxicos de palavras para análise de sentimento em português
POSITIVE_WORDS = [
    'aprender', 'aprimorar', 'ajudar', 'oportunidade', 'conhecimento', 'crescer', 'crescimento',
    'positivo', 'legal', 'ótimo', 'bom', 'excelente', 'fantástico', 'incrível', 'feliz',
    'contribuir', 'vontade', 'propósito', 'realização', 'estabilidade', 'sucesso', 'amigos',
    'conexão', 'entusiasmado', 'animado', 'dedicação', 'desenvolver', 'enriquecer', 'melhorar',
    'paixão', 'inspirador', 'esperança', 'superar', 'orgulho', 'uniao', 'ajuda', 'apoio',
    'acolhedor', 'inclusivo', 'prazer', 'fácil', 'divertido', 'engajamento'
]

NEGATIVE_WORDS = [
    'dúvidas', 'dificuldades', 'barreiras', 'desafios', 'perdido', 'perdida', 'não', 'incerteza',
    'escasso', 'escassos', 'problemas', 'medo', 'desistir', 'cansativo', 'difícil', 'ruim',
    'fracasso', 'desmotivar', 'tristeza', 'preocupação', 'resistência', 'escassez', 'subempregos',
    'complicado', 'chato', 'isolamento', 'sozinho', 'sozinha', 'tímido', 'tímida', 'procrastinação'
]

# Nomes dos grupos de trabalho
GROUP_NAMES = [
    'G1 - Automações Wix', 'G2 - API de Orquestração', 'G3 - Integração WhatsApp', 'G4 - SUPABASE (Banco de Dados)'
]

# Critérios para liderança
LEADERSHIP_TYPES = {
    'DIRETA': "Sim, me sinto a vontade estando a frente e guiando o grupo",
    'SUPORTE': "Sim, me sinto a vontade ajudando quem estiver a frente do grupo",
    'EXECUCAO': "Não, quero apenas executar as atividades"
}

# Nomes das colunas de sentimento para análise de liderança
LEADERSHIP_SENTIMENT_COLS = [
    'objetivo_proposito_sentiment', 'bagagem_contribuicao_sentiment', 'compromisso_pessoal_sentiment'
]

# Mapeamento Tópico-Grupo para calcular aptidão de liderança
TOPIC_TO_GROUP_APTITUDE_MAP = {
    1: {'G1 - Automações Wix': 0.4, 'G2 - API de Orquestração': 0.4, 'G3 - Integração WhatsApp': 0.4, 'G4 - SUPABASE (Banco de Dados)': 0.4},
    2: {'G1 - Automações Wix': 0.6, 'G2 - API de Orquestração': 0.6, 'G3 - Integração WhatsApp': 0.6, 'G4 - SUPABASE (Banco de Dados)': 0.6},
    3: {'G1 - Automações Wix': 0.7, 'G2 - API de Orquestração': 0.7, 'G3 - Integração WhatsApp': 0.7, 'G4 - SUPABASE (Banco de Dados)': 0.7},
    4: {'G1 - Automações Wix': 0.3, 'G2 - API de Orquestração': 0.8, 'G3 - Integração WhatsApp': 0.3, 'G4 - SUPABASE (Banco de Dados)': 0.9},
    5: {'G1 - Automações Wix': 0.9, 'G2 - API de Orquestração': 0.4, 'G3 - Integração WhatsApp': 0.9, 'G4 - SUPABASE (Banco de Dados)': 0.4},
}

# Nome da nova coluna de sentimento geral
OVERALL_SENTIMENT_COL = 'overall_sentiment'