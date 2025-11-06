# Projeto TransDevs TechExperience: Conectando Inovação e Inclusão

## Visão Geral

Este projeto é uma iniciativa do TransDevs TechExperience, focado em utilizar Data Analytics e Machine Learning para aprimorar a experiência de suas pessoas participantes. Nosso objetivo é transformar dados de check-in em *insights* acionáveis, promovendo a inclusão e o desenvolvimento na comunidade tech. O dashboard interativo, construído com Streamlit, captura o lado humano e o "feeling" das pessoas inscritas, auxiliando na construção de perfis e no match para futuras conexões e oportunidades.

Trabalhamos com foco em soluções reprodutíveis, explicáveis e alinhadas aos princípios de Diversidade, Equidade e Inclusão (DE&I).

## Objetivos do Dashboard

O dashboard visa responder a perguntas estratégicas e fornecer uma compreensão aprofundada da comunidade:

*   **Perfil de Interesse:** Quais são os interesses predominantes (grupos de trabalho, aspirações profissionais) das pessoas inscritas?
*   **Engajamento e Liderança:** Qual o nível de interesse em papéis de liderança ou suporte, e como podemos identificar e alocar esses talentos nos grupos?
*   **Tópicos e Temas:** Quais são os principais tópicos e temas de interesse da comunidade (habilidades, objetivos, expectativas de projeto), extraídos de campos de texto livre?
*   **Sentimento da Comunidade:** Como o sentimento geral e específico (por tema) da comunidade se manifesta em relação ao projeto, seus objetivos e desafios, incluindo a percepção de desafios e otimismo?
*   **Barreiras e Oportunidades:** Identificar as principais expectativas e possíveis desafios enfrentados pelas pessoas em suas jornadas e como o projeto pode contribuir.

## 📁 Estrutura do Projeto

A aplicação segue uma arquitetura modular para facilitar a manutenção e escalabilidade:

```text
transdevs_techexperience/
├── .streamlit/                # Configurações do Streamlit (inclui secrets.toml)
│   └── secrets.toml           # **SENHAS E CHAVES SECRETAS (NÃO VAI PARA GIT!)**
├── assets/                    # Ativos estáticos como imagens e ícones
│   └── images/                # Imagens do projeto (ex: logo da DiversificaDev)
│       └── diversificadev_logo.png
├── data/                      # Armazena os dados
│   ├── raw/                   # Dados brutos originais (CSV do formulário)
│   │   └── Checkin TransDevs TechExperience (respostas) - Respostas ao formulário 1.csv
│   └── processed/             # Dados limpos, transformados e insights gerados (CSVs processados)
├── models/                    # Modelos de Machine Learning treinados (LDA, TF-IDF Vectorizer)
├── notebooks/                 # Jupyter Notebooks para exploração e prototipagem
│   └── 01_Exploratory_Leadership_Analysis.ipynb
├── src/                       # Código fonte da aplicação
│   ├── analysis/              # Módulos para EDA, NLP e análises específicas
│   │   ├── __init__.py        # Indica que 'analysis' é um pacote Python
│   │   ├── eda.py             # Funções de Análise Exploratória de Dados
│   │   ├── leadership_analysis.py # Lógica de identificação de líderes
│   │   └── nlp_processing.py  # Funções de Processamento de Linguagem Natural
│   ├── app/                   # Módulos da aplicação Streamlit
│   │   ├── __init__.py        # Indica que 'app' é um pacote Python
│   │   ├── main.py            # Script principal do Dashboard Streamlit
│   │   └── utils.py           # Funções utilitárias e estilos do Dashboard
│   ├── config.py              # Variáveis de configuração, caminhos, constantes, léxicos
│   ├── data_ingestion.py      # Lógica de carregamento de dados brutos
│   └── data_processing.py     # Lógica de limpeza e padronização de dados
├── .env                       # Variáveis de ambiente (opcional, mas boa prática)
├── .gitignore                 # Arquivos e pastas a serem ignorados pelo Git
├── nltk_download_script.py    # Script para baixar recursos do NLTK e spaCy
├── README.md                  # Este arquivo de documentação
├── requirements.txt           # Lista de dependências Python
├── run_eda.py                 # Script para executar o pipeline de EDA e gerar insights
└── run_pipeline.py            # Script para executar o pipeline ETL inicial
```

## Tecnologias Utilizadas

*   **Python:** Linguagem de programação principal (versão 3.8+).
*   **Pandas:** Biblioteca para manipulação e análise de dados tabulares.
*   **NLTK & spaCy:** Bibliotecas essenciais para Processamento de Linguagem Natural (NLP), incluindo tokenização, lematização e remoção de stopwords. O spaCy é preferido para lematização em português devido à sua precisão.
*   **Scikit-learn:** Biblioteca de Machine Learning utilizada para Modelagem de Tópicos (LDA) e Vetorização de texto (TF-IDF).
*   **WordCloud:** Biblioteca para geração de nuvens de palavras impactantes.
*   **Streamlit:** Framework de código aberto para criação rápida de dashboards e aplicações web interativas em Python.
*   **Plotly Express:** Biblioteca para geração de gráficos interativos e esteticamente alinhados à identidade visual.
*   **Google Sheets:** Atua como a fonte de dados primária do projeto (lido via arquivo CSV exportado).

## Como Configurar e Executar o Projeto

### Pré-requisitos

*   Python 3.8+ instalado.
*   `pip` (gerenciador de pacotes Python).
*   `git` (para clonar o repositório, se aplicável).

### 1. Clonar o Repositório (se aplicável)

```bash
git clone <URL_DO_SEU_REPOSITORIO_GIT>
cd transdevs_techexperience
```

### 2. Criar e Ativar o Ambiente Virtual (macOS/Linux)

É altamente recomendado usar um ambiente virtual para gerenciar as dependências do projeto, garantindo isolamento e reprodutibilidade.

```bash
python3 -m venv .venv
source ./.venv/bin/activate
```
*   Após ativar, o prompt do seu terminal deve exibir `(.venv)` no início.

### 3. Instalar Dependências Python

Com o ambiente virtual **ativado**, instale todas as bibliotecas listadas no `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Baixar Recursos de NLP (NLTK e spaCy)

O NLTK e o spaCy precisam de dados adicionais para funcionar corretamente, como dicionários de stopwords e modelos de linguagem.

```bash
# Para NLTK:
python nltk_download_script.py

# Para spaCy (CRÍTICO: Baixa o modelo de português):
python -m spacy download pt_core_news_sm
```

### 5. Configurar Dados e Ativos Visuais

*   **Arquivo de Dados Brutos:** Coloque o arquivo CSV de respostas do formulário (`Checkin TransDevs TechExperience (respostas) - Respostas ao formulário 1.csv`) na pasta `data/raw/`. Este arquivo **não deve ser commitado no Git**.
*   **Logo da DiversificaDev:** Coloque o arquivo da logo (ex: `diversificadev_logo.png`) na pasta `assets/images/`.

### 6. Configurar Credenciais do Dashboard (Segurança)

Para proteger o acesso ao seu dashboard Streamlit, utilize o arquivo `secrets.toml`:

*   Crie uma pasta `.streamlit/` na raiz do seu projeto, se ainda não existir.
*   Crie (ou edite) o arquivo `secrets.toml` dentro de `.streamlit/`.

**Conteúdo de `transdevs_techexperience/.streamlit/secrets.toml`:**
```toml
# .streamlit/secrets.toml
# Este arquivo contém segredos e credenciais sensíveis.
# DEVE ser adicionado ao .gitignore e NUNCA commitado em repositórios públicos!

[user_credentials]
username = "transdevs" # Seu nome de usuário para login no dashboard
password = "sua_senha_secreta" # **MUDE ESTA SENHA PARA ALGO SEGURO E ÚNICO!**
```
**IMPORTANTE:** Certifique-se de que o arquivo `.gitignore` (descrito abaixo) inclui `/.streamlit/secrets.toml` para que suas credenciais não sejam publicadas.

### 7. Executar os Pipelines de Processamento de Dados (ETL e EDA)

Estes scripts irão processar os dados brutos, realizar as análises de NLP e gerar todos os *insights* necessários em arquivos CSV na pasta `data/processed/`.

```bash
# 1. Executar o pipeline de ETL inicial (carregamento e tratamento de PII)
python run_pipeline.py

# 2. Executar o pipeline de Análise Exploratória de Dados (NLP, Tópicos, Sentimento, Liderança)
# Este comando gerará os arquivos CSV finais na pasta 'data/processed/'
python run_eda.py
```
*   **Nota:** Se você alterar `src/config.py`, `src/analysis/nlp_processing.py`, `src/analysis/eda.py` ou `src/analysis/leadership_analysis.py`, você precisará re-executar `python run_eda.py` para gerar os arquivos `data/processed/` atualizados antes de ver as mudanças no dashboard.

### 8. Executar o Dashboard Streamlit

Com os dados processados, inicie a aplicação Streamlit:

```bash
streamlit run src/app/main.py
```
O dashboard será aberto no seu navegador padrão (geralmente `http://localhost:8501`). Uma tela de login solicitará o `username` e `password` configurados no seu `secrets.toml`.

## Privacidade e Segurança de Dados

Aderimos aos princípios de privacidade e segurança de dados, seguindo as melhores práticas (LGPD/GDPR):

*   **Anonimização/Pseudonimização:** Informações Pessoais Identificáveis (PII) sensíveis (como nome e telefone) são imediatamente pseudonimizadas ou removidas no início do pipeline. O mapeamento (`anonymized_pii_mapping.csv`) é para referência interna e **NUNCA deve ser exposto publicamente**.
*   **Controle de Acesso:** O dashboard Streamlit é protegido por um sistema de login com credenciais armazenadas de forma segura via `secrets.toml` (localmente) ou `st.secrets` (no Streamlit Cloud).
*   **`.gitignore`:** Arquivos sensíveis, dados processados e modelos treinados são explicitamente ignorados pelo controle de versão para evitar exposição acidental.

## `.gitignore`

O arquivo `.gitignore` garante que arquivos temporários, de ambiente e sensíveis não sejam incluídos no controle de versão.

```
# Python
__pycache__/
*.pyc
*.o
*.so
*.egg
*.egg-info/
.pytest_cache/
.tox/
.venv/
env/
venv/
pip-log.txt
pip-delete-this-directory.txt

# IDEs
.idea/
.vscode/

# Logs
*.log

# Dados gerados e processados (CRÍTICO)
data/processed/

# Dados brutos (CRÍTICO)
data/raw/Checkin TransDevs TechExperience (respostas) - Respostas ao formulário 1.csv

# Modelos de Machine Learning (CRÍTICO)
models/

# Streamlit secrets (CRÍTICO)
.streamlit/secrets.toml

# Notebook checkpoints
.ipynb_checkpoints/

# Variáveis de ambiente
.env
```

## Exportar Bibliotecas para `requirements.txt`

Sempre que novas bibliotecas forem instaladas com `pip install`, atualize o `requirements.txt` para manter a reprodutibilidade do ambiente:

```bash
pip freeze > requirements.txt
```
*   Execute este comando no terminal, com o ambiente virtual **ativado**, na raiz do projeto.

## Próximos Passos e Melhorias Potenciais

*   **Refinamento Contínuo:** Ajuste dos léxicos de sentimento, `TYPO_CORRECTION_MAP` e `TOPIC_TO_GROUP_APTITUDE_MAP` com base em mais dados ou feedback.
*   **Matchmaking Avançado:** Implementar funcionalidades para sugerir conexões entre participantes ou oportunidades com base em perfis e lacunas.
*   **Feedback Loop:** Adicionar mecanismos para coletar feedback direto dos usuários do dashboard.
*   **Monitoramento de Vieses:** Integrar ferramentas de explicabilidade (XAI) para garantir que os modelos de ML sejam justos e transparentes.

## Colaboradores

*   **Desenvolvimento:** Felipe Freire
