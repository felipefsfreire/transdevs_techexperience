# transdevs_techexperience/src/app/utils.py

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from src.config import EDA_FINAL_PATH, LEADERSHIP_ANALYSIS_PATH, ANONYMIZED_PII_PATH

# --- IDENTIDADE VISUAL DIVERSIFICADEV ---
# Paleta de Cores (ajustadas para alto contraste em fundo escuro)
COLORS = {
    "Solid Black": "#2D2926",      # Fundo principal escuro
    "Pure White": "#FEFEFE",       # Texto claro principal
    "Inclusive Pink": "#FF6CC9",   # Títulos, destaques, borda de sucesso
    "Diverse Purple": "#301982",   # Subtítulos, elementos interativos
    "Dark Purple": "#370051",      # Contraste escuro, fundos de tabela
    "Light Lavender": "#D6A0FF",   # Aviso (warning)
    "Gentle Pink": "#FFA8E1",      # Uso secundário, hover
    "Identity Blue": "#0C0091",    # Informação (info)
    # Para o degradê (se usarmos em fundos ou elementos gráficos específicos)
    "Degrade Pink": "#F462C2",
    "Degrade Purple Light": "#C075CB",
    "Degrade Purple Dark": "#212429", # Cor de fundo padronizada para alertas
}

# Fonte principal (Mona Sans, assumindo que será carregada via CSS ou link do Google Fonts)
FONT_PRINCIPAL = "Mona Sans, sans-serif"

# Caminhos para os dados (importados do config.py)
processed_data_path = EDA_FINAL_PATH
leadership_data_path = LEADERSHIP_ANALYSIS_PATH
pii_mapping_path = ANONYMIZED_PII_PATH

@st.cache_data(show_spinner=False) # Adiciona cache para evitar re-executar tudo se o estado do app mudar
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega os DataFrames necessários para o dashboard.
    Esta função agora espera que os dados já estejam gerados ANTES da execução do app.
    """
    try:
        df_eda = pd.read_csv(EDA_FINAL_PATH)
        df_leadership = pd.read_csv(LEADERSHIP_ANALYSIS_PATH)
        df_pii = pd.read_csv(ANONYMIZED_PII_PATH)
        return df_eda, df_leadership, df_pii
    except FileNotFoundError as e:
        st.error(f"Erro ao carregar dados: {e}. Certifique-se de que os scripts de ETL e EDA foram executados no ambiente de deploy (ou localmente).", icon="❌")
        st.stop() # Interrompe o app se os dados essenciais não forem encontrados
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao carregar os dados: {e}", icon="❗")
        st.stop()
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def get_logo_path(logo_filename: str = "diversificadev_logo.png") -> str:
    """
    Retorna o caminho completo para a logo da DiversificaDev.
    Assume que a logo está em `assets/images/`.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    logo_path = os.path.join(project_root, 'assets', 'images', logo_filename)
    
    if not os.path.exists(logo_path):
        st.error(f"Erro: Logo não encontrada em '{logo_path}'. Verifique o caminho e a existência do arquivo.", icon="🖼️")
        return None
    return logo_path

def set_page_config():
    """Configurações iniciais da página Streamlit, incluindo o ícone."""
    logo_path = get_logo_path()
    page_icon = logo_path if logo_path else "💡"

    st.set_page_config(
        page_title="TransDevs TechExperience - Insights",
        page_icon=page_icon,
        layout="wide"
    )

def apply_custom_css():
    """Aplica estilos CSS customizados para a identidade visual."""
    custom_css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Mona+Sans:wght@400;600;700&display=swap');

        /* Cores de fundo e texto padrão para Streamlit Dark Mode */
        body {{
            background-color: {COLORS["Solid Black"]};
            color: {COLORS["Pure White"]};
        }}
        .stApp {{
            background-color: {COLORS["Solid Black"]};
            color: {COLORS["Pure White"]};
        }}

        /* Títulos */
        h1 {{
            font-family: "{FONT_PRINCIPAL}";
            color: {COLORS["Inclusive Pink"]};
        }}
        h2 {{
            font-family: "{FONT_PRINCIPAL}";
            color: {COLORS["Diverse Purple"]};
        }}
        h3, h4, h5, h6 {{
            font-family: "{FONT_PRINCIPAL}";
            color: {COLORS["Light Lavender"]};
        }}
        p, .stMarkdown, .stText {{
            font-family: "{FONT_PRINCIPAL}";
            color: {COLORS["Pure White"]};
            font-size: 1.1em;
        }}

        /* Botões */
        .stButton>button {{
            background-color: {COLORS["Diverse Purple"]};
            color: {COLORS["Pure White"]};
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
        }}
        .stButton>button:hover {{
            background-color: {COLORS["Inclusive Pink"]};
            color: {COLORS["Solid Black"]};
        }}
        
        /* Abas (Tabs) */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
        }}
        .stTabs [data-baseweb="tab-list"] button {{
            background-color: {COLORS["Solid Black"]};
            border-radius: 4px;
            border: 1px solid {COLORS["Diverse Purple"]};
        }}
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
            font-size: 1.2rem;
            color: {COLORS["Pure White"]};
            font-weight: 600;
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            background-color: {COLORS["Inclusive Pink"]};
            border: 1px solid {COLORS["Inclusive Pink"]};
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {{
            color: {COLORS["Solid Black"]};
        }}

        div[data-testid="stAlert"] {{
            background-color: {COLORS["Degrade Purple Dark"]} !important;
            color: {COLORS["Pure White"]} !important;
            border-left: 8px solid;
            border-radius: 8px;
            padding: 12px;
        }}
        
        div[data-testid="stAlert"] div[data-testid="stMarkdownContainer"] p {{
            color: {COLORS["Pure White"]} !important;
            font-size: 1em !important;
            margin-bottom: 0px;
        }}

        div[data-testid="stAlert"].info {{ border-color: {COLORS["Identity Blue"]} !important; }}
        div[data-testid="stAlert"].warning {{ border-color: {COLORS["Light Lavender"]} !important; }}
        div[data-testid="stAlert"].success {{ border-color: {COLORS["Inclusive Pink"]} !important; }}
        div[data-testid="stAlert"].error {{ border-color: {COLORS["Dark Purple"]} !important; }}
        

        .stDataFrame {{
            color: {COLORS["Pure White"]};
            font-family: "{FONT_PRINCIPAL}";
        }}
        .stDataFrame thead th {{
            color: {COLORS["Inclusive Pink"]} !important;
            background-color: {COLORS["Dark Purple"]} !important;
        }}
        .stDataFrame tbody tr:nth-child(even) {{
            background-color: #3b3a3d;
        }}
        .stDataFrame tbody tr:nth-child(odd) {{
            background-color: {COLORS["Solid Black"]};
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def plot_bar_chart(df: pd.DataFrame, column: str, title: str, x_axis_title: str, y_axis_title: str):
    """
    Gera um gráfico de barras com cores e fonte da identidade visual, adaptado para fundo escuro.
    """
    data = df[column].value_counts().reset_index()
    data.columns = [column, 'count']
    
    fig = px.bar(data, 
                 x=column, 
                 y='count', 
                 title=title,
                 color_discrete_sequence=[COLORS["Diverse Purple"], COLORS["Inclusive Pink"], COLORS["Light Lavender"], COLORS["Dark Purple"], COLORS["Identity Blue"], COLORS["Gentle Pink"]],
                 text_auto=True)
    
    fig.update_layout(
        title_font_family=FONT_PRINCIPAL,
        title_font_color=COLORS["Inclusive Pink"],
        title_font_size=24,
        font_family=FONT_PRINCIPAL,
        font_color=COLORS["Pure White"],
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        plot_bgcolor=COLORS["Solid Black"],
        paper_bgcolor=COLORS["Solid Black"],
        xaxis=dict(showgrid=False, tickfont=dict(color=COLORS["Pure White"])),
        yaxis=dict(showgrid=True, gridcolor='gray', tickfont=dict(color=COLORS["Pure White"])),
        hoverlabel=dict(bgcolor="white", font_size=16, font_family=FONT_PRINCIPAL, font_color="black")
    )
    fig.update_traces(textfont_color=COLORS["Pure White"])
    st.plotly_chart(fig, use_container_width=True)

def plot_pie_chart(df: pd.DataFrame, column: str, title: str):
    """
    Gera um gráfico de pizza com cores e fonte da identidade visual, adaptado para fundo escuro.
    """
    data = df[column].value_counts().reset_index()
    data.columns = [column, 'count']

    fig = px.pie(data, 
                 values='count', 
                 names=column, 
                 title=title,
                 color_discrete_sequence=[COLORS["Diverse Purple"], COLORS["Inclusive Pink"], COLORS["Light Lavender"], COLORS["Dark Purple"], COLORS["Identity Blue"], COLORS["Gentle Pink"]],
                 hole=0.3)

    fig.update_layout(
        title_font_family=FONT_PRINCIPAL,
        title_font_color=COLORS["Inclusive Pink"],
        font_family=FONT_PRINCIPAL,
        font_color=COLORS["Pure White"],
        plot_bgcolor=COLORS["Solid Black"],
        paper_bgcolor=COLORS["Solid Black"],
        legend_font_color=COLORS["Pure White"],
        hoverlabel=dict(bgcolor="white", font_size=16, font_family=FONT_PRINCIPAL, font_color="black")
    )
    fig.update_traces(textinfo='percent+label', textfont_color=COLORS["Pure White"], pull=[0.05 if i == data['count'].idxmax() else 0 for i in range(len(data))])
    st.plotly_chart(fig, use_container_width=True)