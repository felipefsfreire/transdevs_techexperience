# transdevs_techexperience/src/app/main.py

import sys
import os

# Adiciona o diretório raiz do projeto ao sys.path para que 'src' seja reconhecido como um pacote
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from src.app.utils import load_dashboard_data, set_page_config, apply_custom_css, get_logo_path, COLORS, FONT_PRINCIPAL, plot_bar_chart, plot_pie_chart
from src.config import GROUP_NAMES, LEADERSHIP_TYPES, TEXT_COLUMNS_FOR_NLP, OVERALL_SENTIMENT_COL, TOPIC_TO_GROUP_APTITUDE_MAP
from src.analysis.nlp_processing import get_ngram_text_for_wordcloud


# --- Configurações Iniciais da Página ---
set_page_config()
apply_custom_css()

# --- Funções de Autenticação ---
def check_password():
    """Returns `True` if the user authenticates, else `False`."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] == st.secrets["user_credentials"]["username"]
            and st.session_state["password"] == st.secrets["user_credentials"]["password"]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.markdown(f'<h2 style="color:{COLORS["Inclusive Pink"]}; font-family:{FONT_PRINCIPAL};">Acesso Restrito ao Dashboard</h2>', unsafe_allow_html=True)
        st.text_input("Usuário", on_change=password_entered, key="username")
        st.text_input(
            "Senha", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input again with error.
        st.text_input("Usuário", on_change=password_entered, key="username")
        st.text_input(
            "Senha", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Usuário ou senha incorretos")
        return False
    else:
        # Password correct.
        return True

# --- Lógica Principal do Dashboard (dentro do if check_password()) ---
if check_password():
    # --- Carregar Dados (só carrega se o login for bem-sucedido) ---
    df_eda, df_leadership, df_pii_mapping = load_dashboard_data()

    # --- Header do Dashboard ---
    st.image(get_logo_path(), width=150)
    st.markdown(f'<h1 style="color:{COLORS["Inclusive Pink"]}; font-family:{FONT_PRINCIPAL};">TransDevs TechExperience: Conectando Inovação e Inclusão</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:1.2em; font-family:{FONT_PRINCIPAL};">Análise detalhada do perfil, aspirações e potencial de liderança da nossa comunidade.</p>', unsafe_allow_html=True)

    st.divider()

    # --- Abas do Dashboard ---
    tab_about, tab_overview, tab_leadership, tab_profiles, tab_sentiment = st.tabs([
        "Sobre o Projeto e Dashboard",
        "Visão Geral e Demografia",
        "Potencial de Liderança",
        "Perfis e Tópicos",
        "Sentimento da Comunidade"
    ])

    with tab_about:
        st.markdown(f'<h2 style="color:{COLORS["Inclusive Pink"]}; font-family:{FONT_PRINCIPAL};">Apresentamos o Dashboard TransDevs TechExperience!</h2>', unsafe_allow_html=True)
        st.markdown(f'<p>Com o objetivo de conectar inovação e inclusão, este dashboard transforma os dados de check-in em insights cruciais para aprimorar a experiência de nosses participantes do projeto TransDevs TechExperience.</p>', unsafe_allow_html=True)
        st.markdown(f'<p>Nosses objetivos com este projeto são:</p>', unsafe_allow_html=True)
        st.markdown(f'<ul>'
                    f'<li>Construir perfis detalhados das pessoas inscritas.</li>'
                    f'<li>Identificar as principais aspirações e barreiras de entrada percebidas.</li>'
                    f'<li>Otimizar o "match" entre as habilidades e interesses dos participantes e as oportunidades do projeto.</li>'
                    f'<li>Apoiar a liderança da DiversificaDev e TransEmpregos na tomada de decisões estratégicas.</li>'
                    f'<li>Garantir que todas as análises capturem o lado humano e o "feeling" das pessoas, sempre alinhadas aos princípios de Diversidade, Equidade e Inclusão (DE&I).</li>'
                    f'</ul>', unsafe_allow_html=True)

        st.markdown(f'<h3 style="color:{COLORS["Inclusive Pink"]}; font-family:{FONT_PRINCIPAL};">Perguntas Chave que Este Dashboard Responde:</h3>', unsafe_allow_html=True)
        st.markdown(f'<ul>'
                    f'<li>Qual o perfil de interesse (grupos de trabalho, aspirações) das pessoas inscritas?</li>'
                    f'<li>Qual o nível de interesse em liderança e como podemos alocar esses talentos?</li>'
                    f'<li>Quais são os principais tópicos e temas de interesse da comunidade (conhecimentos, objetivos, expectativas)?</li>'
                    f'<li>Como o sentimento geral e específico (por tema) da comunidade se manifesta?</li>'
                    f'<li>Existem potenciais líderes para os grupos que ainda precisam de gestão?</li>'
                    f'</ul>', unsafe_allow_html=True)
        st.info("Este dashboard é uma ferramenta viva e será continuamente aprimorada para melhor servir à comunidade TransDevs.", icon="💡")


    with tab_overview:
        st.markdown(f'<h2 style="color:{COLORS["Inclusive Pink"]}; font-family:{FONT_PRINCIPAL};">Panorama Geral dos Participantes</h2>', unsafe_allow_html=True)
        st.write(f"Total de participantes ativos na análise: **{len(df_eda)}**")

        df_conscience_summary_temp = df_eda['consciencia_escopo_padronizada'].value_counts().reset_index()
        df_conscience_summary_temp.columns = ['Status', 'Count']

        st.markdown(f'<h3>Nível de Consciência sobre o Escopo do Projeto</h3>', unsafe_allow_html=True)
        fig_conscience = px.pie(df_conscience_summary_temp,
                                values='Count',
                                names='Status',
                                title='Consciência sobre o Escopo do Projeto',
                                color_discrete_sequence=[COLORS["Diverse Purple"], COLORS["Inclusive Pink"], COLORS["Light Lavender"], COLORS["Dark Purple"]],
                                hole=0.3)
        fig_conscience.update_layout(
            title_font_family=FONT_PRINCIPAL,
            title_font_color=COLORS["Inclusive Pink"],
            font_family=FONT_PRINCIPAL,
            font_color=COLORS["Pure White"],
            plot_bgcolor=COLORS["Solid Black"],
            paper_bgcolor=COLORS["Solid Black"],
            legend_font_color=COLORS["Pure White"]
        )
        fig_conscience.update_traces(textinfo='percent+label', textfont_color=COLORS["Pure White"])
        st.plotly_chart(fig_conscience, use_container_width=True)

        st.markdown(f'<h3>Preferência por Grupo Principal</h3>', unsafe_allow_html=True)
        plot_bar_chart(df_eda, 'grupo_principal', 'Distribuição de Preferência por Grupo Principal', 'Grupo de Trabalho', 'Número de Pessoas')

        st.markdown(f'<h3>Interesse em Liderança Declarado</h3>', unsafe_allow_html=True)
        plot_pie_chart(df_eda, 'interesse_lideranca', 'Interesse em Exercer Liderança')

    with tab_leadership:
        st.markdown(f'<h2 style="color:{COLORS["Inclusive Pink"]}; font-family:{FONT_PRINCIPAL};">Identificação de Lideranças para os Grupos</h2>', unsafe_allow_html=True)
        st.markdown(f'<p>Esta seção apresenta sugestões de liderança baseadas em interesse declarado, preferências de grupo e análises de texto (bagagem, tópicos, sentimento).</p>', unsafe_allow_html=True)

        # Caixa de texto explicativa para a análise de liderança
        st.markdown(f'<p style="font-size:1.1em; color:{COLORS["Pure White"]};">A análise de liderança é dividida em duas frentes:</p>', unsafe_allow_html=True)
        st.markdown(f'<ul style="color:{COLORS["Pure White"]};">'
                    f'<li><b>Líderes Diretos Atribuídos:</b> São as pessoas que declararam um interesse explícito em "guiar o grupo" e foram atribuídas a um grupo com base em sua preferência principal ou alternativa, se o grupo estivesse disponível.</li>'
                    f'<li><b>Potenciais Líderes de Suporte:</b> São pessoas que se mostraram dispostas a "ajudar na liderança". Para estas, aplicamos um algoritmo de pontuação que considera:<br/>'
                    f'  <ul style="margin-top: 5px; margin-left: 20px;">'
                    f'    <li>A afinidade entre seus grupos preferidos e os grupos que ainda precisam de líderes.</li>'
                    f'    <li>O alinhamento do seu <b>tópico de interesse principal (LDA)</b> com as necessidades dos grupos.</li>'
                    f'    <li>Um <b>score de sentimento</b> (otimismo no propósito, bagagem e compromisso) que indica proatividade.</li>'
                    f'  </ul>'
                    f'  As sugestões são ordenadas pelo seu "score de aptidão geral".'
                    f'</li>'
                    f'</ul>', unsafe_allow_html=True)

        st.divider()

        direct_leaders_assigned = df_leadership[df_leadership['status_lideranca_final'].fillna('').str.contains(r'Líder Direto Atribuído')].copy()

        st.markdown(f'<h3>Líderes Diretos Atribuídos</h3>', unsafe_allow_html=True)
        if not direct_leaders_assigned.empty:
            df_display_leaders = direct_leaders_assigned.merge(df_pii_mapping[['participant_id', 'nome_completo']], on='participant_id', how='left')
            st.dataframe(df_display_leaders[['nome_completo', 'grupo_principal_preferido', 'grupo_alternativo_preferido', 'sugestao_lideranca_grupo', 'tipo_sugestao']], use_container_width=True)
        else:
            st.info("Nenhum líder direto atribuído ainda.")

        st.markdown(f'<h3>Grupos Atualmente sem Líder Direto</h3>', unsafe_allow_html=True)
        groups_with_leaders = direct_leaders_assigned['sugestao_lideranca_grupo'].unique().tolist()
        groups_with_leaders = [g for g in groups_with_leaders if g in GROUP_NAMES]
        groups_needing_leaders = [group for group in GROUP_NAMES if group not in groups_with_leaders]

        if groups_needing_leaders:
            st.warning(f"Os seguintes grupos ainda precisam de liderança direta: **{', '.join(groups_needing_leaders)}**")

            st.markdown(f'<h3>Potenciais Líderes de Suporte para Preencher Lacunas</h3>', unsafe_allow_html=True)
            potential_support_leaders_suggested = df_leadership[
                (df_leadership['status_lideranca_final'].fillna('') == 'Potencial Líder para Suporte') &
                (df_leadership['sugestao_lideranca_grupo'].fillna('').astype(str).apply(lambda x: any(g in x for g in groups_needing_leaders)))
            ].copy().sort_values(by='aptidao_score_geral', ascending=False)

            if not potential_support_leaders_suggested.empty:
                df_display_potential = potential_support_leaders_suggested.merge(df_pii_mapping[['participant_id', 'nome_completo']], on='participant_id', how='left')
                st.dataframe(df_display_potential[['nome_completo', 'sugestao_lideranca_grupo', 'aptidao_score_geral', 'justificativa_bagagem', 'justificativa_topico_lda', 'justificativa_sentimento']], use_container_width=True)
            else:
                st.info("Nenhum participante com interesse em suporte identificado como potencial líder para os grupos carentes, mesmo com lógica avançada.")

        else:
            st.success("Todos os grupos já possuem um líder direto atribuído ou sugerido!")


    with tab_profiles:
        st.markdown(f'<h2 style="color:{COLORS["Inclusive Pink"]}; font-family:{FONT_PRINCIPAL};">Perfis de Interesse e Tópicos Emergentes</h2>', unsafe_allow_html=True)
        st.markdown(f'<p>Esta seção explora os principais temas e interesses que unem os participantes do TechExperience.</p>', unsafe_allow_html=True)

        st.markdown(f'<p style="font-size:1.1em; color:{COLORS["Pure White"]};">A <b>Modelagem de Tópicos (LDA)</b> é uma técnica de Machine Learning que analisa grandes volumes de texto para descobrir os "temas" ou "tópicos" ocultos. Ela agrupa palavras que frequentemente aparecem juntas, e nós interpretamos esses agrupamentos para dar nome aos tópicos. Isso nos ajuda a entender quais são os principais interesses e focos de conhecimento da comunidade, mesmo que não sejam explicitamente declarados.</p>', unsafe_allow_html=True)
        st.info("No contexto do TechExperience, os tópicos revelam as aspirações e o perfil técnico/comunitário dos participantes.", icon="🧐")

        st.markdown(f'<h3>Tópicos Identificados por LDA</h3>', unsafe_allow_html=True)
        st.markdown(f"""
        <p>
        <b>Tópico 1 (Busca por Conhecimento):</b> de com que em mais para aprender conhecimento me experiência<br/>
        <b>Tópico 2 (Recursos/Métodos de Aprendizado):</b> pois aperfeiçoar estabelecidos prazos vídeos livros processos cursos dev comunicação<br/>
        <b>Tópico 3 (Comunidade, Inclusão):</b> ideas trans legais colegas nao información personas otras necesito ganas<br/>
        <b>Tópico 4 (Análise/Gestão/Network):</b> anteriores foco reunião analise união algumas ampliação informações network horizontes<br/>
        <b>Tópico 5 (Projetos/Criação/Colaboração):</b> nada contra projetos empresas si contato socialização boa criar nesse
        </p>
        """, unsafe_allow_html=True)

        st.markdown(f'<h3>Distribuição dos Participantes pelos Tópicos Principais</h3>', unsafe_allow_html=True)
        if 'main_topic' in df_eda.columns and not df_eda['main_topic'].isnull().all():
            df_eda['main_topic_str'] = df_eda['main_topic'].astype(str)
            plot_bar_chart(df_eda, 'main_topic_str', 'Tópicos Principais dos Participantes', 'Tópico (ID)', 'Número de Participantes')
            st.info("Nota: A maioria dos participantes se alinha ao Tópico 1, focado em busca de conhecimento e experiência.")
        else:
            st.info("Dados de tópicos não disponíveis ou insuficientes para visualização.")

        st.markdown(f'<h3>Palavras/Conceitos Mais Frequentes</h3>', unsafe_allow_html=True)
        ngram_choice = st.radio(
            "Selecione o tipo de unidade para a nuvem de palavras:",
            ('Palavras Únicas (Unigrams)', 'Bigrams', 'Trigrams'),
            horizontal=True
        )

        all_lemmas_combined_for_wc = df_eda['all_lemmas_combined'].explode().tolist()

        wc_text = ""
        if ngram_choice == 'Palavras Únicas (Unigrams)':
            wc_text = get_ngram_text_for_wordcloud(all_lemmas_combined_for_wc, n=1)
        elif ngram_choice == 'Bigrams':
            wc_text = get_ngram_text_for_wordcloud(all_lemmas_combined_for_wc, n=2)
        elif ngram_choice == 'Trigrams':
            wc_text = get_ngram_text_for_wordcloud(all_lemmas_combined_for_wc, n=3)

        if wc_text:
            wordcloud = WordCloud(width=800, height=400, background_color=COLORS["Solid Black"], collocations=False, colormap='magma', max_words=100).generate(wc_text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Nenhum texto combinado para gerar a nuvem de palavras.")


    with tab_sentiment:
        st.markdown(f'<h2 style="color:{COLORS["Inclusive Pink"]}; font-family:{FONT_PRINCIPAL};">O Sentimento da Comunidade</h2>', unsafe_allow_html=True)
        st.markdown(f'<p>Uma análise do tom emocional nas respostas, revelando os <i>feelings</i> e percepções dos participantes sobre o projeto e seus objetivos.</p>', unsafe_allow_html=True)

        st.markdown(f'<h3>Sentimento Geral dos Participantes</h3>', unsafe_allow_html=True)
        if OVERALL_SENTIMENT_COL in df_eda.columns and not df_eda[OVERALL_SENTIMENT_COL].isnull().all():
            plot_pie_chart(df_eda, OVERALL_SENTIMENT_COL, 'Sentimento Geral')
        else:
            st.info("Dados de sentimento geral não disponíveis para visualização.")

        st.divider()

        st.markdown(f'<h3>Sentimento por Tema Específico</h3>', unsafe_allow_html=True)
        sentiment_cols_specific = [col for col in df_eda.columns if col.endswith('_sentiment') and col != OVERALL_SENTIMENT_COL]

        num_cols = 2
        cols = st.columns(num_cols)

        for i, col in enumerate(sentiment_cols_specific):
            with cols[i % num_cols]:
                st.markdown(f'<h4>Sentimento em "{col.replace("_sentiment", "").replace("_", " ").title()}"</h4>', unsafe_allow_html=True)
                if not df_eda[col].isnull().all():
                    plot_pie_chart(df_eda, col, f'Sentimento sobre {col.replace("_sentiment", "").replace("_", " ").title()}')
                else:
                    st.info(f"Dados de sentimento não disponíveis para {col.replace('_sentiment', '').replace('_', ' ').title()}.")

        st.markdown(f'<p><b>Insights sobre Sentimento:</b> Observa-se um forte sentimento positivo em relação aos objetivos e contribuições, enquanto o compromisso pessoal e as expectativas da experiência tendem a ser mais neutros, indicando um senso de desafio e seriedade.</p>', unsafe_allow_html=True)