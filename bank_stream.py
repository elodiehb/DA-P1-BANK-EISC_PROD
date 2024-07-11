#imports
import streamlit as st
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
st. set_page_config(layout="wide")
#création du df
df=pd.read_csv("bank.csv")
#sidebar
st.sidebar.title("Sommaire")
pages=["Projet", "Jeu de données", "DataVizualisation", "Pre-processing","Modélisation","Conclusion"]
page=st.sidebar.radio("Menu", pages)
st.sidebar.title("Auteurs")
st.sidebar.markdown('<a href="https://www.linkedin.com/in/elodie-barnay-henriet-916a6311a/" title="LinkedIn Elodie">Elodie Barnay henriet</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="https://www.linkedin.com/in/irinagrankina/" title="LinkedIn Irina">Irina Grankina</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="https://www.linkedin.com/in/samanthaebrard/" title="LinkedIn Samantha">Samantha Ebrard</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="https://www.linkedin.com/in/cedric-le-stunff-profile/" title="LinkedIn Cédric">Cédric Le Stunff</a>', unsafe_allow_html=True)
st.sidebar.title("Liens")
st.sidebar.markdown('<a href="https://drive.google.com/file/d/1oj_DeLvLWaQn907xsUK0laTjgFFn3bmc/view?usp=sharing" title="Rapport de projet">Rapport</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="https://colab.research.google.com/drive/1eY6XSFy2pTywJwGjXQacfjSf8dD_PNqB?usp=sharing" title="Notebook Google Colab">Notebook</a>', unsafe_allow_html=True)
#page0 PROJET
if page == pages[0] : 
  st.title("Prédiction du succès d'une campagne Marketing d'une banque")
  st.header("Contexte")
  # Création de deux colonnes
  col1, col2 = st.columns(2)
  with col1:
    st.markdown("""
Les données du dataset `bank.csv` sont liées à une campagne de marketing direct d'une institution bancaire portugaise menée entre Mai 2008 et Novembre 2010. \n
Les campagnes de marketing étaient basées sur des appels téléphoniques. Plusieurs contacts avec le même client ont eu lieu pour savoir si le dernier a, oui ou non, **souscrit au produit : dépôt bancaire à terme.**
        """)

    with col2:
        st.markdown("""
Le dépôt à terme est un type d'investissement proposé par les banques et les institutions financières. \n
Dans le cadre d'un dépôt à terme, un particulier dépose une certaine somme d'argent auprès de la banque pour une période déterminée, appelée durée ou échéance.
L'argent est conservé par la banque pendant la durée spécifiée, au cours de laquelle il est rémunéré à un taux d'intérêt fixe.
                    """)

  st.write("")
  st.divider()
  st.header("Objectif")

  # Création de deux colonnes
  col1, col2 = st.columns(2)
  with col1:
    st.markdown("""
Notre objectif est d'analyser l'ensemble des données afin d'**identifier les tendances et les facteurs qui influencent la souscription d'un produit dépôt à terme** par les clients. \n
Sur le plan technique, l'analyse de ce dataset représente une occasion unique de mettre en pratique nos compétences en matière d’analyse de données, de visualisation avec Plotly, de pre-processing, de modélisation d’algorithmes de classification jusqu'à la publication sur Streamlit et Github.
        """)


  with col2:
        st.markdown("""
D'un point de vue stratégique, ce projet vise à fournir des **insights précieux pour augmenter les taux de souscription au produit "dépôt à terme"**. \n
En identifiant les facteurs clés de succès des campagnes précédentes, nous pouvons aider l'institution financière à optimiser ses ressources marketing, à cibler plus efficacement ses clients et à améliorer ses taux de conversion, générant ainsi des revenus supplémentaires.
                    """)

  st.write("")
  st.divider()
  # Création de deux colonnes
  col1, col2 = st.columns(2)
  with col1:
        # Ajouter un bouton
        st.markdown(
            """
            <a href="https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data" target="_blank" style="
            display: inline-block; 
            background-color: #56CEB2; 
            color: white; 
            font-size: 16px; 
            font-weight: bold; 
            text-align: center; 
            padding: 10px 20px; 
            border-radius: 5px; 
            text-decoration: none; 
            width: 100%;
            ">Source du Dataset</a>
            """, unsafe_allow_html=True
        )
        
  with col2:
        # Chemin de l'image
        image_path_intro = "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC_PROD/main/img/intro.jpeg"
        #Afficher l'image
        st.image(image_path_intro)








#page1 JEU DE DONNEES
if page == pages[1] :
  st.title('Le jeu de données')
  st.markdown('Le jeu de données bank.csv est basé sur le jeu de données UCI Bank Marketing dont on peut lire la description ici : http://archive.ics.uci.edu/ml/datasets/Bank+Marketing). Créateurs: S. Moro, P. Rita, P. Cortez.')
  st.header('Les variables')
  df2=pd.read_excel("variables.xlsx")
  st.dataframe(df2)
  st.divider()
  st.header('Aperçu des données')
  # Insert containers separated into tabs:
  tab1, tab2, tab3, tab4, tab5,tab6= st.tabs(["Aperçu", "Dimensions","Statistiques","Types","Valeurs nulles", "Doublons"])
  with tab1:
    st.code('df.head(10)')
    st.markdown("Aperçu des 10 premières lignes")
    st.dataframe(df.head(10))
  with tab2:
    st.markdown("Les dimensions du jeu de données : 11 162 lignes et 17 colonnes")
    st.code('df.shape')
    st.write(df.shape)
  with tab3:
    st.code('df.describe()')
    st.write(df.describe())
    st.subheader("Constat")
    st.markdown("""
- age : 50% des valeurs sont entre 32 et 49 ans. Beaucoup de valeurs extrêmes : max 95.
- balance : 50% des valeurs sont entre 122 et 1708. Présence de valeurs négatives et de valeurs extrêmes : min -6 847, max 81 204.
- duration : 50% des valeurs sont entre 138 sec (2min) et 496 (8min). Présence de valeurs extrêmes : max 3 881.
- campaign : 50% des valeurs sont entre 1 et 3 contacts.Présence de valeurs extrêmes : max 63.
- pdays : 50% des valeurs sont entre - 1 et 20. La médiane est à -1 ce qui signifie que la moitié des clients n'ont jamais été contacté avant cette campagne. Présence de valeurs extrêmes : max 854.
- previous : 50% des valeurs sont entre 0 et 1. Présence de valeurs extrêmes : max 58.
""")
  with tab4:
    st.markdown("Les types de données : ")
    st.code('df.dtypes')
    st.write(df.dtypes)
  with tab5:
    st.markdown("Aucune valeur manquante : ")
    st.code('df.isna().sum()')
    st.write(df.isna().sum())
  with tab6:
    st.markdown("Aucun doublon : ")
    st.code('df[df.duplicated()]')
    st.write(df[df.duplicated()])
  st.divider()
  st.header('Quelques statistiques')
  # Insert containers separated into tabs:
  tab1, tab2, tab3, tab4= st.tabs(["Balance", "Duration","Previous","pdays-previous"])
  #TAB1 MOYENNE BALANCE
  with tab1:
    st.markdown("Moyenne du Solde de compte pour clients dépositeurs ou non et pour le total clients.")
    code = '''
# Calcul de la moyenne du solde
mean_balance_all = df['balance'].mean()
mean_balance_yes = df[df['deposit'] == 'yes']['balance'].mean()
mean_balance_no = df[df['deposit'] == 'no']['balance'].mean()
    '''
    st.code(code, language='python')
    # Calcul de la moyenne du solde
    mean_balance_all = df['balance'].mean()
    mean_balance_yes = df[df['deposit'] == 'yes']['balance'].mean()
    mean_balance_no = df[df['deposit'] == 'no']['balance'].mean()
    # Afficher les résultats
    st.write("La moyenne du solde bancaire pour le jeu de données est de :", round(mean_balance_all, 2))
    st.write("La moyenne du solde pour les clients ayant effectué un dépôt est de :", round(mean_balance_yes, 2))
    st.write("La moyenne du solde pour les clients n'ayant pas effectué de dépôt est de :", round(mean_balance_no, 2))
  #TAB2 MOYENNE DURATION
  with tab2:
    st.markdown("Médiane de la durée de contact en minutes pour clients dépositeurs ou non et pour le total clients.")
    # Afficher le code
    code = '''
# Conversion de la colonne duration de secondes à minutes
df['duration_minutes'] = round((df['duration'] / 60.0), 2)
# Calculer la durée médiane pour l'ensemble des données
median_duration_all = df['duration_minutes'].median()
# Calculer la durée médiane pour ceux ayant fait un dépôt (deposit = yes)
median_duration_deposit_yes = df[df['deposit'] == 'yes']['duration_minutes'].median()
median_duration_deposit_no = df[df['deposit'] == 'no']['duration_minutes'].median()
    '''
    st.code(code, language='python')
    # Exécuter le code
    # Conversion de la colonne duration de secondes à minutes
    df['duration_minutes'] = round((df['duration'] / 60.0), 2)
    # Calculer la durée médiane pour l'ensemble des données
    median_duration_all = df['duration_minutes'].median()
    # Calculer la durée médiane pour ceux ayant fait un dépôt (deposit = yes)
    median_duration_deposit_yes = df[df['deposit'] == 'yes']['duration_minutes'].median()
    median_duration_deposit_no = df[df['deposit'] == 'no']['duration_minutes'].median()
    # Afficher les résultats
    st.write("La durée médiane du contact client de la campagne est de :", round(median_duration_all, 2), "minutes.")
    st.write("Pour les clients ayant effectué un dépôt :", round(median_duration_deposit_yes, 2), "minutes.")
    st.write("Pour les clients n'ayant pas effectué de dépôt :", round(median_duration_deposit_no, 2), "minutes.")
  #TAB3 PREVIOUS
  with tab3:
    st.markdown("Moyenne du nombre de contacts clients avant cette campagne pour clients dépositeurs ou non et pour le total clients.")
    # Afficher le code
    code = '''
# Calculer les moyennes de previous pour chaque groupe de deposit
mean_previous_all = np.mean(df['previous'])
mean_previous_yes = np.mean(df[df['deposit'] == 'yes']['previous'])
mean_previous_no = np.mean(df[df['deposit'] == 'no']['previous'])
    '''
    st.code(code, language='python')
    # Exécuter le code
    # Calculer les moyennes de previous pour chaque groupe de deposit
    mean_previous_all = np.mean(df['previous'])
    mean_previous_yes = np.mean(df[df['deposit'] == 'yes']['previous'])
    mean_previous_no = np.mean(df[df['deposit'] == 'no']['previous'])
    # Afficher les résultats
    st.write("Le nombre moyen de contacts clients avant cette campagne est de :", round(mean_previous_all, 2), "contacts.")
    st.write("Pour les clients ayant effectué un dépôt :", round(mean_previous_yes, 2), " contacts en moyenne.")
    st.write("Pour les clients n'ayant pas effectué de dépôt :", round(mean_previous_no, 2), "contacts en moyenne.")
  with tab4:
    # Afficher le texte avec des bullet points
    st.markdown("""
    La valeur -1 de pdays équivaut-elle à la valeur 0 de previous ?
    - pdays : Nombre de jours depuis le dernier contact d'une campagne précédente (valeur -1 équivaut à pas de contact)
    - previous : Nombre de contacts avant cette campagne (valeur 0 équivaut à pas de contact)
    """)
    # Afficher le code
    code = '''
# Filtrer les lignes où pdays est égal à -1
filtered_df = df[df['pdays'] == -1]
# Vérifier si, pour ces lignes, previous est égal à 0
equivaut = (filtered_df['previous'] == 0).all()
    '''
    st.code(code, language='python')
    # Exécuter le code
    # Filtrer les lignes où pdays est égal à -1
    filtered_df = df[df['pdays'] == -1]
    # Vérifier si, pour ces lignes, previous est égal à 0
    equivaut = (filtered_df['previous'] == 0).all()
    # Afficher le résultat
    st.write(f"Est-ce que toutes les lignes avec pdays = -1 ont une valeur previous = 0 ? {equivaut}")

#page2 DATAVIZ
if page == pages[2] :
  st.title('DataVizualisation')
  col1, col2, col3, col4 = st.columns([1, 1, 1,1])
  button1 = col1.button("Variable cible Deposit")
  button2 = col2.button("Variables Numériques")
  button3 = col3.button("Variables Catégorielles")
  button4 = col4.button("Variables versus Cible")

  # Vérifiez si un bouton est cliqué
  button_clicked = button1 or button2 or button3 or button4

  #Définir button1 par défaut à l'ouverture de la page
  if not button_clicked or button1:
        
        # Code pour afficher le graphique avec Plotly
        count_deposit = df['deposit'].value_counts()
        color_sequence = ['#FACA5E', '#5242EA']
        # pie chart
        pie_chart = go.Pie(
            labels=count_deposit.index,
            values=count_deposit.values,
            marker=dict(colors=color_sequence),
            pull=[0.05, 0]
        )
        # bar chart
        bar_chart = go.Bar(
            x=count_deposit.index,
            y=count_deposit.values,
            text=count_deposit.values,
            textposition='auto',
            marker=dict(color=color_sequence),
            showlegend=False
        )
        # figure avec deux sous-plots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]],
            subplot_titles=("Distribution", "Nombre de dépôts")
        )
        # Ajouter pie chart et bar chart à la figure
        fig.add_trace(pie_chart, row=1, col=1)
        fig.add_trace(bar_chart, row=1, col=2)
        # Mise à jour
        fig.update_layout(
            title_text="<b>Analyse de la variable cible : dépôt à terme ou non</b>",
            legend_title="<b>Dépôt</b>"
        )
        # Affichage avec Streamlit
        st.plotly_chart(fig)
        st.subheader("Constat")
        st.markdown("""
La répartition entre les clients qui ont souscrit à un dépôt à terme et ceux qui n'ont pas soucrit est relativement équilibrée, 
avec une différence de 5.2 points. 
Toutefois, il y a légèrement plus de personnes qui n'ont pas contracté de dépôt (52,6 %) 
par rapport à celles qui l'ont fait (47,4 %).
              """)

  if button2:
        # Code pour afficher les histogrammes des variables numériques
        num_columns = ['balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        # Création des sous-graphiques
        fig = make_subplots(rows=2, cols=3, subplot_titles=num_columns)
        # Position du subplot
        row = 1
        col = 1
        # Création des histogrammes pour chaque variable numérique
        for num_column in num_columns:
            fig.add_trace(
                go.Histogram(
                    x=df[num_column],
                    marker_color='#56CEB2',
                    opacity=0.6,
                    marker_line_width=0.5,
                    showlegend=False,
                    name=num_column
                ),
                row=row,
                col=col
            )
            fig.update_xaxes(title_text=num_column, row=row, col=col)
            fig.update_yaxes(title_text='Count', row=row, col=col)

            col += 1
            if col > 3:
                row += 1
                col = 1
        # Mise à jour de la mise en page du graphique
        fig.update_layout(
            height=800,
            width=1000,
            title_text="<b>Histogrammes des variables numériques</b>"
        )
        # Affichage du graphique avec Streamlit
        st.plotly_chart(fig)
        st.subheader("Constat")
        st.markdown("""
- **Solde moyen du compte bancaire (balance)** : Forte concentration des données autour de 0. Présence de valeurs négatives et de valeurs extrêmes.
- **Jour de contact (days)** : la campagne de télémarketing semble avoir lieu tous les jours du mois, avec une baisse notable en moyenne le 10 du mois et entre le 22 et le 27 du mois. Il est à noter que cette variable est lissée sur tous les mois de plusieurs années, avec l'absence de l'information année, ni celle du jour de la semaine, ne nous permettant pas de déduire de grosses tendances à partir de cette variable.
- **Durée du contact (duration)** : exprimée en secondes, présence de valeurs extrêmes.
- **Nombre de contacts de la campagne(campaign)** : présence de valeurs extrêmes.
- **Nombre de jours depuis le contact précédent (pdays)** : forte présence de valeurs négatives, distribution asymétrique, et nombreuses valeurs extrêmes.
- **Nombre de contacts précédents (previous)** : Très forte concentration autour de 0 qui signifie pas de contacts précédemment et présence de valeurs extrêmes.
              """)
        st.divider()
        # Convertir la variable cible 'deposit' en numérique
        df['deposit_num'] = df['deposit'].apply(lambda x: 1 if x == 'yes' else 0)
        # Sélection des variables numériques
        var_num_cible = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'deposit_num']
        # Calcul de la matrice de corrélation
        corr_matrix_cible = df[var_num_cible].corr()
        # Création du heatmap avec Plotly
        heatmap_fig = px.imshow(corr_matrix_cible, text_auto=True, aspect="auto", color_continuous_scale='Turbo')
        # Mise à jour du layout
        heatmap_fig.update_layout(
            title="<b>Heatmap des Variables Numériques avec la variable cible deposit</b>",
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        # Affichage du heatmap avec Streamlit
        st.plotly_chart(heatmap_fig)
        st.subheader("Constat")
        st.markdown("""
- Dans ce graphique de corrélation, on note un lien entre les variables pdays et previous ; 
ce qui semble cohérent puisque pdays représente le nombre de jours depuis le dernier contact client et previous représente le nombre de contacts précédant cette campagne.
- La variable duration - durée du contact client durant la campagne - semble influencer la variable cible deposit. Nous étudierons plus spécifiquement cette variable exprimée en secondes, et présent des valeurs extrêmes.
- Dans une très moindre mesure, les variables pdays, previous et balance semble légèrement influencer la variable cible deposit.
              """)

  if button3:
        # Catégories à afficher
        cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
        # Palette de couleurs
        color_pal4 = ['#56CEB2', '#28DCE0', '#57CF8A', '#579DCF']
        # Création des sous-graphiques
        fig = make_subplots(rows=3, cols=3, subplot_titles=cat_columns)
        # Fonction d'application des couleurs
        counter = 0
        for cat_column in cat_columns:
            value_counts = df[cat_column].value_counts()
            x_pos = np.arange(0, len(value_counts))
            # Mélanger les couleurs de la palette de manière aléatoire
            random_colors = color_pal4.copy()
            random.shuffle(random_colors)
            # Appliquer les couleurs mélangées aux barres de la catégorie
            colors = [random_colors[i % len(random_colors)] for i in range(len(value_counts))]
            trace_x = counter // 3 + 1
            trace_y = counter % 3 + 1
            # Ajout de la barre
            fig.add_trace(
                go.Bar(
                    x=x_pos,
                    y=value_counts.values,
                    text=value_counts.values,
                    textposition='auto',
                    hoverinfo='text+x',
                    name=cat_column,
                    marker_color=colors,
                    opacity=0.8,
                    showlegend=False,
                ),
                row=trace_x,
                col=trace_y
            )
            # Mise en forme de l'axe x
            fig.update_xaxes(
                tickvals=x_pos,
                ticktext=value_counts.index,
                row=trace_x,
                col=trace_y
            )
            # Rotation des étiquettes de l'axe x
            fig.update_xaxes(tickangle=45, row=trace_x, col=trace_y)
            counter += 1
        # Mise à jour de la mise en page du graphique
        fig.update_layout(
            height=800,
            width=1000,
            title_text="<b>Distribution des modalités des variables catégorielles</b>",
        )
        # Affichage du graphique avec Streamlit
        st.plotly_chart(fig)
        st.markdown("**Constat**")
        st.markdown("""
- Profession (job) : Les professions les plus fréquentes sont 'management','blue-collar' (ouvriers) et 'technician'”'.
- État civil (marital) : La majorité des clients sont 'married' (mariés).
- Niveau d'études (education) : La catégorie 'secondary' (enseignement secondaire) est la plus fréquente parmi ceux qui ont souscrit au produit dépôt à terme.
- Défaut de paiement (default) : Très faible part des clients en défaut de paiement.
- Crédit immobilier (housing) : plutôt équilibré entre les clients ayant un crédit immobilier ou non.
- Prêt personnel (loan) : Très faible part de clients avec un prêt personnel.
- Type de contact (contact) : Le contact par mobile est le plus fréquent.
- Mois de contact (month) : Les mois de mai, juin, juillet, et août sont les mois avec le plus de contacts pour cette campagne.
- Résultat précédente campagne (poutcome) : Une bonne partie des résultats de la précédente campagne est inconnue.
              """)

  if button4:
        # Sous-menu pour naviguer dans différentes sections de la page
        st.divider()
        st.markdown("""
        ### Analyse en 4 axes :
        - [Le profil client](#le-profil-client)
        - [Le profil bancaire](#le-profil-bancaire)
        - [Analyse des contacts clients durant la campagne télémarketing](#analyse-des-contacts-clients-durant-la-campagne-télémarketing)
        - [Analyse de la campagne précédente et son influence sur la campagne actuelle](#analyse-de-la-campagne-précédente-et-son-influence-sur-la-campagne-actuelle)
        """)

        # Section 1: Le profil client
        st.markdown("""
        <a id="le-profil-client"></a>
        ### Le profil client
        """, unsafe_allow_html=True)

        # Graphiques pour le profil client

        # Définir les couleurs spécifiques pour chaque catégorie
        color_sequence = ['#5242EA', '#FACA5E']

        # 1er graphique : Distribution de l'âge versus dépôt
        fig1 = px.box(df, x='age', y='deposit', points='all',
                  color='deposit',
                  title="Distribution de l'âge versus dépôt",
                  color_discrete_sequence=color_sequence,
                  labels={'deposit': 'Statut de Dépôt'},
                  category_orders={"deposit": ["yes", "no"]}  #"yes" est avant "no"
                 )

        # 2ème graphique : Répartition des dépôts en fonction de l'âge
        count_deposit = df.groupby(['age', 'deposit']).size().reset_index(name='count')
        fig2 = px.bar(count_deposit, x='age', y='count', color='deposit',
                  barmode='group',
                  title="Répartition des dépôts en fonction de l'âge",
                  labels={'age': 'Âge', 'count': 'Nombre de dépôts', 'deposit': 'Statut de Dépôt'},
                  category_orders={"deposit": ["yes", "no"]},  # "yes" est avant "no"
                  color_discrete_sequence=color_sequence
                 )

        # Assemblage des graphiques
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "Distribution de l'âge versus dépôt",
        "Répartition des dépôts en fonction de l'âge"
        ])

        # Ajouter fig1 sans légende pour éviter les doublons
        for trace in fig1['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=1)

        # Ajouter fig2 avec légende
        for trace in fig2['data']:
            fig.add_trace(trace, row=1, col=2)

        # Mise à jour de la mise en page
        fig.update_layout(
        height=500,
        width=1500,
        title_text="<b>Analyse de l'âge en fonction de deposit",
        showlegend=True,
        bargap=0.1,
        legend=dict(
            title="Dépôt")
    )

        fig.update_xaxes(title_text='Âge du client', row=1, col=1)
        fig.update_yaxes(title_text='Deposit', row=1, col=1)

        fig.update_xaxes(title_text='Âge du client', row=1, col=2)
        fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=2)

        # Affichage du graphique
        st.plotly_chart(fig)

        # Texte explicatif
        st.markdown("**Constat**")
        st.markdown("""
    - Nous pouvons constater ci-dessous que les clients qui ont souscrit au dépôt à terme sont en moyenne plus âgés que ceux n'ayant pas souscrit (78 ans contre 70 ans).
    - Le nuage de points qui suit met en exergue que ceux n'ayant pas souscrit sont plus dispersés après 60 ans. 
    - Nous constatons également la présence de nombreuses valeurs extrêmes (outliers).
    - Enfin, il apparaît nettement une plus forte proportion de dépôt à terme chez les moins de 30 ans et chez les plus de 60 ans.
    
    Pour la suite de l'analyse, nous avons fait le choix de discrétiser la variable 'age' par tranche d'âge pour atténuer 
    le rôle des valeurs extrêmes et pour afficher ensuite plusieurs graphiques par catégorie.
    
    """)
        st.divider()
        # 2ème graphique : Discrétisation de l'âge
        df['age_cat'] = pd.cut(df.age, bins=[18,29,40,50,60,96], labels=['18-29ans','30-40ans','40-50ans','50-60ans','Plus de 60 ans'])
        df['age_cat'].value_counts()

        # 1ER GRAPHIQUE AGE
        counts_age = df.groupby(['age_cat', 'deposit']).size().unstack()
        total_counts_age = counts_age.sum(axis=1)
        percent_yes_age = (counts_age['yes'] / total_counts_age * 100).round(2)
        percent_no_age = (counts_age['no'] / total_counts_age * 100).round(2)
        df_plot_age = pd.melt(counts_age.reset_index(), id_vars=['age_cat'], value_vars=['yes', 'no'],
                      var_name='deposit', value_name='count')
        df_plot_age['percent'] = percent_yes_age.tolist() + percent_no_age.tolist()

        fig_age = px.bar(df_plot_age, x='age_cat', y='count', color='deposit', barmode='group',
                 title="Répartition des dépôts en fonction de la tranche d'âge",
                 labels={'age_cat': 'Age', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
                 category_orders={"age_cat": ['18-29ans','30-40ans','40-50ans','50-60ans','Plus de 60 ans']},
                 text=df_plot_age['percent'].apply(lambda x: f"{x:.2f}%"),  # ajouter le signe % aux pourcentages
                 color_discrete_sequence=['#5242EA', '#FACA5E'],  # configurer les couleurs correctes
                 hover_data={'count': True, 'percent': ':.2f%'}  # afficher les détails au survol
                 )

        # Mettre à jour le layout
        fig_age.update_layout(yaxis_title="Nombre de dépôts",
                      legend_title_text='Statut du dépôt',
                      xaxis_tickangle=30)

        # 2EME GRAPHIQUE JOB
        counts_job = df.groupby(['job', 'deposit']).size().unstack()
        job_order = df.groupby('job')['deposit'].count().reset_index(name='total_deposits')
        job_order = job_order.sort_values(by='total_deposits', ascending=False)['job']
        job_order = job_order.tolist()
        total_counts_job = counts_job.sum(axis=1)
        percent_yes_job = (counts_job['yes'] / total_counts_job * 100).round(2)
        percent_no_job = (counts_job['no'] / total_counts_job * 100).round(2)
        df_plot_job = pd.melt(counts_job.reset_index(), id_vars=['job'], value_vars=['yes', 'no'],
                      var_name='deposit', value_name='count')
        df_plot_job['percent'] = percent_yes_job.tolist() + percent_no_job.tolist()

        fig_job = px.bar(df_plot_job, x='job', y='count', color='deposit', barmode='group',
                 title="Répartition des dépôts en fonction du type d'emploi",
                 labels={'job': 'Métier', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
                 category_orders={'job': job_order},
                 color_discrete_sequence=['#5242EA', '#FACA5E'],  # configurer les couleurs correctes
                 hover_data={'count': True}  # afficher les détails au survol
                 )

        # Mettre à jour le layout
        fig_job.update_layout(yaxis_title="Nombre de dépôts",
                      legend_title_text='Statut du dépôt',
                      xaxis_tickangle=30,
                      bargap=0.1)

        # 3EME GRAPHIQUE MARITAL
        marital_order = ['married', 'single', 'divorced']
        counts_marital = df.groupby(['marital', 'deposit']).size().unstack()
        total_counts_marital = counts_marital.sum(axis=1)
        percent_yes_marital = (counts_marital['yes'] / total_counts_marital * 100).round(2)
        percent_no_marital = (counts_marital['no'] / total_counts_marital * 100).round(2)
        df_plot_marital = pd.melt(counts_marital.reset_index(), id_vars=['marital'], value_vars=['yes', 'no'],
                      var_name='deposit', value_name='count')
        df_plot_marital['percent'] = percent_yes_marital.tolist() + percent_no_marital.tolist()

        fig_marital = px.bar(df_plot_marital, x='marital', y='count', color='deposit', barmode='stack',
                 title="Répartition des dépôts en fonction du statut marital",
                 labels={'marital': 'Statut marital', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
                 category_orders={'marital': marital_order},
                 text=df_plot_marital['percent'].apply(lambda x: f"{x:.2f}%"),  # ajouter le signe % aux pourcentages
                 color_discrete_sequence=['#5242EA', '#FACA5E'],  # configurer les couleurs correctes
                 hover_data={'count': True, 'percent': ':.2f%'}  # afficher les détails au survol
                 )

        # Mettre à jour le layout
        fig_marital.update_layout(yaxis_title="Nombre de dépôts",
                      legend_title_text='Statut du dépôt',
                      xaxis_tickangle=30)

        # 4EME GRAPHIQUE EDUCATION
        education_order = df.groupby('education')['deposit'].count().reset_index(name='total_deposits')
        education_order = education_order.sort_values(by='total_deposits', ascending=False)['education']
        education_order = education_order.tolist()
        counts_education = df.groupby(['education', 'deposit']).size().unstack()
        total_counts_education = counts_education.sum(axis=1)
        percent_yes_education = (counts_education['yes'] / total_counts_education * 100).round(2)
        percent_no_education = (counts_education['no'] / total_counts_education * 100).round(2)
        df_plot_education = pd.melt(counts_education.reset_index(), id_vars=['education'], value_vars=['yes', 'no'],
                      var_name='deposit', value_name='count')
        df_plot_education['percent'] = percent_yes_education.tolist() + percent_no_education.tolist()

        fig_education = px.bar(df_plot_education, x='education', y='count', color='deposit', barmode='stack',
                 title="Répartition des dépôts en fonction du niveau d'études ",
                 labels={'education': "Niveau d'études", 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
                 category_orders={'education': education_order},
                 text=df_plot_education['percent'].apply(lambda x: f"{x:.2f}%"),  # ajouter le signe % aux pourcentages
                 color_discrete_sequence=['#5242EA', '#FACA5E'],  # configurer les couleurs correctes
                 hover_data={'count': True, 'percent': ':.2f%'}  # afficher les détails au survol
                 )

        # Mettre à jour le layout
        fig_education.update_layout(yaxis_title="Nombre de dépôts",
                      legend_title_text='Statut du dépôt',
                      xaxis_tickangle=30)

        # Création des subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Répartition des tranches d'âge en fonction du dépôt",
                "Répartition des jobs en fonction du dépôt",
                "Répartition du statut marital en fonction du dépôt",
                "Répartition du niveau d'études en fonction du dépôt"
            )
        )

        # Ajouter fig_age
        for trace in fig_age['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=1)

        # Ajouter fig_job
        for trace in fig_job['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=2)

        # Ajouter fig_marital
        for trace in fig_marital['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=2, col=1)

        # Ajouter fig_education
        for trace in fig_education['data']:
            fig.add_trace(trace, row=2, col=2)

        # Mettre à jour les axes avec les orders spécifiés
        fig.update_xaxes(categoryorder='array', categoryarray=job_order, row=1, col=2)
        fig.update_xaxes(categoryorder='array', categoryarray=marital_order, row=2, col=1)

        # Mise à jour de la mise en page
        fig.update_layout(
        height=900,
        width=1200,
        title_text="<b>Analyse du profil client selon les résultats de dépôt",
        legend_title="Dépôt"
        )

        fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=1)
        fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=2)
        fig.update_xaxes(title_text='Statut Marital', row=2, col=1)
        fig.update_yaxes(title_text='Nombre de dépôts', row=2, col=1)
        fig.update_xaxes(title_text="Niveau d'études", row=2, col=2)
        fig.update_yaxes(title_text='Nombre de dépôts', row=2, col=2)

        # Afficher les graphiques
        st.plotly_chart(fig)
        st.markdown("**Constat**")
        st.markdown("""
- **Âge**: Une tendance significative se dégage parmi les jeunes et les aînés à souscrire aux dépôts à terme, avec près de 60 % des moins de 30 ans et environ 82 % des plus de 60 ans ayant opté pour cette option.
- **Emploi** : Alors que les managers, ouvriers, techniciens et administratifs représentent une part substantielle des clients de la banque, les retraités, étudiants, sans emploi et managers sont plus enclins à souscrire au dépôt à terme.
- **Statut marital**: Bien que les clients mariés constituent une proportion significative de la clientèle, les célibataires montrent une plus forte propension à souscrire au dépôt, avec plus de 54 % d'entre eux ayant opté pour cette option.
- **Niveau d'études**: Bien que la majorité des clients ait un niveau d'études secondaire, une proportion plus élevée de souscripteurs au dépôt est observée parmi ceux ayant un niveau d'études supérieur (tertiaire), atteignant 54 %. En revanche, les niveaux d'études inférieurs sont associés à des taux moindres de souscription.
    """)
        st.divider()
        # Section 2: Le profil bancaire
        st.markdown("""
        <a id="le-profil-bancaire"></a>
        ### Le profil bancaire
        """, unsafe_allow_html=True)

        ## 1ER GRAPHIQUE DEFAULT
        # Calculer les décomptes pour chaque catégorie de default et deposit
        counts_default = df.groupby(['default', 'deposit']).size().unstack()
        # Calculer les pourcentages
        total_counts_default = counts_default.sum(axis=1)
        percent_yes_default = (counts_default['yes'] / total_counts_default * 100).round(2)
        percent_no_default = (counts_default['no'] / total_counts_default * 100).round(2)
        # Transformer les données pour Plotly Express
        df_plot_default = pd.melt(counts_default.reset_index(), id_vars=['default'], value_vars=['yes', 'no'],var_name='deposit', value_name='count')

        # Ajouter les pourcentages calculés
        df_plot_default['percent'] = percent_yes_default.tolist() + percent_no_default.tolist()

        # Créer le graphique avec Plotly Express
        fig_default = px.bar(df_plot_default, x='default', y='count', color='deposit', barmode='stack',
             title="Répartition des dépôts en fonction du défaut de paiement",
             labels={'default': 'Défaut de paiement', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
             text=df_plot_default['percent'].apply(lambda x: f"{x:.2f}%"),  # ajouter le signe % aux pourcentages
             color_discrete_sequence=['#5242EA', '#FACA5E'],  # configurer les couleurs correctes
             hover_data={'count': True, 'percent': ':.2f%'}  # afficher les détails au survol
             )

        # Mettre à jour le layout
        fig_default.update_layout(yaxis_title="Nombre de dépôts",
                  legend_title_text='Statut du dépôt')


        # 2EME GRAPHIQUE LOAN
        # Calculer les décomptes pour chaque catégorie de loan et deposit
        counts_loan = df.groupby(['loan', 'deposit']).size().unstack()
        # Calculer les pourcentages
        total_counts_loan = counts_loan.sum(axis=1)
        percent_yes_loan = (counts_loan['yes'] / total_counts_loan * 100).round(2)
        percent_no_loan = (counts_loan['no'] / total_counts_loan * 100).round(2)
        # Transformer les données pour Plotly Express
        df_plot_loan = pd.melt(counts_loan.reset_index(), id_vars=['loan'], value_vars=['yes', 'no'],
                  var_name='deposit', value_name='count')

        # Ajouter les pourcentages calculés
        df_plot_loan['percent'] = percent_yes_loan.tolist() + percent_no_loan.tolist()

        # Créer le graphique avec Plotly Express
        fig_loan = px.bar(df_plot_loan, x='loan', y='count', color='deposit', barmode='stack',
             title="Répartition des dépôts en fonction du prêt personnel",
             labels={'loan': 'Prêt personnel', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
             text=df_plot_loan['percent'].apply(lambda x: f"{x:.2f}%"),  # ajouter le signe % aux pourcentages
             color_discrete_sequence=['#5242EA', '#FACA5E'],  # configurer les couleurs correctes
             hover_data={'count': True, 'percent': ':.2f%'}  # afficher les détails au survol
             )

        # Mettre à jour le layout
        fig_loan.update_layout(yaxis_title="Nombre de dépôts",
                  legend_title_text='Statut du dépôt')


        # 3EME GRAPHIQUE HOUSING
        # Calculer les décomptes pour chaque catégorie de housing et deposit
        counts_housing = df.groupby(['housing', 'deposit']).size().unstack()
        # Calculer les pourcentages
        total_counts_housing = counts_housing.sum(axis=1)
        percent_yes_housing = (counts_housing['yes'] / total_counts_housing * 100).round(2)
        percent_no_housing = (counts_housing['no'] / total_counts_housing * 100).round(2)
        # Transformer les données pour Plotly Express
        df_plot_housing = pd.melt(counts_housing.reset_index(), id_vars=['housing'], value_vars=['yes', 'no'],
                  var_name='deposit', value_name='count')

        # Ajouter les pourcentages calculés
        df_plot_housing['percent'] = percent_yes_housing.tolist() + percent_no_housing.tolist()

        # Créer le graphique avec Plotly Express
        fig_housing = px.bar(df_plot_housing, x='housing', y='count', color='deposit', barmode='stack',
             title="Répartition des dépôts en fonction du Prêt immobilier",
             labels={'housing': 'Prêt immobilier', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
             text=df_plot_housing['percent'].apply(lambda x: f"{x:.2f}%"),  # ajouter le signe % aux pourcentages
             color_discrete_sequence=['#5242EA', '#FACA5E'],  # configurer les couleurs correctes
             hover_data={'count': True, 'percent': ':.2f%'}  # afficher les détails au survol
             )

        # Mettre à jour le layout
        fig_housing.update_layout(yaxis_title="Nombre de dépôts",
                  legend_title_text='Statut du dépôt')


        # 4EME GRAPHIQUE BALANCE
        # Distribution de balance versus dépôt
        fig_balance = px.box(df, x='deposit', y='balance',
              color='deposit',
              title="Distribution du solde moyen de compte",
              color_discrete_sequence=['#5242EA', '#FACA5E'],  # configurer les couleurs correctes,
              labels={'deposit': 'Statut de Dépôt'},
              category_orders={"deposit": ["yes", "no"]}  #"yes" est avant "no"
             )

        ## CREATION SUBPLOTS
        fig = make_subplots(
            rows=1, cols=4,
            subplot_titles=(
            "Défaut de paiement",
            "Prêt personnel",
            "Prêt immobilier",
            "Solde moyen de compte"
            )
        )

        # Ajouter fig_default
        for trace in fig_default['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=1)
        # Ajouter fig_loan
        for trace in fig_loan['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=2)
        # Ajouter fig_housing
        for trace in fig_housing['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=3)
        # Ajouter fig_balance
        for trace in fig_balance['data']:
            fig.add_trace(trace, row=1, col=4)

        # Mise à jour de la mise en page
        fig.update_layout(
            height=500,
            width=1400,
            title_text="<b>Analyse du profil bancaire selon les résultats de dépôt",
            legend_title= "Dépôt"
            )

        fig.update_xaxes(title_text='default (yes, no)', row=1, col=1)
        fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=1)

        fig.update_xaxes(title_text='loan (yes, no)', row=1, col=2)
        fig.update_xaxes(title_text='housing (yes, no)', row=1, col=3)
        fig.update_xaxes(title_text='deposit (yes, no)', row=1, col=4)

        fig.update_xaxes(title_text='deposit (yes, no)', row=1, col=4)
        fig.update_yaxes(title_text='balance', row=1, col=4)

        ## AFFICHER LA FIGURE
        st.plotly_chart(fig)
        st.markdown("**Constat**")
        st.markdown("""
- **Défaut de paiement (default)** : Très faible part de clients en défaut de paiement dans le jeu de données. 
En revanche on constate nettement que ceux en défaut de paiement sont beaucoup moins enclins à souscrire au dépôt (69% n'ont pas souscrit, alors que 52% des autres clients ont souscrit ce qui correspond à la la moyenne générale des souscripteurs)
- **Prêt personnel (loan)** : faible part des clients de la banque ayant un crédit consommation ou prêt personnel. Néanmoins, on constate que ceux ayant un prêt personnel en cours ont moins souscrit au dépôt (69% n'ont pas souscrit, alors que 50,4% des autres clients ont souscrit ce qui s'approche de la moyenne générale des souscripteurs)
- **Prêt immobilier (housing)**: On note nettement ici que les clients n'ayant pas de crédit immobilier sont une majorité à souscrire au dépôt (57% d'entre eux) et inversement les clients ayant un crédit immobilier en cours sont moins enclins à souscrire (63% d'entre eux n'ont pas souscrit)
- **Solde moyen de compte (balance)**: les données sont étendues avec beaucoup de valeurs extrêmes. On constate un solde médian à 733€ pour les clients ayant souscrit et un solde médian plus faible à 414€ pour les clients n'ayant pas souscrit .
    """)
        st.divider()



        # Section 3: Analyse des contacts clients durant la campagne télémarketing
        st.markdown("""
        <a id="analyse-des-contacts-clients-durant-la-campagne-télémarketing"></a>
        ### Analyse des contacts clients durant la campagne télémarketing
        """, unsafe_allow_html=True)
        # 1. Graphique Contact
        # Calculer les décomptes pour chaque catégorie de contact et deposit
        counts_contact = df.groupby(['contact', 'deposit']).size().unstack(fill_value=0)
        # Calculer les pourcentages
        total_counts_contact = counts_contact.sum(axis=1)
        percent_yes_contact = (counts_contact['yes'] / total_counts_contact * 100).round(2)
        percent_no_contact = (counts_contact['no'] / total_counts_contact * 100).round(2)
        # Transformer les données pour Plotly Express
        df_plot_contact = pd.melt(counts_contact.reset_index(), id_vars=['contact'], value_vars=['yes', 'no'],
                  var_name='deposit', value_name='count')
        # Ajouter les pourcentages calculés
        df_plot_contact['percent'] = percent_yes_contact.tolist() + percent_no_contact.tolist()

        # Créer le graphique
        fig_contact = px.bar(df_plot_contact, x='contact', y='count', color='deposit', barmode='group',
             title="Mode de contact client et résultats de dépôt",
             labels={'contact': 'Mode de contact client', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
             color_discrete_sequence=['#5242EA', '#FACA5E'],
             hover_data={'count': True, 'percent': ':.2f%'}
             )

        # Mettre à jour le layout
        fig_contact.update_layout(yaxis_title="Nombre de dépôts",
                  legend_title_text='Statut du dépôt')


        # 2. Graphique Duration
        fig_duration = px.box(df,
           x='duration',  # Change 'duration_minutes' to 'duration'
           y='deposit',
           color='deposit',
           color_discrete_sequence=['#5242EA', '#FACA5E'],
           title='<b>Influence de la durée de contact sur le résultat de la campagne')

        # 3. Graphique Month
        # Calculer le nombre total de dépôts pour chaque mois
        month_order = df.groupby('month')['deposit'].count().reset_index(name='total_deposits')
        month_order = month_order.sort_values(by='total_deposits', ascending=False)['month']

        # Convertir en liste pour utilisation dans category_orders
        month_order = month_order.tolist()

        # Création de l'histogramme
        fig_month = px.histogram(df, x='month', color='deposit', barmode='group',
                   title="Répartition de dépôt en fonction des mois",
                   labels={'month': 'Mois de contact', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
                   category_orders={"month": month_order},
                   color_discrete_sequence=['#5242EA', '#FACA5E'])

        # Mettre à jour le layout
        fig_month.update_layout(yaxis_title="Nombre de dépôts",
                  legend_title_text='Statut du dépôt',
                  xaxis_tickangle=30,
                  bargap=0.1)

        # 4. Graphique M Contact
        # Grouper par mois et agréger les décomptes
        data_month = df.groupby('month').agg(
            campaign_count=('campaign', 'sum'),
            deposit_yes_count=('deposit', lambda x: (x == 'yes').sum()),
            deposit_no_count=('deposit', lambda x: (x == 'no').sum())
        ).reset_index()
        # Ajouter une nouvelle colonne avec des valeurs manuelles
        manual_values = [4, 8, 12, 2, 1, 7, 6, 3, 5, 11, 10, 9]
        # Assigner les valeurs manuelles à la colonne 'manual_order'
        data_month['manual_order'] = manual_values
        # Tri du DataFrame par la colonne 'manual_order'
        data_month_sorted = data_month.sort_values(by='manual_order').reset_index(drop=True)
        # Création du graphique
        fig_m_contact = px.line()
        # Ajout des courbes sur le graphique
        fig_m_contact.add_scatter(x=data_month_sorted['month'], y=data_month_sorted['campaign_count'], mode='lines', name='Nombre de contact', line=dict(color='#034F84', dash='dash'))
        fig_m_contact.add_scatter(x=data_month_sorted['month'], y=data_month_sorted['deposit_yes_count'], mode='lines', name='Dépôts Yes', line=dict(color='#5242EA'))
        fig_m_contact.add_scatter(x=data_month_sorted['month'], y=data_month_sorted['deposit_no_count'], mode='lines', name='Dépôts No', line=dict(color='#FACA5E'))
        fig_m_contact.update_layout(title='Nombre de contacts et dépôts par mois')
        # Ajout des axes
        fig_m_contact.update_xaxes(title_text='Mois')
        fig_m_contact.update_yaxes(title_text='Nombre de contacts')

        # Création des subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Mode de contact",
                "Durée du contact",
                "Répartition par mois",
                "Nombre de contacts et dépôts par mois"
            )
        )

        # Ajouter fig_contact
        for trace in fig_contact['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=1)

        # Ajouter fig_duration
        for trace in fig_duration['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=2)

        # Ajouter fig_month
        for trace in fig_month['data']:
            fig.add_trace(trace, row=2, col=1)

        # Ajouter fig_m_contact
        for trace in fig_m_contact['data']:
            fig.add_trace(trace, row=2, col=2)

        # Mise à jour de la mise en page
        fig.update_layout(
            height=600,
            width=1400,
            title_text="<b>Analyse de la campagne : type de contact, nombre de contacts, période et durée",
            legend_title="Dépôt"
            )

        fig.update_xaxes(title_text='Modalité de contact', row=1, col=1)
        fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=1)

        fig.update_xaxes(title_text='Durée de contact en minutes', row=1, col=2)
        fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=2)

        fig.update_xaxes(title_text='Mois de contact', row=2, col=1)
        fig.update_yaxes(title_text='Nombre de dépôts', row=2, col=1)

        fig.update_xaxes(title_text='Mois de contact', row=2, col=2)
        fig.update_yaxes(title_text='Nombre de dépôts', row=2, col=2)

        # Afficher la figure dans Streamlit
        st.plotly_chart(fig)
        st.markdown("**Constat**")
        st.markdown("""
- **Type de contact** : Une majorité de contact pour cette campagne de télémarketing a été opérée sur mobile (cellular) et on constate une plus forte proporition de souscription pour les clients ayant été contacté par ce biais (plus de 54%).
Néanmoins une part importante du type de contact est inconnue.
- **Durée du contact**: Il apparaît que la durée du contact client influence sur le résultat du dépôt à terme : plus le contact est long, plus les clients ont tendance à souscrire.
- **Mois de contact** : Il est intéressant de constater sur les 2 graphiques du bas, que si les mois de mai, juin, juillet, Août sont les mois de plus forte activité de la campagne,
ce ne sont pas les mois où la part de souscription est la plus importante. On constate en effet une **plus forte proportion de clients effectuant un dépôt durant les mois de février, mars, avril, septembre, octobre et décembre**.
L'exemple flagrant est le mois de mai qui semble être le mois avec la plus forte activité de la campagne et pour lequel la part de dépôt est plus faible.
    """)
        st.divider()
        # Section 4: Analyse de la campagne précédente et son influence sur la campagne actuelle
        st.markdown("""
        <a id="analyse-de-la-campagne-précédente-et-son-influence-sur-la-campagne-actuelle"></a>
        ### Analyse de la campagne précédente et son influence sur la campagne actuelle
        """, unsafe_allow_html=True)

        # 1. Graphique Contacts Précédents ou Non Contactés
        # Diviser en deux groupes
        df['group'] = df['previous'].apply(lambda x: 'non contactés' if x == 0 else 'contactés')

        # Compter les valeurs de deposit pour chaque groupe
        count_df = df.groupby(['group', 'deposit']).size().reset_index(name='count')

        # Calculer les pourcentages
        total_counts = count_df.groupby('group')['count'].transform('sum')
        count_df['percentage'] = (count_df['count'] / total_counts * 100).round(2)

        # Création du bar plot avec Plotly Express
        fig_previous = px.bar(
            count_df,
            x='group',
            y='count',
            color='deposit',
            text=count_df['percentage'].astype(str) + '%',
            color_discrete_sequence=['#5242EA', '#FACA5E'],
        )

        # 2. Graphique Nombre de Jours depuis le Dernier Contact (pdays)
        # Filtrer les données pour exclure les valeurs de 'pdays' égales à -1
        df_filtered = df[df['pdays'] != -1]

        # Créer le box plot
        fig_pdays = px.box(df_filtered,
             x='deposit',
             y='pdays',
             color='deposit',
             color_discrete_sequence=['#5242EA', '#FACA5E'],
             )

        # 3. Graphique Résultats de la Précédente Campagne (poutcome)
        # Calculer les décomptes pour chaque catégorie de poutcome et deposit
        counts_poutcome = df.groupby(['poutcome', 'deposit']).size().unstack()
        # Calculer les pourcentages
        total_counts_poutcome = counts_poutcome.sum(axis=1)
        percent_yes_poutcome = (counts_poutcome['yes'] / total_counts_poutcome * 100).round(2)
        percent_no_poutcome = (counts_poutcome['no'] / total_counts_poutcome * 100).round(2)
        # Transformer les données pour Plotly Express
        df_plot_poutcome = pd.melt(counts_poutcome.reset_index(), id_vars=['poutcome'], value_vars=['yes', 'no'],
                  var_name='deposit', value_name='count')

        # Ajouter les pourcentages calculés
        df_plot_poutcome['percent'] = percent_yes_poutcome.tolist() + percent_no_poutcome.tolist()

        # Créer le graphique avec Plotly Express
        fig_poutcome = px.bar(df_plot_poutcome, x='poutcome', y='count', color='deposit', barmode='group',
             text=df_plot_poutcome['percent'].apply(lambda x: f"{x:.2f}%"),  # ajouter le signe % aux pourcentages
             color_discrete_sequence=['#5242EA', '#FACA5E'],  # configurer les couleurs correctes
             hover_data={'count': True, 'percent': ':.2f%'}  # afficher les détails au survol
             )

        # Création des subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Contacts précédents ou non",
                "Nombre de jours depuis le dernier contact",
                "Succès de la précédente campagne"
            )
        )

        # Ajouter fig_previous
        for trace in fig_previous['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=1)

        # Ajouter fig_pdays
        for trace in fig_pdays['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=2)

        # Ajouter fig_poutcome
        for trace in fig_poutcome['data']:
            fig.add_trace(trace, row=1, col=3)

        # Mise à jour de la mise en page
        fig.update_layout(
            height=600,
            width=1400,
            title_text="<b>Analyses de la précédente campagne (pdays, previous, poutcome) et influence sur la campagne actuelle",
            legend_title="Dépôt"
            )

        fig.update_xaxes(title_text='Groupe de clients si contactés précédemment ou non (previous)', row=1, col=1)
        fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=1)

        fig.update_xaxes(title_text='Deposit', row=1, col=2)
        fig.update_yaxes(title_text='Nombre de jours depuis le dernier contact (pdays)', row=1, col=2)

        fig.update_xaxes(title_text='Résultats de la campagne précédente (poutcome)', row=1, col=3)
        fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=3)

        # Afficher la figure dans Streamlit
        st.plotly_chart(fig)

        st.markdown("**Constat**")
        st.markdown("""
- **Contacts précédents** : Une forte proportion des clients n'ont pas été contacté précédemment. 
Néanmoins, il est intéressant de noter que les clients ayant déjà été contactés avant cette campagne (lors d'une campagne précédente) 
sont plus enclins à souscrire au dépôt : 67% des clients contactés précédemment, ont souscrit au dépôt lors de cette campagne, et inversement ceux n'ayant pas été contactés précédemment ont été près de 60% à ne pas souscrire au dépôt durant cette campagne. 
Ceci indique que la multiplication des contacts sur différentes campagnes peut inciter les clients et influencer la réussite d'une campagne suivante.
- **Nombre de jours depuis le dernier contact** : On peut remarquer que moins de temps a passé depuis le dernier contact chez les clients souscrivant au dépôt sur cette campagne.
Avec une étendue moins large (entre 94 et 246 jours) que ceux n'ayant pas souscrit au dépôt (étendue 148 à 332 jours).
En sus, on peut constater de nombreuses valeurs extrêmes, notamment chez ceux ayant souscrit au dépôt.
- **Succès de la précédente campagne** : Une grande part des données sont inconnues. Il est tout de même intéressant de noter qu'un client ayant souscrit à un produit d'une campagne précédente (success), sont très enclins à souscrire au dépôt de la campagne actuelle : 91% d'entre eux ont souscrit au dépôt.
    """)
        st.divider()
        # Section 5: Conclusion Analyse
        #st.markdown("""
        #<a id="conclusion_analyse"></a>
        ### Conclusion Analyse variables explicatives vs variable cible
        #""", unsafe_allow_html=True)


#page3 PREPROCESSING
if page == pages[3]:
   st.title('Pre-processing')
   st.markdown("#### Préparation des données avant d'appliquer des algorithmes de classification")
   st.header("Démarche de pre-processing")

   # Functions for each step
   def discretize_age(df):
    df['age_cat'] = pd.cut(df.age, bins=[18, 29, 40, 50, 60, np.inf], 
                           labels=['18-29ans', '30-40ans', '40-50ans', '50-60ans', 'Plus de 60 ans'], right=False)
    df = df.drop('age', axis=1)
    return df


   def replace_unknown_education(df):
    most_frequent = df[df['education'] != 'unknown']['education'].mode()[0]
    df['education'] = df['education'].replace('unknown', most_frequent)
    return df


   def transform_pdays(df):
    df['pdays_contact'] = df['pdays'].apply(lambda x: 'no' if x == -1 else 'yes')
    df['pdays_days'] = df['pdays'].apply(lambda x: 0 if x == -1 else x)
    df = df.drop('pdays', axis=1)
    return df
   
   # Insert containers separated into tabs:
   tab1, tab2, tab3 = st.tabs(["Pré-traitement", "Séparation Train/Test","Standardisation et Encodages"])
   with tab1:
    st.markdown("#### Pré-traitement")
    #AGE
    st.markdown("""
- **Discrétiser la variable 'age'** par tranches d'âge pour atténuer le rôle des valeurs extrêmes.
- **Remplacer la modalité 'unknown' de la variable 'education'** par la modalité la plus fréquente.
- **Diviser la variable 'pdays'** en 2 variables distinctes : pdays_contact et la variable pdays_days. 
    - **pdays_contact** : valeur no pour les valeurs -1 de pdays, et valeur yes pour les autres valeurs.
    - **pdays_days** : valeur 0 pour les valeurs -1 de pdays et valeurs de pdays >= 0. 
                """)
    st.code("""
df['age_cat'] = pd.cut(df.age, bins=[18,29,40,50,60,np.inf], labels = ['18-29ans','30-40ans','40-50ans','50-60ans','Plus de 60 ans'],right=False)
df = df.drop('age', axis = 1)
pd.DataFrame(df['age_cat'].unique(), columns=['Catégories d\'âge'])
            """)
    df = discretize_age(df)
    st.write("Modalités de l'âge après discrétisation :")
    st.table(pd.DataFrame(df['age_cat'].unique(), columns=['Catégories d\'âge']))

    st.write("") 
    
    #EDUCATION 
    st.code("""
most_frequent = df[df['education'] != 'unknown']['education'].mode()[0]
df['education'] = df['education'].replace('unknown', most_frequent)
df['education'].unique()
            """)
    st.write("Modalités de Education après remplacemement de la valeurs 'unknown' par le mode le plus fréquent :")
    df = replace_unknown_education(df)
    st.write(df['education'].unique())
    st.write("")

    #PDAYS
    st.code("""
df['pdays_contact'] = df['pdays'].apply(lambda x: 'no' if x == -1 else 'yes')
df['pdays_days'] = df['pdays'].apply(lambda x: 0 if x == -1 else x)
df = df.drop('pdays', axis = 1)
df.head()
            """)
    st.write("DataFrame df après Feature Engineering de Pdays :")
    df = transform_pdays(df)
    st.write(df.head())

   with tab2:
    st.markdown("#### Séparation Features et Target")
    st.markdown("Séparer les variables explicatives de la cible en deux jeux de données.")
    st.code("""
features = df.drop('deposit', axis = 1)
target = df['deposit']
            """)
    st.markdown("#### Séparation Train / Test")
    # Chemin de l'image
    image_path_traintest = "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC_PROD/main/img/traintest.jpg"
    #Afficher l'image
    st.image(image_path_traintest)
    st.markdown("""
- Séparer le jeu de données en : 
    - un jeu d'entraînement (X_train,y_train) et 
    - un jeu de test (X_test, y_test) 
- la partie de test contient 25% du jeu de données initial
                """)
    st.code("""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.25, random_state = 42)
X_train.shape, X_test.shape
            """)
    features = df.drop('deposit', axis = 1)
    target = df['deposit']
    # Séparer les données
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    # Afficher les dimensions des ensembles d'entraînement et de test
    st.write(f"Dimensions de X_train : {X_train.shape}")
    st.write(f"Dimensions de X_test : {X_test.shape}")
   with tab3:
    st.markdown("#### Standardisation et encodages")
    st.markdown("""
- **LabelEncoder** de la variable cible ‘deposit’.
- **Encodage cyclique** des variables temporelles 'month', 'day'.
- **RobustScaler** sur les variables numériques 'balance', 'duration', 'campaign','previous', ‘pdays_days’.
- **LabelEncoder** des modalités binaires des variables explicatives 'default', 'housing', loan', ‘pdays_contact’.
- **OneHotEncoder** des modalités des variables explicatives 'job', 'marital', 'contact', 'poutcome'.
- **OrdinalEncoder** des modalités des variables explicatives 'age', 'education'.
                """)
   st.divider()
   st.header("Démarche Pipeline")
   # Chemin de l'image
   image_path_pipeline = "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC_PROD/main/img/pipeline.jpg"
   #Afficher l'image
   st.image(image_path_pipeline)
   st.markdown("Grâce à une pipeline, nous avons pu générer rapidement 4 pre-processing différents, testés ensuite sur différents algorithmes de Machine Learning :")
   st.markdown("""
- **Pre-processing 1** : 
    - sans feature engineering de p_days, avec l'âge discrétisé, et un encodage Robust Scaler sur les variables numériques.
- **Pre-processing 2** : 
    - avec la division de pdays en 2 variables, âge discrétisé, Robust Scaler sur les variables numériques.
- **Pre-processing 3** : 
    - équivalent au précédant mais avec un Standard Scaler sur les variables numériques.
- **Pre-processing 4** : 
    - avec l'âge sans discrétisation et Standard Scaler sur les variables numériques.
               """)
   st.markdown("""Après avoir fait tourner différents algorithmes de classification, 
               le pré-processing 2 a eu de meileurs scores - même si très légèrement supérieurs.
               C'est donc cette pipeline utilisée ci-après :  """)
   st.divider()
   # Fonction pour charger les données
   @st.cache_data
   def load_data():
    df=pd.read_csv("bank.csv")
    return df

   #Fonction pour prétraiter les données
   @st.cache_data
   def preprocess_data(df):
        # Conversion de 'month' en int en utilisant un mapping
        month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12 }
        df['month'] = df['month'].map(month_mapping)

        # Discrétisation de la variable 'age'
        df['age_cat'] = pd.cut(df.age, bins=[18, 29, 40, 50, 60, np.inf], labels=['18-29ans', '30-40ans', '40-50ans', '50-60ans', 'Plus de 60 ans'], right=False)

        # Diviser la colonne 'pdays' en deux colonnes
        df['pdays_contact'] = df['pdays'].apply(lambda x: 'no' if x == -1 else 'yes')
        df['pdays_days'] = df['pdays'].apply(lambda x: 0 if x == -1 else x)

        # Séparation de features et target
        features = df.drop(columns=['deposit', 'age', 'pdays'])
        target = df['deposit']

        # Séparation en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
        return X_train, X_test, y_train, y_test

   #Fonction pour prétraiter et transformer les données
   @st.cache_data
   def preprocess_and_transform(X_train, X_test, y_train, y_test):
        # Définition des features
        binary_features = ['default', 'housing', 'loan', 'pdays_contact']
        categorical_features = ['job', 'marital', 'contact', 'poutcome']
        ordinal_features = ['education', 'age_cat']
        numerical_features = ['balance', 'campaign', 'duration', 'previous', 'pdays_days']
        cyclic_features = ['day', 'month']

        # Pipeline pour les variables binaires
        binary_pipeline = Pipeline([
            ('binary_encoding', FunctionTransformer(lambda x: x.replace({'yes': 1, 'no': 0}))),
        ])

        # Pipeline pour les variables catégorielles
        categorical_pipeline = Pipeline([
            ('onehot_encoding', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Pré-traitement de 'education'
        most_frequent_education = X_train['education'].mode()[0]
        X_train['education'] = X_train['education'].replace('unknown', most_frequent_education)
        X_test['education'] = X_test['education'].replace('unknown', most_frequent_education)

        # Pipeline pour l'encodage ordinal de 'education' et 'age_cat'
        education_categories = ['primary', 'secondary', 'tertiary']
        age_cat_categories = ['18-29ans', '30-40ans', '40-50ans', '50-60ans', 'Plus de 60 ans']

        ordinal_pipeline = Pipeline([
            ('ordinal_encoding', OrdinalEncoder(categories=[education_categories, age_cat_categories]))
        ])

        # Pipeline pour les variables numériques
        numerical_pipeline = Pipeline([
            ('duration_minutes', FunctionTransformer(lambda x: x / 60.0 if 'duration' in x else x)),
            ('scaler', RobustScaler())
        ])

        # Encodage cyclique pour 'day' et 'month'
        def encode_cyclic(df, column, max_value):
            df[column + '_sin'] = np.sin(2 * np.pi * df[column] / max_value)
            df[column + '_cos'] = np.cos(2 * np.pi * df[column] / max_value)
            return df

        X_train = encode_cyclic(X_train, 'day', 31)
        X_train = encode_cyclic(X_train, 'month', 12)

        X_test = encode_cyclic(X_test, 'day', 31)
        X_test = encode_cyclic(X_test, 'month', 12)

        # Pré-processing complet avec ColumnTransformer
        preprocessor = ColumnTransformer([
            ('binary', binary_pipeline, binary_features),
            ('categorical', categorical_pipeline, categorical_features),
            ('ordinal', ordinal_pipeline, ordinal_features),
            ('cyclic', 'passthrough', ['day_sin', 'day_cos', 'month_sin', 'month_cos']),
            ('numerical', numerical_pipeline, numerical_features),
        ])

        # Application du preprocessor sur X_train et X_test
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Encodage de la variable cible 'deposit'
        label_encoder = LabelEncoder()
        y_train_processed = label_encoder.fit_transform(y_train)
        y_test_processed = label_encoder.transform(y_test)

        # Conversion en DataFrame pour visualisation
        columns = (binary_features + 
               list(preprocessor.named_transformers_['categorical'].named_steps['onehot_encoding'].get_feature_names_out(categorical_features)) +
               ordinal_features + 
               ['day_sin', 'day_cos', 'month_sin', 'month_cos'] + 
               numerical_features)

        X_train_processed_df = pd.DataFrame(X_train_processed, columns=columns, index=X_train.index).sort_index()
        X_test_processed_df = pd.DataFrame(X_test_processed, columns=columns, index=X_test.index).sort_index()
        y_train_processed_df = pd.DataFrame({'Deposit': y_train_processed}, index=y_train.index).sort_index()
        y_test_processed_df = pd.DataFrame({'Deposit': y_test_processed}, index=y_test.index).sort_index()

        return X_train_processed_df, X_test_processed_df, y_train_processed_df, y_test_processed_df

   # Utilisation des fonctions mises en cache dans Streamlit
   if __name__ == "__main__":
      st.subheader("Résultats du pre-processing")

   df = load_data()
   # Prétraitement pour obtenir les données avant et après traitement
   X_train, X_test, y_train, y_test = preprocess_data(df)
   # Menu déroulant pour choisir l'affichage des résultats
   option = st.selectbox(
        "**Afficher les jeux de données avant ou après pre-processing**",
        ["Avant Pipeline", "Après Pipeline"]
        )
   st.write("")
   st.write("")
   if option == "Avant Pipeline":
      st.markdown("##### Avant Pipeline")
      st.write("**Shapes des ensembles de données :**")
      st.write(f"Shape de X_train avant prétraitement : {X_train.shape}")
      st.write(f"Shape de X_test avant prétraitement : {X_test.shape}")
      st.write(f"Shape de y_train avant prétraitement : {y_train.shape}")
      st.write(f"Shape de y_test avant prétraitement : {y_test.shape}")
      st.write("**X_train avant prétraitement :**")
      st.dataframe(X_train.head())
      st.write("**X_test avant prétraitement :**")
      st.dataframe(X_test.head())
      st.write("**y_train avant prétraitement :**")
      st.dataframe(pd.DataFrame({'Deposit': y_train.values}, index=y_train.index).head())
      st.write("**y_test avant prétraitement :**")
      st.dataframe(pd.DataFrame({'Deposit': y_test.values}, index=y_test.index).head())

   elif option == "Après Pipeline":
      #Appliquer le prétraitement
      X_train_processed_df, X_test_processed_df, y_train_processed_df, y_test_processed_df = preprocess_and_transform(X_train, X_test, y_train, y_test)
      st.markdown("##### Après Pipeline")
      st.write("**Shapes des ensembles de données traitées :**")
      st.write(f"Shape de X_train_processed après prétraitement : {X_train_processed_df.shape}")
      st.write(f"Shape de X_test_processed après prétraitement : {X_test_processed_df.shape}")
      st.write("**Colonnes de X_train_processed :**")
      st.write(X_train_processed_df.columns)
      st.write("**X_train_processed après prétraitement :**")
      st.dataframe(X_train_processed_df.head())
      st.write("**X_test_processed après prétraitement :**")
      st.dataframe(X_test_processed_df.head())
      st.write("**y_train_processed après prétraitement :**")
      st.dataframe(y_train_processed_df.head())
      st.write("**y_test_processed après prétraitement :**")
      st.dataframe(y_test_processed_df.head())

#page4 MODELISATION
if page == pages[4]:
   st.title('Modélisation')
   st.markdown("#### Entraînement de modèles de classification")
   st.header("Démarche de Modélisation")
   # Insert containers separated into tabs:
   tab1, tab2, tab3, tab4 = st.tabs(["Etapes", "11 modèles testés", "Entrainement sans Duration", "GridSearch"])
   with tab1:
    st.markdown("##### Etapes")
    st.markdown("""
- Premier entrainement de 11 modèles de classification 
- Analyse des performances
- Choix de supprimer la variable duration 
- Entrainement des 11 modèles sans duration
- Choix des 3 modèles les plus performants
- Grille de recherche d'hyperparamètres optimaux (GridSeachCV)
- Entrainement des 3 modèles avec leurs meilleurs paramètres
- Analyse des performances 
    """)

   with tab2:
    st.markdown("##### 11 modèles de classification testés")
    st.markdown("""
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine
- K-Nearest Neighbors'
- AdaBoost:
- Naive Bayes
- MLP Classifier
- XGBoost
- LightGBM
- Decision Tree
                """)
    st.write("")     
    st.markdown("##### Performances")
    # Chemin de l'image
    image_path_perf11 = "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC_PROD/main/img/perf11.png"
    #Afficher l'image
    st.image(image_path_perf11)
    st.write("")
    st.markdown("##### Feature importances")        
    st.markdown("Pour les 3 modèles les plus performants")
    # Créez un menu déroulant pour sélectionner l'image à afficher
    option = st.selectbox(
       'Choisir le modèle à afficher:',
       ['Random Forest', 'XGBoost', 'LightGBM']
       )
    # Définir les chemins des images
    image_paths = {
       'Random Forest': "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC_PROD/main/img/fi_rf.png",
       'XGBoost': "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC_PROD/main/img/fi_xg.png",
       'LightGBM': "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC_PROD/main/img/fi_gbm.png"
       }
    # Afficher l'image en fonction de la sélection
    st.image(image_paths[option])
    st.markdown("##### Constat")              
    st.markdown("""
- La variable **'duration'** (durée de l'appel durant la campagne) est la variable explicative qui ressort comme étant la plus importante dans la prédiction des résultats. 
Ceci était démontré dans la matrice de corrélation et désormais confirmé lors du scoring des modèles entrainés.
Néanmoins, cette variable est connue à posteriori de la campagne puisque c'est après l'appel télémarketing que la durée de l'appel est connue.
Il apparaît donc intéressant de **tester nos modèles en supprimant cette variable de nos jeux d'entrainement et de test**. Nous avons également testé en conservant la variable duration avec uniquement sa valeur médiane et le résultat n'a pas été concluant.
- A noter que la variable **'balance'** (solde moyen du compte) est celle qui a le plus d'importance après 'duration'.
- A noter également que le modèle XGBoost accorde plus d'importance à la variable **'poutcome_success'** (résultats de la précédente campagne) dans ses prédictions.
                """)   

   with tab3:
    st.markdown("#### Entraînement sans Duration")
    st.write("")     
    st.markdown("##### Performances")
    # Chemin de l'image
    image_path_perfsd = "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC_PROD/main/img/perfsd.png"
    #Afficher l'image
    st.image(image_path_perfsd)
    st.write("")     
    st.markdown("###### Constat")
    st.markdown("""
Sans la variable Duration, les scores et performances sont nettement moins bons, 
ce qui démontre le poids de cette variable dans la modélisation prédictive.
\nNous allons chercher à optimiser les modèles avec une grille de recherche les meilleurs hyperparamètres sur les 3 modèles les plus performants à l'aide d'une GridSearchCV. 
\nRegardons juste avant les **Feature importances** sans la prépondérance de duration. 
                """)
    st.divider()
    st.write("")
    st.markdown("##### Feature importances")        
    st.markdown("Pour les 3 modèles les plus performants")
    # Créez un menu déroulant pour sélectionner l'image à afficher
    option2 = st.selectbox(
       'Choisir le modèle à afficher:',
       ['Random Forest', 'XGBoost', 'LightGBM'],
       key='feature_importances_selectbox'
       )
    # Définir les chemins des images
    image_paths2 = {
       'Random Forest': "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC_PROD/main/img/fisd_rf.png",
       'XGBoost': "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC_PROD/main/img/fisd_xg.png",
       'LightGBM': "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC_PROD/main/img/fisd_gbm.jpg"
       }
    # Afficher l'image en fonction de la sélection
    st.image(image_paths2[option2])
    st.markdown("###### Constat")
    st.markdown("""
- La variable **'Balance'** a pris l'ascendant de la feature la plus importante pour les modèles LightGBM et Random Forest. 
- La variable **"poutcome_success'** est la plus importante pour le modèle XGBoost.
                """)

   with tab4:
    st.markdown("#### GridSearchCV")
    # Créez un menu déroulant pour sélectionner le modèle
    model_option = st.selectbox(
        '**Afficher les meilleurs paramètres des modèles** :',
        ['Random Forest', 'XGBoost', 'LightGBM'],
        key='model_selectbox'
    )
    # Définir les textes pour chaque modèle
    model_texts = {
        'Random Forest': """
        Random Forest est un modèle d'arbres décisionnels qui utilise plusieurs arbres pour améliorer la précision et éviter le surapprentissage.
        - Temps d'exécution : 4min 5s
        - Meilleurs paramètres trouvés pour Random Forest: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
        - Meilleur score : 0.7336
        """,
        'XGBoost': """
        XGBoost est une implémentation de gradient boosting qui est efficace et performante pour les tâches de classification et de régression.
        - Temps d'exécution : 1min 29s
        - Meilleurs paramètres trouvés pour XGBoost : {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 300, 'subsample': 0.8}
        - Meilleur score : 0.7360
        """,
        'LightGBM': """
        LightGBM est un cadre de gradient boosting basé sur les arbres qui est conçu pour être distribué et efficace avec une grande capacité de données.
        - Temps d'exécution : 35s
        - Meilleurs paramètres trouvés pour LightGBM : {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'n_estimators': 200, 'num_leaves': 31, 'subsample': 0.8}
        - Meilleur score : 0.7373
        """
    }
    # Afficher le texte correspondant au modèle sélectionné
    st.markdown(model_texts[model_option])

   st.divider()
   st.write()
   st.header(" Entrainement des 3 modèles avec Hyperparamètres")

   def binary_encoding_func(x):
    return x.replace({'yes': 1, 'no': 0})   
   def pdays_contact_lambda(x):
      return 'no' if x == -1 else 'yes'
   def pdays_days_lambda(x):
      return 0 if x == -1 else x

   # Fonction de prétraitement des données
   @st.cache_data
   def preprocess_data2(df):
    # Discrétisation de la variable 'age' avec labels ordinaux
    bins = [18, 29, 40, 50, 60, 96]
    labels = [1, 2, 3, 4, 5]
    df['age_cat'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    
    # Conversion de 'month' en int en utilisant un mapping
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month'] = df['month'].map(month_mapping)
    # Diviser la colonne 'pdays' en deux colonnes
    df['pdays_contact'] = df['pdays'].apply(pdays_contact_lambda)
    df['pdays_days'] = df['pdays'].apply(pdays_days_lambda)
    
    # Séparation de features et target
    features = df.drop(columns=['deposit', 'age', 'pdays', 'duration'])
    target = df['deposit']
    
    # Séparation en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    
    # Encodage cyclique pour day et month
    def encode_cyclic(df, column, max_value):
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column + '_sin'] = np.sin(2 * np.pi * df[column] / max_value)
        df[column + '_cos'] = np.cos(2 * np.pi * df[column] / max_value)
        return df
    
    X_train = encode_cyclic(X_train, 'day', 31)
    X_train = encode_cyclic(X_train, 'month', 12)
    X_test = encode_cyclic(X_test, 'day', 31)
    X_test = encode_cyclic(X_test, 'month', 12)

    # Définition des features
    binary_features = ['default', 'housing', 'loan', 'pdays_contact']
    categorical_features = ['job', 'marital', 'contact', 'poutcome']
    ordinal_features = ['education', 'age_cat']
    numerical_features = ['balance', 'campaign', 'previous', 'pdays_days']

    # Pipeline pour les variables binaires
    binary_pipeline = Pipeline([
        ('binary_encoding', FunctionTransformer(binary_encoding_func))
        ])

    # Pipeline pour les variables catégorielles
    categorical_pipeline = Pipeline([
       ('onehot_encoding', OneHotEncoder(handle_unknown='ignore'))
       ])

    # Pré-traitement de education (remplacer 'unknown' par la valeur la plus fréquente)
    most_frequent_education = X_train['education'].mode()[0]
    X_train['education'] = X_train['education'].replace('unknown', most_frequent_education)
    X_test['education'] = X_test['education'].replace('unknown', most_frequent_education)

    # Pipeline pour l'encodage ordinal de 'education' et age_cat
    education_categories = ['primary', 'secondary', 'tertiary']
    age_cat_categories = [1, 2, 3, 4, 5]
    ordinal_pipeline = Pipeline([
        ('ordinal_encoding', OrdinalEncoder(categories=[education_categories, age_cat_categories]))
        ])

    # Pipeline pour les variables numériques
    numerical_pipeline = Pipeline([
        ('scaler', RobustScaler())
        ])

    # Pipeline pour les variables cycliques (si besoin de transformations spécifiques)
    cyclic_pipeline = Pipeline([
       ('passthrough', 'passthrough')  # Passe les colonnes sans transformation supplémentaire
       ])
    
    # Pré-processing complet avec ColumnTransformer
    preprocessor2 = ColumnTransformer([
        ('binary', binary_pipeline, binary_features),
        ('categorical', categorical_pipeline, categorical_features),
        ('ordinal', ordinal_pipeline, ordinal_features),
        ('cyclic', cyclic_pipeline, ['day_sin', 'day_cos', 'month_sin', 'month_cos']),
        ('numerical', numerical_pipeline, numerical_features),
    ])

    # Application du preprocessor2 sur X_train et X_test
    X_train_processed2 = preprocessor2.fit_transform(X_train)
    X_test_processed2 = preprocessor2.transform(X_test)

    # Encodage de la variable cible 'deposit'
    label_encoder = LabelEncoder()
    y_train_processed2 = label_encoder.fit_transform(y_train)
    y_test_processed2 = label_encoder.transform(y_test)

    return X_train_processed2, X_test_processed2, y_train_processed2, y_test_processed2, preprocessor2, label_encoder
   

   # Fonction pour entraîner et évaluer les modèles
   @st.cache_data
   def train_and_evaluate_models(X_train, X_test, y_train, y_test):
      params_rf = {
         'max_depth': 10,
         'max_features': 'sqrt',
         'min_samples_leaf': 1,
         'min_samples_split': 2,
         'n_estimators': 100,
         'random_state': 42
         }
      params_xg = {
         'colsample_bytree': 0.8,
         'learning_rate': 0.01,
         'max_depth': 7,
         'n_estimators': 300,
         'subsample': 0.8,
         'random_state': 42
         }
      params_light = {
         'colsample_bytree': 0.8,
         'learning_rate': 0.01,
         'n_estimators': 200,
         'num_leaves': 31,
         'subsample': 0.8,
         'random_state': 42
         }
      
      models = {
         'Random Forest': RandomForestClassifier(**params_rf),
         'XGBoost': XGBClassifier(**params_xg),
         'LightGBM': LGBMClassifier(**params_light)
         }
      
      results = {}
      for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Classification Report': report,
            'Confusion Matrix': cm,
            'Predictions': y_pred
        }
        
        joblib.dump(model, f"{model_name}_model.joblib")

      return results


   # Charger les données
   df = pd.read_csv("bank.csv")
   # Prétraiter les données
   X_train_processed2, X_test_processed2, y_train_processed2, y_test_processed2, preprocessor2, label_encoder = preprocess_data2(df)

   # Entraîner et évaluer les modèles
   results = train_and_evaluate_models(X_train_processed2, X_test_processed2, y_train_processed2, y_test_processed2)
   # Affichage des résultats
   st.write("### Récapitulatif des performances des modèles")
   results_df = pd.DataFrame({
      k: {metric: v for metric, v in results.items() if metric not in ['Classification Report', 'Confusion Matrix', 'Predictions']} 
      for k, results in results.items()})
   results_df_rounded = results_df.round(3)
   # Affichage des métriques de performance dans un tableau
   st.dataframe(results_df_rounded.transpose())
   
   #Affichage des rapports de classification et des matrices de confusion pour chaque modèle
   for model_name, metrics in results.items():
        st.write(f"#### Modèle : {model_name}")
        st.write("**Rapport de classification :**")
        st.text(metrics['Classification Report'])

        st.write("**Matrice de confusion :**")
        st.write(pd.crosstab(index=metrics['Predictions'], columns=y_test_processed2, rownames=['Classes prédites'], colnames=['Classes réelles']))

   # Affichage du graphique des performances
   results_df = results_df.transpose()
   results_long_df = results_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
   results_long_df.rename(columns={'index': 'Model'}, inplace=True)
   results_long_df['Score'] = results_long_df['Score'].round(3)
   
   fig = px.bar(
      results_long_df,
      x='Score',
      y='Model',
      color='Metric',
      barmode='group',
      orientation='h',
      title='<b>Métriques de performance des modèles</b>',
      color_discrete_sequence=px.colors.qualitative.Pastel
      )
   fig.update_layout(
      xaxis_title='Scores',
      yaxis_title='Modèles',
      legend_title='Metrics',
      height=800,
      template='plotly_white'
      )
   
   fig.update_traces(marker_line_width=1.5)
   st.plotly_chart(fig)

   st.divider()
   
   # Section pour afficher les importances des caractéristiques
   st.header("Importances des caractéristiques")
   # Charger les modèles entraînés
   models = {
      'Random Forest': joblib.load("Random Forest_model.joblib"),
      'XGBoost': joblib.load("XGBoost_model.joblib"),
      'LightGBM': joblib.load("LightGBM_model.joblib")
      }
   # Choix du modèle pour voir les importances des caractéristiques
   model_choice = st.selectbox("Choisissez un modèle pour voir les importances des caractéristiques", list(models.keys()))
   
   # Récupérer le modèle sélectionné
   selected_model = models[model_choice]
   
   # Récupérer les noms des caractéristiques 
   feature_names = [
      'default', 'housing', 'loan', 'pdays_contact',
      'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
      'job_management', 'job_retired', 'job_self-employed', 'job_services',
      'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
      'marital_divorced', 'marital_married', 'marital_single',
      'contact_cellular', 'contact_telephone', 'contact_unknown',
      'poutcome_failure', 'poutcome_other', 'poutcome_success',
      'poutcome_unknown', 'education', 'age_cat', 'day_sin', 'day_cos',
      'month_sin', 'month_cos', 'balance', 'campaign', 'previous',
      'pdays_days'
      ]
   
   # Afficher les importances des caractéristiques si elles existent
   if hasattr(selected_model, 'feature_importances_'):
    importances = selected_model.feature_importances_
    if len(feature_names) == len(importances):
       feature_importances_df = pd.DataFrame({
          'Feature': feature_names,
          'Importance': importances
          }).sort_values(by='Importance', ascending=False)
       
       
       st.write(f"### Importances des caractéristiques pour le modèle {model_choice}")
       
       # Visualisation des importances
       fig = px.bar(feature_importances_df,
                    x='Importance', y='Feature', 
                    orientation='h', title=f'Importances des caractéristiques - {model_choice}',
                    color='Feature', color_discrete_sequence=px.colors.qualitative.Pastel)
       # Ajustements du graphique
       fig.update_layout(yaxis={'categoryorder': 'total ascending'})
       st.plotly_chart(fig)
       
       # Affichage du DataFrame des importances des caractéristiques
       st.dataframe(feature_importances_df)
    else:
       st.write(f"Erreur : Le nombre de caractéristiques ({len(feature_names)}) ne correspond pas au nombre d'importances des caractéristiques ({len(importances)}).")
   else:
      st.write(f"Le modèle {model_choice} ne supporte pas l'extraction des importances des caractéristiques.")




if page == pages[5] : 
    st.title("Conclusion et Perspective")

    st.markdown("""
    L’analyse des données a permis de mieux cibler les clients potentiellement souscripteurs et l'utilisation d’algorithme de Machine Learning (ML) peut offrir de nombreuses opportunités pour la banque, pour prédire notamment les dits-souscripteurs.""")
    st.markdown("\n")

    st.subheader("Algorithme de Machine Learning à privilégier lors des prochaines campagnes") 
    st.markdown("""
    En prenant en compte la durée de l’appel, les modèles sont très performants. 
    Néanmoins, il est à privilégier un modèle un peu moins performant mais sans aucun doute plus prédictif du potentiel client souscripteur (sans la donnée de durée d’appel). 

    Nous recommandons ainsi d’utiliser les modèles de ML optimisés, tels que **XGBoost** ou **LightGBM**, pour prédire les clients susceptibles de souscrire à ce type de produit bancaire ‘dépôt à terme’. L’un ou l’autre permettra une prédiction correcte en moyenne de 74% à 75% des clients susceptibles de souscrire au produit.""")


    st.subheader("Recommandations pour améliorer les modèles prédictifs") 
    st.markdown("""
    Les modèles se nourrissant de la data, il faut noter que leurs performances seront optimisées à chaque campagne et avec l’apport de nouvelles données. 
    - Aussi il peut être intéressant d’envisager un apport de données par des sources supplémentaires comme certaines données transactionnelles, ou les interactions sur les réseaux sociaux.
    - Il serait également utile d’enrichir les données en recueillant et analysant les feedbacks et retours clients.  Les clients peuvent fournir des informations précieuses sur les raisons pour lesquelles ils ont souscrit ou non au produit.
    Ces données enrichies pourraient ainsi améliorer l’entrainement des modèles prédictifs.""")
    st.markdown("\n")

    st.subheader("Profil client à cibler en priorité") 
    image_PersonaCible = "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC/main/img/Persona Cible.png"
    st.image(image_PersonaCible)
    #st.markdown(
    #    f"""
    #    <style>
    #   .center {{
    #        display: block;
    #        margin-left: auto;
    #        margin-right: auto;
    #    }}
    #    </style>
    #    <img src="{image_PersonaCible}" class="center">
    #    """,
     #   unsafe_allow_html=True
    #)
    st.markdown("\n")
    st.subheader("Stratégie et optimisation continue") 
    st.markdown("""
    Nous recommandons donc l’utilisation d’un modèle prédictif de type **LightGBM** ou **XGBoost** selon les contraintes d’infrastructure et de données pour la prochaine campagne. 
    Il permettra de cibler une bonne part de clients susceptibles de souscrire. 
    Les résultats de la prochaine campagne permettront alors d’alimenter le système prédictif mis en place et de l’optimiser dans un processus continu. 

    Coupler à une bonne sensibilisation des agents marketing à la cible client et à l’amélioration du démarchage telle que recommandé, la prochaine campagne devrait connaître de meilleurs résultats. 

    L’analyse des prochains résultats à chaque campagne permettra un processus d’amélioration continue pour toujours mieux cibler les clients, augmenter le taux de souscription au dépôt à terme et assurer une meilleure efficacité des futures campagnes.""")
    st.markdown("\n")


