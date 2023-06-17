#Dashboard projet 7

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import shap

# DATASETS

FILE_BEST_MODELE = 'C:\\Users\\Marwa\\Downloads\\Projet7\\best_model.pickle'
FILE_TEST_SET = 'C:\\Users\\Marwa\\Downloads\\Projet7\\test_fs_lightgbm_80.pickle'
FILE_DASHBOARD = 'C:\\Users\\Marwa\\Downloads\\Projet7\\dashboard\\df_dashboard.pickle'
FILE_CLIENT_INFO = 'C:\\Users\\Marwa\\Downloads\\Projet7\\dashboard\\df_info_client.pickle'
FILE_CLIENT_PRET = 'C:\\Users\\Marwa\\Downloads\\Projet7\\dashboard\\df_pret_client.pickle'
FILE_VOISINS_INFO = 'C:\\Users\\Marwa\\Downloads\\Projet7\\dashboard\\df_info_voisins.pickle'
FILE_VOISIN_AGG = 'C:\\Users\\Marwa\\Downloads\\Projet7\\dashboard\\df_voisin_train_agg.pickle'
FILE_ALL_TRAIN_AGG = 'C:\\Users\\Marwa\\Downloads\\Projet7\\dashboard\\df_all_train_agg.pickle'

# ====================================================================
# VARIABLES
# ====================================================================
group_val1 = ['AMT_ANNUITY',
              'INST_PAY_DAYS_PAYMENT_RATIO_MAX',
              'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN',
              'INST_PAY_AMT_INSTALMENT_SUM',
              'PREV_APP_INTEREST_SHARE_MAX']

group_val2 = ['EXT_SOURCE_SUM',
            'CODE_GENDER',
            'CREDIT_GOODS_RATIO',
            'CREDIT_ANNUITY_RATIO',
            'NAME_EDUCATION_TYPE_HIGHER_EDUCATION',
            'NAME_FAMILY_STATUS_MARRIED',
            'EXT_SOURCE_1',
            'DAYS_BIRTH']

group_val3 = ['AMT_ANNUITY_MEAN',
              'INST_PAY_DAYS_PAYMENT_RATIO_MAX_MEAN',
              'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN_MEAN',
              'INST_PAY_AMT_INSTALMENT_SUM_MEAN',
              'PREV_APP_INTEREST_SHARE_MAX_MEAN']

group_val4 = ['EXT_SOURCE_SUM_MEAN',
            'CODE_GENDER_MEAN',
            'CREDIT_GOODS_RATIO_MEAN',
            'CREDIT_ANNUITY_RATIO_MEAN',
            'NAME_EDUCATION_TYPE_HIGHER_EDUCATION_MEAN',
            'NAME_FAMILY_STATUS_MARRIED_MEAN',
            'EXT_SOURCE_1_MEAN',
            'DAYS_BIRTH_MEAN']

# ====================================================================
# IMAGES
# ====================================================================
# Légende des courbes
legende =  Image.open("C:\\Users\\Marwa\\Downloads\\Projet7\\images\\legende.png") 

# ====================================================================
# HTML MARKDOWN
# ====================================================================
html_AMT_ANNUITY = "<h4 style='text-align: center'>AMT_ANNUITY</h4> <br/> <h5 style='text-align: center'>Annuité du prêt</h5> <hr/>"
html_BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN = "<h4 style='text-align: center'>BUREAU_CURRENT_CREDIT_DEBT_DIFF_MIN</h4> <br/> <h5 style='text-align: center'>Valeur minimale de la différence entre la limite de crédit actuelle de la carte de crédit et la dette actuelle sur le crédit</h5> <hr/>" 
html_BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN = "<h4 style='text-align: center'>BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN</h4> <br/> <h5 style='text-align: center'>Valeur moyenne de la différence entre la limite de crédit actuelle de la carte de crédit et la dette actuelle sur le crédit</h5> <hr/>" 
html_INST_PAY_AMT_INSTALMENT_SUM = "<h4 style='text-align: center'>INST_PAY_AMT_INSTALMENT_SUM</h4> <br/> <h5 style='text-align: center'>Somme du montant de l'acompte prescrit des crédits précédents sur cet acompte</h5> <hr/>" 
html_BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN = "<h4 style='text-align: center'>BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN</h4> <br/> <h5 style='text-align: center'>Moyenne du ratio des prêts précédents sur d'autres institution de : la dette actuelle sur le crédit et la limite de crédit actuelle de la carte de crédit (valeur * 100)</h5> <hr/>" 
html_CAR_EMPLOYED_RATIO = "<h4 style='text-align: center'>CAR_EMPLOYED_RATIO</h4> <br/> <h5 style='text-align: center'>Ratio : Âge de la voiture du demandeur / Ancienneté dans l'emploi à la date de la demande (valeur * 1000)</h5> <hr/>" 
html_CODE_GENDER = "<h4 style='text-align: center'>CODE_GENDER</h4> <br/> <h5 style='text-align: center'>Sexe</h5> <hr/>" 
html_CREDIT_ANNUITY_RATIO = "<h4 style='text-align: center'>CREDIT_ANNUITY_RATIO</h4> <br/> <h5 style='text-align: center'>Ratio : montant du crédit du prêt / Annuité de prêt</h5> <hr/>" 
html_CREDIT_GOODS_RATIO = "<h4 style='text-align: center'>CREDIT_GOODS_RATIO</h4> <br/> <h5 style='text-align: center'>Ratio : Montant du crédit du prêt / prix des biens pour lesquels le prêt est accordé / Crédit est supérieur au prix des biens ? (valeur * 100)</h5> <hr/>" 
html_YEAR_BIRTH = "<h4 style='text-align: center'>YEAR_BIRTH</h4> <br/> <h5 style='text-align: center'>Âge (ans)</h5> <hr/>" 
html_YEAR_ID_PUBLISH = "<h4 style='text-align: center'>YEAR_ID_PUBLISH</h4> <br/> <h5 style='text-align: center'>Combien de jours avant la demande le client a-t-il changé la pièce d'identité avec laquelle il a demandé le prêt ? (ans)</h5> <hr/>" 
html_EXT_SOURCE_1 = "<h4 style='text-align: center'>EXT_SOURCE_1</h4> <br/> <h5 style='text-align: center'>Source externe normalisée (valeur * 100)</h5> <hr/>" 
html_EXT_SOURCE_SUM = "<h4 style='text-align: center'>EXT_SOURCE_SUM</h4> <br/> <h5 style='text-align: center'>Somme des 3 sources externes normalisées (EXT_SOURCE_1, EXT_SOURCE_2 et EXT_SOURCE_3, valeur * 100)</h5> <hr/>" 
html_NAME_EDUCATION_TYPE_HIGHER_EDUCATION = "<h4 style='text-align: center'>NAME_EDUCATION_TYPE_HIGHER_EDUCATION</h4> <br/> <h5 style='text-align: center'>Niveau d'éducation le plus élévé</h5> <hr/>" 
html_PREV_APP_INTEREST_SHARE_MAX = "<h4 style='text-align: center'>PREV_APP_INTEREST_SHARE_MAX</h4> <br/> <h5 style='text-align: center'>La valeur maximale de tous les précédents crédit dans d'autres institution : de la durée du crédit multiplié par l'annuité du prêt moins le montant final du crédit</h5> <hr/>" 

# ====================================================================
# HEADER - TITRE
# ====================================================================
#Title display
html_header="""
     <div style="background-color: Gold; padding:10px; border-radius:10px">
     <h1 style="color: white; text-align:center">Dashboard Scoring Credit</h1>
     </div>
     <p style="font-size: 25px; font-weight: bold; text-align:center">Projet 7 : Implémentez un modèle de scoring</p>
     """
st.markdown(html_header, unsafe_allow_html=True)


# CHARGEMENT DES DONNEES
# ====================================================================


# Chargement du modèle et des différents dataframes
# Optimisation en conservant les données non modifiées en cache mémoire
@st.cache(allow_output_mutation = True)
def load():
    with st.spinner('Chargement des données'):
        
        # Import du dataframe des informations des traits stricts du client
        fic_client_info = FILE_CLIENT_INFO
        with open(fic_client_info, 'rb') as df_info_client:
            df_info_client = pickle.load(df_info_client)    
            
        # Import du dataframe des informations sur le prêt du client
        fic_client_pret = FILE_CLIENT_PRET
        with open(fic_client_pret, 'rb') as df_pret_client:
            df_pret_client = pickle.load(df_pret_client)
            
        # Import du dataframe des informations des traits stricts des voisins
        fic_voisin_info = FILE_VOISINS_INFO
        with open(fic_voisin_info, 'rb') as df_info_voisins:
            df_info_voisins = pickle.load(df_info_voisins)
            

        # Import du dataframe des informations sur le dashboard
        fic_dashboard = FILE_DASHBOARD
        with open(fic_dashboard, 'rb') as df_dashboard:
            df_dashboard = pickle.load(df_dashboard)

        # Import du dataframe des informations sur les voisins aggrégés
        fic_voisin_train_agg = FILE_VOISIN_AGG
        with open(fic_voisin_train_agg, 'rb') as df_voisin_train_agg:
            df_voisin_train_agg = pickle.load(df_voisin_train_agg)

        # Import du dataframe des informations sur les voisins aggrégés
        fic_all_train_agg = FILE_ALL_TRAIN_AGG
        with open(fic_all_train_agg, 'rb') as df_all_train_agg:
            df_all_train_agg = pickle.load(df_all_train_agg)

        # Import du dataframe des informations sur les voisins aggrégés
        with open(FILE_TEST_SET, 'rb') as df_test_set:
            test_set = pickle.load(df_test_set)

                     
    # Import du meilleur modèle lgbm entrainé
    with st.spinner('Import du modèle'):
        
        # Import du meilleur modèle lgbm entrainé
        fic_best_model = FILE_BEST_MODELE
        with open(fic_best_model, 'rb') as model_lgbm:
            best_model = pickle.load(model_lgbm)
  
    # SHAP values
    with st.spinner('Chargement SHAP values'):
   
        # Test set sans l'identifiant
        X_bar = test_set.set_index('SK_ID_CURR')
        # Entraînement de shap sur le train set
        bar_explainer = shap.Explainer(best_model, X_bar)
        bar_values = bar_explainer(X_bar, check_additivity=False)
                    
    return df_info_client, df_pret_client, df_info_voisins, \
        df_dashboard, df_voisin_train_agg, df_all_train_agg, test_set, \
            best_model, bar_values

# Chargement des dataframes et du modèle
df_info_client, df_pret_client, df_info_voisins, \
    df_dashboard, df_voisin_train_agg, df_all_train_agg, test_set, \
             best_model, bar_values = load()



# ====================================================================
# CHOIX DU CLIENT
# ====================================================================

html_select_client="""
    <div class="card">
      <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #fff; padding-top: 6px; width: auto;
                  height: 40px;">
        <h3 class="card-title" style="background-color:#fff; color:Black;
                   font-family:Georgia; text-align: center; padding: 0px 0;">
          Informations descriptives des clients
        </h3>
      </div>
    </div>
    """

st.markdown(html_select_client, unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns([1,3])
    with col1:
        st.write("")
        col1.header("**ID Client**")
        client_id = col1.selectbox('Sélectionnez un client :',
                                   df_info_voisins['ID_CLIENT'].unique())
    with col2:
        # Infos principales client
        # st.write("*Traits stricts*")
        client_info = df_info_client[df_info_client['SK_ID_CURR'] == client_id].iloc[:, :]
        client_info.set_index('SK_ID_CURR', inplace=True)
        st.table(client_info)
        # Infos principales sur la demande de prêt
        # st.write("*Demande de prêt*")
        client_pret = df_pret_client[df_pret_client['SK_ID_CURR'] == client_id].iloc[:, :]
        client_pret.set_index('SK_ID_CURR', inplace=True)
        st.table(client_pret)


# ====================================================================
# SCORE - PREDICTIONS
# ====================================================================

html_score="""
    <div class="card">
      <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: white; padding-top: 5px; width: auto;
                  height: 40px;">
        <h3 class="card-title" style="background-color: white; color:Black;
                   font-family:Georgia; text-align: center; padding: 0px 0;">
          Crédit Score
        </h3>
      </div>
    </div>
    """

st.markdown(html_score, unsafe_allow_html=True)

# Préparation des données à afficher dans la jauge ==============================================

# ============== Score du client en pourcentage ==> en utilisant le modèle ======================
# Sélection des variables du clients
X_test = test_set[test_set['SK_ID_CURR'] == client_id]
# Score des prédictions de probabiltés
y_proba = best_model.predict_proba(X_test.drop('SK_ID_CURR', axis=1))[:, 1]
# Score du client en pourcentage arrondi et nombre entier
score_client = int(np.rint(y_proba * 100))

# ============== Score moyen des 10 plus proches voisins du test set en pourcentage =============

# Score moyen des 10 plus proches voisins du test set en pourcentage
score_moy_voisins_test = int(np.rint(df_dashboard[
    df_dashboard['SK_ID_CURR'] == client_id]['SCORE_10_VOISINS_MEAN_TEST'] * 100))

# ============== Pourcentage de clients voisins défaillants dans l'historique des clients =======
pourc_def_voisins_train = int(np.rint(df_dashboard[
    df_dashboard['SK_ID_CURR'] == client_id]['%_NB_10_VOISINS_DEFAILLANT_TRAIN']))

# ============== Pourcentage de clients voisins défaillants prédits parmi les nouveaux clients ==
pourc_def_voisins_test = int(np.rint(df_dashboard[
    df_dashboard['SK_ID_CURR'] == client_id]['%_NB_10_VOISINS_DEFAILLANT_TEST']))


# Graphique de jauge du cédit score ==========================================
fig_jauge = go.Figure(go.Indicator(
    mode = 'gauge+number+delta',
    # Score du client en % df_dashboard['SCORE_CLIENT_%']
    value = score_client,  
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': 'Crédit score du client', 'font': {'size': 24}},
    # Score des 10 voisins test set
    # df_dashboard['SCORE_10_VOISINS_MEAN_TEST']
    delta = {'reference': score_moy_voisins_test,
             'increasing': {'color': 'Black'},
             'decreasing': {'color': 'Green'}},
    gauge = {'axis': {'range': [None, 100],
                      'tickwidth': 3,
                      'tickcolor': 'Black'},
             'bar': {'color': 'white', 'thickness' : 0.25},
             'bgcolor': 'white',
             'borderwidth': 2,
             'bordercolor': 'gray',
             'steps': [{'range': [0, 25], 'color': 'Green'},
                       {'range': [25, 49.49], 'color': 'LimeGreen'},
                       {'range': [49.5, 50.5], 'color': 'red'},
                       {'range': [50.51, 75], 'color': 'Orange'},
                       {'range': [75, 100], 'color': 'Blue'}],
             'threshold': {'line': {'color': 'white', 'width': 10},
                           'thickness': 0.8,
                           # Score du client en %
                           # df_dashboard['SCORE_CLIENT_%']
                           'value': score_client}}))

fig_jauge.update_layout(paper_bgcolor='white',
                        height=400, width=400,
                        font={'color': 'Black', 'family': 'Arial'},
                        margin=dict(l=0, r=0, b=0, t=0, pad=0))

with st.container():
    # JAUGE + récapitulatif du score moyen des voisins
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.plotly_chart(fig_jauge)
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        # Texte d'accompagnement de la jauge
        if 0 <= score_client < 25:
            score_text = 'Crédit score : EXCELLENT'
            st.success(score_text)
        elif 25 <= score_client < 50:
            score_text = 'Crédit score : BON'
            st.success(score_text)
        elif 50 <= score_client < 75:
            score_text = 'Crédit score : MOYEN'
            st.warning(score_text)
        else :
            score_text = 'Crédit score : BAS'
            st.error(score_text)
        st.write("")    
        st.markdown(f'Crédit score moyen des 10 clients similaires : **{score_moy_voisins_test}**')
        st.markdown(f'**{pourc_def_voisins_train}**% de clients voisins réellement défaillants dans l\'historique')
        st.markdown(f'**{pourc_def_voisins_test}**% de clients voisins défaillants prédits pour les nouveaux clients')
   

# --------------------------------------------------------------------
# Interpretabilité du modèle : SHAP VALUE
# --------------------------------------------------------------------
    
def interpretabilite():
    ''' Affiche l'interpretabilite du modèle
    '''
    html_interpretabilite="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: white; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:white; color:Black;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      SHAP Values
                  </h3>
            </div>
        </div>
        """
    
    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    if st.checkbox("Interpretabilité du modèle"):     
        
        st.markdown(html_interpretabilite, unsafe_allow_html=True)

        with st.spinner('**Affiche l\'interpretabilité du modèle...**'):                 
                       
            with st.expander('interpretabilité du modèle',
                              expanded=True):
                
                explainer = shap.TreeExplainer(best_model)
                
                client_index = test_set[test_set['SK_ID_CURR'] == client_id].index.item()
                X_shap = test_set.set_index('SK_ID_CURR')
                X_test_courant = X_shap.iloc[client_index]
                X_test_courant_array = X_test_courant.values.reshape(1, -1)
                
                shap_values_courant = explainer.shap_values(X_test_courant_array)
                
                col1, col2 = st.columns([1, 1])
                # BarPlot du client courant
                with col1:

                    plt.clf()
                    

                    # BarPlot du client courant
                    shap.plots.bar(bar_values[client_index], max_display=40)
                    
                    fig = plt.gcf()
                    fig.set_size_inches((10, 20))
                    # Plot the graph on the dashboard
                    st.pyplot(fig)
     
                # Décision plot du client courant
                with col2:
                    plt.clf()

                    # Décision Plot
                    shap.decision_plot(explainer.expected_value[1], shap_values_courant[1],
                                    X_test_courant)
                
                    fig2 = plt.gcf()
                    fig2.set_size_inches((10, 15))
                    # Plot the graph on the dashboard
                    st.pyplot(fig2)
                    
st.subheader('Interpretabilité du modèle : SHAPLEY Values')
interpretabilite()

# ====================================================================
# COMPRAISON CLIENT AUX CLIENTS SIMILAIRES
# ====================================================================

# Toutes Les informations non modifiées du client courant
#df_client_origin = application_test[application_test['SK_ID_CURR'] == client_id]

# Toutes Les informations non modifiées du client courant
df_client_test = test_set[test_set['SK_ID_CURR'] == client_id]

# Toutes les informations du client courant
df_client_courant = df_dashboard[df_dashboard['SK_ID_CURR'] == client_id]


# --------------------------------------------------------------------
# CLIENTS SIMILAIRES 
# --------------------------------------------------------------------
def infos_clients_similaires():
    ''' Affiche les informations sur les clients similaires :
            - traits stricts.
            - demande de prêt
    '''
    html_clients_similaires="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: white; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:white; color:Black;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      Clients similaires
                  </h3>
            </div>
        </div>
        """
    
    titre = True

    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    if st.checkbox("Comparaison client / Clients similaires"):     
        
        if titre:
            st.markdown(html_clients_similaires, unsafe_allow_html=True)
            titre = False

        with st.spinner('**Affiche les graphiques comparant le client courant et les clients similaires...**'):                 
                       
            with st.expander('Comparaison variables impactantes client courant/moyennes des clients similaires',
                             expanded=True):
                with st.container():
                    # Préparatifs dataframe
                    df_client = df_voisin_train_agg[df_voisin_train_agg['ID_CLIENT'] == client_id].astype(int)
                    # ====================================================================
                    # Lineplot comparatif features importances client courant/voisins
                    # ====================================================================
                    
                    # ===================== Valeurs moyennes des features importances pour le client courant =====================
                    df_feat_client  = df_client_courant[['SK_ID_CURR',
                               'EXT_SOURCE_SUM',
                               'INST_PAY_DAYS_PAYMENT_RATIO_MAX',
                               'CODE_GENDER',
                               'CREDIT_GOODS_RATIO',
                               'CREDIT_ANNUITY_RATIO',
                               'PREV_APP_INTEREST_SHARE_MAX',
                               'AMT_ANNUITY',
                               'NAME_EDUCATION_TYPE_HIGHER_EDUCATION',
                               'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN',
                               'NAME_FAMILY_STATUS_MARRIED',
                               'INST_PAY_AMT_INSTALMENT_SUM',
                               'EXT_SOURCE_1',
                               'DAYS_BIRTH']]
                    df_feat_client['YEAR_BIRTH'] = \
                        np.trunc(np.abs(df_feat_client['DAYS_BIRTH'] / 365)).astype('int8')

                    df_feat_client_gp1 = df_feat_client[group_val1]
                    df_feat_client_gp2 = df_feat_client[group_val2]
                    # X
                    x_gp1 = df_feat_client_gp1.columns.to_list()
                    x_gp2 = df_feat_client_gp2.columns.to_list()
                    # y
                    y_feat_client_gp1 = df_feat_client_gp1.values[0].tolist()
                    y_feat_client_gp2 = df_feat_client_gp2.values[0].tolist()
                    
                    
                    # ===================== Valeurs moyennes de tous les clients non-défaillants/défaillants du train sets =======================
                    df_all_train = df_all_train_agg[['TARGET',
                               'EXT_SOURCE_SUM_MEAN',
                               'INST_PAY_DAYS_PAYMENT_RATIO_MAX_MEAN',
                               'CODE_GENDER_MEAN',
                               'CREDIT_GOODS_RATIO_MEAN',
                               'CREDIT_ANNUITY_RATIO_MEAN',
                               'PREV_APP_INTEREST_SHARE_MAX_MEAN',
                               'AMT_ANNUITY_MEAN',
                               'NAME_EDUCATION_TYPE_HIGHER_EDUCATION_MEAN',
                               'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN_MEAN',
                               'NAME_FAMILY_STATUS_MARRIED_MEAN',
                               'INST_PAY_AMT_INSTALMENT_SUM_MEAN',
                               'EXT_SOURCE_1_MEAN',
                               'DAYS_BIRTH_MEAN']]
                    
                    # Non-défaillants
                    df_all_train_nondef_gp3 = df_all_train[df_all_train['TARGET'] == 0][group_val3]
                    df_all_train_nondef_gp4 = df_all_train[df_all_train['TARGET'] == 0][group_val4]
                    # Défaillants
                    df_all_train_def_gp3 = df_all_train[df_all_train['TARGET'] == 1][group_val3]
                    df_all_train_def_gp4 = df_all_train[df_all_train['TARGET'] == 1][group_val4]
                    # y
                    # Non-défaillants
                    y_all_train_nondef_gp3 = df_all_train_nondef_gp3.values[0].tolist()
                    y_all_train_nondef_gp4 = df_all_train_nondef_gp4.values[0].tolist()
                    # Défaillants
                    y_all_train_def_gp3 = df_all_train_def_gp3.values[0].tolist()
                    y_all_train_def_gp4 = df_all_train_def_gp4.values[0].tolist()

                    # Légende des courbes
                    st.image(legende)
                                                  
                    col1, col2 = st.columns([1, 1.5])
                    with col1:
                        # Lineplot de comparaison des features importances client courant/all ================
                        plt.figure(figsize=(6, 6))
                        plt.plot(x_gp1, y_feat_client_gp1, color='Orange')
                        plt.plot(x_gp1, y_all_train_nondef_gp3, color='Green')
                        plt.plot(x_gp1, y_all_train_def_gp3, color='Crimson')
                        plt.xticks(rotation=90)
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                    with col2: 
                        # Lineplot de comparaison des features importances client courant/all ================
                        plt.figure(figsize=(8, 5))
                        plt.plot(x_gp2, y_feat_client_gp2, color='Orange')
                        plt.plot(x_gp2, y_all_train_nondef_gp4, color='Green')
                        plt.plot(x_gp2, y_all_train_def_gp4, color='Crimson')
                        plt.xticks(rotation=90)
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                        
                    with st.container(): 
                        
                        vars_select = ['EXT_SOURCE_SUM',
            'INST_PAY_DAYS_PAYMENT_RATIO_MAX',
            'CODE_GENDER',
            'CREDIT_GOODS_RATIO',
            'CREDIT_ANNUITY_RATIO',
            'PREV_APP_INTEREST_SHARE_MAX',
            'AMT_ANNUITY',
            'NAME_EDUCATION_TYPE_HIGHER_EDUCATION',
            'BUREAU_CURRENT_CREDIT_DEBT_DIFF_MEAN',
            'NAME_FAMILY_STATUS_MARRIED',
            'INST_PAY_AMT_INSTALMENT_SUM',
            'EXT_SOURCE_1',
            'DAYS_BIRTH']

                        feat_imp_to_show = st.multiselect("Feature(s) importance(s) à visualiser : ",
                                                          vars_select)

st.subheader('Clients similaires')
infos_clients_similaires()

# ====================================================================
# FOOTER
# ====================================================================
html_line="""
<br>
<br>
<br>
<br>
<hr style= "  display: block;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
  margin-left: auto;
  margin-right: auto;
  border-style: inset;
  border-width: 3px;">
<p style="color:Black; text-align: center; font-size:20px;">Auteur : marwa.dhifallah@outlook.com - 30/08/2022</p>
"""
st.markdown(html_line, unsafe_allow_html=True)