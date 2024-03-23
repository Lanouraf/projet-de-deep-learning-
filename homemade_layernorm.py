import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google_drive_downloader import GoogleDriveDownloader as gdd
from Sequences import prep_data
import torch
from LNtrain import LNtest
from LNModule import BagOfWordsClassifier, BagOfWordsClassifierLayer, BagOfWordsClassifierLayerHM
from DATALOAD import data_review
import json

st.set_option('deprecation.showPyplotGlobalUse', False)

def homemade_layernorm():
    st.title("Homemade Layer Normalization")
    
    # Afficher le DataFrame avec les données de review
    st.subheader("Data Review")
    df = data_review()  # Charger les données depuis un fichier CSV
    st.dataframe(df.head())  # Afficher les premières lignes du DataFrame
    
    # Instructions sur le but de la page
    st.write("Nous voulons prédire le sentiment d'un review, où 0 correspond à un sentiment négatif et 1 à un sentiment positif.")
    
    # Instructions sur les modèles entraînés
    st.write("Nous avons entraîné trois modèles :")
    st.write("- Un modèle simple Bag of Words")
    st.write("- Un modèle qui applique la layer norm de PyTorch")
    st.write("- Un modèle qui applique la layer norm implémentée à la main")
    
    train_dataset, train_loader, test_dataset, test_loader = prep_data()
    
    # Chargement des pertes depuis le fichier losses.csv
    losses_df = pd.read_csv("losses.csv")
    
    # Lecture des accuracies depuis le fichier accuracies.csv dans votre répertoire Git
    accuracies_df = pd.read_csv("accuracies.csv", header=None, index_col=0)  # Charger sans utiliser la première colonne comme index
    
    # Création d'un dictionnaire à partir des données du fichier CSV
    accuracies_dict = accuracies_df.to_dict()[1]

    # Associer les noms des modèles aux clés des accuracies
    model_names = {
        'Simple Bag of Words': 'BOW',
        'Bag of Words with Torch LN': 'BOW_LN',
        'Bag of Words with Homemade LN': 'BOW_HM_LN'
    }

    # Sélection des modèles à comparer
    selected_models_ui = list(model_names.keys())
    selected_models_keys = [model_names[name] for name in selected_models_ui]
    selected_models = st.multiselect("Select models to compare", selected_models_ui)

    st.subheader("Loss")
    for model_name in selected_models:
        key = model_names.get(model_name)
        if key in losses_df['Model'].values:
            loss_values = json.loads(losses_df[losses_df['Model'] == key]['Loss'].iloc[0])
            plt.plot(loss_values, label=model_name)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot()

    # Affichage des accuracies pour les modèles sélectionnés
    st.subheader("Accuracy")
    for model_name in selected_models:
        key = model_names.get(model_name)
        if key in accuracies_dict:
            st.write(f"Accuracy for {model_name}: {accuracies_dict[key]}")

    # Affichage des courbes de perte pour les modèles sélectionnés


