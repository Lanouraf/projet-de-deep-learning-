"""
1. Définition de la fonction 'homemade_layernorm':
   - Cette fonction crée une application Streamlit pour visualiser les performances des modèles de classification de sentiment.
   - Elle affiche les données de review, les instructions sur le but de la page et les modèles entraînés.

2. Chargement des données et des modèles :
   - Les données de review sont chargées à partir d'un fichier CSV.
   - Trois modèles de classification de sentiment sont chargés : Bag of Words simple, Bag of Words avec Layer Norm de PyTorch, et Bag of Words avec Layer Norm implémentée à la main.
   - Les pertes et les accuracies des modèles sont également chargées à partir de fichiers.

3. Affichage des graphiques et des informations :
   - Un graphique des pertes est affiché pour chaque modèle sélectionné.
   - Les accuracies sont affichées pour chaque modèle sélectionné.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google_drive_downloader import GoogleDriveDownloader as gdd
from LNprepdata import prep_data
import torch
from LNModule import BagOfWordsClassifier, BagOfWordsClassifierLayer, BagOfWordsClassifierLayerHM
from LN_dataload import data_review

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
    gdd.download_file_from_google_drive(file_id='1lESMtwwy_qzfWtpfpGDQYKzLOkABrIyp', dest_path='./losses.pth', overwrite=True, showsize=True)
    losses=torch.load('losses.pth')
    # Lecture des accuracies depuis le fichier accuracies.csv dans votre répertoire Git
    gdd.download_file_from_google_drive(file_id='1Yng2XlYhJtC7HBr8LtzJGiRQijFLQr0z', dest_path='./accuracies.csv', overwrite=True, showsize=True)
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
    fig_loss, ax_loss = plt.subplots()
    for model_name in selected_models:
        key = model_names.get(model_name)
        if key in losses:
            ax_loss.plot(losses[key], label=model_name)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    st.pyplot(fig_loss)

    # Affichage des accuracies pour les modèles sélectionnés
    st.subheader("Accuracy")
    for model_name in selected_models:
        key = model_names.get(model_name)
        if key in accuracies_dict:
            st.write(f"Accuracy for {model_name}: {accuracies_dict[key]}")

    # Affichage des courbes de perte pour les modèles sélectionnés


