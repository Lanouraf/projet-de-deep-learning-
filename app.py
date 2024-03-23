import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google_drive_downloader import GoogleDriveDownloader as gdd
from Sequences import prep_data
import torch
from LNtrain import LNtest
from LNModule import BagOfWordsClassifier, BagOfWordsClassifierLayer, BagOfWordsClassifierLayerHM

def homemade_layernorm():
    st.title("Homemade Layer Normalization")
    train_dataset, train_loader, test_dataset, test_loader = prep_data()
    
    # Téléchargement des fichiers depuis Google Drive
    gdd.download_file_from_google_drive(file_id='1lESMtwwy_qzfWtpfpGDQKzLOkABrIyp', dest_path='./losses.pth')
    
    # Chargement des pertes depuis le fichier losses.pth
    losses = torch.load('./losses.pth')
    
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

    # Affichage des pertes pour les modèles sélectionnés
    st.subheader("Losses")
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
    st.subheader("Accuracies")
    for model_name in selected_models:
        key = model_names.get(model_name)
        if key in accuracies_dict:
            st.write(f"Accuracy for {model_name}: {accuracies_dict[key]}")

def main():
    st.title("Batch and Layer Normalization")
    app_mode = st.sidebar.selectbox(
        "Choose the experiment",
        [
            "Home",
            "Homemade Layer Normalization"
        ],
    )
    if app_mode == "Home":
        st.write("To continue select a mode in the selection box to the left.")
    elif app_mode == "Homemade Layer Normalization":
        homemade_layernorm()

if __name__ == "__main__":
    main()
