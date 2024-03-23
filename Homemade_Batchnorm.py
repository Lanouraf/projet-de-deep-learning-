import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google_drive_downloader import GoogleDriveDownloader as gdd
import torch
import os

def homemade_batchnormalisation():
    st.title("Homemade Batch Normalisation")
    
    # Instructions sur le but de la page
    st.write("Nous voulons comparer les performances entre la batchnormalisation implémenté dans pytorch et notre batch normalisation implémentée par nous même.")
    st.write("Nous utilisons le jeu de données MNIST pour entraîner des modèles suivant l'architecture LeNet-5 (Yann LeCun et al. (1998)) avec et sans notre batchnormalisation")
    # Instructions sur les modèles entraînés
    st.write("Nous avons pour cette expérience entrainées 2 modèles:")
    st.write("- Un modèle LeNet-5  avec la BatchNorm de Pytorch")
    st.write("- Un modèle LeNet-5 avec notre BatchNorm fait main")
    
    # Instructions sur les fichiers de loss     
    # Téléchargement des fichiers de loss de nos modèles déjà entrainés depuis Google Drive
    
    # Vérifier si les fichiers de loss existent déjà
    if not os.path.exists('valeur loss-models-batch//losses_vanilla.npy') or not os.path.exists('valeur loss-models-batch//losses_vanilla.npy') or not os.path.exists('valeur loss-models-batch//losses_vanilla.npy') or not os.path.exists('valeur loss-models-batch//losses_vanilla.npy'):
       # Liste des IDs de vos fichiers sur Google Drive
        file_ids = ['11IH2ZXJ3b_tZezDk8kN3Doar2-cs5YIZ', '1HEeypE2pBz7KpogT0KW4eQNTEp7hJhSd', '1ENI0CFZjgyM9A2rW_tIg6_nAXs6531oS','1YVzklFpo5ty6ApDK-XBzdb18zuogkQsY']

        # Liste des noms de fichiers de sortie
        output_filenames = ['losses_vanilla.npy', 'losses_bn.npy', 'val_losses_vanilla.npy','val_losses_bn.npy']

        # Boucle pour télécharger chaque fichier de loss depuis Google Drive
        for file_id, output_filename in zip(file_ids, output_filenames):
            url = f'https://drive.google.com/file/d/{file_id}/view?usp=sharing'
            dest_path = './valeur loss-models-batch/' + output_filename
            gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_path, overwrite=True, showsize=True)
    else:
        st.write("Les fichiers de loss existent déjà.")

# Charger les fichiers de loss
    losses_vanilla = np.load('valeur loss-models-batch//losses_vanilla.npy')
    losses_bn = np.load('valeur loss-models-batch//losses_bn.npy')
    val_losses_vanilla = np.load('valeur loss-models-batch//val_losses_vanilla.npy')
    val_losses_bn = np.load('valeur loss-models-batch//val_losses_bn.npy')

    st.write(val_losses_bn)
    if losses_vanilla is not None and losses_bn is not None and val_losses_vanilla is not None and val_losses_bn is not None:
        st.write("Les fichiers de loss ont été téléchargés avec succès.")
    
