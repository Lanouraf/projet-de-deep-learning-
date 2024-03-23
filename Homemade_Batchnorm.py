import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google_drive_downloader import GoogleDriveDownloader as gdd
import torch

def homemade_batchnormalisation():
    st.title("Homemade Batch Normalisation")
    
    # Instructions sur le but de la page
    st.write("Nous voulons comparer les performances entre la batchnormalisation implémenté dans pytorch et notre batch normalisation implémentée par nous même.")
    st.write("Nous utilisons le jeu de données MNIST pour entraîner des modèles suivant l'architecture LeNet-5 avec et sans notre batchnormalisation")
    # Instructions sur les modèles entraînés
    st.write("Nous avons pour cette expérience entrainées 2 modèles:")
    st.write("- Un modèle LeNet-5  avec la BatchNorm de Pytorch")
    st.write("- Un modèle LeNet-5 avec notre BatchNorm fait main")
        
    
# # Téléchargement des fichiers de loss de nos modèles déjà entrainés depuis Google Drive
   
# # Liste des IDs de vos fichiers sur Google Drive
# file_ids = ['FILE_ID_1', 'FILE_ID_2', 'FILE_ID_3','FILE_ID_3']

# # Liste des noms de fichiers de sortie
# output_filenames = ['OUTPUT_FILENAME_1.npy', 'OUTPUT_FILENAME_2.npy', 'OUTPUT_FILENAME_3.npy','OUTPUT_FILENAME_3.npy']

# # Boucle pour télécharger chaque fichier
# for file_id, output_filename in zip(file_ids, output_filenames):
#     url = f'https://drive.google.com/uc?id={file_id}'
#     gdown.download(url, output_filename, quiet=False)

# # Maintenant, vous pouvez charger les fichiers .npy
# data = []
# for output_filename in output_filenames:
#     data.append(np.load(output_filename))
    
    
