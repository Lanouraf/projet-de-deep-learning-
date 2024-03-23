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
        
    # Téléchargement des fichiers depuis Google Drive

    
    
