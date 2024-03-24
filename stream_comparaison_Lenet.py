"""
Utilisé dans Stream_comparaison_Lenet.py

Ce script utilise Streamlit pour comparer les performances des modèles d'architecture LeNet-5 avec Layer Normalization (LN) et Batch Normalization (BN).

Il télécharge les données d'accuracy depuis un fichier .pth et de losses sur Google Drive, puis trace les graphiques des pertes (loss) pour les modèles LeNet-5 avec LN et LeNet-5 avec BN. Ensuite, il affiche les valeurs d'accuracy pour chaque modèle.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
import os
import numpy as np
import time
from BNfonctions import plot_comparetrain

def comparaison_Lenet():
    st.title("Comparaison LeNet-5")
    
    placeholder = st.empty()
    
    # Vérifier si les fichiers de loss existent déjà
    # Vérifier si les fichiers de loss sont présents dans le dossier
    if not os.path.exists('valeur loss-models-batch//losses_layer.npy') or not os.path.exists('valeur loss-models-batch//val_losses_layer.npy') or not os.path.exists('valeur loss-models-batch//losses_bn.npy') or not os.path.exists('valeur loss-models-batch//val_losses_bn.npy'):
        placeholder.text("Les fichiers de loss ne sont pas présents dans le dossier.")
        file_ids = [ '1HEeypE2pBz7KpogT0KW4eQNTEp7hJhSd','1YVzklFpo5ty6ApDK-XBzdb18zuogkQsY','1KD9vVGuc9sS-Ba0dNMO7T-Tk-SnNeDDS','1YVzklFpo5ty6ApDK-XBzdb18zuogkQsY']
        output_filenames = [ 'losses_bn.npy' , 'val_losses_bn.npy' , 'losses_layer.npy' , 'val_losses_layer.npy']
    # Boucle pour télécharger chaque fichier de loss depuis Google Drive
        for file_id, output_filename in zip(file_ids, output_filenames):
            url = f'https://drive.google.com/file/d/{file_id}/view?usp=sharing'
            dest_path = './valeur loss-models-batch/' + output_filename
         
            gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_path, overwrite=True, showsize=True)
    
    else:
        placeholder.text("Les fichiers de loss sont présents dans le dossier.")
        time.sleep(3)
        placeholder.empty()
    
    
    losses_bn = np.load('valeur loss-models-batch/losses_bn.npy')
    val_losses_bn = np.load('valeur loss-models-batch/val_losses_bn.npy')
    losses_layer = np.load('valeur loss-models-batch/losses_layer.npy')
    val_losses_layer = np.load('valeur loss-models-batch/val_losses_layer.npy')
    
    
    all_losses_bn = []
    all_losses_bn.append((losses_bn, val_losses_bn))
    
    all_losses_layer = []
    all_losses_layer.append((losses_layer, val_losses_layer))
    
    #on fait le plot de comparaison de training loss car dans le cas de lenet-5 layer on a pas de validation loss
    
    plot_comparetrain(all_losses_layer, all_losses_bn,mode="streamlit", legend_a="LeNet-5 with layerNorm from Pytorch", legend_b="LeNet-5 with Homemade-BatchNorm", save_to="result_plot_batch/layerNorm_vs_BatchNorm")
    
    all_losses_bn = []
    all_losses_bn.append((losses_bn, val_losses_bn))
    
    if not os.path.exists('batch/accuracies.pth'):
        file_id = '157Iol19Wp0AjaS7VL2UtD-Jx2ePVofVL'
        dest_path = './batch/accuracies.pth'
        gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_path, overwrite=True, showsize=True)
    else:
        placeholder.text("Le fichier d'accuracies existe déjà.")
        time.sleep(3)
        placeholder.empty()
    
    
    #on charge les accuracies    
    accuracies = torch.load('batch/accuracies.pth')
    
    st.write("La précision du modèle LeNet-5 Avec notre Batch normalisation sur le jeu de test est de {:.2f}%.".format(accuracies['modèle_bn'] * 100))
    st.write("La précision du modèle LeNet-5 Avec une Layer Normalisation sur le jeu de test est de {:.2f}%.".format(accuracies['modèle_layer'] * 100))
    
    st.write("Le modèle LeNet-5 Avec notre Batch normalisation a une précision de {:.2f}% supérieure à celle du modèle LeNet-5 Avec une Layer Normalisation.".format((accuracies['modèle_bn'] - accuracies['modèle_layer']) * 100))
    st.write("on peut donc dire que notre Batch Normalisation est plus performante que la Layer Normalisation sur le jeu de test MNIST  avec le modèle LeNet-5.")
    
