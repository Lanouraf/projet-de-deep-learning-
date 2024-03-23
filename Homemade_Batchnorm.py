import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google_drive_downloader import GoogleDriveDownloader as gdd
import torch
import os
import time



def descriptionMNISt():
    """
    Affiche la description du jeu de données MNIST.
    en fonction de l'interaction de l'utilisateur avec le bouton.
    """
    
    if st.button("Afficher la description du jeu MNIST", key='button1'):
        st.write("""
Le jeu de données MNIST (Modified National Institute of Standards and Technology) est une grande base de données de chiffres écrits à la main qui est couramment utilisée pour l'entraînement de divers systèmes de traitement d'images. Cette base de données est également largement utilisée pour l'entraînement et les tests dans le domaine de l'apprentissage automatique.

Le jeu de données MNIST contient 60 000 images d'entraînement et 10 000 images de test. Chaque image est une image en niveaux de gris de 28x28 pixels, associée à une étiquette de 0 à 9. Ce jeu de données est un bon point de départ pour ceux qui souhaitent essayer des techniques d'apprentissage et des méthodes de reconnaissance de formes sur des données du monde réel tout en minimisant les efforts de prétraitement et de formatage.

L'une des raisons pour lesquelles MNIST est si populaire en apprentissage automatique et en apprentissage profond est due à sa simplicité et sa taille. Il est suffisamment petit pour entraîner un modèle sur un ordinateur personnel et les données sont également faciles à comprendre. Par conséquent, il est souvent le premier jeu de données utilisé par les personnes qui apprennent l'apprentissage profond.

Chaque image dans le jeu de données MNIST est une image en niveaux de gris de 28x28 pixels d'un chiffre écrit à la main, de 0 à 9, ce qui donne 10 classes différentes. Les valeurs de pixels en niveaux de gris sont comprises entre 0 et 255, ce qui représente l'intensité de la couleur. Une valeur de 0 représente le noir, qui est la couleur de fond, et une valeur de 255 représente le blanc, qui est la couleur du chiffre.

La tâche consiste à classer ces images dans l'une des 10 classes. Il s'agit d'un problème de classification multiclasse, qui est un problème courant en apprentissage automatique. Le jeu de données MNIST a été largement utilisé pour évaluer les algorithmes de classification, et l'apprentissage profond a atteint des performances proches de celles de l'homme sur ce jeu de données.
""")
    
    else: 
        pass



def homemade_batchnormalisation():
    
    st.write("Nous voulons comparer les performances entre la batchnormalisation implémenté dans pytorch et notre batch normalisation implémentée par nous même.")
    st.write("Nous utilisons le jeu de données MNIST pour entraîner des modèles suivant l'architecture LeNet-5 (Yann LeCun et al. (1998)) avec et sans notre batchnormalisation.")
    
    descriptionMNISt()
    
    # Instructions sur les modèles entraînés
    st.write("Nous avons pour cette expérience entrainées 2 modèles:")
    st.write("- Un modèle LeNet-5  avec la BatchNorm de Pytorch")
    st.write("- Un modèle LeNet-5 avec notre BatchNorm fait main")
    
    # Instructions sur les fichiers de loss     
    

    placeholder = st.empty()
    
    # Vérifier si les fichiers de loss existent déjà
    if not os.path.exists('valeur loss-models-batch//losses_vanilla.npy') or not os.path.exists('valeur loss-models-batch//losses_vanilla.npy') or not os.path.exists('valeur loss-models-batch//losses_vanilla.npy') or not os.path.exists('valeur loss-models-batch//losses_vanilla.npy'):
        # Télécharger les fichiers de loss
        placeholder.text("Les fichiers de loss n'ont pas été trouvés. Téléchargement en cours...")
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
        placeholder.text("Les fichiers de loss existent déjà.")

    # Charger les fichiers de loss
    losses_vanilla = np.load('valeur loss-models-batch//losses_vanilla.npy')
    losses_bn = np.load('valeur loss-models-batch//losses_bn.npy')
    val_losses_vanilla = np.load('valeur loss-models-batch//val_losses_vanilla.npy')
    val_losses_bn = np.load('valeur loss-models-batch//val_losses_bn.npy')

    if losses_vanilla is not None and losses_bn is not None and val_losses_vanilla is not None and val_losses_bn is not None:
        placeholder.text("Les fichiers de loss ont été téléchargés avec succès.")
        time.sleep(7)
        placeholder.empty()
        
        

    
    
        

    
