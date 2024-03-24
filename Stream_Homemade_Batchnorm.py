import streamlit as st
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
from BNfonctions import plot_compare  
from BNfonctions import test





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
    """
    Cette fonction réalise une comparaison des performances entre la batchnormalisation implémentée dans PyTorch et notre batch normalisation implémentée par nous-même.
    Elle utilise le jeu de données MNIST pour entraîner des modèles suivant l'architecture LeNet-5 (Yann LeCun et al. (1998)) avec et sans notre batchnormalisation.
    
    La fonction télécharge les fichiers de loss nécessaires pour l'expérience et les charge en mémoire.
    Ensuite, elle affiche un graphique comparant les courbes de perte d'entraînement et de perte de validation pour les modèles LeNet-5 avec et sans BatchNorm.
    
    """
    
    # Code de la fonction
    ...
    
    
    st.write("Nous voulons comparer les performances entre la batch normalisation implémentée dans pytorch et notre batch normalisation implémentée par nous même.")
    st.write("Nous utilisons le jeu de données MNIST pour entraîner des modèles suivant l'architecture LeNet-5 (Yann LeCun et al. (1998)) avec et sans notre batchnormalisation.")
    
    descriptionMNISt()
    
    # Instructions sur les modèles entraînés si le bouton est cliqué
    if st.button("Afficher le détails des différents modèles entrainés", key='button2'):
        st.write("Nous avons pour cette expérience entrainés 6 modèles:")
        st.write("- Un modèle LeNet-5  avec la BatchNorm de Pytorch et un learning rate de 1e-3")
        st.write("- Un modèle LeNet-5 avec notre BatchNorm fait main et un learning rate de 1e-3")
        st.write("- Un modèle LeNet-5 avec notre BatchNorm fait main et un learning rate de 1e-2")
        st.write("- Un modèle LeNet-5 avec notre BatchNorm fait main et un learning rate de 5e-2")
        st.write("- Un modèle LeNet-5 simple sans batch normalisation et un learning rate de 1e-3")
        st.write("- Un modèle LeNet-5 simple sans batch normalisation et un learning rate de 5e-2")
    else:
        pass
    
    st.write(" Tout d'abord nous allons comparer les courbes de perte d'entraînement et de perte de validation pour les modèles LeNet-5 avec la batchnorm Pytorch et la homemade BatchNorm.")
    
    # Instructions sur les fichiers de loss     
    

    placeholder = st.empty()
    
    # Vérifier si les fichiers de loss existent déjà
    # Vérifier si les fichiers de loss sont présents dans le dossier
    if not os.path.exists('valeur loss-models-batch//losses_vanilla.npy') or not os.path.exists('valeur loss-models-batch//val_losses_vanilla.npy') or not os.path.exists('valeur loss-models-batch//losses_bn.npy') or not os.path.exists('valeur loss-models-batch//val_losses_bn.npy') or not os.path.exists('valeur loss-models-batch//losses_stockbn.npy') or not os.path.exists('valeur loss-models-batch//val_losses_stockbn.npy') or not os.path.exists('valeur loss-models-batch//losses_bn2.npy') or not os.path.exists('valeur loss-models-batch//val_losses_bn2.npy') or not os.path.exists('valeur loss-models-batch//losses_bn3.npy') or not os.path.exists('valeur loss-models-batch//val_losses_bn3.npy') or not os.path.exists('valeur loss-models-batch//losses_vanilla2.npy') or not os.path.exists('valeur loss-models-batch//val_losses_vanilla2.npy'):
        placeholder.text("Les fichiers de loss ne sont pas présents dans le dossier.")


       # Liste des IDs de vos fichiers sur Google Drive
        file_ids = ['19dkl_J39vGJAIVhhG51IOjJavhkP4_fr', '1HEeypE2pBz7KpogT0KW4eQNTEp7hJhSd','11IH2ZXJ3b_tZezDk8kN3Doar2-cs5YIZ','1h798r9UZZWu89zgtZ75L37YhiYawK6yY','19c2MB_l_DqM_ACIbJiQ8HxbWo5O0jQni','1CFV-3f3B_gBgSVwVr8a-Okr1sxGORL8R','1ENI0CFZjgyM9A2rW_tIg6_nAXs6531oS','1Zv725JX9_WRH1WMcajV9RQjQ-zDLuoY8','1ZWC-0GKxJkZP8XbujXyWzIpEVMyov9qa','1xNR6YIpXYTyxdt6Tc8eWqTiC0_y3P8RM' ,'1TIkKrY9uV-oWjYHQE4zvuk4kfpLJgwdV','1YVzklFpo5ty6ApDK-XBzdb18zuogkQsY']

        # Liste des noms de fichiers de sortie
        output_filenames = ['losses_stockbn.npy', 'losses_bn.npy','losses_vanilla.npy','losses_vanilla2.npy', 'losses_bn2.npy','losses_bn3.npy' ,'val_losses_vanilla.npy','val_losses_vanilla2.npy' ,'val_losses_bn3.npy','val_losses_bn2.npy','val_losses_stockbn.npy','val_losses_bn.npy']

        # Boucle pour télécharger chaque fichier de loss depuis Google Drive
        for file_id, output_filename in zip(file_ids, output_filenames):
            url = f'https://drive.google.com/file/d/{file_id}/view?usp=sharing'
            dest_path = './valeur loss-models-batch/' + output_filename
         
            gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_path, overwrite=True, showsize=True)
    
    # Sinon, afficher un message indiquant que les fichiers de loss existent déjà
    else:
        placeholder.text("Les fichiers de loss sont présents dans le dossier.")
        time.sleep(3)
        placeholder.empty()

    # Charger les fichiers de loss en mémoire
    losses_stockbn = np.load('valeur loss-models-batch/losses_stockbn.npy')
    val_losses_stockbn = np.load('valeur loss-models-batch/val_losses_stockbn.npy')
    losses_bn = np.load('valeur loss-models-batch/losses_bn.npy')
    val_losses_bn = np.load('valeur loss-models-batch/val_losses_bn.npy')
    losses_vanilla = np.load('valeur loss-models-batch/losses_vanilla.npy')
    val_losses_vanilla = np.load('valeur loss-models-batch/val_losses_vanilla.npy')
    losses_bn2 = np.load('valeur loss-models-batch/losses_bn2.npy')
    val_losses_bn2 = np.load('valeur loss-models-batch/val_losses_bn2.npy')
    losses_bn3 = np.load('valeur loss-models-batch/losses_bn3.npy')
    val_losses_bn3 = np.load('valeur loss-models-batch/val_losses_bn3.npy')
    losses_vanilla2 = np.load('valeur loss-models-batch/losses_vanilla2.npy')
    val_losses_vanilla2 = np.load('valeur loss-models-batch/val_losses_vanilla2.npy')
    
    
    # Vérifier si les fichiers de loss ont été téléchargés avec succès

    if losses_stockbn is not None and losses_bn is not None and val_losses_stockbn is not None and val_losses_bn is not None and losses_vanilla is not None and val_losses_vanilla is not None and losses_bn2 is not None and val_losses_bn2 is not None and losses_bn3 is not None and val_losses_bn3 is not None and losses_vanilla2 is not None and val_losses_vanilla2 is not None:
        placeholder.text("Les fichiers de loss ont été téléchargés avec succès.")
        time.sleep(3)
        placeholder.empty()
 
 
 # on charge les loss dans des listes pour les afficher sur le même graphique       
    all_losses_stockbn = []
    all_losses_stockbn.append((losses_stockbn, val_losses_stockbn))

    all_losses_bn = []
    all_losses_bn.append((losses_bn, val_losses_bn))
    
    all_losses_vanilla = []
    all_losses_vanilla.append((losses_vanilla, val_losses_vanilla))

    all_losses_bn = []
    all_losses_bn.append((losses_bn, val_losses_bn))

    all_losses_bn2 = []
    all_losses_bn2.append((losses_bn2, val_losses_bn2))

    all_losses_bn3 = []
    all_losses_bn3.append((losses_bn3, val_losses_bn3))

    all_losses_vanilla2 = []
    all_losses_vanilla2.append((losses_vanilla2, val_losses_vanilla2))

    all_losses_stockbn = []
    all_losses_stockbn.append((losses_stockbn, val_losses_stockbn))
    
    #on affiche le graphique de comparaison de la home batchnorm et la stock batchnorm

    plot_compare(all_losses_stockbn, all_losses_bn,mode="streamlit", legend_a="LeNet-5 with Pytorch-BatchNorm", legend_b="LeNet-5 with Homemade-BatchNorm", save_to="result_plot_batch/StockBN_vs_BatchNorm")
    
    st.write("On peut remarquer que la batch normalisation Pytorch à des performances similaires à notre batch normalisation homemade, ce qui veut dire que notre implémentation est efficace.")
    #on recupères les modèles entrainées depuis google drive et on calcule la précision sur le jeu de test
    
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
    
    
    st.write("La précision du modèle LeNet-5 avec la  homemade Batch Normalisation sur le jeu de test est de {:.2f}%.".format(accuracies['modèle_bn'] * 100))
    st.write("La précision du modèle LeNet-5 avec la batch normalisation de pytorch sur le jeu de test est de {:.2f}%.".format(accuracies['modèle_stockbn']* 100))
    
    
    # Demander à l'utilisateur de choisir les 2 modèles qu'il veut comparer
    models = st.multiselect("Choisir les modèles", ["LeNet-5 with Pytorch-BatchNorm", "LeNet-5 with Homemade-BatchNorm", "LeNet-5 with Homemade-BatchNorm with a 1e-2 learning rate", "LeNet-5 with Homemade-BatchNorm with a 5e-2 learning rate", "LeNet-5 without modification", "LeNet-5 without modification and a 5e-2 learning rate"], key='models')

    # Vérifier les choix de l'utilisateur et afficher le graphique de comparaison correspondant
    if len(models) == 2:
        
        #tous les choix possibles de comparaison avec StockBN
        if "LeNet-5 with Pytorch-BatchNorm" in models and "LeNet-5 with Homemade-BatchNorm" in models:
            plot_compare(all_losses_stockbn, all_losses_bn, mode="streamlit", legend_a="LeNet-5 with Pytorch-BatchNorm", legend_b="LeNet-5 with Homemade-BatchNorm", save_to="result_plot_batch/StockBN_vs_BatchNorm")
        elif "LeNet-5 with Pytorch-BatchNorm" in models and "LeNet-5 with Homemade-BatchNorm with a 1e-2 learning rate" in models:
            plot_compare(all_losses_stockbn, all_losses_bn2, mode="streamlit", legend_a="LeNet-5 with Pytorch-BatchNorm", legend_b="LeNet-5 with Homemade-BatchNorm (learning rate: 1e-2)", save_to="result_plot_batch/StockBN_vs_BatchNorm_lr1e-2")
        elif "LeNet-5 with Pytorch-BatchNorm" in models and "LeNet-5 with Homemade-BatchNorm with a 5e-2 learning rate" in models:
            plot_compare(all_losses_stockbn, all_losses_bn3, mode="streamlit", legend_a="LeNet-5 with Pytorch-BatchNorm", legend_b="LeNet-5 with Homemade-BatchNorm (learning rate: 5e-2)", save_to="result_plot_batch/StockBN_vs_BatchNorm_lr5e-2")
        elif "LeNet-5 with Pytorch-BatchNorm" in models and "LeNet-5 without modification" in models:
            plot_compare(all_losses_stockbn, all_losses_vanilla, mode="streamlit", legend_a="LeNet-5 with Pytorch-BatchNorm", legend_b= "LeNet-5 without modification", save_to="result_plot_batch/BatchNorm_vs_vanilla")
        elif "LeNet-5 with Pytorch-BatchNorm" in models and  "LeNet-5 without modification and a 5e-2 learning rate" in models:
            plot_compare(all_losses_stockbn, all_losses_vanilla2, mode="streamlit", legend_a="LeNet-5 with Pytorch-BatchNorm", legend_b= "LeNet-5 without modification and a 5e-2 learning rate", save_to="result_plot_batch/BatchNorm_vs_vanilla_lr5e-2")
        
        
        elif "LeNet-5 with Homemade-BatchNorm" in models and "LeNet-5 with Homemade-BatchNorm with a 1e-2 learning rate" in models:
            plot_compare(all_losses_bn, all_losses_bn2, mode="streamlit", legend_a="LeNet-5 with Homemade-BatchNorm", legend_b="LeNet-5 with Homemade-BatchNorm (learning rate: 1e-2)", save_to="result_plot_batch/BatchNorm_vs_BatchNorm_lr1e-2")
        elif "LeNet-5 with Homemade-BatchNorm" in models and "LeNet-5 with Homemade-BatchNorm with a 5e-2 learning rate" in models:
            plot_compare(all_losses_bn, all_losses_bn3, mode="streamlit", legend_a="LeNet-5 with Homemade-BatchNorm", legend_b="LeNet-5 with Homemade-BatchNorm (learning rate: 5e-2)", save_to="result_plot_batch/_BatchNorm_vs_BatchNorm_lr5e-2")
        elif "LeNet-5 with Homemade-BatchNorm" in models and "LeNet-5 without modification" in models:
            plot_compare(all_losses_bn, all_losses_vanilla, mode="streamlit", legend_a="LeNet-5 with Homemade-BatchNorm", legend_b= "LeNet-5 without modification", save_to="result_plot_batch/BatchNorm_vs_vanilla")
        elif "LeNet-5 with Homemade-BatchNorm" in models and  "LeNet-5 without modification and a 5e-2 learning rate" in models:
            plot_compare(all_losses_bn, all_losses_vanilla2, mode="streamlit", legend_a="LeNet-5 with Homemade-BatchNorm", legend_b= "LeNet-5 without modification and a 5e-2 learning rate", save_to="result_plot_batch/_BatchNormvs_vanilla_lr5e-2")
        
        elif "LeNet-5 with Homemade-BatchNorm with a 1e-2 learning rate" in models and "LeNet-5 with Homemade-BatchNorm with a 5e-2 learning rate" in models:
            plot_compare(all_losses_bn2, all_losses_bn3, mode="streamlit", legend_a="LeNet-5 with Homemade-BatchNorm with a 1e-2 learning rate", legend_b="LeNet-5 with Homemade-BatchNorm with a 5e-2 learning rate", save_to="result_plot_batch/BatchNorm_lr1e-2_vs_BatchNorm_lr5e-2")
        elif "LeNet-5 with Homemade-BatchNorm with a 1e-2 learning rate" in models and  "LeNet-5 without modification and a 5e-2 learning rate" in models:
            plot_compare(all_losses_bn2, all_losses_vanilla2, mode="streamlit", legend_a="LeNet-5 with Homemade-BatchNorm with a 1e-2 learning rate", legend_b= "LeNet-5 without modification and a 5e-2 learning rate", save_to="result_plot_batch/BatchNorm_lr1e-2vs_vanilla_lr5e-2")
        elif "LeNet-5 with Homemade-BatchNorm with a 1e-2 learning rate" in models and "LeNet-5 without modification" in models:
            plot_compare(all_losses_bn2, all_losses_vanilla, mode="streamlit", legend_a="LeNet-5 with Homemade-BatchNorm with a 1e-2 learning rate", legend_b= "LeNet-5 without modification", save_to="result_plot_batch/BatchNorm_lr1e-2__vs_vanilla")    

        elif "LeNet-5 with Homemade-BatchNorm with a 5e-2 learning rate"  in models and  "LeNet-5 without modification and a 5e-2 learning rate" in models:
            plot_compare(all_losses_bn3, all_losses_vanilla2, mode="streamlit", legend_a="LeNet-5 with Homemade-BatchNorm with a 5e-2 learning rate", legend_b= "LeNet-5 without modification and a 5e-2 learning rate", save_to="result_plot_batch/BatchNorm_lr5e-2vs_vanilla_lr5e-2")
        elif "LeNet-5 with Homemade-BatchNorm with a 5e-2 learning rate"  in models and "LeNet-5 without modification" in models:
            plot_compare(all_losses_bn3, all_losses_vanilla, mode="streamlit", legend_a="LeNet-5 with Homemade-BatchNorm with a 5e-2 learning rate", legend_b= "LeNet-5 without modification", save_to="result_plot_batch/BatchNorm_lr5e-2__vs_vanilla")    

        elif "LeNet-5 without modification"  in models and "LeNet-5 without modification and a 5e-2 learning rate" in models:
            plot_compare(all_losses_vanilla2, all_losses_vanilla, mode="streamlit", legend_a="LeNet-5 without modification and a 5e-2 learning rate", legend_b= "LeNet-5 without modification", save_to="result_plot_batch/vanilla_lr5e-2__vs_vanilla")
        
    else:
        st.write("Veuillez sélectionner exactement deux modèles.")
            
    if "LeNet-5 with Pytorch-BatchNorm" in models:
            st.write("La précision du modèle LeNet-5 avec la batch normalisation de pytorch sur le jeu de test est de {:.2f}%.".format(accuracies['modèle_stockbn']* 100))
    if "LeNet-5 with Homemade-BatchNorm" in models:
            st.write("La précision du modèle LeNet-5 avec la homemade Batch Normalisation sur le jeu de test est de {:.2f}%.".format(accuracies['modèle_bn'] * 100))
    if "LeNet-5 with Homemade-BatchNorm with a 1e-2 learning rate" in models:
            st.write("La précision du modèle LeNet-5 avec la homemade Batch Normalisation (learning rate: 1e-2) sur le jeu de test est de {:.2f}%.".format(accuracies['modèle_bn2'] * 100))  
    
    if "LeNet-5 with Homemade-BatchNorm with a 5e-2 learning rate" in models:
        st.write("La précision du modèle LeNet-5 avec la homemade Batch Normalisation (learning rate: 5e-2) sur le jeu de test est de {:.2f}%.".format(accuracies['modèle_bn3'] * 100))
    
    if "LeNet-5 without modification" in models:
        st.write("La précision du modèle LeNet-5 sans modification sur le jeu de test est de {:.2f}%.".format(accuracies['modèle_vanilla'] * 100))   
    
    if "LeNet-5 without modification and a 5e-2 learning rate" in models:
        st.write("La précision du modèle LeNet-5 sans modification (learning rate: 5e-2) sur le jeu de test est de {:.2f}%.".format(accuracies['modèle_vanilla2'] * 100))
    
       

    
    
        

    
