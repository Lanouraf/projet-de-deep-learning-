import streamlit as st
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
from Batchfonctions import plot_compare  
from Batchfonctions import test
from Batcharchitecture import LeNet, LeNetBN , LeNetStockBN




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
    if not os.path.exists('valeur loss-models-batch//losses_stockbn.npy') or not os.path.exists('valeur loss-models-batch//losses_stockbn.npy') or not os.path.exists('valeur loss-models-batch//losses_bn.npy') or not os.path.exists('valeur loss-models-batch//val_losses_bn.npy'):
        # Télécharger les fichiers de loss depuis Google Drive si les fichiers n'existent pas
        placeholder.text("Les fichiers de loss n'ont pas été trouvés. Téléchargement en cours...")
       # Liste des IDs de vos fichiers sur Google Drive
        file_ids = ['19dkl_J39vGJAIVhhG51IOjJavhkP4_fr', '1HEeypE2pBz7KpogT0KW4eQNTEp7hJhSd', '1TIkKrY9uV-oWjYHQE4zvuk4kfpLJgwdV','1YVzklFpo5ty6ApDK-XBzdb18zuogkQsY']

        # Liste des noms de fichiers de sortie
        output_filenames = ['losses_stockbn.npy', 'losses_bn.npy', 'val_losses_stockbn.npy','val_losses_bn.npy']

        # Boucle pour télécharger chaque fichier de loss depuis Google Drive
        for file_id, output_filename in zip(file_ids, output_filenames):
            url = f'https://drive.google.com/file/d/{file_id}/view?usp=sharing'
            dest_path = './valeur loss-models-batch/' + output_filename
         
            gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_path, overwrite=True, showsize=True)
    
    # Sinon, afficher un message indiquant que les fichiers de loss existent déjà
    else:
        placeholder.text("Les fichiers de loss existent déjà.")

    # Charger les fichiers de loss suite au téléchargement
    losses_vanilla = np.load('valeur loss-models-batch//losses_vanilla.npy')
    val_losses_vanilla = np.load('valeur loss-models-batch//val_losses_vanilla.npy')
    losses_bn = np.load('valeur loss-models-batch//losses_bn.npy')
    val_losses_bn = np.load('valeur loss-models-batch//val_losses_bn.npy')
    losses_stockbn = np.load('valeur loss-models-batch//losses_stockbn.npy')
    val_losses_stockbn = np.load('valeur loss-models-batch//val_losses_stockbn.npy')
    losses_bn2 = np.load('valeur loss-models-batch//losses_bn2.npy')
    val_losses_bn2 = np.load('valeur loss-models-batch//val_losses_bn2.npy')
    losses_bn3 = np.load('valeur loss-models-batch//losses_bn3.npy')
    val_losses_bn3 = np.load('valeur loss-models-batch//val_losses_bn3.npy')
    losses_vanilla2 = np.load('valeur loss-models-batch//losses_vanilla2.npy')
    val_losses_vanilla2 = np.load('valeur loss-models-batch//val_losses_vanilla2.npy')

    if losses_stockbn is not None and losses_bn is not None and val_losses_stockbn is not None and val_losses_bn is not None:
        placeholder.text("Les fichiers de loss ont été téléchargés avec succès.")
        time.sleep(3)
        placeholder.empty()
        
    all_losses_stockbn = []
    all_losses_stockbn.append((losses_stockbn, val_losses_stockbn))

    all_losses_bn = []
    all_losses_bn.append((losses_bn, val_losses_bn))
    
    plot_compare(all_losses_stockbn, all_losses_bn,mode="streamlit", legend_a="LeNet-5 with Pytorch-BatchNorm", legend_b="LeNet-5 with Homemade-BatchNorm", save_to="result_plot_batch/StockBN_vs_BatchNorm")
    
    st.write("On peut remarquer que la batch normalisation Pytorch à des performances similaires à notre batch normalisation homemade, ce qui veut dire que notre implémentation est efficace.")
    #on recupères les modèles entrainées depuis google drive et on calcule la précision sur le jeu de test
    
    #on recupère les données de test 
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    test1 = datasets.MNIST(root="dataset", train=False, transform=transform, download=True)
    test_dataloader = DataLoader(test1, batch_size=512, shuffle=False)
    
    
    placeholder = st.empty()
    
    # Vérifier si les fichiers de loss existent déjà
    if not os.path.exists('batch/modèles_stockbn_entrainé.pth') or not os.path.exists('batch/modèle_bn_entrainé.pth') :
        # Télécharger les fichiers de loss
        placeholder.text("Les fichiers de modèles n'ont pas été trouvés. Téléchargement en cours...")
       # Liste des IDs de vos fichiers sur Google Drive
        file_ids = ['1x9EpK8hLu4obAFoy14_tN2OdsKMB_-xq', '1_LdtVZNoiwb2BzfqI6yjSAbOxDGMRTny']

        # Liste des noms de fichiers de sortie
        output_filenames = ['modèles_stockbn_entrainé.pth', 'modèle_bn_entrainé.pth']

        # Boucle pour télécharger chaque fichier de loss depuis Google Drive
        for file_id, output_filename in zip(file_ids, output_filenames):
            url = f'https://drive.google.com/file/d/{file_id}/view?usp=sharing'
            dest_path = './batch/' + output_filename
         
            gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_path, overwrite=True, showsize=True)
    
    
    else:
        placeholder.text("Les fichiers de modèles existent déjà.")
        
    #charger les modèles 
    modèle_stockbn=LeNetStockBN() 
    modèles_stockbn_entrainé=torch.load('batch/modèles_stockbn_entrainé.pth')
    modèle_stockbn.load_state_dict(modèles_stockbn_entrainé)
    
    
    modèle_bn=LeNetBN()
    modèle_bn_entrainé=torch.load('batch/modèle_bn_entrainé.pth')
    modèle_bn.load_state_dict(modèle_bn_entrainé)
    
    # Vérifier si les modèles ont été importés avec succès
    if modèles_stockbn_entrainé is not None and modèle_bn_entrainé is not None:
        placeholder.text("Les modèles ont été importés avec succès.")
        time.sleep(3)
        placeholder.empty()
    else:
        placeholder.text("Erreur lors de l'importation des modèles.")
        
    # Calculer la précision sur le jeu de test
    _, testacc_stockbn = test(modèle_stockbn, test_dataloader)
    _, testacc_bn = test(modèle_bn, test_dataloader)
    
    st.write("La précision du modèle LeNet-5 avec BatchNorm sur le jeu de test est de {:.2f}%.".format(testacc_bn * 100))
    st.write("La précision du modèle LeNet-5 sans BatchNorm sur le jeu de test est de {:.2f}%.".format(testacc_stockbn * 100))
    
    
    # Demander à l'utilisateur de choisir les 2 modèles qu'il veut comparer
    models = st.multiselect("Choisir les modèles", ["LeNet-5 with Pytorch-BatchNorm", "LeNet-5 with Homemade-BatchNorm"])

    # Vérifier les choix de l'utilisateur et afficher le graphique de comparaison correspondant
    if len(models) == 2:
        if "LeNet-5 with Pytorch-BatchNorm" in models and "LeNet-5 with Homemade-BatchNorm" in models:
            plot_compare(all_losses_stockbn, all_losses_bn, mode="streamlit", legend_a="LeNet-5 with Pytorch-BatchNorm", legend_b="LeNet-5 with Homemade-BatchNorm", save_to="result_plot_batch/StockBN_vs_BatchNorm")
            
            # Calculer la précision sur le jeu de test pour les modèles choisis
            if "LeNet-5 with Pytorch-BatchNorm" in models:
                _, testacc_modèle1 = test(modèle_stockbn, test_dataloader)
                st.write("La précision du modèle LeNet-5 with Pytorch-BatchNorm sur le jeu de test est de {:.2f}%.".format(testacc_modèle1 * 100))
            
            if "LeNet-5 with Homemade-BatchNorm" in models:
                _, testacc_model2 = test(modèle_bn, test_dataloader)
                st.write("La précision du modèle LeNet-5 with Homemade-BatchNorm sur le jeu de test est de {:.2f}%.".format(testacc_model2 * 100))

        else:
            st.write("Veuillez sélectionner les modèles corrects.")
    else:
        st.write("Veuillez sélectionner exactement deux modèles.")
            
    
    
     
       
        

    
    
        

    
