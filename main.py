"""
Ce fichier charge des modèles préalablement entraînés et évalue leurs performances sur un jeu de données de test.

1. Import des modules et des classes nécessaires :
   - LNtest: Fonction pour évaluer les modèles sur le jeu de données de test.
   - prep_data: Fonction pour préparer les données d'apprentissage et de test.
   - BagOfWordsClassifier, BagOfWordsClassifierLayer, BagOfWordsClassifierLayerHM, BagOfWordsClassifierBatchNorm: Modèles de classification Bag-of-Words.
   - GoogleDriveDownloader: Pour télécharger les fichiers des modèles et des pertes depuis Google Drive.
   - torch, numpy, pandas: Modules pour les calculs et la manipulation des données.

2. Téléchargement des fichiers de modèles et de pertes :
   - Les fichiers des modèles et des pertes sont téléchargés depuis Google Drive.

3. Chargement des modèles préalablement entraînés :
   - Les poids des modèles sont chargés à partir des fichiers .pth correspondants.

4. Calcul des moyennes de pertes :
   - Les valeurs moyennes de pertes d'entraînement sont calculées pour chaque modèle à partir des fichiers de pertes.

5. Évaluation des performances des modèles sur le jeu de données de test :
   - La fonction LNtest est utilisée pour évaluer chaque modèle sur le jeu de données de test.
   - Les accuracies sont calculées pour chaque modèle.

6. Affichage des résultats :
   - Les moyennes des pertes d'entraînement pour chaque modèle sont affichées.
   - Les accuracies pour chaque modèle sont affichées.
   - Les accuracies sont enregistrées dans un DataFrame pandas et sauvegardées dans un fichier CSV.
"""
from LNtrain import LNtest
import numpy as np
from Sequences import prep_data
from LNModule import BagOfWordsClassifier, BagOfWordsClassifierLayer, BagOfWordsClassifierLayerHM, BagOfWordsClassifierBatchNorm
from google_drive_downloader import GoogleDriveDownloader as gdd
import torch
import pandas as pd

# Préparation des données
train_dataset, train_loader, test_dataset, test_loader = prep_data()

# Téléchargement des fichiers de modèles et de pertes depuis Google Drive
gdd.download_file_from_google_drive(file_id='1P28VRYGSueQ7DO7zghG5-pSdcp8TkzBT', dest_path='./BOW.pth', overwrite=True, showsize=True)
gdd.download_file_from_google_drive(file_id='1INsBCiiYbPC7zNbP7Ufp5jfb0Ku2rTA-', dest_path='./BOW_LN.pth', overwrite=True, showsize=True)
gdd.download_file_from_google_drive(file_id='1TstrD3ic2NDqxy_rpFpFp4gYhyvtPZGQ', dest_path='./BOW_HM_LN.pth', overwrite=True, showsize=True)
gdd.download_file_from_google_drive(file_id='1lESMtwwy_qzfWtpfpGDQYKzLOkABrIyp', dest_path='./losses.pth', overwrite=True, showsize=True)
gdd.download_file_from_google_drive(file_id='18LR3O6GcrwUUmF9QEsR_mShAAYNtslL2', dest_path='./BOW_BN.pth', overwrite=True, showsize=True)
gdd.download_file_from_google_drive(file_id='1NZ0VpdekC7WCfYeJlDHe_WGhuh2z1ON3', dest_path='./loss_BN.pth', overwrite=True, showsize=True)

# Définition des paramètres des modèles
vocab_size = len(train_dataset.vectorizer.vocabulary_)
hidden1 = 128
hidden2 = 64
output_shape = 1

# Initialisation des modèles
model1 = BagOfWordsClassifier(vocab_size, hidden1, hidden2, output_shape)
model2 = BagOfWordsClassifierLayer(vocab_size, hidden1, hidden2, output_shape)
model3 = BagOfWordsClassifierLayerHM(vocab_size, hidden1, hidden2, output_shape)
model4 = BagOfWordsClassifierBatchNorm(vocab_size, hidden1, hidden2, output_shape)

# Chargement des poids des modèles
model1.load_state_dict(torch.load('BOW.pth'))
model2.load_state_dict(torch.load('BOW_LN.pth'))
model3.load_state_dict(torch.load('BOW_HM_LN.pth'))
model4.load_state_dict(torch.load('BOW_BN.pth'))

# Chargement des fichiers de pertes
losses = torch.load('./losses.pth')
loss_BN = torch.load('./loss_BN.pth')

# Calcul des moyennes de pertes
averages = {}
for model, loss in losses.items():
    averages[model] = sum(loss) / len(loss)

loss_BN = {model: [float(loss_val) for loss_val in loss_list] for model, loss_list in loss_BN.items()}
for model, loss in loss_BN.items():
    averages['BOW_BN'] = sum(loss) / len(loss)

# Affichage des moyennes de pertes
for model, average_loss in averages.items():
    print(f"Moyenne des pertes pour {model}: {average_loss}")

# Évaluation des modèles sur le jeu de données de test
acc1 = LNtest(model1, test_loader)
acc2 = LNtest(model2, test_loader)
acc3 = LNtest(model3, test_loader)
acc4 = LNtest(model4, test_loader)

# Affichage des accuracies
print("Accuracy modèle 1:", acc1)
print("Accuracy modèle 2:", acc2)
print("Accuracy modèle 3:", acc3)
print("Accuracy modèle 4:", acc4)

# Enregistrement des accuracies dans un fichier CSV
accuracies_dict = {"BOW": acc1, "BOW_LN": acc2, "BOW_HM_LN": acc3, "BOW_BN": acc4}
accuracies_df = pd.DataFrame.from_dict(accuracies_dict, orient='index', columns=['Accuracy'])
accuracies_df.to_csv('accuracies.csv')
