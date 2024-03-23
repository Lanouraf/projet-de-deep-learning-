"""
Ce fichier contient le processus d'entraînement de différents modèles de classification Bag-of-Words avec différentes configurations de normalisation.
Les données sont préparées à l'aide de la fonction prep_data du module Sequences, puis utilisées pour entraîner plusieurs modèles de classification.

1. Import des modules et des classes nécessaires :
   - prep_data: Fonction pour préparer les données d'apprentissage et de test.
   - BagOfWordsClassifierLayer: Modèle de classification Bag-of-Words avec normalisation de couche.
   - BagOfWordsClassifier: Modèle de classification Bag-of-Words sans normalisation.
   - BagOfWordsClassifierLayerHM: Modèle de classification Bag-of-Words avec normalisation de couche faite maison.
   - BagOfWordsClassifierBatchNorm: Modèle de classification Bag-of-Words avec normalisation par lot.
   - LNtrain: Fonction pour l'entraînement des modèles.
   - LNtest: Fonction pour l'évaluation des modèles.

2. Préparation des données :
   - Les données sont préparées à l'aide de la fonction prep_data, qui retourne les ensembles de données d'entraînement et de test, ainsi que les chargeurs de données correspondants.

3. Définition des paramètres des modèles :
   - vocab_size: Taille du vocabulaire.
   - hidden1, hidden2: Nombre d'unités dans les couches cachées.
   - output_shape: Forme de sortie du modèle.
   - criterion: Fonction de perte (Binary Cross Entropy Loss).

4. Initialisation des modèles et des optimiseurs :
   - Quatre modèles différents sont initialisés avec différentes configurations de normalisation.
   - Quatre optimiseurs Adam sont utilisés pour l'entraînement des modèles.

5. Entraînement des modèles :
   - Chaque modèle est entraîné en utilisant la fonction LNtrain qui prend le modèle, l'optimiseur et la fonction de perte comme entrées.
   - Les pertes d'entraînement sont enregistrées pour chaque modèle.

6. Sauvegarde des modèles et des pertes :
   - Les poids des modèles sont sauvegardés dans des fichiers .pth.
   - Les valeurs de perte d'entraînement sont sauvegardées dans des fichiers .pth pour chaque modèle.
"""

from Sequences import prep_data
from LNModule import BagOfWordsClassifierLayer,BagOfWordsClassifier,BagOfWordsClassifierLayerHM,BagOfWordsClassifierBatchNorm
from LNtrain import LNtrain,LNtest
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import pandas as pd

train_dataset,train_loader,test_dataset,test_loader=prep_data()

vocab_size = len(train_dataset.vectorizer.vocabulary_)
hidden1 = 128
hidden2 = 64
output_shape = 1

criterion = nn.BCELoss()

model = BagOfWordsClassifier(vocab_size, hidden1, hidden2, output_shape)
model2=BagOfWordsClassifier(vocab_size, hidden1, hidden2, output_shape)
model3=BagOfWordsClassifierLayerHM(vocab_size, hidden1, hidden2, output_shape)
model4=BagOfWordsClassifierBatchNorm(vocab_size, hidden1, hidden2, output_shape)

optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
optimizer3 = optim.Adam(model3.parameters(), lr=0.001)
optimizer4 = optim.Adam(model4.parameters(), lr=0.001)

BOW_LN,loss=LNtrain(model,optimizer,criterion)
BOW,loss2=LNtrain(model2,optimizer2,criterion)
BOW_HM_LN,loss3=LNtrain(model3,optimizer3,criterion)
BOW_BN,loss4=LNtrain(model4,optimizer4,criterion)

# Créer un dictionnaire de pertes pour chaque modèle
losses_dict = {
    'BOW': loss2,
    'BOW_LN': loss,
    'BOW_HM_LN': loss3
}

losses_dict_BN = {
    'BOW_BN': loss}


torch.save(losses_dict_BN, 'loss_BN.pth')
torch.save(losses_dict, 'losses.pth')


torch.save(BOW.state_dict(), 'BOW.pth')
torch.save(BOW_LN.state_dict(), 'BOW_LN.pth')
torch.save(BOW_HM_LN.state_dict(), 'BOW_HM_LN.pth')
torch.save(BOW_BN.state_dict(), 'BOW_BN.pth')


