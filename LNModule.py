"""
Ce fichier contient plusieurs classes de modèles de classification de type Bag-of-Words, utilisés dans le cadre de l'apprentissage de réseaux de neurones avec PyTorch.
Chaque classe de modèle implémente une architecture spécifique avec différentes couches de normalisation.

1. BagOfWordsClassifier:
   - Modèle de classification Bag-of-Words sans couches de normalisation.
   - Il consiste en trois couches linéaires avec des fonctions d'activation ReLU.

2. BagOfWordsClassifierLayer:
   - Modèle de classification Bag-of-Words avec normalisation de couche (Layer Normalization).
   - Applique la normalisation de couche après chaque couche linéaire.
   - Il consiste en trois couches linéaires avec des fonctions d'activation ReLU et de Layer Normalization.

3. BagOfWordsClassifierBatchNorm:
   - Modèle de classification Bag-of-Words avec normalisation par lot (Batch Normalization).
   - Applique la normalisation par lot après chaque couche linéaire.
   - Il consiste en trois couches linéaires avec des fonctions d'activation ReLU et de Batch Normalization.

4. BagOfWordsClassifierLayerHM:
   - Modèle de classification Bag-of-Words avec normalisation de couche faite maison (Homemade Layer Normalization).
   - Utilise une implémentation personnalisée de la normalisation de couche après chaque couche linéaire.
   - Il consiste en trois couches linéaires avec des fonctions d'activation ReLU et de Homemade Layer Normalization.
"""


import torch.nn as nn
import torch
from HOMEMADE_LN import LayerNorm

class BagOfWordsClassifier(nn.Module):
    """
    Bag-of-Words classifier without normalization layers.

    This class defines a simple Bag-of-Words classifier model without any normalization layers.
    It consists of three linear layers with ReLU activation functions.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    hidden1 : int
        Number of units in the first hidden layer.
    hidden2 : int
        Number of units in the second hidden layer.
    out_shape : int
        Output shape of the model.

    Methods
    -------
    forward(x)
        Defines the forward pass of the model.
    """
    def __init__(self, vocab_size, hidden1, hidden2, out_shape):
        super(BagOfWordsClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(vocab_size, hidden1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden2, out_shape),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.squeeze()
        x = nn.Sigmoid()(x)
        return x

class BagOfWordsClassifierLayer(nn.Module):
    """
    Bag-of-Words classifier with Layer Normalization.

    This class defines a Bag-of-Words classifier model with Layer Normalization applied after each linear layer.
    It consists of three linear layers with ReLU activation functions and Layer Normalization.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    hidden1 : int
        Number of units in the first hidden layer.
    hidden2 : int
        Number of units in the second hidden layer.
    out_shape : int
        Output shape of the model.

    Methods
    -------
    forward(x)
        Defines the forward pass of the model.
    """
    def __init__(self, vocab_size, hidden1, hidden2, out_shape):
        super(BagOfWordsClassifierLayer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(vocab_size, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden2, out_shape),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.squeeze()
        x = nn.Sigmoid()(x)
        return x

class BagOfWordsClassifierBatchNorm(nn.Module):
    """
    Bag-of-Words classifier with Batch Normalization.

    This class defines a Bag-of-Words classifier model with Batch Normalization applied after each linear layer.
    It consists of three linear layers with ReLU activation functions and Batch Normalization.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    hidden1 : int
        Number of units in the first hidden layer.
    hidden2 : int
        Number of units in the second hidden layer.
    out_shape : int
        Output shape of the model.

    Methods
    -------
    forward(x)
        Defines the forward pass of the model.
    """
    def __init__(self, vocab_size, hidden1, hidden2, out_shape):
        super(BagOfWordsClassifierBatchNorm, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(vocab_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden2, out_shape),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.squeeze()
        x = nn.Sigmoid()(x)
        return x

class BagOfWordsClassifierLayerHM(nn.Module):
    """
    Bag-of-Words classifier with Homemade Layer Normalization.

    This class defines a Bag-of-Words classifier model with homemade Layer Normalization applied after each linear layer.
    It consists of three linear layers with ReLU activation functions and homemade Layer Normalization.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    hidden1 : int
        Number of units in the first hidden layer.
    hidden2 : int
        Number of units in the second hidden layer.
    out_shape : int
        Output shape of the model.

    Methods
    -------
    forward(x)
        Defines the forward pass of the model.
    """
    def __init__(self, vocab_size, hidden1, hidden2, out_shape):
        super(BagOfWordsClassifierLayerHM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(vocab_size, hidden1),
            LayerNorm(hidden1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            LayerNorm(hidden2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden2, out_shape),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self
