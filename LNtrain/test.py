"""
Ce script contient des fonctions pour l'entraînement et l'évaluation de modèles de réseaux de neurones avec normalisation de couche (Layer Normalization).
Il utilise les bibliothèques PyTorch pour la création et l'entraînement des modèles, ainsi que numpy pour le traitement des données.
Les principales fonctions incluses sont LNtrain, qui entraîne un modèle avec Layer Normalization, et LNtest, qui évalue la performance du modèle sur un ensemble de données de test.
Ces fonctions sont conçues pour être utilisées dans le contexte de l'apprentissage de réseaux de neurones pour des tâches de classification.
"""


from tqdm import tqdm, tqdm_notebook
from LNprepdata import prep_data
import torch
import numpy as np

def LNtrain(model, optimizer, criterion):
    """
    Function to train a neural network model with Layer Normalization.

    This function performs training of the given neural network model using Layer Normalization.
    It iterates over the training data for a fixed number of epochs, updating the model parameters based on the
    computed loss and using the specified optimizer and loss function.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for updating model parameters during training.
    criterion : torch.nn.Module
        The loss function used for computing the loss.

    Returns
    -------
    model : torch.nn.Module
        The trained neural network model.
    train_losses : list
        A list containing the training losses for each epoch.

    """
    model.train()
    train_losses = []
    for epoch in range(20):
        progress_bar = tqdm_notebook(train_loader, leave=False)
        losses = []
        total = 0
        for inputs, target in progress_bar:
            inputs = inputs.squeeze().float()
            targets = target.float()
            model.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f'Loss: {loss.item():.3f}')
            losses.append(loss.item())
            total += 1
        epoch_loss = sum(losses) / total
        train_losses.append(epoch_loss)
        tqdm.write(f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.3f}')
    return model, train_losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def LNtest(model, test_loader):
    """
    Function to test a neural network model with Layer Normalization.

    This function evaluates the performance of the given neural network model using Layer Normalization
    on a test dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The trained neural network model to be evaluated.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.

    Returns
    -------
    accuracy : float
        The accuracy of the model on the test dataset.

    """
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(test_loader):
            features = features.to(device)
            targets = targets.float().to(device)
            logits = model(features.squeeze().float())
            predicted_labels = torch.round(logits)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets.squeeze()).sum()
    accuracy = np.round(float((correct_pred.float() / num_examples)), 4) * 100
    return accuracy
