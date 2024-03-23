import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Sequences import prep_data
from LNtrain import LNtest
from LNModule import BagOfWordsClassifier, BagOfWordsClassifierLayer, BagOfWordsClassifierLayerHM
from google_drive_downloader import GoogleDriveDownloader as gdd

# Charger les données
train_dataset, train_loader, test_dataset, test_loader = prep_data()

# Télécharger les modèles
gdd.download_file_from_google_drive(file_id='1P28VRYGSueQ7DO7zghG5', dest_path='./BOW.pth')
gdd.download_file_from_google_drive(file_id='1INsBCiiYbPC7zNbP7Ufp5jfb0Ku2rTA', dest_path='./BOW_LN.pth')
gdd.download_file_from_google_drive(file_id='1TstrD3ic2NDqxy_rpFpFp4gYhyvtPZGQ', dest_path='./BOW_HM_LN.pth')
gdd.download_file_from_google_drive(file_id='1lESMtwwy_qzfWtpfpGDQYKzLOkABrIyp', dest_path='./losses.pth')

# Charger les modèles
def load_models():
    vocab_size = len(train_dataset.vectorizer.vocabulary_)
    hidden1 = 128
    hidden2 = 64
    output_shape = 1
    model1 = BagOfWordsClassifier(vocab_size, hidden1, hidden2, output_shape)
    model1.load_state_dict(torch.load('BOW.pth'))
    model2 = BagOfWordsClassifierLayer(vocab_size, hidden1, hidden2, output_shape)
    model2.load_state_dict(torch.load('BOW_LN.pth'))
    model3 = BagOfWordsClassifierLayerHM(vocab_size, hidden1, hidden2, output_shape)
    model3.load_state_dict(torch.load('BOW_HM_LN.pth'))
    return model1, model2, model3

model1, model2, model3 = load_models()

# Charger les losses
losses = torch.load('./losses.pth')

# Fonction pour calculer l'accuracy
def calculate_accuracy(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, targets in test_loader:
            outputs = model(features.float())
            predicted = torch.round(outputs)
            total += targets.size(0)
            correct += (predicted == targets.float()).sum().item()
    return correct / total

# Fonction pour afficher les graphiques de pertes
def plot_losses(losses):
    for model_name, loss in losses.items():
        plt.plot(loss, label=model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses of Models')
    plt.legend()
    st.pyplot()

# Interface utilisateur Streamlit
st.title('Comparaison des Modèles')

# Sélection des modèles à comparer
selected_models = st.multiselect('Choisissez les modèles à comparer', list(losses.keys()))

# Afficher les graphiques de pertes
if selected_models:
    selected_losses = {model_name: losses[model_name] for model_name in selected_models}
    plot_losses(selected_losses)

# Afficher les accuracies
st.subheader('Accuracies des modèles sélectionnés')
if 'Simple Model' in selected_models:
    acc1 = calculate_accuracy(model1, test_loader)
    st.write('Accuracy du modèle simple:', acc1)
if 'Model with Homemade LayerNorm' in selected_models:
    acc2 = calculate_accuracy(model2, test_loader)
    st.write('Accuracy du modèle avec Homemade LayerNorm:', acc2)
if 'Model with Torch LayerNorm' in selected_models:
    acc3 = calculate_accuracy(model3, test_loader)
    st.write('Accuracy du modèle avec Torch LayerNorm:', acc3)
