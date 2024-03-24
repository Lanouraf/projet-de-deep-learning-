import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd

def comparaison_bag_of_words():
    st.title("Comparaison Bag of Words")

    # Charger les pertes depuis le fichier .pth correspondant
    gdd.download_file_from_google_drive(file_id='1Yng2XlYhJtC7HBr8LtzJGiRQijFLQr0z', dest_path='./accuracies.csv', overwrite=True, showsize=True)

    accuracies_df = pd.read_csv("accuracies.csv", header=None, index_col=0)
    accuracies_dict = accuracies_df.to_dict()[1]

    # Pertes pour le modèle BOW_BN
    losses_bn_values = [0.7055076360702515, 0.6567512804811652, 0.625949577851729, 0.5994973914189772, 0.5777895396406, 0.5590631772171367, 0.5467127371918071, 0.52896648645401, 0.508357825604352, 0.4908906573599035, 0.4729962078007785, 0.4611104184930975, 0.4428300613706762, 0.42585317113182763, 0.42101734605702484, 0.42463927377354016, 0.40683170882138336, 0.3997000510042364, 0.3884751336141066, 0.3464887656948783]

    # Pertes pour le modèle BOW_LN
    losses_ln_values = [0.7124035683545199, 0.6793716008012946, 0.6638796004382047, 0.640407453883778, 0.6062358482317491, 0.5624494877728549, 0.5102048299529336, 0.45655843344601715, 0.40757705677639355, 0.3679321516643871, 0.33734892173246905, 0.31466537578539416, 0.29774568433111365, 0.2851671441034837, 0.2753246602686969, 0.267489492893219, 0.2606926750053059, 0.2546943453225223, 0.24980780482292175, 0.24394505674188788]

    # Accuracy pour le modèle BOW_BN
    accuracy_bn = accuracies_dict.get("BOW_BN")

    # Accuracy pour le modèle BOW_LN
    accuracy_ln = accuracies_dict.get("BOW_LN")

    # Afficher les graphiques de pertes
    st.subheader("Graphiques de Pertes")
    fig, ax = plt.subplots()
    ax.plot(losses_bn_values, label="Batch Normalization")
    ax.plot(losses_ln_values, label="Layer Normalization")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig)

    # Afficher les accuracies
    st.subheader("Accuracy")
    st.write(f"Accuracy Batch Normalization: {accuracy_bn}")
    st.write(f"Accuracy Layer Normalization: {accuracy_ln}")

