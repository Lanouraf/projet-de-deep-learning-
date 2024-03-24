import streamlit as st
from stream_comparaison_BOW import comparaison_bag_of_words
from Stream_comparaison_Lenet import comparaison_Lenet
def comparaison():
    st.title("Comparaison BN/LN")
    st.write(
        "Sur cet onglet, nous comparons les performances de deux modèles :"
        " l'un avec Batch Normalization (BN) et l'autre avec Layer Normalization (LN)."
        " Nous évaluons leurs performances sur différentes tâches."
    )

    # Section pour choisir la tâche
    st.subheader("Choix de la tâche")
    task = st.selectbox("Sélectionnez la tâche", ["Bag of Words", "Lenet-5"])

    # Section pour afficher les datasets et les performances des modèles
    st.subheader("Comparaison des performances")
    if task == "Bag of Words":
        comparaison_bag_of_words()
    elif task == "Lenet-5":
        comparaison_Lenet()