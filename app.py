import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from stream_homemade_layernorm import homemade_layernorm
from stream_comparaison import comparaison
from stream_homemade_batchnorm import homemade_batchnormalisation

def main():
    st.title("Batch and Layer Normalization")
    app_mode = st.sidebar.selectbox(
        "Choisissez l'expérience",
        [
            "Home",
            "Homemade Layer Normalisation",
            "Homemade Batch Normalisation", 
            "Comparaison Layer et Batch"
        ],
    )
    if app_mode == "Home":
        st.write("Bienvenue sur l'application Streamlit de Layer Normalisation et de Batch Normalisation")
        st.write("Cette application explore les concepts de Layer Normalisation et de Batch Normalisation.")
        st.write("Pour continuer, veuillez sélectionner une expérience dans la barre latérale.")
        
    
    elif app_mode == "Homemade Layer Normalisation":
        homemade_layernorm()
        
    elif app_mode == "Homemade Batch Normalisation":
        homemade_batchnormalisation()
        
    elif app_mode == "Comparaison Layer et Batch":
        comparaison()

if __name__ == "__main__":
    main()