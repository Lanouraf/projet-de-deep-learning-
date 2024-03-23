import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from homemade_layernorm import homemade_layernorm

def main():
    st.title("Batch and Layer Normalization")
    app_mode = st.sidebar.selectbox(
        "Choisissez l'expérience",
        [
            "Home",
            "Homemade Layer Normalisation"
        ],
    )
    if app_mode == "Home":
        st.write("Bienvenue sur l'application Streamlit de Layer Normalisation et de Batch Normalisation")
        st.write("Cette application explore les concepts de Layer Normalisation et de Batch Normalisation.")
        st.write("Pour continuer, veuillez sélectionner une expérience dans la barre latérale.")
        
    
    elif app_mode == "Homemade Layer Normalisation":
        homemade_layernorm()

if __name__ == "__main__":
    main()