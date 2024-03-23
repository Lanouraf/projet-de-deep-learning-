import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from homemade_layernorm import homemade_layernorm

def main():
    st.title("Batch and Layer Normalization")
    app_mode = st.sidebar.selectbox(
        "Choose the experiment",
        [
            "Home",
            "Homemade Layer Normalization",
            "homemade batch normalisation"
        ],
    )
    if app_mode == "Home":
        st.write("To continue select a mode in the selection box to the left.")
    elif app_mode == "Homemade Layer Normalization":
        homemade_layernorm()

if __name__ == "__main__":
    main()
