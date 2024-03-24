"""
utilisé dans stream_comparaison.py

Ce script utilise Streamlit pour comparer les performances des modèles Bag of Words avec Batch Normalization (BN) et Layer Normalization (LN).

Il télécharge les données d'accuracy depuis un fichier CSV sur Google Drive, puis trace les graphiques des pertes (loss) pour les modèles BOW_BN (avec BN) et BOW_LN (avec LN). Ensuite, il affiche les valeurs d'accuracy pour chaque modèle.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd