"""
=======================================================================
SLIDERS PRODUITS STRUCTURES
Tom Uzan
=======================================================================

Point d'entree : langue, puis le menu (principal.py). Le pricing est dans
moteur.py. Version anglaise : traduction.py.

Lancer avec :  streamlit run app.py
=======================================================================
"""

import streamlit as st

import traduction

st.set_page_config(page_title="Sliders Produits Structurés", layout="wide")

anglais = st.sidebar.toggle(
    "English", key="anglais", bind="query-params", on_change=traduction.bascule,
    help="Affiche l'appli en anglais. / Switches the app to English.")
traduction.module("principal", anglais).lance()
