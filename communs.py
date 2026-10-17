"""Constantes et simulations mises en cache, partagees par toutes les pages."""

import streamlit as st

import moteur as M


ROUGE, VERT, BLEU, GRIS, ORANGE = "#C8102E", "#2E7D32", "#1565C0", "#6E6E6E", "#E07B00"
MAX_CACHE = 6


# =====================================================================
# ENVELOPPES MISES EN CACHE
# =====================================================================

@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def sim_bs(spot, T, r, q, sigma, n, seed=42):
    return M.simule_bs(spot, T, r, q, sigma, n, seed=seed)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def sim_bates(spot, T, r, q, n, v0, theta, kappa, xi, rho, lam, mu_j, sig_j, seed=42):
    return M.simule_bates(spot, T, r, q, n, v0, theta, kappa, xi, rho,
                          lam, mu_j, sig_j, seed=seed)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def sim_panier(n_actifs, corr, spot, T, r, q, sigma, n, seed=42):
    return M.simule_panier(n_actifs, corr, spot, T, r, q, sigma, n, seed=seed)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def sim_defaut(n, T, spread_bps, recovery, seed=123):
    return M.tire_defaut(n, T, spread_bps, recovery, seed)
