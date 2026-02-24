# pages/Analyse.py
"""
Analysis page: loads scored (matched + converted) samples from the database,
runs CBC evaluation, and renders interactive visuals.
"""

import os
import sys

import streamlit as st
import pandas as pd
from datetime import datetime

# Make parent folder importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.samples_store import (
    get_conn as get_samples_conn,
    load_scored_sample,
    list_sample_pairs,
)
from cbc.cbc_core_advanced import run_cbc
from FeatureList import REQUIRED_COLS
from visuals.visuals_advanced import show_sample_visuals_advanced
from app_common import setup_page

# Page setup 
st.set_page_config(
    page_title="Analyse",
    layout="wide",
    page_icon="user_profile_logos/bagger_consortium_logo.png",
)

if not setup_page():
    st.stop()

user_id = st.session_state.get("username")
st.title("Bekijk bestaande PDF monsters")

# Fetch available scored samples  
samples_conn = get_samples_conn()
sample_pairs = list_sample_pairs(samples_conn, user_id)

if not sample_pairs:
    st.info("Nog geen opgeslagen monsters gevonden voor deze gebruiker.")
    samples_conn.close()
    st.stop()

selected_samples = st.multiselect(
    "Selecteer monsters om te visualiseren",
    sample_pairs,
    format_func=lambda pair: f"{pair[0]} — {pair[1]}",
)

if not selected_samples:
    st.info("Kies minimaal één monster om de visualisaties te tonen.")
    samples_conn.close()
    st.stop()

# Validate rules DB  
db_path = "BB_CBC.db"
if not os.path.exists(db_path):
    st.error("Lokale regeldatabase 'baggerTool_v7.db' niet gevonden.")
    samples_conn.close()
    st.stop()

# CBC evaluation loop 
result_rows        = []
scoring_matrices   = {}
distance_dicts     = {}
abs_distance_dicts = {}
spec_type_dicts    = {}

for pdf_id, sample_name in selected_samples:
    # Load pre-matched wide DataFrame — no mapping step needed
    wide = load_scored_sample(samples_conn, user_id, pdf_id, sample_name, REQUIRED_COLS)

    if wide is None or wide.empty:
        st.warning(
            f"Geen scored data gevonden voor monster "
            f"{sample_name} in {pdf_id}, deze wordt overgeslagen."
        )
        continue

    sample_label = f"{pdf_id} — {sample_name}"
    wide["SampleID"] = sample_label

    # Run the advanced CBC engine (handles its own DB connection)
    results_dict, scoring_mat, dist_mat, abs_dist_mat, spec_mat = run_cbc(
        wide, db_path
    )

    # Build a 1-row results DataFrame
    results_dict["SampleID"] = sample_label
    results_dict["DateProcessed"] = wide.iloc[0].get(
        "DateProcessed", datetime.now().strftime("%Y-%m-%d")
    )
    result_rows.append(pd.DataFrame([results_dict]))

    # Store per-sample matrices for the visual layer
    scoring_matrices[sample_label]   = scoring_mat
    distance_dicts[sample_label]     = dist_mat
    abs_distance_dicts[sample_label] = abs_dist_mat
    spec_type_dicts[sample_label]    = spec_mat

samples_conn.close()

# Render visuals
if not result_rows:
    st.warning(
        "Er konden geen CBC resultaten berekend worden voor de geselecteerde monsters."
    )
else:
    result = pd.concat(result_rows, ignore_index=True)
    show_sample_visuals_advanced(
        result,
        scoring_matrices,
        distance_dicts,
        abs_distance_dicts,
        spec_type_dicts,
    )
