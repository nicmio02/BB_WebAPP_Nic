# pages/Parameter_Configuration.py
import os
import sys
import sqlite3
import streamlit as st
from app_common import setup_page

# Make parent folder importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auth_config import get_authenticator
from db.samples_store import (
    get_conn as get_samples_conn,
    update_parameter_mapping,
    get_global_mappings,
    get_local_mappings
)

st.set_page_config(page_title="Parameter Configuratie", layout="wide",
    page_icon="user_profile_logos/bagger_consortium_logo.png")

# 1. Run the page setup function
if not setup_page():
    st.stop()



user_id = st.session_state.get("username")
st.title("Parameter Configuratie")

# --- DATABASE CONNECTIONS ---
samples_conn = get_samples_conn()
bagger_db_path = "baggerTool_v7.db"

if not os.path.exists(bagger_db_path):
    st.error(f"Regeldatabase '{bagger_db_path}' niet gevonden.")
    st.stop()

rules_conn = sqlite3.connect(bagger_db_path)

# --- 1. SELECT PDF ---
pdf_query = samples_conn.execute(
    "SELECT DISTINCT pdf_id FROM extracted_samples WHERE user_id = ? ORDER BY pdf_id",
    (user_id,)
)
pdf_list = [r[0] for r in pdf_query.fetchall()]

if not pdf_list:
    st.info("Geen geëxtraheerde PDF's gevonden.")
    st.stop()

st.sidebar.header("Context Selectie")
selected_pdf = st.sidebar.selectbox("Actief Rapport (PDF)", pdf_list)

# --- 2. SELECT USAGE TARGET ---
target_query = rules_conn.execute("SELECT TargetID, Name FROM TARGET ORDER BY Name")
targets = {r[1]: r[0] for r in target_query.fetchall()}

selected_target_name = st.sidebar.selectbox("Gebruiksscenario (Doel)", list(targets.keys()))
selected_target_id = targets[selected_target_name]

# --- 3. DATA FETCHING ---

# [UPDATED] Get Min/Max along with Name
req_query = """
    SELECT E.Name, H.Weight, H.Min, H.Max
    FROM HEEFT H
    JOIN EIGENSCHAP E ON H.EigID = E.EigID
    WHERE H.TargetID = ? AND H.Weight > 0
    ORDER BY E.Name
"""
req_rows = rules_conn.execute(req_query, (selected_target_id,)).fetchall()
# Store as list of dicts for easier access
required_props = [
    {"Name": r[0], "Weight": r[1], "Min": r[2], "Max": r[3]}
    for r in req_rows
]

param_query = """
    SELECT DISTINCT parameter, unit 
    FROM extracted_samples 
    WHERE user_id = ? AND pdf_id = ?
    ORDER BY parameter, unit
"""
param_rows = samples_conn.execute(param_query, (user_id, selected_pdf)).fetchall()


def format_full_id(param, unit):
    p = str(param).strip()
    u = str(unit).strip() if unit else ""
    if u: return f"{p} ({u})"
    return p


available_params = [format_full_id(r[0], r[1]) for r in param_rows]
available_params_set = set(available_params)

# Get mappings
global_map = get_global_mappings(samples_conn)
local_map = get_local_mappings(samples_conn, user_id, selected_pdf)


def get_best_source_for_target(target_name, local_map, global_map, available_set):
    for source, target in local_map.items():
        if target == target_name:
            return source, "Manual"
    candidates = [source for source, target in global_map.items() if target == target_name]
    matches_in_pdf = [c for c in candidates if c in available_set]
    if matches_in_pdf:
        return matches_in_pdf[0], "Global"
    return None, "None"


# --- 4. MAPPING INTERFACE ---
st.markdown(f"### Koppelingsregels voor: **{selected_target_name}**")
st.caption("Koppel de geëxtraheerde parameters uit uw PDF aan de eigenschappen die vereist zijn voor de rekenregels.")

mapping_container = st.container()

with mapping_container:
    # [HEADER]
    c1, c2, c3, c4 = st.columns([3, 1.5, 3, 1.5])
    c1.markdown("**Vereiste Eigenschap**")
    c2.markdown("**Specificaties (Min - Max)**")
    c3.markdown("**Geëxtraheerde Parameter**")
    c4.markdown("**Status**")

    st.divider()

    for prop_data in required_props:
        prop_name = prop_data["Name"]
        min_val = prop_data["Min"]
        max_val = prop_data["Max"]

        # Format Range String
        if min_val is not None and max_val is not None:
            range_str = f"{min_val} - {max_val}"
        elif min_val is not None:
            range_str = f"> {min_val}"
        elif max_val is not None:
            range_str = f"< {max_val}"
        else:
            range_str = "-"

        # [COLUMNS]
        col1, col2, col3, col4 = st.columns([3, 1.5, 3, 1.5])

        # 1. Property Name
        col1.markdown(f"**{prop_name}**")

        # 2. Specs/Range
        col2.caption(range_str)

        # 3. Dropdown
        active_source, source_type = get_best_source_for_target(prop_name, local_map, global_map, available_params_set)
        options = ["(Niet gekoppeld)"] + available_params

        try:
            current_index = options.index(active_source) if active_source in options else 0
        except ValueError:
            current_index = 0

        selected_source = col3.selectbox(
            f"Selecteer koppeling voor {prop_name}",
            options,
            index=current_index,
            key=f"sel_{prop_name}",
            label_visibility="collapsed"
        )

        # 4. Status Badge
        if selected_source != "(Niet gekoppeld)":
            # Logic to determine badge color
            if source_type == "Manual" or (source_type == "Global" and selected_source != active_source):
                if selected_source != active_source:
                    col4.info("Opgeslagen")
                elif source_type == "Manual":
                    col4.info("Handmatig")
                else:
                    col4.success("Globaal")
            elif source_type == "Global":
                col4.success("Globaal")
            else:
                col4.info("Handmatig")

            # Save Logic
            if selected_source != active_source:
                update_parameter_mapping(samples_conn, user_id, selected_pdf, selected_source, prop_name)
                st.toast(f"Opgeslagen: {selected_source}")
                local_map[selected_source] = prop_name

        else:
            col4.caption("—")  # Cleaner than "Missing" text

            # Reset Logic
            if active_source is not None and source_type == "Manual":
                update_parameter_mapping(samples_conn, user_id, selected_pdf, active_source, "RESET")
                st.toast(f"Ontkoppeld: {prop_name}")
                if active_source in local_map:
                    del local_map[active_source]

samples_conn.close()
rules_conn.close()