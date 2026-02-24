# Home.py
"""
Upload page: accepts a soil report PDF, extracts parameters via the two-pass
Gemini parser, then matches + converts values to canonical scoring columns
using the AI matcher. Results are stored for analysis on the Analyse page.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from db.samples_store import (
    get_conn as get_samples_conn,
    save_extraction_results,
    save_scored_sample,
    list_sample_pairs,
)
from smart_parser_two_pass import process_generic_report
from matcher import match_parameters
from app_common import setup_page

st.set_page_config(
    page_title="Soil Report Intelligence",
    layout="wide",
    page_icon="user_profile_logos/bagger_consortium_logo.png",
)

# 1. Gatekeeper
if not setup_page():
    st.stop()

# --- SECRETS ---
api_key = os.getenv("GEMINI_KEY") or st.secrets.get("GEMINI_KEY")
if not api_key:
    st.error("🚨 GEMINI_KEY is missing! Please add it to .streamlit/secrets.toml or Streamlit Cloud Secrets.")
    st.stop()

# Also set for the matcher (it reads GEMINI_API_KEY)
os.environ.setdefault("GEMINI_API_KEY", api_key)

username = st.session_state.get("username")
name = st.session_state.get("name", username)
user_id = username

# --- HEADER ---
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("Het Circulaire Bagger Consortium")
    st.markdown("### Bagger Classificatie Tool")
with header_col2:
    st.image("user_profile_logos/bagger_consortium_logo.png", width=150)

col1, col2 = st.columns(2)
col1.info(f"👤 **Ingelogd als:** {name}")

st.divider()

# --- UPLOAD ---
st.subheader("Importeer Waterbodem Onderzoek")
agreement_checked = st.checkbox(
    "Ik ga akkoord met de [Gebruiksvoorwaarden](https://www.circulairebaggerconsortium.nl/) "
    "en bevestig dat ik dit rapport mag verwerken."
)

uploaded_file = st.file_uploader("Kies een PDF", type=["pdf"], disabled=not agreement_checked)

if uploaded_file is not None:
    pdf_id = uploaded_file.name
    pdf_bytes = uploaded_file.getvalue()

    # Persist PDF locally for the parser
    cache_dir = Path("extractions")
    cache_dir.mkdir(exist_ok=True)
    pdf_cache_path = cache_dir / pdf_id
    with open(pdf_cache_path, "wb") as f:
        f.write(pdf_bytes)

    # ── Phase 1 + 2: Extract raw parameters from PDF ─────────────────
    extracted_df = None
    with st.spinner("📄 Extracting parameters from PDF..."):
        try:
            extracted_df = process_generic_report(
                str(pdf_cache_path),
                api_key=api_key,
            )
        except Exception as e:
            st.error(f"Parsing failed: {e}")

    if extracted_df is None or extracted_df.empty:
        st.warning("No samples were extracted from this upload.")
    else:
        sample_names = sorted(extracted_df["sample_id"].unique())
        parameters = sorted(extracted_df["parameter"].unique())

        st.success(f"Extracted {len(parameters)} parameters across {len(sample_names)} samples.")
        st.write("**Samples detected:**", ", ".join(sample_names))

        # ── Phase 3: AI Matching + Unit Conversion per sample ─────────
        samples_conn = get_samples_conn()

        # Store raw extraction for traceability
        save_extraction_results(samples_conn, user_id, pdf_id, extracted_df)

        # Check if scored results already exist for this PDF
        existing_pairs = list_sample_pairs(samples_conn, user_id)
        existing_for_pdf = {s for p, s in existing_pairs if p == pdf_id}

        if existing_for_pdf and not st.checkbox(
            "Scored results already exist for this PDF. Re-run matching?",
            value=False,
            key="force_rematch",
        ):
            st.info(
                f"Using existing scored results for {len(existing_for_pdf)} samples. "
                f"Go to **Analyse** to view results."
            )
            samples_conn.close()
        else:
            progress_bar = st.progress(0, text="Matching parameters...")
            total = len(sample_names)

            for idx, sample_id in enumerate(sample_names):
                sample_rows = extracted_df[extracted_df["sample_id"] == sample_id]

                # Build the list-of-dicts that matcher expects
                extracted_list = [
                    {
                        "parameter": row["parameter"],
                        "value":     row["value"],
                        "unit":      row.get("unit", ""),
                        "sample_id": sample_id,
                    }
                    for _, row in sample_rows.iterrows()
                ]

                progress_bar.progress(
                    (idx + 1) / total,
                    text=f"Matching sample {idx + 1}/{total}: {sample_id}...",
                )

                try:
                    scoring_df = match_parameters(
                        extracted_list,
                        min_confidence=0.7,
                        diagnostic=False,
                    )
                    save_scored_sample(samples_conn, user_id, pdf_id, sample_id, scoring_df)
                except Exception as e:
                    st.warning(f"Matching failed for sample '{sample_id}': {e}")

            progress_bar.empty()
            samples_conn.close()

            st.success(
                f"✅ Stored {len(sample_names)} matched samples from '{pdf_id}' for {user_id}. "
                f"Go to **Analyse** to view results."
            )
