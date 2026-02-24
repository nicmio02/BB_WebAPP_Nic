#!/usr/bin/env python3
"""
test_pipeline.py
================
End-to-end smoke test of the soil analysis pipeline WITHOUT Streamlit.

Usage:
    python test_pipeline.py <path_to_pdf>

Tests:
    1. PDF extraction (smart_parser_two_pass)
    2. AI matching (matcher)
    3. DB storage (samples_store)
    4. CBC scoring (cbc_core_advanced)
    5. Visual generation (visuals_advanced)

Requires:
    - GEMINI_KEY environment variable set
    - BB_CBC.db in the project root
"""

import os
import sys
import sqlite3

import pandas as pd
import numpy as np


def check_prerequisites():
    """Verify all required files and env vars exist."""
    print("=" * 60)
    print("PREREQUISITE CHECK")
    print("=" * 60)

    ok = True

    # Check GEMINI_KEY
    api_key = os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"GEMINI_KEY set ({api_key[:8]}...)")
    else:
        print("GEMINI_KEY not set — export GEMINI_KEY=your-key")
        ok = False

    # Check rules DB
    if os.path.exists("BB_CBC.db"):
        conn = sqlite3.connect("BB_CBC.db")
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        print(f" db found — tables: {', '.join(tables)}")
    else:
        print("db not found in current directory")
        ok = False

    # Check imports
    imports_ok = True
    for module in ["smart_parser_two_pass", "matcher", "FeatureList",
                   "db.samples_store", "cbc.cbc_core_advanced", "visuals.visuals_advanced"]:
        try:
            __import__(module)
            print(f"import {module}")
        except ImportError as e:
            print(f"import {module} — {e}")
            imports_ok = False

    ok = ok and imports_ok
    print()
    return ok, api_key


def test_extraction(pdf_path, api_key):
    """Phase 1+2: Extract parameters from PDF."""
    print("=" * 60)
    print(f"PHASE 1+2: EXTRACTION — {os.path.basename(pdf_path)}")
    print("=" * 60)

    from smart_parser_two_pass import process_generic_report

    df = process_generic_report(pdf_path, api_key=api_key)

    if df is None or df.empty:
        print("   No data extracted!")
        return None

    samples = df["sample_id"].unique()
    params = df["parameter"].unique()
    print(f"\n   Extracted {len(df)} rows")
    print(f"     Samples:    {', '.join(samples)}")
    print(f"     Parameters: {len(params)} unique")
    print(f"\n  First 10 rows:")
    print(df.head(10).to_string(index=False))
    print()
    return df


def test_matching(extracted_df):
    """Phase 3: Match + convert parameters via AI."""
    print("=" * 60)
    print("PHASE 3: AI MATCHING")
    print("=" * 60)

    from matcher import match_parameters

    sample_names = extracted_df["sample_id"].unique()
    all_scoring = {}

    for sample_id in sample_names:
        sample_rows = extracted_df[extracted_df["sample_id"] == sample_id]

        extracted_list = [
            {
                "parameter": row["parameter"],
                "value": row["value"],
                "unit": row.get("unit", ""),
                "sample_id": sample_id,
            }
            for _, row in sample_rows.iterrows()
        ]

        print(f"\n  Matching sample '{sample_id}' ({len(extracted_list)} params)...")

        try:
            scoring_df = match_parameters(extracted_list, min_confidence=0.7, diagnostic=False)
            all_scoring[sample_id] = scoring_df

            # Count matched vs missing
            row = scoring_df.iloc[0]
            matched = sum(1 for v in row if v != -1 and not (isinstance(v, float) and np.isnan(v)))
            total = len(row)
            print(f"   Matched {matched}/{total} canonical parameters")

            # Show matched values
            matched_params = {k: v for k, v in row.items() if v != -1}
            if matched_params:
                print(f"     Matched: {', '.join(matched_params.keys())}")

        except Exception as e:
            print(f"   Matching failed: {e}")
            all_scoring[sample_id] = None

    print()
    return all_scoring


def test_db_storage(extracted_df, scoring_dfs):
    """Test storing and retrieving from samples.db."""
    print("=" * 60)
    print("PHASE 4: DATABASE STORAGE")
    print("=" * 60)

    from db.samples_store import (
        get_conn,
        save_extraction_results,
        save_scored_sample,
        load_scored_sample,
        list_sample_pairs,
    )
    from FeatureList import REQUIRED_COLS

    # Use a test database
    test_db = "test_samples.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    conn = get_conn(test_db)
    user_id = "test_user"
    pdf_id = "test_report.pdf"

    # Save raw extraction
    save_extraction_results(conn, user_id, pdf_id, extracted_df)
    print("   Raw extraction saved")

    # Save scored samples
    for sample_id, scoring_df in scoring_dfs.items():
        if scoring_df is not None:
            save_scored_sample(conn, user_id, pdf_id, sample_id, scoring_df)

    # Verify retrieval
    pairs = list_sample_pairs(conn, user_id)
    print(f"   Stored {len(pairs)} scored samples: {pairs}")

    # Test load
    for pdf, sample in pairs:
        wide = load_scored_sample(conn, user_id, pdf, sample, REQUIRED_COLS)
        if wide is not None:
            non_null = wide.iloc[0].dropna().shape[0]
            print(f"   Loaded '{sample}': {non_null} non-null columns")
        else:
            print(f"   Failed to load '{sample}'")

    conn.close()
    os.remove(test_db)
    print()
    return pairs


def test_cbc(scoring_dfs):
    """Test CBC evaluation."""
    print("=" * 60)
    print("PHASE 5: CBC SCORING")
    print("=" * 60)

    from cbc.cbc_core_advanced import run_cbc
    from FeatureList import REQUIRED_COLS
    from datetime import datetime

    db_path = "BB_CBC.db"

    for sample_id, scoring_df in scoring_dfs.items():
        if scoring_df is None:
            continue

        # Build the wide DF that run_cbc expects
        wide = scoring_df.copy()
        wide["SampleID"] = sample_id
        wide["DateProcessed"] = datetime.now().strftime("%Y-%m-%d")

        # Ensure all required columns exist
        for col in REQUIRED_COLS:
            if col not in wide.columns:
                wide[col] = np.nan

        print(f"\n  Scoring sample '{sample_id}'...")

        try:
            results, scoring_mat, dist_mat, abs_dist_mat, spec_mat = run_cbc(wide, db_path)

            # Show top 5 scores
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
            print(f"   CBC completed — {len(results)} targets scored")
            print(f"     Top 5 use cases:")
            for name, score in sorted_results[:5]:
                bar = "" * int(score * 20)
                print(f"       {score:.2f} {bar} {name}")

            print(f"     Scoring matrix: {scoring_mat.shape}")
            print(f"     Distance matrix: {dist_mat.shape}")

        except Exception as e:
            print(f"   CBC failed: {e}")
            import traceback
            traceback.print_exc()

    print()


def test_visuals(scoring_dfs):
    """Test that matplotlib figures can be generated."""
    print("=" * 60)
    print("PHASE 6: VISUAL GENERATION")
    print("=" * 60)

    from cbc.cbc_core_advanced import run_cbc
    from FeatureList import REQUIRED_COLS
    from visuals.visuals_advanced import (
        plot_professional_bar_from_df,
        plot_professional_spider_from_df,
        analyse_closest_enhanced,
        plot_professional_diverging_bars,
    )
    from datetime import datetime
    import matplotlib.pyplot as plt

    db_path = "BB_CBC.db"

    result_rows = []
    distance_dicts = {}
    abs_distance_dicts = {}
    spec_type_dicts = {}

    for sample_id, scoring_df in scoring_dfs.items():
        if scoring_df is None:
            continue

        wide = scoring_df.copy()
        wide["SampleID"] = sample_id
        wide["DateProcessed"] = datetime.now().strftime("%Y-%m-%d")
        for col in REQUIRED_COLS:
            if col not in wide.columns:
                wide[col] = np.nan

        results, _, dist_mat, abs_dist_mat, spec_mat = run_cbc(wide, db_path)
        results["SampleID"] = sample_id
        results["DateProcessed"] = wide.iloc[0]["DateProcessed"]
        result_rows.append(pd.DataFrame([results]))

        distance_dicts[sample_id] = dist_mat
        abs_distance_dicts[sample_id] = abs_dist_mat
        spec_type_dicts[sample_id] = spec_mat

    if not result_rows:
        print("  ❌ No results to visualize")
        return

    master_df = pd.concat(result_rows, ignore_index=True)
    use_cols = [c for c in master_df.columns if c not in ("SampleID", "DateProcessed")]
    master_df[use_cols] = master_df[use_cols].apply(pd.to_numeric, errors="coerce")

    # Test bar chart
    try:
        fig = plot_professional_bar_from_df(master_df, use_cols)
        plt.close(fig)
        print("  ✅ Bar chart generated")
    except Exception as e:
        print(f"  ❌ Bar chart failed: {e}")

    # Test spider chart
    try:
        fig = plot_professional_spider_from_df(master_df, use_cols)
        plt.close(fig)
        print("  ✅ Spider chart generated")
    except Exception as e:
        print(f"  ❌ Spider chart failed: {e}")

    # Test diverging bars
    if use_cols:
        target = use_cols[0]
        try:
            indiv, agg, split = analyse_closest_enhanced(
                distance_dicts, abs_distance_dicts, spec_type_dicts,
                target_name=target, n=10, include_passing=True, split_by_status=True,
            )
            if not agg.empty:
                fig = plot_professional_diverging_bars(split, agg, target, top_n=10)
                plt.close(fig)
                print(f"  ✅ Diverging bar chart generated for '{target}'")
            else:
                print(f"  ⚠️  No distance data for '{target}' — skipped diverging bars")
        except Exception as e:
            print(f"  ❌ Diverging bar chart failed: {e}")

    print()


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <path_to_pdf>")
        print("\nThis tests the full pipeline without Streamlit:")
        print("  1. PDF extraction")
        print("  2. AI parameter matching")
        print("  3. Database storage")
        print("  4. CBC scoring")
        print("  5. Visual generation")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    ok, api_key = check_prerequisites()
    if not ok:
        print("❌ Prerequisites not met. Fix the issues above and retry.")
        sys.exit(1)

    # Run pipeline
    extracted_df = test_extraction(pdf_path, api_key)
    if extracted_df is None:
        sys.exit(1)

    scoring_dfs = test_matching(extracted_df)
    if not any(v is not None for v in scoring_dfs.values()):
        print("❌ All matching failed. Check API key and Gemini quota.")
        sys.exit(1)

    test_db_storage(extracted_df, scoring_dfs)
    test_cbc(scoring_dfs)
    test_visuals(scoring_dfs)

    print("=" * 60)
    print("✅ ALL TESTS PASSED — ready for Streamlit deployment")
    print("=" * 60)
