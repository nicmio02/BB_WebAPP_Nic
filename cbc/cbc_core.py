import sqlite3
import pandas as pd

# ---------------------------
# 1) Parameter rename map (Updated based on provided DB export)
# ---------------------------
# rename_map = {
#     # --- Carbon / Organische stof ---
#     "Anorg. koolstof (CaCO3) (% (m/m) ds)": "Kalk (CaCO3)",
#     "Anorganisch koolstof (als C) (g/kg ds)": "Kalk (CaCO3)",
#     "C-anorganisch (%)": "Kalk (CaCO3)",
#     "Koolzure kalk (%)": "Kalk (CaCO3)",
#
#     "C-organisch (%)": "Gehalte organische koolstof (TOC)",
#
#     "Organische stof (% (m/m) ds)": "Gehalte organische stof",
#     "Organische stof (%)": "Gehalte organische stof",
#
#     # --- Nutriënten (Concentraties voor regels) ---
#     "N-totale bodemvoorraad (mg N/kg)": "Stikstof totaal (N-Kjeldahl)",
#
#     "S-totale bodemvoorraad (mg S/kg)": "Zwavel totaal",
#     "S-plantbeschikbaar (mg S/kg)": "Zwavel beschikbaar",
#
#     # Fosfor
#     "P-bodemvoorraad (mg P/100 g)": "Fosfor totaal (destructie)",
#     "P-bodemvoorraad (mg P2O5/100 g)": "Fosfor totaal (destructie)",
#     "P-plantbeschikbaar (mg P/kg)": "Fosfor beschikbaar",
#
#     # Kalium
#     "K-bodemvoorraad (mmol+/kg)": "Kalium totaal",
#     "K-plantbeschikbaar (mg K/kg)": "Kalium beschikbaar",
#
#     # Overige kationen (CEC context)
#     "Ca-bodemvoorraad (mmol+/kg)": "Calcium totaal",
#     "Mg-bodemvoorraad (mmol+/kg)": "Magnesium totaal",
#
#     # pH
#     "Zuurgraad (pH)": "pH-waarde",
#     "Zuurgraad (pH-CaCl2) (pH unit)": "pH-waarde",  # Alternatief indien gebruikt
#
#     # --- Fracties / Textuur ---
#     # Lutum (< 2um)
#     "Klei (<2 µm) (%)": "Lutum (fractie < 2um)",
#     "Korrelgrootte < 2 µm, gravimetrisch (% (m/m) ds)": "Lutum (fractie < 2um)",
#     "Korrelgrootte < 2 µm, laser (% min. delen)": "Lutum (fractie < 2um)",  # Fallback
#
#     # Silt (2-63um, maar 2-50um komt vaak voor in landbouw)
#     "Silt (2-50 µm) (%)": "Silt (2um < fractie < 63um)",
#
#     # Zand (> 50/63um)
#     "Zand (>50 µm) (%)": "Zand (63um < fractie < 2mm)",
#
#     # Overig textuur (Mapping naar dichtstbijzijnde CBC definitie indien mogelijk)
#     "Korrelgrootte < 63 µm (% (m/m) ds)": "(63um < fractie < 250um)",
#     # Ruwe schatting, vaak 'fractie < 63' is totaal fijn
#
#     # --- Metalen ---
#     "Koper (Cu) (mg/kg ds)": "Koper totaal",
#     "Zink (Zn) (mg/kg ds)": "Zink totaal",
#
#     # --- Entries to drop / ignore (Landbouwkundige voorraden in kg/ha) ---
#     "N-totale bodemvoorraad (kg N/ha)": None,
#     "S-totale bodemvoorraad (kg S/ha)": None,
#     "P-bodemvoorraad (kg P/ha)": None,
#     "K-bodemvoorraad (kg K/ha)": None,
#     "Ca-bodemvoorraad (kg Ca/ha)": None,
#     "Mg-bodemvoorraad (kg Mg/ha)": None,
#     "Na-bodemvoorraad (kg Na/ha)": None,
#
#     "Aanvoer effectieve organische stof (gewasresten) (kg/ha)": None,
#     "Fosfaat (P2O5) - Bodemgericht Advies (kg/ha)": None,
#     "Kali (K2O) - Bodemgericht Advies (kg/ha)": None,
#
#     # --- Overige te negeren parameters ---
#     "Gloeirest (% (m/m) ds)": None,
#     "Droge stof (% (m/m))": None,
#     "Arseen (As) (mg/kg ds)": None,
#     "Barium (Ba) (mg/kg ds)": None,
#     "Cadmium (Cd) (mg/kg ds)": None,
#     "Chroom (Cr) (mg/kg ds)": None,
#     "Kwik (Hg) (mg/kg ds)": None,
#     "Lood (Pb) (mg/kg ds)": None,
#     "Nikkel (Ni) (mg/kg ds)": None,
# }

required_cols = [
    "Mineralen delen ten opzichte DS",
    "Zand (63um < fractie < 2mm)",
    "(250um < fractie < 2mm)",
    "(63um < fractie < 250um)",
    "Silt (2um < fractie < 63um)",
    "(20 < fractie < 63um)",
    "Leemfractie (fractie < 10um)",
    "Lutum (fractie < 2um)",
    "Leem (lutum+silt)",
    "Korrelverdeling (D60/D10)",
    "Korrelverdeling (M50)",
    "Gehalte organische koolstof (TOC)",
    "Gehalte organische stof",
    "Kalk (CaCO3)",
    "pH-waarde",
    "Grof materiaal (fractie)",
    "Bodemvreemd",
    "Fosfor totaal (destructie)",
    "Fosfor beschikbaar",
    "Stikstof totaal (N-Kjeldahl)",
    "Ammonium totaal",
    "Stikstof levering",
    "C/N verhouding",
    "Zwavel totaal",
    "Zwavel beschikbaar",
    "Kalium totaal",
    "Kalium beschikbaar",
    "Koper totaal",
    "Zink totaal",
]

# ---------------------------
# 6) CBC rule engine
# ---------------------------
def run_cbc(sample_wide_df: pd.DataFrame, db_conn: sqlite3.Connection, custom_mappings: dict = None):
    """
    Evaluate CBC rules for a single-sample wide dataframe.

    Args:
        sample_wide_df: The data (one row).
        db_conn: Connection to rules DB.
        custom_mappings: Dict { 'extracted_col_name': 'standard_eigenschap_name' }
    Returns:
      - results_df: 1-row DataFrame with columns [<targets...>, SampleID, DateProcessed]
      - compact_matrix: DataFrame EigName x TargetName with values {-1, 0, 1}
      - detail_df: DataFrame with per-rule details (Target/Eigenschap, ranges, values, pass/fail)
    """

    # --- 0. Apply Mappings (Single Pass with Collision Protection) ---
    if custom_mappings:
        # 1. Identify targets
        new_targets = list(custom_mappings.values())

        # 2. Collision Check: Drop placeholders (NaN) before renaming
        collisions = [col for col in new_targets if col in sample_wide_df.columns]
        if collisions:
            sample_wide_df = sample_wide_df.drop(columns=collisions)

        # 3. Rename
        sample_wide_df = sample_wide_df.rename(columns=custom_mappings)

    def get_table(name: str):
        cs = db_conn.execute(f"SELECT * FROM {name}")
        cols = [d[0] for d in cs.description]
        return [dict(zip(cols, row)) for row in cs]

    target = get_table("TARGET")
    eigenschap = get_table("EIGENSCHAP")
    heeft = get_table("HEEFT")

    # --- build sample dict: key = EigID, value = sample value ---
    row = sample_wide_df.iloc[0]
    sample = {}

    for e in eigenschap:
        eig_id = str(e["EigID"])
        name = str(e["Name"])  # expected column name in sample_wide_df (now after rename)
        if name in sample_wide_df.columns:
            try:
                val = float(row[name]) if pd.notna(row[name]) else -1
            except Exception:
                val = -1
        else:
            val = -1  # sentinel for "missing"
        sample[eig_id] = val

    # --- initialise scores and tracking ---
    target_scores = {str(t["TargetID"]): 0.0 for t in target}
    target_max = {str(t["TargetID"]): 0.0 for t in target}
    pass_fail_matrix = []
    target_map = {str(t["TargetID"]): t["Name"] for t in target}
    eig_map = {str(e["EigID"]): e["Name"] for e in eigenschap}

    # --- evaluate rules (non-zero weights only) ---
    for h in heeft:
        weight = h["Weight"]
        if weight == 0:
            continue

        tid = str(h["TargetID"])
        eig_id = str(h["EigID"])
        s = sample.get(eig_id, -1)

        # skip missing sentinel (-1) completely
        min_val = h["Min"]
        max_val = h["Max"]

        if s == -1:
            passed = -1
        elif (min_val is not None) and (max_val is not None) and (min_val <= s <= max_val):
            target_scores[tid] += weight
            passed = 1
        else:
            passed = 0

        if s != -1:
            target_max[tid] += weight

        pass_fail_matrix.append({
            "TargetID": h["TargetID"],
            "EigID": h["EigID"],
            "TargetName": target_map.get(tid),
            "EigName": eig_map.get(eig_id),
            "SampleValue": s,
            "Min": min_val,
            "Max": max_val,
            "Weight": weight,
            "Passed": passed,
        })

    # --- compute suitability scores per target ---
    results = {}
    for t in target:
        tid = str(t["TargetID"])
        denom = target_max[tid]
        results[t["Name"]] = (target_scores[tid] / denom) if denom > 0 else 0.0

    # convert results dict -> 1-row DataFrame and attach IDs
    results_row = results.copy()
    results_row["SampleID"] = sample_wide_df.loc[0, "SampleID"]
    results_row["DateProcessed"] = sample_wide_df.loc[0, "DateProcessed"]
    results_df = pd.DataFrame([results_row])

    # --- build pass/fail matrix as in your original code ---
    pass_fail_df = pd.DataFrame(pass_fail_matrix)
    if pass_fail_df.empty:
        return results_df, pd.DataFrame(), pd.DataFrame()

    if "TargetName" not in pass_fail_df.columns:
        pass_fail_df["TargetName"] = pass_fail_df["TargetID"].map(target_map)

    compact_matrix = pass_fail_df.pivot_table(
        index="EigName",
        columns="TargetName",
        values="Passed",
        aggfunc="first",
    ).fillna(-1).astype(int)

    detail_df = pass_fail_df[
        [
            "TargetID",
            "TargetName",
            "EigID",
            "EigName",
            "Weight",
            "Min",
            "Max",
            "SampleValue",
            "Passed",
        ]
    ].copy()

    return results_df, compact_matrix, detail_df
