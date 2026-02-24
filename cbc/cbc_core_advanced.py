import sqlite3
import pandas as pd

try:
    from FeatureList import REQUIRED_COLS
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
    from FeatureList import REQUIRED_COLS


# ---------------------------
# CBC rule engine (advanced)
# ---------------------------
def run_cbc(input_df: pd.DataFrame, dbFile: str):
    """
    Evaluate a sample against target criteria stored in an SQLite database.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame containing one or more samples with columns matching EIGENSCHAP.
        Only the first row is evaluated. Must contain 'SampleID' and 'DateProcessed'.
    dbFile : str
        Path to SQLite database containing TARGET, EIGENSCHAP, and HEEFT tables.

    Returns
    -------
    results : dict
        Normalised scores per target (0–1).
    scoring_matrix : pd.DataFrame
        EigName × TargetName with values {-1, 0, 1}.
    distance_matrix : pd.DataFrame
        Percentage distances from spec limits.
    absolute_distance_matrix : pd.DataFrame
        Absolute distances from spec limits.
    spec_type_matrix : pd.DataFrame
        Spec type per cell ('range', 'min_only', 'max_only', 'none').

    Notes
    -----
    Column-name mappings (extracted → canonical eigenschap) are expected to be
    applied to ``input_df`` *before* calling this function.  A future AI-driven
    mapping layer will handle this automatically.
    """

    # ------------------------------------------------------------------
    # DB helper — single connection for all table fetches
    # ------------------------------------------------------------------
    def get_tables(*table_names):
        db = sqlite3.connect(dbFile, timeout=5)
        try:
            tables = {}
            for name in table_names:
                cs = db.execute(f"SELECT * FROM {name}")
                cols = [d[0] for d in cs.description]
                tables[name] = [dict(zip(cols, row)) for row in cs]
        finally:
            db.close()
        return tables

    tables     = get_tables("TARGET", "EIGENSCHAP", "HEEFT")
    target     = tables["TARGET"]
    eigenschap = tables["EIGENSCHAP"]
    heeft      = tables["HEEFT"]

    # ------------------------------------------------------------------
    # Build sample dict: EigID → float value (or -1 sentinel for missing)
    # ------------------------------------------------------------------
    row = input_df.iloc[0]
    sample = {}
    for e in eigenschap:
        name = str(e["Name"])
        if name in input_df.columns and pd.notna(row[name]):
            try:
                sample[str(e["EigID"])] = float(row[name])
            except (ValueError, TypeError):
                sample[str(e["EigID"])] = -1
        else:
            sample[str(e["EigID"])] = -1

    # EigID → Name lookup
    eig_name_map = {str(e["EigID"]): e["Name"] for e in eigenschap}

    target_scores = {str(t["TargetID"]): 0 for t in target}
    target_max    = {str(t["TargetID"]): 0 for t in target}
    pass_fail_matrix = []

    UPPER_SENTINEL = 999999.0
    LOWER_SENTINEL = 0.0

    for h in heeft:
        if h["Weight"] == 0:
            continue

        eig_id   = str(h["EigID"])
        s        = sample.get(eig_id, -1)
        eig_name = eig_name_map.get(eig_id, eig_id)

        # Missing → excluded from scoring entirely
        if s == -1:
            pass_fail_matrix.append({
                "TargetID":        h["TargetID"],
                "EigID":           h["EigID"],
                "EigName":         eig_name,
                "SampleValue":     s,
                "Passed":          -1,
                "Distance":        None,
                "AbsoluteDistance": None,
                "SpecType":        None,
            })
            continue

        target_max[str(h["TargetID"])] += h["Weight"]

        has_min = h["Min"] is not None and h["Min"] > LOWER_SENTINEL
        has_max = h["Max"] is not None and h["Max"] < UPPER_SENTINEL

        if has_min and has_max:
            spec_type = "range"
        elif has_min:
            spec_type = "min_only"
        elif has_max:
            spec_type = "max_only"
        else:
            spec_type = "none"

        passed            = 0
        distance          = None
        absolute_distance = None

        if has_min and has_max:
            distance_low  = s - h["Min"]
            distance_high = h["Max"] - s

            passed = 1 if h["Min"] <= s <= h["Max"] else 0

            absolute_distance = (distance_low, distance_high)

            spec_range = h["Max"] - h["Min"]
            if distance_low < 0:
                distance_low_pct = (distance_low / abs(h["Min"])) * 100 if h["Min"] != 0 else distance_low
            else:
                distance_low_pct = (distance_low / spec_range) * 100 if spec_range > 0 else 0

            if distance_high < 0:
                distance_high_pct = (distance_high / abs(h["Max"])) * 100 if h["Max"] != 0 else distance_high
            else:
                distance_high_pct = (distance_high / spec_range) * 100 if spec_range > 0 else 0

            distance = (distance_low_pct, distance_high_pct)

        elif has_min:
            distance_low      = s - h["Min"]
            passed            = 1 if s >= h["Min"] else 0
            absolute_distance = distance_low
            distance          = (distance_low / abs(h["Min"])) * 100 if h["Min"] != 0 else distance_low

        elif has_max:
            distance_high     = h["Max"] - s
            passed            = 1 if s <= h["Max"] else 0
            absolute_distance = distance_high
            distance          = (distance_high / abs(h["Max"])) * 100 if h["Max"] != 0 else distance_high

        else:
            passed = -1

        if passed == 1:
            target_scores[str(h["TargetID"])] += h["Weight"]

        pass_fail_matrix.append({
            "TargetID":         h["TargetID"],
            "EigID":            h["EigID"],
            "EigName":          eig_name,
            "SampleValue":      s,
            "Passed":           passed,
            "Distance":         distance,
            "AbsoluteDistance":  absolute_distance,
            "SpecType":         spec_type,
        })

    # Normalised scores
    results = {}
    for t in target:
        tid = str(t["TargetID"])
        results[t["Name"]] = (
            target_scores[tid] / target_max[tid] if target_max[tid] > 0 else 0
        )

    pass_fail_df = pd.DataFrame(pass_fail_matrix)
    if pass_fail_df.empty:
        return results, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    target_map = {t["TargetID"]: t["Name"] for t in target}
    pass_fail_df["TargetName"] = pass_fail_df["TargetID"].map(target_map)

    def pivot(values_col, fill=None, dtype=None):
        p = pass_fail_df.pivot_table(
            index="EigName",
            columns="TargetName",
            values=values_col,
            aggfunc="first",
        )
        if fill is not None:
            p = p.fillna(fill)
        if dtype is not None:
            p = p.astype(dtype)
        return p

    scoring_matrix           = pivot("Passed",          fill=-1, dtype=int)
    distance_matrix          = pivot("Distance")
    absolute_distance_matrix = pivot("AbsoluteDistance")
    spec_type_matrix         = pivot("SpecType")

    # Align all matrices to the scoring matrix index/columns
    for mat in (distance_matrix, absolute_distance_matrix, spec_type_matrix):
        mat = mat.reindex(index=scoring_matrix.index, columns=scoring_matrix.columns)

    return results, scoring_matrix, distance_matrix, absolute_distance_matrix, spec_type_matrix
