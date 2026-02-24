"""
matcher.py
----------
Matches extracted soil sample parameters to required scoring columns and
converts values to canonical target units via a single Gemini call.

Grain size fractions that are not directly
reported by labs are derived from either:
  1. Differential/range bins  (e.g. "0-2 um", "2-16 µm" - See ZuidHolland example)
  2. Cumulative passing values (e.g. "Korrelgrootte < 250 um", "% < 63 µm" - see GemeenteAmsterdam example)
When both sources are present for the same fraction, the raw_range_* path
takes precedence. 

Each input dict must have: 'parameter', 'value', 'unit'
Output is either:
  - diagnostic=False: wide single-row DataFrame (one column per canonical param) - Legacy debug code
  - diagnostic=True: long DataFrame with one row per extracted parameter - required for processing

"""

import os
import json
import re
import logging
import time

import pandas as pd
from google import genai

from FeatureList import REQUIRED_COLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Target unit map (moved to seperate file)
#from unit_map import TARGET_UNITS
TARGET_UNITS: dict[str, str] = {
    "Mineralen delen ten opzichte DS":       "%",
    "Zand (63um < fractie < 2mm)":           "%",
    "(250um < fractie < 2mm)":               "%",
    "(63um < fractie < 250um)":              "%",
    "Silt (2um < fractie < 63um)":           "%",
    "(20 < fractie < 63um)":                 "%",
    "Leemfractie (fractie < 10um)":          "%",
    "Lutum (fractie < 2um)":                 "%",
    "Leem (lutum+silt)":                     "%",
    "Korrelverdeling (D60/D10)":             "-",
    "Korrelverdeling (M50)":                 "um",
    "Gehalte organische koolstof (TOC)":     "%",
    "Gehalte organische stof":               "%",
    "Kalk (CaCO3)":                          "%",
    "pH-waarde":                             "pH",
    "Grof materiaal (fractie)":              "%",
    "Bodemvreemd":                           "",
    "Fosfor totaal (destructie)":            "mg P/kg",
    "Fosfor beschikbaar":                    "mg P/kg",
    "Stikstof totaal (N-Kjeldahl)":          "mg N/kg",
    "Ammonium totaal":                       "mg NH4+/kg",
    "Stikstof levering":                     "kg N/ha",
    "C/N verhouding":                        "-",
    "Zwavel totaal":                         "%",
    "Zwavel beschikbaar":                    "mg S/kg",
    "Wateroplosbare Zouten (chloride)":      "%",
    "CEC":                                   "mmol+/kg",
    "CEC bezetting totaal":                  "%",
    "CEC bezetting calcium":                 "%",
    "CEC bezetting kalium":                  "%",
    "CEC bezetting magnesium":               "%",
    "Kalium totaal":                         "mmol+/kg",
    "Kalium beschikbaar":                    "mg K/kg",
    "Magnesium totaal":                      "mmol+/kg",
    "Magnesium beschikbaar":                 "mg Mg/kg",
    "Calcium totaal":                        "mmol+/kg",
    "Calcium beschikbaar":                   "mmol Ca/L",
    "Aluminium totaal":                      "mg Al/kg",
    "Natrium totaal":                        "mmol+/kg",
    "Zink totaal":                           "ug/kg",
    "Koper totaal":                          "ug/kg",
}

# Parameters where /ha units cannot be converted without bulk density + depth.
_DROP_IF_PER_HA: set[str] = {
    "Zwavel beschikbaar", "Zwavel totaal",
    "Fosfor beschikbaar", "Fosfor totaal (destructie)",
    "Kalium beschikbaar", "Kalium totaal",
    "Calcium beschikbaar", "Calcium totaal",
    "Magnesium beschikbaar", "Magnesium totaal",
    "Natrium totaal", "Stikstof totaal (N-Kjeldahl)",
}

# Molar mass context for Gemini
_MOLAR_MASS_NOTE = """
Molar masses for reference:
  K=39.10, K2O/2=47.10, Mg=24.31, Ca=40.08, Na=22.99, Al=26.98
  P=30.97, P2O5->P factor=0.4365, NH4=18.04, N=14.01, S=32.06
"""


# Grain size: raw-range bin names and derived fraction definitions

# Canonical bin names produced by Gemini for particle-size range measurements.
# Gemini is instructed to map e.g. "0-2 um", "fractie 0-2 µm" -> "raw_range_0-2um"
RAW_RANGE_CANONICAL = "raw_range"   # prefix — these are NOT in REQUIRED_COLS
RAW_CUM_CANONICAL   = "raw_cum"     # prefix for cumulative passing values

# All raw range bins we recognise (ordered by lower bound).
# Add new bins here if new lab formats introduce them.
ALL_RAW_BINS: list[str] = [
    "raw_range_0-2um",
    "raw_range_2-16um",
    "raw_range_16-50um",
    "raw_range_50-63um",
    "raw_range_63-125um",
    "raw_range_125-180um",
    "raw_range_180-250um",
    "raw_range_250-355um",
    "raw_range_355-500um",
    "raw_range_500-1000um",
    "raw_range_1000-2000um",
]

# All cumulative passing bins we recognise (descending order)
# Add new sizes here if new lab formats introduce them; keep descending order.
ALL_RAW_CUM_BINS: list[str] = [
    "raw_cum_2000um",
    "raw_cum_1000um",
    "raw_cum_500um",
    "raw_cum_250um",
    "raw_cum_125um",
    "raw_cum_63um",
    "raw_cum_50um",
    "raw_cum_32um",
    "raw_cum_16um",
    "raw_cum_8um",
    "raw_cum_2um",
]

# Map each cumulative bin name to its threshold in um (for arithmetic).
_CUM_BIN_SIZE_UM: dict[str, float] = {
    "raw_cum_2000um": 2000.0,
    "raw_cum_1000um": 1000.0,
    "raw_cum_500um":   500.0,
    "raw_cum_250um":   250.0,
    "raw_cum_125um":   125.0,
    "raw_cum_63um":     63.0,
    "raw_cum_50um":     50.0,
    "raw_cum_32um":     32.0,
    "raw_cum_16um":     16.0,
    "raw_cum_8um":       8.0,
    "raw_cum_2um":       2.0,
}

# Derived fractions: canonical name -> list of raw bins to sum.
# Bins that are missing from the data are skipped
RANGE_FRACTIONS: dict[str, list[str]] = {
    "Lutum (fractie < 2um)": [
        "raw_range_0-2um",
    ],
    "Slib (fractie < 16um)": [
        "raw_range_0-2um", "raw_range_2-16um",
    ],
    "Silt (2um < fractie < 63um)": [
        "raw_range_2-16um", "raw_range_16-50um", "raw_range_50-63um",
    ],
    "Zand (63um < fractie < 2mm)": [
        "raw_range_63-125um", "raw_range_125-180um", "raw_range_180-250um",
        "raw_range_250-355um", "raw_range_355-500um", "raw_range_500-1000um",
        "raw_range_1000-2000um",
    ],
    "(250um < fractie < 2mm)": [
        "raw_range_250-355um", "raw_range_355-500um",
        "raw_range_500-1000um", "raw_range_1000-2000um",
    ],
    "(63um < fractie < 250um)": [
        "raw_range_63-125um", "raw_range_125-180um", "raw_range_180-250um",
    ],
    "(20 < fractie < 63um)": [
        "raw_range_16-50um", "raw_range_50-63um",
    ],
    "Leemfractie (fractie < 10um)": [
        "raw_range_0-2um", "raw_range_2-16um",
    ],
    "Leem (lutum+silt)": [
        "raw_range_0-2um", "raw_range_2-16um",
        "raw_range_16-50um", "raw_range_50-63um",
    ],
}

# Notes explaining approximations or non-obvious bin choices
    # Only available in diagnostic_output debug code
# Shown in the calculation_detail column of diagnostic output.
RANGE_FRACTION_NOTES: dict[str, str] = {
    "(20 < fractie < 63um)":    "APPROX: no 20µm sieve — using 16µm bin boundary as proxy for lower limit",
    "Leemfractie (fractie < 10um)": "APPROX: no 10µm sieve — using sum of bins <16µm as proxy for <10µm",
}

# Prompt builder
def _build_prompt(extracted: list[dict]) -> str:
    """Build a single prompt for Gemini: parameter matching + unit conversion."""

    canonical_lines = "\n".join(
        f"  - {col}  [target: {TARGET_UNITS.get(col, '?')}]"
        for col in REQUIRED_COLS
    )

    raw_bin_lines = "\n".join(f"  - {b}" for b in ALL_RAW_BINS)
    raw_cum_bin_lines = "\n".join(f"  - {b}" for b in ALL_RAW_CUM_BINS)

    extracted_lines = "\n".join(
        f"  {i+1}. parameter=\"{e.get('parameter', '')}\"  "
        f"value=\"{e.get('value', '')}\"  unit=\"{e.get('unit', '')}\""
        for i, e in enumerate(extracted)
    )

    return f"""You are a soil science expert processing Dutch laboratory soil reports.

CANONICAL PARAMETER LIST (with required target units):
{canonical_lines}

PARTICLE-SIZE RAW BIN NAMES (use these when a parameter is a grain-size range, i.e. a differential/fraction value):
{raw_bin_lines}

CUMULATIVE PARTICLE-SIZE BIN NAMES (use these when a parameter is a cumulative passing value, e.g. "Korrelgrootte < 250 um", "% < 63 µm"):
{raw_cum_bin_lines}

EXTRACTED PARAMETERS FROM REPORT:
{extracted_lines}

{_MOLAR_MASS_NOTE}

TASK:
For each extracted parameter:
1. Determine if it matches a canonical parameter OR a raw particle-size bin.

   a) Canonical parameters:
      - Parameters may be in Dutch or abbreviated differently across labs.
      - 'bodemvoorraad' -> 'totaal', 'plantbeschikbaar' -> 'beschikbaar'.
      - One extracted parameter maps to at most one canonical.
      - If no match: set canonical=null, converted_value=null, target_unit=null.

   b) Particle-size range bins (e.g. "0-2 um", "fractie 2-16 µm"):
      - Map these to the PARTICLE-SIZE RAW BIN NAMES above.
      - Use canonical = "raw_range_<lo>-<hi>um" (e.g. "raw_range_0-2um").
      - converted_value should be the percentage value (already in %).
      - target_unit = "%".
      - If the range does not match any known bin, set canonical=null.

   c) Cumulative particle-size passing values (e.g. "Korrelgrootte < 250 um", "% passend < 63 µm"):
      - These represent the cumulative % of material finer than a given size.
      - Map these to the CUMULATIVE PARTICLE-SIZE BIN NAMES above.
      - Use canonical = "raw_cum_<size>um" (e.g. "raw_cum_250um", "raw_cum_63um").
      - converted_value should be the cumulative percentage value (already in %).
      - target_unit = "%".
      - If the size does not match any known cumulative bin, set canonical=null.

2. If matched to a canonical parameter, convert the numeric value to the
   canonical target unit.
   - Strip any '<' prefix before converting (e.g. '<0.05' -> 0.05).
   - Replace commas with periods (e.g. '1,23' -> 1.23).
   - If the value is already in the target unit, factor=1.0.
   - If the incoming unit is kg/ha AND the canonical is NOT 'Stikstof levering',
     set converted_value=null (will be dropped).
   - If the value cannot be parsed as a number, set converted_value=null.

3. Return "extracted" EXACTLY as the parameter string appears in the input
   (no number prefix, no surrounding quotes).

Return ONLY a valid JSON array — no explanation, no markdown fences. Example:
[
  {{
    "extracted": "N-totale Bodemvoorraad",
    "canonical": "Stikstof totaal (N-Kjeldahl)",
    "confidence": 0.85,
    "converted_value": 2450.0,
    "target_unit": "mg N/kg"
  }},
  {{
    "extracted": "Fractie 0-2 µm",
    "canonical": "raw_range_0-2um",
    "confidence": 0.99,
    "converted_value": 12.5,
    "target_unit": "%"
  }},
  {{
    "extracted": "Korrelgrootte < 250 um",
    "canonical": "raw_cum_250um",
    "confidence": 0.99,
    "converted_value": 68.3,
    "target_unit": "%"
  }},
  {{
    "extracted": "Microbiele Activiteit",
    "canonical": null,
    "confidence": 0.0,
    "converted_value": null,
    "target_unit": null
  }}
]
"""



# Gemini API call
def _call_gemini(prompt: str) -> list[dict]:
    """Send prompt to Gemini and return parsed JSON list. Retries on 429."""

    api_key = os.environ.get("GEMINI_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            break
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                logger.warning(
                    f"Rate limited (429), retrying in {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
            else:
                raise

    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini JSON: {e}\nRaw:\n{raw}")
        raise



# Grain size derived fraction computation
def _compute_grain_fractions(
    bin_values: dict[str, float],
    sample_id: str,
    diagnostic_rows: list,
    bin_details: dict[str, str] | None = None,
) -> dict[str, float]:
    """
    Given a mapping of {raw_range_bin: value_%}, compute all derived fractions
    defined in RANGE_FRACTIONS by summing the available constituent bins.

    Only fractions whose canonical name appears in REQUIRED_COLS are kept.

    Returns {canonical_name: summed_value} for fractions that could be computed
    (i.e. at least one constituent bin was present in bin_values).
    """
    derived: dict[str, float] = {}

    for fraction_name, bins in RANGE_FRACTIONS.items():
        if fraction_name not in REQUIRED_COLS:
            continue  # e.g. "Slib (fractie < 16um)" not required — skip

        available = [b for b in bins if b in bin_values]
        if not available:
            continue  # no constituent data present for this fraction

        summed = sum(bin_values[b] for b in available)
        derived[fraction_name] = round(summed, 6)

        missing = set(bins) - set(available)
        if missing:
            logger.warning(
                f"[{sample_id}] '{fraction_name}': missing bins {missing} "
                f"— partial sum used ({len(available)}/{len(bins)} bins)."
            )

        # Build human-readable formula for debugging.
        # If bin_details is provided (cumulative path), show derivation of each sub-bin.
        if bin_details:
            term_parts = []
            for b in available:
                sub = bin_details.get(b, f"{b}({bin_values[b]:.4f})")
                term_parts.append(f"[{b}: {sub}]")
            terms = " + ".join(term_parts)
        else:
            terms = " + ".join(f"{b}({bin_values[b]:.4f})" for b in available)
        if missing:
            terms += f"  [MISSING: {', '.join(sorted(missing))}]"
        detail = f"{terms} = {summed:.6f}"
        if fraction_name in RANGE_FRACTION_NOTES:
            detail += f" | NOTE: {RANGE_FRACTION_NOTES[fraction_name]}"

        _add_row(
            diagnostic_rows,
            sample_id=sample_id,
            extracted_parameter=f"[computed] {fraction_name}",
            extracted_value=str(summed),
            extracted_unit="%",
            canonical_parameter=fraction_name,
            converted_value=summed,
            converted_unit="%",
            confidence=1.0,
            status="ok",
            conversion_source="grain_size_computed",
            calculation_detail=detail,
        )

    return derived


def _compute_grain_fractions_from_cumulative(
    cum_bin_values: dict[str, float],
    sample_id: str,
    diagnostic_rows: list,
    already_scored: set[str],
) -> dict[str, float]:
    """
    Given a mapping of {raw_cum_<size>um: cumulative_%_passing}, derive
    canonical grain-size fractions by differencing adjacent cumulative values.

    The cumulative value at size X means "% of material with diameter < X µm".
    The fraction between sizes A and B (A < B) is:  cum_B - cum_A.

    Fractions are derived using a dynamically-constructed lookup that maps
    each RANGE_FRACTIONS bin to the cumulative bin pair that spans it.
    Any fraction whose canonical name is in already_scored (i.e. already
    computed from raw_range bins) is skipped to avoid double-counting.

    Strategy
    --------
    For each canonical fraction in RANGE_FRACTIONS:
      - Determine the total size range [lo_um, hi_um] it spans.
      - Attempt to read cum[hi_um] − cum[lo_um] directly from cum_bin_values.
      - If one or both endpoints are unavailable, log a warning and skip.

    Returns {canonical_name: computed_value} for fractions that could be derived.
    """

    # Build a lookup from threshold (um) -> cumulative value, for fast access.
    cum_by_size: dict[float, float] = {
        _CUM_BIN_SIZE_UM[name]: val
        for name, val in cum_bin_values.items()
        if name in _CUM_BIN_SIZE_UM
    }

    # For each raw_range bin, determine its [lo, hi] in um from the bin name,
    # then store the corresponding cumulative difference as a synthetic range value.
    # This lets us reuse _compute_grain_fractions's summation logic.
    import re as _re
    synthetic_range_bins: dict[str, float] = {}
    synthetic_range_bin_details: dict[str, str] = {}

    for bin_name in ALL_RAW_BINS:
        # Parse "raw_range_<lo>-<hi>um" -> (lo, hi)
        m = _re.match(r"raw_range_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)um$", bin_name)
        if not m:
            continue
        lo_um = float(m.group(1))
        hi_um = float(m.group(2))

        cum_hi = cum_by_size.get(hi_um)
        cum_lo = cum_by_size.get(lo_um)

        if cum_hi is None or cum_lo is None:
            # Missing endpoint — this range bin cannot be computed from these data.
            continue

        diff = cum_hi - cum_lo
        if diff < 0:
            logger.warning(
                f"[{sample_id}] Cumulative inversion for '{bin_name}': "
                f"cum({hi_um}µm)={cum_hi} < cum({lo_um}µm)={cum_lo}. Clamping to 0."
            )
            diff = 0.0

        synthetic_range_bins[bin_name] = round(diff, 6)
        # Record how this synthetic bin was derived for diagnostic output
        synthetic_range_bin_details[bin_name] = (
            f"cum({hi_um:.0f}um)={cum_hi:.4f} - cum({lo_um:.0f}um)={cum_lo:.4f} = {diff:.6f}"
        )

    if not synthetic_range_bins:
        return {}

    # Now delegate to the standard range-based fraction computation.
    # Pass an empty diagnostic_rows list and merge below to avoid double entries.
    temp_diag: list[dict] = []
    derived = _compute_grain_fractions(synthetic_range_bins, sample_id, temp_diag, bin_details=synthetic_range_bin_details)

    # Re-label conversion_source to make provenance clear in diagnostic output.
    for row in temp_diag:
        row["conversion_source"] = "cum_grain_size_computed"
        # Skip fractions that were already computed from raw_range bins.
        if row["canonical_parameter"] not in already_scored:
            diagnostic_rows.append(row)

    # Filter out anything already scored by raw_range path.
    return {k: v for k, v in derived.items() if k not in already_scored}



# Public interface
def match_parameters(
    extracted: list[dict],
    min_confidence: float = 0.7,
    diagnostic: bool = False,
) -> pd.DataFrame:
    """
    Match extracted soil parameters to canonical scoring columns and convert
    values to target units using Gemini.

    Particle-size range bins returned by Gemini are accumulated and used to
    compute derived grain-size fractions (Lutum, Silt, Zand, …) by summation.

    Parameters
    ----------
    extracted : list of dict
        Each dict must have keys: 'parameter', 'value', 'unit'.
        Optionally include 'sample_id' for diagnostic output.

    min_confidence : float
        Matches below this threshold are dropped. Set to 0.0 to see all.

    diagnostic : bool
        If True, returns a tuple ``(diagnostic_df, scoring_df)`` where
        *diagnostic_df* is the long-format DataFrame (one row per extracted
        parameter, showing match details, status, and conversion_source) and
        *scoring_df* is the wide single-row scored DataFrame — identical to
        what is returned when ``diagnostic=False``.  Missing canonical columns
        in *scoring_df* are filled with ``-1``.
        If False (default), returns only the wide single-row scored DataFrame.
    """

    if not extracted:
        logger.warning("No extracted parameters provided.")
        empty_scoring = pd.DataFrame([{col: -1 for col in REQUIRED_COLS}])
        empty_scoring.index = ["Value"]
        empty_scoring.columns.name = "Parameter"
        if diagnostic:
            return pd.DataFrame(columns=[
                "sample_id", "extracted_parameter", "extracted_value",
                "extracted_unit", "canonical_parameter", "converted_value",
                "converted_unit", "confidence", "status", "conversion_source",
                "calculation_detail",
            ]), empty_scoring
        return empty_scoring

    logger.info(f"Sending {len(extracted)} parameters to Gemini…")
    prompt  = _build_prompt(extracted)
    matches = _call_gemini(prompt)

    extracted_lookup = {e["parameter"]: e for e in extracted}

    scored_result:   dict[str, float] = {}
    bin_values:      dict[str, float] = {}   # raw_range_* bins -> value %
    cum_bin_values:  dict[str, float] = {}   # raw_cum_* bins   -> cumulative %
    diagnostic_rows: list[dict]       = []

    for match in matches:
        extracted_name = re.sub(r"^\d+\.\s*", "", match.get("extracted", "")).strip()
        canonical      = match.get("canonical")
        confidence     = float(match.get("confidence", 0.0))
        ai_value       = match.get("converted_value")
        ai_unit        = match.get("target_unit")

        original  = extracted_lookup.get(extracted_name, {})
        raw_value = original.get("value", None)
        unit      = str(original.get("unit", "")).strip()
        sample_id = original.get("sample_id", "")

        # Gatekeeping                                                          
        if canonical is None:
            _add_row(diagnostic_rows, sample_id, extracted_name, raw_value,
                     unit, canonical, None, None, confidence, "no_match", "none")
            continue

        if confidence < min_confidence:
            _add_row(diagnostic_rows, sample_id, extracted_name, raw_value,
                     unit, canonical, None, None, confidence,
                     f"low_confidence ({confidence:.2f} < {min_confidence:.2f})", "none")
            continue

        # Raw particle-size bin — collect for later summation                 
        if canonical.startswith(RAW_RANGE_CANONICAL):
            if canonical not in ALL_RAW_BINS:
                logger.warning(f"Unknown raw bin '{canonical}' returned by Gemini — skipping.")
                _add_row(diagnostic_rows, sample_id, extracted_name, raw_value,
                         unit, canonical, None, None, confidence, "unknown_raw_bin", "ai")
                continue

            try:
                bin_val = float(ai_value) if ai_value is not None else float(
                    str(raw_value).lstrip("<").replace(",", ".").strip()
                )
            except (TypeError, ValueError):
                _add_row(diagnostic_rows, sample_id, extracted_name, raw_value,
                         unit, canonical, None, None, confidence, "value_parse_error", "ai")
                continue

            bin_values[canonical] = bin_val
            _add_row(diagnostic_rows, sample_id, extracted_name, raw_value,
                     unit, canonical, bin_val, "%", confidence, "ok", "ai",
                     calculation_detail=f"directly extracted: {bin_val:.4f} %")
            continue

        # Cumulative particle-size bin — collect for later differencing       
        if canonical.startswith(RAW_CUM_CANONICAL):
            if canonical not in ALL_RAW_CUM_BINS:
                logger.warning(f"Unknown cumulative bin '{canonical}' returned by Gemini — skipping.")
                _add_row(diagnostic_rows, sample_id, extracted_name, raw_value,
                         unit, canonical, None, None, confidence, "unknown_raw_cum_bin", "ai")
                continue

            try:
                cum_val = float(ai_value) if ai_value is not None else float(
                    str(raw_value).lstrip("<").replace(",", ".").strip()
                )
            except (TypeError, ValueError):
                _add_row(diagnostic_rows, sample_id, extracted_name, raw_value,
                         unit, canonical, None, None, confidence, "value_parse_error", "ai")
                continue

            cum_bin_values[canonical] = cum_val
            _add_row(diagnostic_rows, sample_id, extracted_name, raw_value,
                     unit, canonical, cum_val, "%", confidence, "ok", "ai",
                     calculation_detail=f"cumulative passing (< threshold): {cum_val:.4f} %")
            continue

        # Standard canonical parameter                                         
        if canonical not in REQUIRED_COLS:
            _add_row(diagnostic_rows, sample_id, extracted_name, raw_value,
                     unit, canonical, None, None, confidence,
                     "canonical_not_in_required_cols", "none")
            continue

        # Parse raw value
        clean = str(raw_value).lstrip("<").strip().replace(",", ".")
        try:
            numeric_value = float(clean)
        except (TypeError, ValueError):
            _add_row(diagnostic_rows, sample_id, extracted_name, raw_value,
                     unit, canonical, None, None, confidence, "value_parse_error", "none")
            continue

        # Determine converted value
        converted_value = None
        converted_unit  = None
        status          = "ok"
        conversion_source = "none"

        if ai_value is not None:
            try:
                converted_value   = float(ai_value)
                converted_unit    = ai_unit or TARGET_UNITS.get(canonical, "")
                conversion_source = "ai"
            except (TypeError, ValueError):
                logger.warning(
                    f"AI returned non-numeric converted_value '{ai_value}' "
                    f"for '{extracted_name}'."
                )

        # AI gave null — check for /ha drop
        if converted_value is None:
            unit_lower = unit.lower().strip()
            if "/ha" in unit_lower and canonical in _DROP_IF_PER_HA:
                status = "dropped_per_ha"
                _add_row(diagnostic_rows, sample_id, extracted_name, raw_value,
                         unit, canonical, None, None, confidence, status, "none")
                continue

            # Neither AI nor any fallback — pass through as-is
            converted_value   = numeric_value
            converted_unit    = unit
            conversion_source = "none"
            status            = "unit_not_in_registry"

        # Persist first successful match per canonical column
        if status == "ok" and canonical not in scored_result:
            scored_result[canonical] = converted_value

        _add_row(diagnostic_rows, sample_id, extracted_name, raw_value,
                 unit, canonical, converted_value, converted_unit,
                 confidence, status, conversion_source)

    # Compute grain-size derived fractions from accumulated bins           
    sample_id_repr = extracted[0].get("sample_id", "") if extracted else ""

    # Grain-size computed values perferred over calculated ones
    grain_size_computed: set[str] = set()

    if bin_values:
        derived = _compute_grain_fractions(bin_values, sample_id_repr, diagnostic_rows)
        for canon, val in derived.items():
            if canon in scored_result:
                logger.info(
                    f"[{sample_id_repr}] '{canon}': grain-size computed value "
                    f"({val:.6f}) supersedes directly assigned value "
                    f"({scored_result[canon]:.6f})."
                )
            scored_result[canon] = val
            grain_size_computed.add(canon)

    if cum_bin_values:
        derived_cum = _compute_grain_fractions_from_cumulative(
            cum_bin_values, sample_id_repr, diagnostic_rows,
            already_scored=grain_size_computed,  # only skip raw_range-computed fractions
        )
        for canon, val in derived_cum.items():
            if canon in scored_result and canon not in grain_size_computed:
                logger.info(
                    f"[{sample_id_repr}] '{canon}': cumulative grain-size computed value "
                    f"({val:.6f}) supersedes directly assigned value "
                    f"({scored_result[canon]:.6f})."
                )
            if canon not in grain_size_computed:
                scored_result[canon] = val

    # Return                                                               
    n_ok = sum(1 for r in diagnostic_rows if r["status"] == "ok")
    logger.info(f"Matched {n_ok}/{len(extracted)} extracted parameters successfully.")

    # Build wide scoring DataFrame — missing columns filled with -1
    ordered_result = {col: scored_result.get(col, -1) for col in REQUIRED_COLS}
    scoring_df = pd.DataFrame([ordered_result])
    # Set index label to match the expected format
    scoring_df.index = ["Value"]
    scoring_df.columns.name = "Parameter"

    if diagnostic:
        diagnostic_df = pd.DataFrame(diagnostic_rows)
        return diagnostic_df, scoring_df

    return scoring_df



# Internal helper
def _add_row(
    rows: list, sample_id, extracted_parameter, extracted_value,
    extracted_unit, canonical_parameter, converted_value, converted_unit,
    confidence, status, conversion_source,
    calculation_detail: str = "",
) -> None:
    rows.append({
        "sample_id":           sample_id,
        "extracted_parameter": extracted_parameter,
        "extracted_value":     extracted_value,
        "extracted_unit":      extracted_unit,
        "canonical_parameter": canonical_parameter,
        "converted_value":     (
            round(converted_value, 6)
            if isinstance(converted_value, float)
            else converted_value
        ),
        "converted_unit":      converted_unit,
        "confidence":          confidence,
        "status":              status,
        "conversion_source":   conversion_source,
        "calculation_detail":  calculation_detail,
    })
