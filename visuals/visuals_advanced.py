"""
Consolidated Chart Functions — Soil Analysis Visualizations
============================================================
Self-contained module with everything needed to produce:

  1. plot_professional_bar_from_df      — Average suitability bar chart
  2. plot_professional_spider_from_df   — Average suitability spider/radar chart
  3. plot_professional_diverging_bars   — Pass/fail diverging bar chart

Supporting calculation functions (required inputs for the diverging bar):

  4. get_distance_to_pass               — Extract signed distance + direction from a spec value
  5. analyse_closest_enhanced           — Aggregate per-sample distances into split_df / aggregated_df

Streamlit integration:

  6. show_sample_visuals_advanced       — Drop-in Streamlit page renderer (replaces old show_sample_visuals)

Utility:

  7. save_figure                        — Save a figure to PNG / PDF
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.patches import Wedge

# Optional Streamlit import — only needed for the wrapper function
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 1.5

# Color palettes
_SPIDER_BAR_COLORS = {
    'grond':         '#c5dff8',
    'dijken':        '#f4d1ae',
    'zandaggregaat': '#a8dadc',
    'kleiaggregaat': '#fcd5ce',
}

_DIVERGING_COLORS = {
    'passing': '#27ae60',
    'failing': '#e74c3c',
}


# ===========================================================================
# 4. DISTANCE CALCULATION — get_distance_to_pass
# ===========================================================================

def get_distance_to_pass(value, spec_type=None):
    """
    Extract the signed distance to the passing threshold from a CBC distance value.

    Returns
    -------
    distance : float | None
        Signed scalar distance (negative = failing).
    direction : str | None
        One of 'Below Min', 'Above Max', 'Above Min', 'Below Max',
        'Below', 'Above'.
    """
    if value is None:
        return None, None
    if not isinstance(value, tuple) and pd.isna(value):
        return None, None

    if isinstance(value, tuple):
        lower, upper = value
        if lower < 0:
            return lower, 'Below Min'
        elif upper < 0:
            return upper, 'Above Max'
        else:
            if abs(lower) <= abs(upper):
                return lower, 'Above Min'
            else:
                return upper, 'Below Max'
    else:
        if value < 0:
            if spec_type == 'min_only':
                return value, 'Below Min'
            elif spec_type == 'max_only':
                return value, 'Above Max'
            else:
                return value, 'Below'
        else:
            if spec_type == 'min_only':
                return value, 'Above Min'
            elif spec_type == 'max_only':
                return value, 'Below Max'
            else:
                return value, 'Above'


# ===========================================================================
# 5. DISTANCE AGGREGATION — analyse_closest_enhanced
# ===========================================================================

def analyse_closest_enhanced(data_dict_pct, data_dict_abs, data_dict_spec_type,
                               target_name, n=5,
                               include_passing=True, split_by_status=False):
    """
    Aggregate per-sample CBC distance results for one use-case target.

    Parameters
    ----------
    data_dict_pct : dict[str, pd.DataFrame]
        {sample_id: relative_distance_matrix} — output of run_cbc.
    data_dict_abs : dict[str, pd.DataFrame]
        {sample_id: absolute_distance_matrix} — output of run_cbc.
    data_dict_spec_type : dict[str, pd.DataFrame]
        {sample_id: spec_type_matrix} — output of run_cbc.
    target_name : str
        Column name in the distance matrices.
    n : int
        Number of closest-to-boundary features per sample.
    include_passing : bool
        Include passing features as well as failing ones.
    split_by_status : bool
        If True, return a split_df with separate rows for Passing vs Failing.

    Returns
    -------
    individual_df, aggregated_df, split_df : pd.DataFrame
    """
    all_results = []

    for sample_id in data_dict_pct.keys():
        if target_name not in data_dict_pct[sample_id].columns:
            continue

        df_pct  = data_dict_pct[sample_id]
        df_abs  = data_dict_abs[sample_id]
        df_spec = data_dict_spec_type[sample_id]

        features_data = []
        for feature in df_pct.index:
            value_pct  = df_pct.loc[feature, target_name]
            value_abs  = df_abs.loc[feature, target_name]
            stype      = df_spec.loc[feature, target_name]

            distance_pct, direction = get_distance_to_pass(value_pct, stype)
            distance_abs, _         = get_distance_to_pass(value_abs, stype)

            if distance_pct is None:
                continue

            is_failing = distance_pct < 0
            if is_failing or include_passing:
                features_data.append({
                    'Sample':               sample_id,
                    'Feature':              feature,
                    'Distance_to_Pass_Pct': distance_pct,
                    'Distance_to_Pass_Abs': distance_abs,
                    'Direction':            direction,
                    'Status':               'Failing' if is_failing else 'Passing',
                })

        if features_data:
            df_feat = pd.DataFrame(features_data)
            df_feat['Abs_Distance_Pct'] = df_feat['Distance_to_Pass_Pct'].abs()
            df_feat['Abs_Distance_Abs'] = df_feat['Distance_to_Pass_Abs'].abs()
            df_feat = df_feat.sort_values('Abs_Distance_Pct').head(n)
            all_results.append(df_feat)

    if not all_results:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    individual_df = pd.concat(all_results, ignore_index=True)

    aggregated_df = individual_df.groupby('Feature').agg(
        Samples_Count=('Distance_to_Pass_Pct', 'count'),
        Avg_Distance_to_Pass_Pct=('Distance_to_Pass_Pct', 'mean'),
        Avg_Distance_to_Pass_Abs=('Distance_to_Pass_Abs', 'mean'),
        Avg_Abs_Distance_Pct=('Abs_Distance_Pct', 'mean'),
        Avg_Abs_Distance_Abs=('Abs_Distance_Abs', 'mean'),
        Direction=('Direction',  lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
        Status=('Status',        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
    ).reset_index()

    aggregated_df = aggregated_df.sort_values('Avg_Abs_Distance_Pct')
    aggregated_df = aggregated_df[[
        'Feature', 'Samples_Count', 'Status', 'Direction',
        'Avg_Distance_to_Pass_Pct', 'Avg_Abs_Distance_Pct',
        'Avg_Distance_to_Pass_Abs', 'Avg_Abs_Distance_Abs',
    ]]

    split_df = pd.DataFrame()
    if split_by_status:
        split_df = individual_df.groupby(['Feature', 'Status']).agg(
            Count=('Distance_to_Pass_Pct', 'count'),
            Avg_Distance_to_Pass_Pct=('Distance_to_Pass_Pct', 'mean'),
            Avg_Distance_to_Pass_Abs=('Distance_to_Pass_Abs', 'mean'),
            Avg_Abs_Distance_Pct=('Abs_Distance_Pct', 'mean'),
            Avg_Abs_Distance_Abs=('Abs_Distance_Abs', 'mean'),
            Direction=('Direction', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
        ).reset_index()
        split_df = split_df.sort_values(['Feature', 'Status'])

    return individual_df, aggregated_df, split_df


# ===========================================================================
# 1. AVERAGE SUITABILITY — BAR CHART
# ===========================================================================

def plot_professional_bar_from_df(master_df, use_cols, figsize=(14, 8),
                                   horizontal=True):
    """
    Create a professional average-suitability bar chart.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mean_suitability = master_df[use_cols].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        mean_sorted = mean_suitability.sort_values(ascending=True)
        colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(mean_sorted)))
        y_pos = np.arange(len(mean_sorted))
        bars = ax.barh(y_pos, mean_sorted.values,
                       color=colors, edgecolor='white', linewidth=2,
                       alpha=0.9, height=0.7)

        for bar, value in zip(bars, mean_sorted.values):
            ax.text(bar.get_width() + 0.02,
                    bar.get_y() + bar.get_height() / 2.,
                    f'{value:.2f}', ha='left', va='center',
                    fontsize=11, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(mean_sorted.index, fontsize=11)
        ax.set_xlabel('Gemiddelde Geschiktheidsscore',
                      fontsize=13, fontweight='bold', labelpad=10)
        ax.set_xlim(0, 1.1)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    else:
        colors = plt.cm.Greens(np.linspace(0.5, 0.9, len(mean_suitability)))
        x_pos = np.arange(len(mean_suitability))
        bars = ax.bar(x_pos, mean_suitability.values,
                      color=colors, edgecolor='white', linewidth=2.5, alpha=0.9)

        for bar, value in zip(bars, mean_suitability.values):
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 0.02, f'{value:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(mean_suitability.index, rotation=45,
                           ha='right', fontsize=11)
        ax.set_ylabel('Gemiddelde Geschiktheidsscore',
                      fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    ax.set_title('Gemiddelde Geschiktheid per Gebruikstype',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    return fig


# ===========================================================================
# 2. AVERAGE SUITABILITY — SPIDER / RADAR CHART
# ===========================================================================

def plot_professional_spider_from_df(master_df, use_cols, groups=None,
                                      figsize=(14, 10)):
    """
    Create a professional average-suitability spider (radar) chart.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if groups is None:
        groups = {
            'Grond':         (0,  4),
            'Dijken':        (4,  7),
            'Zandaggregaat': (7,  12),
            'Kleiaggregaat': (12, 15),
        }

    avg_values = master_df[use_cols].mean().tolist()
    avg_values += avg_values[:1]

    angles = np.linspace(0, 2 * np.pi, len(use_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    ax.plot(angles, avg_values, linewidth=3, color='#27ae60',
            label='Gemiddelde Geschiktheid', zorder=3)
    ax.fill(angles, avg_values, color='#27ae60', alpha=0.25)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                       fontsize=10, color='gray')
    ax.set_rlabel_position(180 / len(use_cols))
    ax.grid(color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(2.5)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([str(i + 1) for i in range(len(use_cols))],
                       fontsize=13, fontweight='bold')

    ax.set_title('Gemiddeld Geschiktheidsprofiel (Alle Monsters)',
                 fontsize=16, fontweight='bold', pad=30)

    legend_text = '\n'.join(
        [f"{i + 1}. {label}" for i, label in enumerate(use_cols)]
    )
    ax.text(1.1, 0.5, legend_text, transform=ax.transAxes,
            fontsize=15, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='gray', alpha=0.95, pad=1))

    wedge_handles = []
    for name, (start, end) in groups.items():
        if start >= len(angles) - 1 or end >= len(angles):
            continue
        theta1 = np.degrees(angles[start])
        theta2 = np.degrees(angles[end])
        color = _SPIDER_BAR_COLORS.get(name.lower(), '#cccccc')
        wedge = Wedge(center=(0, 0), r=1,
                      theta1=theta1, theta2=theta2, width=1,
                      transform=ax.transData._b,
                      facecolor=color, alpha=0.15,
                      edgecolor=color, linewidth=2)
        ax.add_patch(wedge)
        wedge_handles.append(
            mpatches.Patch(facecolor=color, edgecolor=color,
                           alpha=0.5, label=name, linewidth=2)
        )

    if wedge_handles:
        ax.legend(handles=wedge_handles,
                  loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fontsize=15, ncol=4,
                  frameon=True, edgecolor='gray', fancybox=True, shadow=True)

    plt.tight_layout()
    return fig


# ===========================================================================
# 3. PASS / FAIL — DIVERGING BAR CHART
# ===========================================================================

def plot_professional_diverging_bars(split_df, aggregated_df, target_name,
                                      top_n=20, figsize=None,
                                      bar_height_per_feature=0.7):
    """
    Create a professional diverging bar chart showing distance to pass/fail
    thresholds for the top-N features of a given use case.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sns.set_style('whitegrid', {
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })

    top_features = aggregated_df.head(top_n)['Feature'].tolist()
    plot_data = split_df[split_df['Feature'].isin(top_features)].copy()

    _dir_sign = {
        'Below Min': -1,
        'Above Max':  1,
        'Above Min':  1,
        'Below Max': -1,
    }

    def _signed(row):
        sign = _dir_sign.get(
            row['Direction'],
            1 if row['Avg_Distance_to_Pass_Pct'] >= 0 else -1
        )
        return sign * abs(row['Avg_Distance_to_Pass_Pct'])

    plot_data['plot_distance'] = plot_data.apply(_signed, axis=1)

    n_features = len(top_features)
    if figsize is None:
        figsize = (14, max(6, n_features * bar_height_per_feature + 2.5))

    fig, ax = plt.subplots(figsize=figsize)

    BAR_H = 0.35
    GAP   = 0.04

    def _draw_side(rows, sign, y_center):
        """Stack bars outward from zero for a group going one direction."""
        cumulative = 0.0
        for _, row in rows.iterrows():
            width  = abs(row['plot_distance'])
            status = row['Status']
            count  = int(row['Count'])
            color  = (_DIVERGING_COLORS['passing']
                      if status == 'Passing'
                      else _DIVERGING_COLORS['failing'])

            # Place bar flush against the previous one (with a small gap)
            start = sign * (abs(cumulative) + GAP if cumulative != 0 else 0)
            ax.barh(y_center, sign * width, left=start,
                    height=BAR_H, color=color, alpha=0.85,
                    edgecolor='white', linewidth=2.5, zorder=3)

            outer_edge = start + sign * width
            nudge      = sign * max(abs(outer_edge) * 0.04, 1)
            text_x     = outer_edge + nudge
            ha         = 'left' if sign > 0 else 'right'

            ax.text(text_x, y_center + 0.07, f'{width:.1f}%',
                    ha=ha, va='center', fontsize=9,
                    fontweight='bold', color=color, zorder=5)
            ax.text(text_x, y_center - 0.13, f'(n={count})',
                    ha=ha, va='center', fontsize=7.5,
                    style='italic', color=color, alpha=0.8, zorder=5)

            cumulative += sign * (width + GAP)

    for idx, feature in enumerate(reversed(top_features)):
        y_center = idx
        feature_data = plot_data[plot_data['Feature'] == feature].copy()

        neg_rows = feature_data[feature_data['plot_distance'] < 0].sort_values(
            'plot_distance', ascending=True)
        pos_rows = feature_data[feature_data['plot_distance'] > 0].sort_values(
            'plot_distance', ascending=True)

        _draw_side(neg_rows, -1, y_center)
        _draw_side(pos_rows, +1, y_center)

    ax.axvline(x=0, color='#2c3e50', linewidth=2.5, alpha=0.8, zorder=2)
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(list(reversed(top_features)), fontsize=11, fontweight='500')
    ax.set_xlabel(
        'Relatieve afstand tot specificatiegrens (%)\n<- Onder limiet  |  Boven limiet ->',
        fontsize=12, fontweight='bold', labelpad=10,
    )
    ax.set_title(f'Afstand tot Specificatiegrens - {target_name}',
                 fontsize=16, fontweight='bold', pad=20, loc='left')
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    xlim = ax.get_xlim()
    rng  = xlim[1] - xlim[0]
    ax.set_xlim(xlim[0] - rng * 0.15, xlim[1] + rng * 0.15)
    ax.set_ylim(-0.6, n_features - 0.4)

    legend_elements = [
        mpatches.Patch(facecolor=_DIVERGING_COLORS['passing'],
                       edgecolor='white', linewidth=1.5,
                       label='Voldoet aan specificatie'),
        mpatches.Patch(facecolor=_DIVERGING_COLORS['failing'],
                       edgecolor='white', linewidth=1.5,
                       label='Voldoet niet aan specificatie'),
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right',
                       frameon=True, fontsize=10, edgecolor='gray',
                       fancybox=True, shadow=True)
    legend.get_frame().set_alpha(0.95)

    plt.tight_layout()
    return fig


# ===========================================================================
# 6. SAVE UTILITY
# ===========================================================================

def save_figure(fig, filename, dpi=300, formats=None):
    """Save a matplotlib figure to one or more file formats."""
    if formats is None:
        formats = ['png']

    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for fmt in formats:
        out = f'{filename}.{fmt}'
        fig.savefig(out, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved -> {out}")


# ===========================================================================
# 7. STREAMLIT INTEGRATION — show_sample_visuals_advanced
# ===========================================================================

def show_sample_visuals_advanced(
    result: pd.DataFrame,
    scoring_matrices: dict,
    distance_dicts: dict,
    abs_distance_dicts: dict,
    spec_type_dicts: dict,
):
    """
    Drop-in Streamlit page renderer for CBC results.

    Replaces the old ``show_sample_visuals(result, matrices, breakdowns)``.

    Parameters
    ----------
    result : pd.DataFrame
        Concatenated results with SampleID, DateProcessed, and score columns.
    scoring_matrices : dict[str, pd.DataFrame]
        {sample_label: scoring_matrix} from run_cbc.
    distance_dicts : dict[str, pd.DataFrame]
        {sample_label: distance_matrix} from run_cbc.
    abs_distance_dicts : dict[str, pd.DataFrame]
        {sample_label: absolute_distance_matrix} from run_cbc.
    spec_type_dicts : dict[str, pd.DataFrame]
        {sample_label: spec_type_matrix} from run_cbc.
    """
    if not _HAS_STREAMLIT:
        raise ImportError("Streamlit is required for show_sample_visuals_advanced()")

    if result.empty:
        st.warning("Geen geschiktheidsresultaten om weer te geven.")
        return

    use_cols = [c for c in result.columns if c not in ("SampleID", "DateProcessed")]
    result[use_cols] = result[use_cols].apply(pd.to_numeric, errors="coerce")

    sample_ids = result["SampleID"].tolist()

    # ------------------------------------------------------------------
    # Section 1: Overall scores table
    # ------------------------------------------------------------------
    st.subheader("Geschiktheidsscores per monster")
    st.dataframe(result, use_container_width=True)

    # ------------------------------------------------------------------
    # Section 2: Average charts (bar + spider) when multiple samples
    # ------------------------------------------------------------------
    if len(sample_ids) >= 1:
        st.subheader("Gemiddelde Geschiktheid")

        col_bar, col_spider = st.columns(2)

        with col_bar:
            fig_bar = plot_professional_bar_from_df(result, use_cols)
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        with col_spider:
            fig_spider = plot_professional_spider_from_df(result, use_cols)
            st.pyplot(fig_spider)
            plt.close(fig_spider)

    # ------------------------------------------------------------------
    # Section 3: Per-target diverging bar analysis
    # ------------------------------------------------------------------
    st.subheader("Afstandsanalyse per Gebruiksdoel")

    # Build target list from the score columns
    target_names = use_cols

    selected_target = st.selectbox(
        "Selecteer een gebruiksdoel voor gedetailleerde analyse",
        target_names,
        key="diverging_target_select",
    )

    if selected_target:
        individual_df, aggregated_df, split_df = analyse_closest_enhanced(
            distance_dicts,
            abs_distance_dicts,
            spec_type_dicts,
            target_name=selected_target,
            n=20,
            include_passing=True,
            split_by_status=True,
        )

        if aggregated_df.empty:
            st.info(f"Geen afstandsdata beschikbaar voor '{selected_target}'.")
        else:
            fig_div = plot_professional_diverging_bars(
                split_df, aggregated_df, selected_target, top_n=15,
            )
            st.pyplot(fig_div)
            plt.close(fig_div)

            with st.expander("Geaggregeerde afstandstabel", expanded=False):
                st.dataframe(aggregated_df, use_container_width=True)

            with st.expander("Individuele monsterdata", expanded=False):
                st.dataframe(individual_df, use_container_width=True)

    # ------------------------------------------------------------------
    # Section 4: Per-sample scoring matrix heatmaps
    # ------------------------------------------------------------------
    st.subheader("Pass/Fail Matrix per Monster")

    for sample_label in sample_ids:
        pf = scoring_matrices.get(sample_label)
        if pf is None or pf.empty:
            continue

        with st.expander(f"Scoring matrix — {sample_label}", expanded=False):
            # Filter out rows that are entirely missing (-1)
            pf_display = pf.loc[~(pf.eq(-1).all(axis=1))].copy()
            pf_display = pf_display.replace({-1: np.nan})

            # Color-code: 1=green, 0=red, NaN=gray
            def _color_cell(val):
                if pd.isna(val):
                    return "background-color: #f0f0f0; color: #999"
                elif val == 1:
                    return "background-color: #d4edda; color: #155724"
                elif val == 0:
                    return "background-color: #f8d7da; color: #721c24"
                return ""

            st.dataframe(
                pf_display.style.map(_color_cell),
                use_container_width=True,
            )


# ===========================================================================
# Quick self-test
# ===========================================================================

if __name__ == '__main__':
    print("visuals_advanced.py — all functions loaded successfully.\n")
    print("Calculation helpers:")
    print("  get_distance_to_pass(value, spec_type)")
    print("  analyse_closest_enhanced(...)\n")
    print("Plot functions (return fig):")
    print("  plot_professional_bar_from_df(master_df, use_cols, ...)")
    print("  plot_professional_spider_from_df(master_df, use_cols, ...)")
    print("  plot_professional_diverging_bars(split_df, aggregated_df, target_name, ...)\n")
    print("Streamlit wrapper:")
    print("  show_sample_visuals_advanced(result, scoring_matrices, distance_dicts, ...)\n")
    print("Utility:")
    print("  save_figure(fig, filename, dpi, formats)")
