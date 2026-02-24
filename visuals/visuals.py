# visuals.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def _status_label(passed: int) -> str:
    if passed == 1:
        return "Pass"
    if passed == 0:
        return "Fail"
    return "Missing"


def _status_color(passed: int) -> str:
    if passed == 1:
        return "#2ca02c"
    if passed == 0:
        return "#d62728"
    return "#7f7f7f"


def _render_target_breakdown(target_df: pd.DataFrame, sample_name: str, target_name: str):
    display_df = target_df.copy()
    display_df["Status"] = display_df["Passed"].apply(_status_label)
    display_df["SampleValue"] = display_df["SampleValue"].replace(-1, np.nan)

    st.markdown(f"**{target_name}**")
    st.dataframe(
        display_df[
            ["EigName", "SampleValue", "Min", "Max", "Weight", "Status"]
        ].rename(columns={"EigName": "Eigenschap"}),
        use_container_width=True,
    )

    with_limits = display_df[display_df["Min"].notna() & display_df["Max"].notna()].copy()
    if with_limits.empty:
        st.info("No acceptable limits defined for this target.")
        return

    with_limits["RangeWidth"] = with_limits["Max"] - with_limits["Min"]
    with_limits["PlotValue"] = with_limits["SampleValue"]
    marker_colors = [_status_color(val) for val in with_limits["Passed"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=with_limits["EigName"],
        x=with_limits["RangeWidth"],
        base=with_limits["Min"],
        orientation="h",
        marker=dict(color="rgba(55, 128, 191, 0.3)"),
        name="Acceptable range",
        hovertemplate="Min: %{base}<br>Max: %{x+base}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        y=with_limits["EigName"],
        x=with_limits["PlotValue"],
        mode="markers",
        marker=dict(color=marker_colors, size=10),
        name="Sample value",
        hovertemplate="Value: %{x}<extra></extra>",
    ))

    x_min = np.nanmin([with_limits["Min"].min(), with_limits["PlotValue"].min()])
    x_max = np.nanmax([with_limits["Max"].max(), with_limits["PlotValue"].max()])
    padding = (x_max - x_min) * 0.1 if x_max > x_min else 1

    fig.update_layout(
        title=f"{target_name} – value ranges ({sample_name})",
        xaxis=dict(range=[x_min - padding, x_max + padding]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=120, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_sample_visuals(
    result: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    breakdowns: dict[str, pd.DataFrame] | None = None,
):
    """
    Render tabs with one tab per sample + optional 'Average' tab.
    Uses Plotly for radar, heatmap, and top-N bar chart.
    """
    if result.empty:
        st.warning("No suitability results to display.")
        return

    st.subheader("Suitability scores per sample")
    st.dataframe(result)

    st.subheader("Visuals")

    sample_ids = [r["SampleID"] for _, r in result.iterrows()]
    tab_labels = sample_ids.copy()

    # Add an extra "Average" tab if we have more than one sample
    has_average = len(sample_ids) > 1
    if has_average:
        tab_labels.append("Average")

    tabs = st.tabs(tab_labels)
    use_cols = [c for c in result.columns if c not in ("SampleID", "DateProcessed")]

    for idx, tab in enumerate(tabs):
        with tab:
            # --- Per-sample tabs ---
            if idx < len(sample_ids):
                sample_name = sample_ids[idx]
                row = result[result["SampleID"] == sample_name].iloc[0]

                c1, c2 = st.columns(2)

                # Radar / spider chart (Plotly)
                with c1:
                    vals = row[use_cols].astype(float).values
                    categories = use_cols

                    radar_fig = go.Figure()
                    radar_fig.add_trace(go.Scatterpolar(
                        r=list(vals) + [vals[0]],
                        theta=categories + [categories[0]],
                        fill="toself",
                        name=sample_name,
                    ))
                    radar_fig.update_layout(
                        title=f"Radar – {sample_name}",
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False,
                        margin=dict(l=30, r=30, t=40, b=20),
                    )
                    st.plotly_chart(radar_fig, use_container_width=True)

                # Pass/fail heatmap (Plotly)
                with c2:
                    pf = matrices.get(sample_name)
                    if pf is None or pf.empty:
                        st.info("No pass/fail matrix available for this sample.")
                    else:
                        pf2 = pf.loc[~(pf.eq(-1).all(axis=1))]
                        pf_plot = pf2.replace({-1: np.nan})

                        heatmap_fig = go.Figure(data=go.Heatmap(
                            z=pf_plot.values,
                            x=list(pf_plot.columns),
                            y=list(pf_plot.index),
                            zmin=0,
                            zmax=1,
                            colorbar=dict(title="Pass/Fail"),
                        ))
                        heatmap_fig.update_layout(
                            title=f"Pass/Fail by Property – {sample_name}",
                            xaxis=dict(side="top"),
                            margin=dict(l=80, r=20, t=60, b=40),
                        )
                        st.plotly_chart(heatmap_fig, use_container_width=True)

                # Top-N bar chart (Plotly)
                top_n = 3
                s = row[use_cols].astype(float).sort_values(ascending=False).head(top_n)
                bar_fig = px.bar(
                    x=s.index,
                    y=s.values,
                    labels={"x": "Use", "y": "Suitability"},
                    title=f"Top {top_n} uses – {sample_name}",
                )
                bar_fig.update_yaxes(range=[0, 1])
                st.plotly_chart(bar_fig, use_container_width=True)

                breakdown_df = (breakdowns or {}).get(sample_name)
                if breakdown_df is not None and not breakdown_df.empty:
                    st.subheader("Score breakdown")
                    targets = (
                        breakdown_df["TargetName"]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                    for target_name in targets:
                        target_df = breakdown_df[breakdown_df["TargetName"] == target_name]
                        if target_df.empty:
                            continue
                        with st.expander(f"{target_name} details", expanded=False):
                            _render_target_breakdown(target_df, sample_name, target_name)
                else:
                    st.info("No detailed breakdown available for this sample.")

            # --- Final "Average" tab ---
            else:
                avg_vals = result[use_cols].astype(float).mean(axis=0)
                avg_name = "Average of samples"

                c1, c2 = st.columns(2)

                # Radar for average
                with c1:
                    categories = use_cols
                    vals = avg_vals.values

                    radar_fig = go.Figure()
                    radar_fig.add_trace(go.Scatterpolar(
                        r=list(vals) + [vals[0]],
                        theta=categories + [categories[0]],
                        fill="toself",
                        name=avg_name,
                    ))
                    radar_fig.update_layout(
                        title=f"Radar – {avg_name}",
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False,
                        margin=dict(l=30, r=30, t=40, b=20),
                    )
                    st.plotly_chart(radar_fig, use_container_width=True)

                # Average pass/fail heatmap
                with c2:
                    pf_list = []
                    for sid in sample_ids:
                        pf = matrices.get(sid)
                        if pf is not None and not pf.empty:
                            pf_list.append(pf.replace({-1: np.nan}))

                    if not pf_list:
                        st.info("No pass/fail matrices available to average.")
                    else:
                        all_index = sorted(set().union(*[pf.index for pf in pf_list]))
                        all_cols = sorted(set().union(*[pf.columns for pf in pf_list]))

                        aligned = [
                            pf.reindex(index=all_index, columns=all_cols)
                            for pf in pf_list
                        ]

                        stack = np.stack([a.values for a in aligned], axis=0)
                        avg_vals_pf = np.nanmean(stack, axis=0)
                        avg_pf = pd.DataFrame(avg_vals_pf, index=all_index, columns=all_cols)

                        heatmap_fig = go.Figure(data=go.Heatmap(
                            z=avg_pf.values,
                            x=list(avg_pf.columns),
                            y=list(avg_pf.index),
                            zmin=0,
                            zmax=1,
                            colorbar=dict(title="Avg pass rate"),
                        ))
                        heatmap_fig.update_layout(
                            title="Average Pass/Fail by Property – all samples",
                            xaxis=dict(side="top"),
                            margin=dict(l=80, r=20, t=60, b=40),
                        )
                        st.plotly_chart(heatmap_fig, use_container_width=True)

                # Top 3 uses from the average scores
                top_n = 3
                s = avg_vals.sort_values(ascending=False).head(top_n)
                bar_fig = px.bar(
                    x=s.index,
                    y=s.values,
                    labels={"x": "Use", "y": "Suitability"},
                    title=f"Top {top_n} uses – {avg_name}",
                )
                bar_fig.update_yaxes(range=[0, 1])
                st.plotly_chart(bar_fig, use_container_width=True)


