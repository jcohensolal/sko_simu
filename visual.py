from io import BytesIO
import json
import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

pd.set_option('future.no_silent_downcasting', True)


###### Logging setup ######
default_log_args = {
    "level": logging.INFO,
    "format": "[%(levelname)s] - %(message)s",
    "datefmt": "%d-%b-%y %H:%M",
    "force": True,
}
logging.basicConfig(**default_log_args)
LOG = logging.getLogger(__name__)


# Seaborn style & palette
sns.set_style("whitegrid")
sns.set_palette("muted")


def global_styling():
    # Remove blank spaces
    st.markdown('''
        <style>
            /* Remove top bar with red gradient */
            [data-testid="stDecoration"] {display: none;}

           /* Remove blank space at top */
           .block-container {
               padding-top: 0rem;
            }

           /* Remove blank space at the center canvas */
           .st-emotion-cache-bm2z3a {
               position: relative;
               top: 0px;
               }
        </style>''',
        unsafe_allow_html=True)

    # Hide anchor symbols
    st.markdown('''
        <style>
            h1 a {
                display: none;
            }
        </style>''',
        unsafe_allow_html=True)


def format_df(df, metric_col):
    df_out = df.copy()

    # Format metric_col as string, others numeric if possible
    df_out[metric_col] = df_out[metric_col].fillna("").astype(str)
    for col in df_out.columns.drop(metric_col):
        if pd.api.types.is_numeric_dtype(df_out[col]):
            df_out[col] = df_out[col].apply(lambda v: "" if pd.isna(v) else f"{v:,.2f}")
        else:
            df_out[col] = df_out[col].fillna("").astype(str)
    return df_out


def get_row_bg_color(metric_val):
    if metric_val.startswith("rank1") or metric_val.startswith("rank10"):
        return "#ebebeb"
    elif metric_val.startswith("rank3"):
        return "#ffffff"
    return ""


def build_html_table(df, metric_col, display_col_map):
    df_fmt = format_df(df, metric_col)
    html = "<table style='width:100%; border-collapse: collapse; font-size: 14px;'>"
    html += "<thead><tr>" + "".join(
        f"<th style='padding:8px; border-bottom:2px solid #ccc; text-align:left; font-weight:bold;'>{display_col_map.get(c, c)}</th>" 
        for c in df_fmt.columns) + "</tr></thead><tbody>"

    for _, row in df_fmt.iterrows():
        bg = get_row_bg_color(row[metric_col])
        tr_style = f"background-color:{bg};" if bg else ""
        html += f"<tr style='{tr_style}'>"
        for i, col in enumerate(df_fmt.columns):
            val = row[col]
            val_html = f"<strong>{val}</strong>" if i == 0 else val
            html += f"<td style='padding:8px; border-bottom:1px solid #eee; text-align:left;'>{val_html}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html


def display_aggregate_table(df_raw):
    metric_col = df_raw.columns[0]
    display_col_map = {
        "avg": "avg", 
        "0pct": "min", 
        "25pct": "25%", 
        "50pct": "50%", 
        "75pct": "75%",
        "90pct": "90%", 
        "95pct": "95%", 
        "99pct": "99%", 
        "100pct": "max"
    }

    df_tournament = df_raw[~df_raw[metric_col].astype(str).str.startswith("rank")]
    df_player = df_raw[df_raw[metric_col].astype(str).str.startswith("rank")]

    st.subheader("🏆 Results")
    st.markdown(build_html_table(df_tournament, metric_col, display_col_map), unsafe_allow_html=True)

    st.subheader("🎯 Rank-Specific Results")
    st.markdown(build_html_table(df_player, metric_col, display_col_map), unsafe_allow_html=True)


def downloadable_jsons(results_bytes, results_title, figure_buf, pp_cfg, levels_cfg):
    cols = st.columns(6)
    with cols[0]:
        st.download_button("📥 Simu Results", results_bytes, file_name=results_title, mime="text/csv")
    with cols[1]:
        st.download_button("📥 Reg Prize Pool", json.dumps(pp_cfg, indent=4), "payouts.json", "application/json")
    with cols[2]:
        st.download_button("📥 Token levels", json.dumps(levels_cfg, indent=4), "levels.json", "application/json")
    with cols[3]:
        st.download_button("📥 Histogram", figure_buf, results_title.replace(".csv", ".png"), "image/png")


def plot_hists_tournaments(all_tournament_results, expected_pp):
    df = pd.DataFrame(all_tournament_results)
    metric = "total_pp"

    if metric not in df or df[metric].dropna().empty:
        st.warning("No data found in all_tournament_results for 'total_pp'.")
        return

    series = pd.to_numeric(df[metric].dropna(), errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(series, bins=125, kde=True, stat="percent", ax=ax)

    mean_val = series.mean()
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1)
    ax.annotate(f"mean: {mean_val:,.2f}", xy=(mean_val, ax.get_ylim()[1]*0.85), xytext=(5, 0),
                textcoords="offset points", color="red", fontsize=9)

    ax.axvline(expected_pp, color="black", linestyle="--", linewidth=1)
    ax.annotate(f"expected: {expected_pp:,.2f}", xy=(expected_pp, ax.get_ylim()[1]*0.75), xytext=(5, 0),
                textcoords="offset points", color="black", fontsize=9)

    ax.set(title="Total Prize Pool (€)", xlabel="", ylabel="Probability Density (%)")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.session_state["figure_buf"] = buf
