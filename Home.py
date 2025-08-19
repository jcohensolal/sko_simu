# Imports
import gc
import logging

import streamlit as st

import process, simu, visual as visual


###### Logging setup ######
default_log_args = {
    "level": logging.INFO,
    "format": "[%(levelname)s] - %(message)s",
    "datefmt": "%d-%b-%y %H:%M",
    "force": True,
}
logging.basicConfig(**default_log_args)
LOG = logging.getLogger(__name__)


# Config icon/title
st.set_page_config(
    page_title="SKO Simulator",
    layout='wide')


def clear_memory():
    # Clear results from previous execution
    keys_to_clear = [
        "settings", "results", "results_bytes", "results_title", "tournament_results", "player_results", 
        "figure_buf", "pp_cfg", "level_cfg"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Garbage collect
    gc.collect()


############################################################################
############################################################################

# --- Sidebar ---
with st.sidebar:
    css = '''
        <style>
            [data-testid="stSidebar"] {
                width: 325px !important;
            }
            [data-testid='stSidebarNav'] > ul {
                min-height: 23vh;
            }
        </style>
        '''
    st.markdown(css, unsafe_allow_html=True)

    # Button to start the simulation
    st.markdown("<div style='text-align: center; font-size: 26px; font-weight: bold;'>Simulation</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        div.stButton > button {
            display: block;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    run_button = st.button("Run Simulation", key="Run", type="primary", args=())
    st.selectbox("Simulations Count", key="Simu_Count", options=[1, 10, 100, 500, 1000, 2500], index=3)

    st.markdown("<div style='text-align: center; font-size: 26px; font-weight: bold;'>Tournament Settings</div>", unsafe_allow_html=True)
    st.selectbox("Players in Tournament", key="Players_Count", options=[100, 500, 1000, 5000], index=2)
    st.selectbox("Buy-In (€)", key="Buy_In", options=[10, 50], index=0)
    st.selectbox("Players Per Table", key="PlayersPerTable", options=[6], index=0)
    st.selectbox("Starting Stack", key="Starting_Stack", options=[5000, 10000, 20000], index=1)

    st.markdown("<div style='text-align: center; font-size: 26px; font-weight: bold;'>Prize Pool Definition</div>", unsafe_allow_html=True)
    st.selectbox("ITM%", key="ITM_Pct", options=[12, 14, 16], index=2)
    st.selectbox("Min Payout Factor", key="Payout_Factor", options=[1.0, 1.25], index=0)
    st.selectbox("Regular Prize Pool (%) (Rake=10%, rest for KO Prize Pool)", key="Regular_Pct", options=[45], index=0)

    st.markdown("<div style='text-align: center; font-size: 26px; font-weight: bold;'>Space Settings</div>", unsafe_allow_html=True)
    st.selectbox("Max Token Level", key="Max_Token", options=[14, 13, 12, 11, 10], index=0)
    st.selectbox("Min Multiplier", key="Min_Multiplier", options=[0.4, 0.5], index=0)
    st.selectbox("Max Multiplier", key="Max_Multiplier", options=[100, 75, 50, 25], index=0)


# General styling
visual.global_styling()

# Title
st.write("")
st.markdown("<div style='text-align: center; font-size: 44px; font-weight: bold;'>SKO Simulator</div>", unsafe_allow_html=True)

# Initialize state variables
if "results" not in st.session_state:
    st.session_state["results"] = None
    st.session_state["results_bytes"] = None
    st.session_state["results_title"] = None
    st.session_state["settings"] = None
    st.session_state["tournament_results"] = None
    st.session_state["player_results"] = None


if run_button:
    # Clear RAM
    clear_memory()

    # Run simulations
    all_tournament_results, all_player_results, settings = \
        simu.run_simus()
    st.session_state["tournament_results"] = all_tournament_results
    st.session_state["player_results"] = all_player_results
    st.session_state["settings"] = settings

    # Store processed results
    results = process.compute_aggregates(all_tournament_results, all_player_results, settings["expected_pp"])
    st.session_state["results"] = results
    
    results_bytes, results_title = process.create_download_results(
        results,
        settings["n_simus"], settings["n_players"], settings["buy_in"],
        settings["players_per_table"], settings["starting_stack"],
        settings["itm_pct"], settings["payout_factor"],
        settings["reg_pp_pct"], settings["max_token"]
    )        
    st.session_state["results_bytes"] = results_bytes
    st.session_state["results_title"] = results_title

# Display results if they exist
if st.session_state["results"] is not None:
    results = st.session_state["results"]
    settings = st.session_state["settings"]

    visual.display_aggregate_table(results)

    visual.plot_hists_tournaments(
        st.session_state["tournament_results"],
        settings["expected_pp"]
    )

    visual.downloadable_jsons(
        st.session_state["results_bytes"],
        st.session_state["results_title"],
        st.session_state["figure_buf"],
        settings["pp_cfg"],
        settings["levels_cfg"]
    )
    
LOG.info("Simulation ended successfully")
