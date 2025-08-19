from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import accumulate
import logging
import json
import os
import random
from time import time

import streamlit as st


###### Logging setup ######
default_log_args = {
    "level": logging.INFO,
    "format": "[%(levelname)s] - %(message)s",
    "datefmt": "%d-%b-%y %H:%M",
    "force": True,
}
logging.basicConfig(**default_log_args)
LOG = logging.getLogger(__name__)


DEFAULT_BUYIN = 10


def preprocess_levels(levels_cfg):
    processed = {}
    for lvl, data in levels_cfg.items():
        probs = data["probas"]
        cum_probs = list(accumulate(p['proba'] for p in probs))
        multipliers = [p['multiplier'] for p in probs]
        processed[lvl] = {
            "cum_probs": cum_probs,
            "multipliers": multipliers,
            "value_base10euro": data["value_base10euro"]
        }
    return processed


def simulate_allin(alive_players):
    """Simulate an all-in confrontation between two players."""
    # 2 random idx in the list, 1st one always wins allin (for simplicity)
    idx1, idx2 = random.sample(range(len(alive_players)), 2)
    p1 = alive_players[idx1]
    p2 = alive_players[idx2]
    LOG.debug("All-in player 1: %s", p1)
    LOG.debug("All-in player 2: %s", p2)

    # Stack in play from loser goes to winner
    stack_in_play = min(p1["stack"], p2["stack"])
    p1["stack"] += stack_in_play
    p2["stack"] -= stack_in_play

    return alive_players, idx1, idx2


def get_random_ko_multiplier(preprocessed_levels, token_level):
    """Get a random KO multiplier based on token level."""
    # Retrieve all possibilities for current token level
    try:
        level_data = preprocessed_levels[str(token_level)]
        cum_probs = level_data["cum_probs"]
        multipliers = level_data["multipliers"]
    except KeyError:
        raise ValueError("Level not found")

    # Draw a random value and find related multiplier from this token level
    draw = random.random()
    idx = bisect_left(cum_probs, draw)
    return multipliers[idx]


def get_token_level(preprocessed_levels, token_value, buyin_scale):
    """Determine the token level based on its value using preprocessed levels."""
    value = token_value / buyin_scale
    LOG.debug("token_value %s", value)

    for level in sorted(preprocessed_levels.keys(), key=int):
        if value <= preprocessed_levels[level]["value_base10euro"]:
            return int(level)

    return len(preprocessed_levels)  # fallback to highest level


def load_payout(itm_pct, payout_factor, reg_pp_pct, players_per_table, n_players, buy_in):
    filename = (f"{itm_pct}_"
                f"{str(payout_factor)[0]}-{str(payout_factor)[2:]}_"
                f"{str(reg_pp_pct).split('.')[-1]}_"
                f"{players_per_table}_"
                f"{n_players}_"
                f"{buy_in}")
    full_name = os.path.join("payouts", f"{filename}.json")
    #LOG.info("Prize pool file: " + full_name)

    try:
        with open(full_name) as pp_cfg:
            pp_cfg = json.load(pp_cfg)
        LOG.info("Prize pool successfully loaded")
        return pp_cfg
    except:
        st.markdown("<div style='text-align: center; font-size: 20px; font-weight: bold;'>Payouts not available for these settings</div>", unsafe_allow_html=True)
        LOG.info("Cannot find prize pool file")
        return None


############################################################################
############################################################################


# TODO: optimize everything for speed
# TODO: parallel computing?
# TODO: Min Payout Factor 1.5
# TODO: hosting


def run_simus():
    start = time()
    print("\n")
    LOG.info("---------------------------------------------------------------------------------------")
    LOG.info("----------------------------------- NEW SIMU ------------------------------------------")
    LOG.info("---------------------------------------------------------------------------------------")
    print("\n")

    # Retrieve entry inputs
    itm_pct = st.session_state.ITM_Pct
    payout_factor = st.session_state.Payout_Factor
    reg_pp_pct = st.session_state.Regular_Pct / 100
    players_per_table = st.session_state.PlayersPerTable
    n_simus = st.session_state.Simu_Count
    n_players = st.session_state.Players_Count
    starting_stack = st.session_state.Starting_Stack
    buy_in = st.session_state.Buy_In
    max_token = st.session_state.Max_Token
    min_multi = st.session_state.Min_Multiplier
    max_multi = st.session_state.Max_Multiplier

    # Load payout
    pp_cfg = load_payout(itm_pct, payout_factor, reg_pp_pct, players_per_table, n_players, buy_in)
    st.session_state["pp_cfg"] = pp_cfg
    LOG.info("pp_cfg: " + str(pp_cfg))
    if pp_cfg is None:
        st.markdown(
            "<div style='text-align: center; font-size: 20px; font-weight: bold; color: red;'>Payouts for these parameters not yet available</div>",
            unsafe_allow_html=True)
        return [], [], {}

    # Load levels, limit based on user input, and store in fast structure
    levels_file = f"levels_{min_multi}_{max_multi}.json"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, 'levels', levels_file)) as levels_cfg:
        levels_cfg = json.load(levels_cfg)
    levels_cfg = {k: v for k, v in levels_cfg.items() if int(k) <= max_token}
    st.session_state["levels_cfg"] = levels_cfg
    LOG.info("levels_cfg: " + str(levels_cfg))
    preprocessed_levels = preprocess_levels(levels_cfg)

    # Initialization
    rake_pct = 0.1  # Fixed
    ko_pp_pct = 1 - rake_pct - reg_pp_pct
    base_ko_value = buy_in * ko_pp_pct
    buyin_scale = buy_in / DEFAULT_BUYIN
    expected_pp_no_rake =  buy_in * n_players
    expected_pp = expected_pp_no_rake - (rake_pct * expected_pp_no_rake)
    pp_all = [pp_cfg["prizes"][str(i)] for i in range(1, len(pp_cfg["prizes"]) + 1)]
    #LOG.info("pp_all: " + str(pp_all))

    players_template = [{
        "id": str(i),
        "token_euros": base_ko_value,
        "stack": starting_stack,
        "token_level": 1
    } for i in range(n_players)]

    st.write("Running " + str(n_simus) + " simulations...")

    if n_simus < 1000:
        return run_simus_mono(
            n_simus, n_players, buy_in, rake_pct, base_ko_value, preprocessed_levels, pp_all, 
            buyin_scale, players_template, players_per_table, starting_stack, itm_pct, payout_factor, 
            max_token, expected_pp, reg_pp_pct, pp_cfg, levels_cfg, start)
    else:
        # Determine workers
        if n_simus < 5000:
            workers = 12
        elif n_simus < 10000:
            workers = 16
        else:
            workers = 32
        st.write("Starting " + str(workers) + " workers")

        return run_simus_multiprocess(
            n_simus, n_players, buy_in, rake_pct, base_ko_value, preprocessed_levels, pp_all, 
            buyin_scale, players_template, players_per_table, starting_stack, itm_pct, payout_factor, 
            max_token, expected_pp, reg_pp_pct, pp_cfg, levels_cfg, start, workers)


def run_simus_multiprocess(
        n_simus, n_players, buy_in, rake_pct, base_ko_value, preprocessed_levels, pp_all, 
        buyin_scale, players_template, players_per_table, starting_stack, itm_pct, payout_factor, 
        max_token, expected_pp, reg_pp_pct, pp_cfg, levels_cfg, start, workers):

    # Use ProcessPoolExecutor for parallel simulation
    all_tournament_results, all_player_results = [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(
                run_tournament,
                i,
                preprocessed_levels,
                pp_all,
                n_players,
                buy_in,
                rake_pct,
                base_ko_value,
                buyin_scale,
                players_template
            )
            for i in range(n_simus)
        ]

        # Progress bar update in Streamlit can be tricky with concurrency,
        # so we update after each finished task
        progress_bar = st.progress(0)
        for i, future in enumerate(as_completed(futures)):
            tournament_results, player_results = future.result()
            all_tournament_results.append(tournament_results)
            all_player_results.append(player_results)

            progress = (i + 1) / n_simus
            progress_bar.progress(progress, text=f"Progress: {int(progress * 100)}%")

    settings = {
        "n_simus": n_simus,
        "n_players": n_players,
        "buy_in": buy_in,
        "players_per_table": players_per_table,
        "starting_stack": starting_stack,
        "itm_pct": itm_pct,
        "payout_factor": payout_factor,
        "max_token": max_token,
        "expected_pp": expected_pp,
        "reg_pp_pct": reg_pp_pct,
        "pp_cfg": pp_cfg,
        "levels_cfg": levels_cfg,
    }

    elapsed = round(time() - start, 2)
    st.write("Took " + str(elapsed) + " secs")

    return all_tournament_results, all_player_results, settings


def run_simus_mono(
        n_simus, n_players, buy_in, rake_pct, base_ko_value, preprocessed_levels, pp_all, 
        buyin_scale, players_template, players_per_table, starting_stack, itm_pct, payout_factor, 
        max_token, expected_pp, reg_pp_pct, pp_cfg, levels_cfg, start):
    
    all_tournament_results, all_player_results = [], []
    progress_bar = st.progress(0)
    for i in range(n_simus):
        # Simulate current tournament
        tournament_results, player_results = run_tournament(
            i, preprocessed_levels, pp_all, n_players, buy_in, rake_pct, base_ko_value,
            buyin_scale, players_template)

        # Store results
        #LOG.info("tournament_results: " + str(tournament_results))
        #LOG.info("player_results: " + str(player_results))
        all_player_results.append(player_results)
        all_tournament_results.append(tournament_results)

        # Update progress bar
        if (i % max(1, n_simus // 100)) == 0 or (i == n_simus - 1):
            progress = (i+1) / n_simus
            progress_bar.progress(progress, text=f"Progress: {int(progress * 100)}%")

    settings = {
        "n_simus": n_simus,
        "n_players": n_players,
        "buy_in": buy_in,
        "players_per_table": players_per_table,
        "starting_stack": starting_stack,
        "itm_pct": itm_pct,
        "payout_factor": payout_factor,
        "max_token": max_token,
        "expected_pp": expected_pp,
        "reg_pp_pct": reg_pp_pct,
        "pp_cfg": pp_cfg,
        "levels_cfg": levels_cfg,
    }

    #LOG.info(all_player_results)
    #LOG.info(all_tournament_results)
    elapsed = round(time() - start, 2)
    st.write("Took " + str(elapsed) + " secs")

    return all_tournament_results, all_player_results, settings


def run_tournament(
        i, preprocessed_levels, pp_all, n_players, buy_in, rake_pct, base_ko_value,
        buyin_scale, players_template):

    # Initialize tournament
    current_place = n_players
    total_rake = buy_in * rake_pct * n_players
    results = defaultdict(lambda: {
        "rank": -1,
        "KOs": 0,
        "KO_winnings": 0,
        "reg_winnings": 0,
        "total_winnings": 0,
        "token_level": 1,
    })

    # Initialize players
    alive_players = [dict(player) for player in players_template]
    #LOG.info(alive_players)

    # Play hands until only one left
    while len(alive_players) > 1:
        LOG.debug("--- New hand - Players left: " + str(len(alive_players)))

        # All-in confrontation
        alive_players, idx1, idx2 = simulate_allin(alive_players)

        if alive_players[idx2]["stack"] == 0:
            loser = alive_players[idx2]
            loser_id = loser["id"]
            winner = alive_players[idx1]
            winner_id = winner["id"]

            # Update the eliminated player's results
            ko_value = max(0, loser["token_euros"] - base_ko_value)
            results[loser_id]["KO_winnings"] = round(ko_value, 2)
            results[loser_id]["rank"] = current_place
            #LOG.info("Busto: " + str(loser))

            # Calculate KO prize for the winner using the standard multiplier
            prize_multiplier = get_random_ko_multiplier(preprocessed_levels, loser["token_level"])
            ko_prize = prize_multiplier * loser["token_euros"]
            half_bounty = ko_prize / 2.0
            #LOG.info("Multi: " + str(prize_multiplier) + " - Prize: " + str(ko_prize))

            # Update the winner results
            alive_players[idx1]["token_euros"] = winner["token_euros"] + half_bounty
            alive_players[idx1]["token_level"] = get_token_level(preprocessed_levels, winner["token_euros"], buyin_scale)
            results[winner_id]["KOs"] += 1
            results[winner_id]["KO_winnings"] += half_bounty
            results[winner_id]["token_level"] = winner["token_level"]
            #LOG.info("Winner: " + str(winner))

            # Prepare for next hand
            current_place -= 1
            alive_players.pop(idx2)

    # Process final results (don't forget the original bounty of the winner)
    winner_id = alive_players[0]["id"]
    results[winner_id]["rank"] = 1
    results[winner_id]["KO_winnings"] = round(results[winner_id]["KO_winnings"] + base_ko_value, 2)
    sorted_results  = sorted(results.items(), key=lambda x: x[1]["rank"])
    player_results = {place + 1: {"id": player_id, **data} for place, (player_id, data) in enumerate(sorted_results )}

    # Store regular + total winnings for each player
    for i in range(len(pp_all)):
        player_results[i + 1]["reg_winnings"] = pp_all[i]
    for result in player_results.values():
        result["total_winnings"] = round(result["KO_winnings"] + result["reg_winnings"], 2)
    #LOG.info(player_results)

    # Compute global figures
    total_ko_winnings = round(sum(result["KO_winnings"] for result in player_results.values()), 2)
    total_reg_winnings = round(sum(result["reg_winnings"] for result in player_results.values()), 2)
    tournament_results = {
        "total_pp": round(total_reg_winnings + total_ko_winnings, 2),
        "total_reg_winnings": total_reg_winnings,
        "total_ko_winnings": total_ko_winnings,
        "total_rake": total_rake,
    }

    return tournament_results, player_results