# Imports
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd


def results_to_df(aggregate):
    # Original stat keys in the data
    data_columns = ["avg", "0pct", "25pct", "50pct", "75pct", "90pct", "95pct", "99pct", "100pct"]

    # Group aggregate into metrics (numeric raw values)
    metrics = defaultdict(dict)
    for key, value in aggregate.items():
        if "_" in key:
            metric, stat = key.rsplit("_", 1)
            if stat in data_columns:
                try:
                    # Keep numeric values as-is
                    metrics[metric][stat] = float(value)
                except Exception:
                    metrics[metric][stat] = value

    # Build a DataFrame with raw numeric values (for CSV)
    rows_raw = [{"Metric": metric, **{col: stat_values.get(col, pd.NA) for col in data_columns}} for metric, stat_values in metrics.items()]
    df_raw = pd.DataFrame(rows_raw)[["Metric"] + data_columns]

    return df_raw, data_columns


def compute_stats(arr, prefix, aggregate):
    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    arr = np.array(arr)
    agg_prefix = prefix + "_"
    aggregate[agg_prefix + "avg"] = float(round(np.mean(arr), 2))
    aggregate[agg_prefix + "0pct"] = float(round(np.min(arr), 2))
    for p in percentiles[1:-1]:
        aggregate[agg_prefix + f"{p}pct"] = float(round(np.percentile(arr, p), 2))
    aggregate[agg_prefix + "100pct"] = float(round(np.max(arr), 2))


def get_stats_from_rank(all_player_results, aggregate, rank):
    stats = {
        "KOWinnings": [x[rank]['KO_winnings'] for x in all_player_results],
        "TotalWinnings": [x[rank]['total_winnings'] for x in all_player_results],
        "NbKO": [x[rank]['KOs'] for x in all_player_results],
        "TokenLvl": [x[rank]['token_level'] for x in all_player_results],
    }

    for stat_name, values in stats.items():
        compute_stats(values, f"rank{rank}-{stat_name}", aggregate)

    return aggregate


def compute_aggregates(all_tournament_results, all_player_results, expected_pp):
    aggregate = {}

    # Tournament-level stats
    total_rake = [x['total_rake'] for x in all_tournament_results]
    total_reg_winnings = [x['total_reg_winnings'] for x in all_tournament_results]
    ko_winnings = [x['total_ko_winnings'] for x in all_tournament_results]
    total_pp = [x['total_pp'] for x in all_tournament_results]

    aggregate["rake_avg"] = float(round(np.mean(total_rake), 2))
    aggregate["expectedPrizePool_avg"] = float(round(expected_pp, 2))
    aggregate["regularPrizePool_avg"] = float(round(np.mean(total_reg_winnings), 2))
    
    compute_stats(ko_winnings, "KOPrizePool", aggregate)
    compute_stats(total_pp, "totalPrizePool", aggregate)

    # Player-level stats
    all_entries = [player for simu in all_player_results for player in simu.values()]
    NbKO = [entry['KOs'] for entry in all_entries]
    TokenLvl = [entry['token_level'] for entry in all_entries]

    compute_stats(NbKO, "NbKO", aggregate)
    compute_stats(TokenLvl, "TokenLvl", aggregate)

    # Stats by ranks
    for rank in [1, 3, 10]:
        aggregate = get_stats_from_rank(all_player_results, aggregate, rank)

    results, _ = results_to_df(aggregate)
    return results


def create_download_results(
        df_raw, n_simus, n_players, buy_in, players_per_table, starting_stack, itm_pct, payout_factor, reg_pp_pct, max_token):

    ts = int(time())
    title_str = f"{ts}_{n_simus}_{n_players}_{buy_in}_{players_per_table}_{starting_stack}_{itm_pct}_{int(payout_factor*100)}_{int(reg_pp_pct*100)}_{max_token}.csv"
    csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
    return csv_bytes, title_str