# player_feature_utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/updatedPlayerStats.csv")
df["gameDate"] = pd.to_datetime(df["gameDate"])
df["playerIdentifier"] = df["firstName"] + " " + df["lastName"]

def create_features(player_name, num_games=150):
    player_data = df[df["playerIdentifier"] == player_name].sort_values("gameDate").tail(num_games).copy()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    opponent_encoded = encoder.fit_transform(player_data[["opponentteamName"]])
    opponent_cols = encoder.get_feature_names_out(["opponentteamName"])
    opponent_df = pd.DataFrame(opponent_encoded, index=player_data.index, columns=opponent_cols)

    for stat in ["points", "assists", "reboundsTotal"]:
        player_data[f"{stat}_avg3"] = player_data[stat].shift(1).rolling(3).mean()
        player_data[f"{stat}_avg5"] = player_data[stat].shift(1).rolling(5).mean()
        player_data[f"{stat}_avg10"] = player_data[stat].shift(1).rolling(10).mean()
        player_data[f"{stat}_avg30"] = player_data[stat].shift(1).rolling(30).mean()

    player_data["PER"] = (
        player_data["points"] + player_data["reboundsTotal"] + player_data["assists"] +
        player_data["steals"] + player_data["blocks"] - player_data["turnovers"]
    ) / (player_data["numMinutes"] + 1e-5)

    base_features = [
        "fieldGoalsAttempted", "freeThrowsAttempted",
        "numMinutes", "fieldGoalsPercentage", "freeThrowsPercentage","fieldGoalsMade", "freeThrowsMade",
        "steals", "blocks", "turnovers", "threePointersMade",
        "home", "win", "plusMinusPoints",
        "points_avg3", "assists_avg3", "reboundsTotal_avg3",
        "points_avg5", "assists_avg5", "reboundsTotal_avg5", "PER", "threePointersPercentage", "reboundsTotal_avg10", "assists_avg10", "points_avg10",
    ]

    full_df = pd.concat([player_data[base_features], opponent_df], axis=1).dropna()
    X = full_df.values
    y_points = player_data.loc[full_df.index, "points"].values
    y_assists = player_data.loc[full_df.index, "assists"].values
    y_rebounds = player_data.loc[full_df.index, "reboundsTotal"].values
    game_dates = player_data.loc[full_df.index, "gameDate"].values

    return X, y_points, y_assists, y_rebounds, game_dates
