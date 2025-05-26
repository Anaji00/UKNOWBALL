# player_model_utils_xgb.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import joblib
print("📋 Model columns:", df.columns.tolist())

# Now we pass df from outside instead of reading a file here
def create_features(player_name, df, num_games=150):
    df["gameDate"] = pd.to_datetime(df["gameDate"])
    df["playerIdentifier"] = df["firstName"] + " " + df["lastName"]

    player_data = df[df["playerIdentifier"] == player_name].sort_values("gameDate").tail(num_games).copy()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    opponent_encoded = encoder.fit_transform(player_data[["opponentteamName"]])
    opponent_cols = encoder.get_feature_names_out(["opponentteamName"])
    opponent_df = pd.DataFrame(opponent_encoded, index=player_data.index, columns=opponent_cols)

    for stat in ["points", "assists", "reboundsTotal"]:
        player_data[f"{stat}_avg3"] = player_data[stat].rolling(3).mean()
        player_data[f"{stat}_avg5"] = player_data[stat].rolling(5).mean()
        player_data[f"{stat}_avg10"] = player_data[stat].shift(1).rolling(10).mean()
        player_data[f"{stat}_avg30"] = player_data[stat].shift(1).rolling(30).mean()

    player_data["PER"] = (
        player_data["points"] + player_data["reboundsTotal"] + player_data["assists"] +
        player_data["steals"] + player_data["blocks"] - player_data["turnovers"]
    ) / (player_data["numMinutes"] + 1e-5)

    base_features = [
        "fieldGoalsAttempted", "freeThrowsAttempted",
        "numMinutes", "fieldGoalsPercentage", "freeThrowsPercentage", "fieldGoalsMade", "freeThrowsMade",
        "steals", "blocks", "turnovers", "threePointersMade",
        "home", "win", "plusMinusPoints",
        "points_avg3", "assists_avg3", "reboundsTotal_avg3",
        "points_avg5", "assists_avg5", "reboundsTotal_avg5", "PER", "threePointersPercentage",
        "reboundsTotal_avg10", "assists_avg10", "points_avg10",
    ]

    full_df = pd.concat([player_data[base_features], opponent_df], axis=1).dropna()

    X = full_df.values
    y_points = player_data.loc[full_df.index, "points"].values
    y_assists = player_data.loc[full_df.index, "assists"].values
    y_rebounds = player_data.loc[full_df.index, "reboundsTotal"].values
    game_dates = player_data.loc[full_df.index, "gameDate"].values

    return X, y_points, y_assists, y_rebounds, game_dates

def predict_next_games(player_name, df, num_games=5, save_models=True):
    X, y_points, y_assists, y_rebounds, game_dates = create_features(player_name, df)

    X_train = X[:-num_games]
    X_test = X[-num_games:]
    y_points_train, y_points_test = y_points[:-num_games], y_points[-num_games:]
    y_assists_train, y_assists_test = y_assists[:-num_games], y_assists[-num_games:]
    y_rebounds_train, y_rebounds_test = y_rebounds[:-num_games], y_rebounds[-num_games:]
    test_dates = game_dates[-num_games:]

    preds = {}

    for stat, y_train, y_test in zip(
        ["points", "assists", "rebounds"],
        [y_points_train, y_assists_train, y_rebounds_train],
        [y_points_test, y_assists_test, y_rebounds_test]
    ):
        model = XGBRegressor(n_estimators=100)
        if stat == "points":
            y_train_log = np.log1p(y_train)
            model.fit(X_train, y_train_log)
            preds[stat] = np.expm1(model.predict(X_test))
        else:
            model.fit(X_train, y_train)
            preds[stat] = model.predict(X_test)

        if save_models:
            joblib.dump(model, f"{player_name}_{stat}_xgb.pkl")

    df_pred = pd.DataFrame({
        "points": preds["points"],
        "assists": preds["assists"],
        "rebounds": preds["rebounds"]
    }, index=pd.to_datetime(test_dates))

    y_true = {
        "points": y_points_test,
        "assists": y_assists_test,
        "rebounds": y_rebounds_test
    }

    return df_pred, y_true
