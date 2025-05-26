import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os

# Include model paths
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

# Import custom model utilities
from player_model_utils_xgb import predict_next_games as predict_with_xgb
from player_model_utils_mlp import predict_next_games_mlp
from player_model_utils import create_features

# Streamlit config
st.set_page_config(page_title="🏀 Player Performance Predictor", page_icon="🏀")
st.title("🏀 Player Performance Predictor \n(Alessio Naji-Sepasgozar)")

# === Load player stats from Google Drive ===
import gdown

# Only download once and cache
@st.cache_data
def load_large_csv_from_drive():
    url = "https://drive.google.com/uc?id=1B_mrKhMBYfmhiLhsjz7-A1QJojWe_uIC"
    output_path = "/tmp/updatedPlayerStats.csv"
    gdown.download(url, output_path, quiet=False)
    df = pd.read_csv(output_path)
    return df

df = load_large_csv_from_drive()


# Normalize column names to avoid case/space/BOM issues
df.columns = df.columns.str.strip().str.lower()

# Check if both columns are present
if "firstname" not in df.columns or "lastname" not in df.columns:
    st.error(f"❌ Required columns missing: 'firstname' and/or 'lastname'")
    st.write("📋 Found columns:", df.columns.tolist())
    st.stop()

# Use first and last name as identifier
df["playerIdentifier"] = df["firstname"] + " " + df["lastname"]

# === UI selections
player_name = st.selectbox("Select a player:", player_list)
model_choice = st.radio("Select model:", ["XGBoost", "MLP"], horizontal=True)

# === Display prediction metrics
def show_stat_metrics(label, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    st.markdown(f"""
    #### 📈 **{label} Prediction Metrics**
    - 🧮 **MAE**: `{mae:.6f}`
    - 🧨 **MSE**: `{mse:.6f}`
    - 🎯 **R² Score**: `{r2:.8f}`
    """)

# === Run prediction
if st.button("Predict Player Performance"):
    try:
        st.info(f"🔮 Predicting next 5 games using **{model_choice}**...")

        if model_choice == "XGBoost":
            preds_df, y_true = predict_with_xgb(player_name, df=df, num_games=5)
        else:
            preds_df, y_true = predict_next_games_mlp(player_name, df=df, num_games=5)

        if isinstance(preds_df, pd.DataFrame):
            st.subheader(f"📊 Predicted Stats for {player_name} (Next 5 Games)")
            for i, (game_date, row) in enumerate(preds_df.iterrows(), 1):
                st.markdown(f"""
                #### 🗓️ Game on {game_date.date()}
                - 🟦 **Points**: {row['points']:.2f}
                - 🟩 **Assists**: {row['assists']:.2f}
                - 🟨 **Rebounds**: {row['rebounds']:.2f}
                """)

            st.divider()
            st.subheader("📊 Prediction Accuracy (Last 5 Games)")
            show_stat_metrics("Points", y_true["points"], preds_df["points"].values)
            show_stat_metrics("Assists", y_true["assists"], preds_df["assists"].values)
            show_stat_metrics("Rebounds", y_true["rebounds"], preds_df["rebounds"].values)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
