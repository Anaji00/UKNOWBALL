from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from player_model_utils import create_features
import pandas as pd
import numpy as np
import joblib


def build_mlp(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256), BatchNormalization(), LeakyReLU(),
        Dropout(0.3),
        Dense(128), BatchNormalization(), LeakyReLU(),
        Dropout(0.3),
        Dense(64), BatchNormalization(), LeakyReLU(),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_next_games_mlp(player_name, num_games=5, save_models=True):
    X, y_points, y_assists, y_rebounds, game_dates = create_features(player_name)

    # Prepare targets
    y_all = np.stack([y_points, y_assists, y_rebounds], axis=1)

    # Train/test split
    X_train, X_test = X[:-num_games], X[-num_games:]
    y_train, y_test = y_all[:-num_games], y_all[-num_games:]
    test_dates = game_dates[-num_games:]

    # Scale inputs and targets
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train)
    
    # Build and train MLP
    mlp = build_mlp(X.shape[1], output_dim=y_all.shape[1])
    mlp.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=64,
            validation_split=0.1, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)

    # Predict
    y_pred_scaled = mlp.predict(X_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    # Save model
    if save_models:
        mlp.save(f"{player_name}_mlp.h5")
        joblib.dump(x_scaler, f"{player_name}_x_scaler.pkl")
        joblib.dump(y_scaler, f"{player_name}_y_scaler.pkl")

    # Sanity check
    print("💡 Sanity check (points):", list(zip(y_test[:, 0], y_pred[:, 0])))

    # Output
    df_pred = pd.DataFrame({
        "points": y_pred[:, 0],
        "assists": y_pred[:, 1],
        "rebounds": y_pred[:, 2]
    }, index=pd.to_datetime(test_dates))

    y_true = {
        "points": y_test[:, 0],
        "assists": y_test[:, 1],
        "rebounds": y_test[:, 2]
    }

    return df_pred, y_true
