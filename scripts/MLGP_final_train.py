import joblib
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "maindata/osu_source_residual_dataset.csv"
MODEL_OUT = "MLTrainData/gp_source_residual_model.joblib"
SCALER_OUT = "MLTrainData/gp_source_input_scaler.joblib"
TRAIN_PRED_OUT = "MLTrainData/osu_source_gp_all_predictions.csv"


def main():
    df = pd.read_csv(DATA_PATH).copy()

    X = df[["temp_C", "oxidation_time_min"]].to_numpy(float)
    y = df["source_residual_nm"].to_numpy(float)

    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e3))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=15,
        random_state=42
    )

    gp.fit(X_scaled, y)

    y_pred, y_std = gp.predict(X_scaled, return_std=True)

    df["gp_pred_residual_nm"] = y_pred
    df["gp_pred_std_nm"] = y_std
    df["hybrid_pred_nm"] = df["door_baseline_pred_nm"] + df["gp_pred_residual_nm"]
    df["hybrid_error_nm"] = df["growth_nm"] - df["hybrid_pred_nm"]

    mae = mean_absolute_error(df["growth_nm"], df["hybrid_pred_nm"])
    rmse = np.sqrt(mean_squared_error(df["growth_nm"], df["hybrid_pred_nm"]))

    print("\n=== FINAL GP TRAINED ON ALL SOURCE DATA ===")
    print(f"n    = {len(df)}")
    print(f"MAE  = {mae:.3f} nm")
    print(f"RMSE = {rmse:.3f} nm")
    print("\nLearned GP kernel:")
    print(gp.kernel_)

    df.to_csv(TRAIN_PRED_OUT, index=False)
    joblib.dump(gp, MODEL_OUT)
    joblib.dump(x_scaler, SCALER_OUT)

    print(f"\nSaved: {TRAIN_PRED_OUT}")
    print(f"Saved: {MODEL_OUT}")
    print(f"Saved: {SCALER_OUT}")


if __name__ == "__main__":
    main()