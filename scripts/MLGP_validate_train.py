import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "maindata/osu_source_residual_dataset.csv"
OUT_TRAIN_PATH = "maindata/osu_source_gp_train_predictions.csv"
OUT_VAL_PATH = "maindata/osu_source_gp_val_predictions.csv"

# Choose held-out SOURCE runs here
VAL_SOURCE_RUN_IDS = {"R02"}   # <-- change this to a real source run you want to hold out


def metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n=== {label} ===")
    print(f"n    = {len(y_true)}")
    print(f"MAE  = {mae:.3f} nm")
    print(f"RMSE = {rmse:.3f} nm")


def main():
    df = pd.read_csv(DATA_PATH).copy()

    # Split by source run_id
    val_df = df[df["run_id"].isin(VAL_SOURCE_RUN_IDS)].copy()
    train_df = df[~df["run_id"].isin(VAL_SOURCE_RUN_IDS)].copy()

    if len(val_df) == 0:
        raise ValueError("Validation set is empty. Check VAL_SOURCE_RUN_IDS.")

    if len(train_df) < 3:
        raise ValueError("Need more source training points after holdout.")

    # Inputs and target
    X_train = train_df[["temp_C", "oxidation_time_min"]].to_numpy(float)
    y_train = train_df["source_residual_nm"].to_numpy(float)

    X_val = val_df[["temp_C", "oxidation_time_min"]].to_numpy(float)
    y_val = val_df["source_residual_nm"].to_numpy(float)

    # Scale inputs using TRAIN only
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)

    # GP kernel
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

    # Fit only on TRAIN
    gp.fit(X_train_scaled, y_train)

    print("\nLearned GP kernel:")
    print(gp.kernel_)

    # Predict TRAIN
    train_resid_pred, train_resid_std = gp.predict(X_train_scaled, return_std=True)
    train_df["gp_pred_residual_nm"] = train_resid_pred
    train_df["gp_pred_std_nm"] = train_resid_std
    train_df["hybrid_pred_nm"] = train_df["door_baseline_pred_nm"] + train_df["gp_pred_residual_nm"]
    train_df["hybrid_error_nm"] = train_df["growth_nm"] - train_df["hybrid_pred_nm"]

    # Predict VAL
    val_resid_pred, val_resid_std = gp.predict(X_val_scaled, return_std=True)
    val_df["gp_pred_residual_nm"] = val_resid_pred
    val_df["gp_pred_std_nm"] = val_resid_std
    val_df["hybrid_pred_nm"] = val_df["door_baseline_pred_nm"] + val_df["gp_pred_residual_nm"]
    val_df["hybrid_error_nm"] = val_df["growth_nm"] - val_df["hybrid_pred_nm"]

    # Metrics on final hybrid thickness
    metrics(train_df["growth_nm"], train_df["hybrid_pred_nm"], "TRAIN SOURCE HYBRID")
    metrics(val_df["growth_nm"], val_df["hybrid_pred_nm"], "VALIDATION SOURCE HYBRID")

    # Save
    train_df.to_csv(OUT_TRAIN_PATH, index=False)
    val_df.to_csv(OUT_VAL_PATH, index=False)

    print(f"\nSaved: {OUT_TRAIN_PATH}")
    print(f"Saved: {OUT_VAL_PATH}")

    # Optional parity plot for validation
    plt.figure(figsize=(6, 6))
    plt.scatter(val_df["growth_nm"], val_df["hybrid_pred_nm"])
    lims = [
        min(val_df["growth_nm"].min(), val_df["hybrid_pred_nm"].min()),
        max(val_df["growth_nm"].max(), val_df["hybrid_pred_nm"].max())
    ]
    plt.plot(lims, lims, "--")
    plt.xlabel("Measured source growth (nm)")
    plt.ylabel("Hybrid predicted growth (nm)")
    plt.title("Validation parity plot, source hybrid model")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()