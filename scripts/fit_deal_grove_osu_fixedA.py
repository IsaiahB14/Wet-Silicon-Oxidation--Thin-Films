import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

R = 8.314  # J/mol-K

# -------------------
# PATHS
# -------------------
DATA_PATH = "data/osu_gp_ready_dataset.csv"

OUT_PER_TEMP = "data/osu_wet_per_temp_fixedA_door_params.csv"
OUT_ARRH = "data/arrhenius_params_osu_fixedA_door.csv"
OUT_PARAM_TABLE = "data/osu_wet_calibrated_params_fixedA_door.csv"
OUT_TRAIN_PRED = "data/osu_door_fixedA_train_predictions.csv"
OUT_VAL_PRED = "data/osu_door_fixedA_val_predictions.csv"
OUT_ALL_VALID_PRED = "data/osu_door_fixedA_all_valid_predictions.csv"

# -------------------
# USER CHOICES
# -------------------
DROP_RUN_IDS = {"R01"}      # anomalous run, drop fully
USE_ONLY_DOOR = True
A_FIXED_UM = 0.20
TAU_FIXED_HR = 0.0

# Hold out one DOOR wafer condition for clean validation
# Change this if you want a different held-out door run
VAL_RUN_IDS = {"R12"}
# Training is all remaining valid door runs after dropping VAL_RUN_IDS

# -------------------
# HELPERS
# -------------------
def deal_grove_x(A_um, B_um2_per_hr, t_hr, tau_hr=0.0):
    t_eff = np.maximum(np.asarray(t_hr, float) + float(tau_hr), 0.0)
    disc = float(A_um)**2 + 4.0 * float(B_um2_per_hr) * t_eff
    disc = np.maximum(disc, 0.0)
    return (-float(A_um) + np.sqrt(disc)) / 2.0

def arrhenius_k(T_K, k0, E):
    return float(k0) * np.exp(-float(E) / (R * float(T_K)))

# -------------------
# LOAD DATA
# -------------------
def load_dataset():
    df = pd.read_csv(DATA_PATH).copy()

    if USE_ONLY_DOOR:
        df = df[df["position"].str.lower() == "door"].copy()

    df = df[df["valid_final_measurement"] == True].copy()
    df = df[~df["run_id"].isin(DROP_RUN_IDS)].copy()

    needed = [
        "run_id",
        "temp_C",
        "oxidation_time_min",
        "growth_nm",
        "final_std_nm",
        "position",
    ]
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    df["time_hr"] = df["oxidation_time_min"].astype(float) / 60.0
    df["growth_um"] = df["growth_nm"].astype(float) / 1000.0

    return df.sort_values(["temp_C", "oxidation_time_min"]).reset_index(drop=True)

def split_train_val(df):
    val_df = df[df["run_id"].isin(VAL_RUN_IDS)].copy()
    train_df = df[~df["run_id"].isin(VAL_RUN_IDS)].copy()

    if len(val_df) == 0:
        raise ValueError("Validation set is empty. Check VAL_RUN_IDS.")

    if len(train_df) < 2:
        raise ValueError("Need at least 2 training points.")

    overlap = set(train_df["run_id"]).intersection(set(val_df["run_id"]))
    if overlap:
        raise ValueError(f"Train/val overlap found: {overlap}")

    return train_df, val_df

# -------------------
# FIT ONE TEMPERATURE, B ONLY
# -------------------
def fit_one_temperature_fixedA(sub, A_fixed_um=A_FIXED_UM, tau_hr=TAU_FIXED_HR):
    T_C = float(sub["temp_C"].iloc[0])

    t_hr = sub["time_hr"].to_numpy(float)
    x_um = sub["growth_um"].to_numpy(float)

    if len(sub) < 1:
        raise ValueError(f"No points available to fit temp {T_C} C")

    def residuals(p):
        lnB = p[0]
        B_um2_per_hr = np.exp(lnB)
        x_pred = deal_grove_x(A_fixed_um, B_um2_per_hr, t_hr, tau_hr=tau_hr)

        res = []
        for i, xp in enumerate(x_pred):
            sigma_um = sub.iloc[i]["final_std_nm"] / 1000.0 if pd.notna(sub.iloc[i]["final_std_nm"]) else 0.005
            sigma_um = max(float(sigma_um), 0.002)
            res.append((xp - x_um[i]) / sigma_um)
        return np.array(res, dtype=float)

    B_guess = max(np.median((x_um**2) / np.maximum(t_hr, 1e-6)), 1e-5)
    p0 = np.array([np.log(B_guess)], dtype=float)

    lb = np.array([np.log(1e-8)], dtype=float)
    ub = np.array([np.log(1e4)], dtype=float)

    res = least_squares(
        residuals,
        p0,
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=20000
    )

    B_um2_per_hr = float(np.exp(res.x[0]))
    BA_um_per_hr = B_um2_per_hr / A_fixed_um

    summary = {
        "temp_C": T_C,
        "n_points": int(len(sub)),
        "A_um": float(A_fixed_um),
        "B_um2_per_hr": B_um2_per_hr,
        "tau_hr": float(tau_hr),
        "BA_um_per_hr": BA_um_per_hr,
        "success": bool(res.success),
        "cost": float(res.cost),
        "message": str(res.message),
    }

    return summary

# -------------------
# FIT ALL TRAIN TEMPS
# -------------------
def fit_per_temp_table(train_df):
    param_rows = []

    for T in sorted(train_df["temp_C"].unique()):
        sub = train_df[train_df["temp_C"] == T].copy().sort_values("oxidation_time_min")

        # If a temperature has no remaining training point, skip it
        if len(sub) == 0:
            continue

        summary = fit_one_temperature_fixedA(sub)
        param_rows.append(summary)

        print(f"\nFitting DOOR baseline at T = {T:.1f} C with {len(sub)} training point(s)")
        print(f"A    = {summary['A_um']:.6g} um (fixed)")
        print(f"B    = {summary['B_um2_per_hr']:.6g} um^2/hr")
        print(f"B/A  = {summary['BA_um_per_hr']:.6g} um/hr")
        print(f"success = {summary['success']}")

    params_df = pd.DataFrame(param_rows).sort_values("temp_C").reset_index(drop=True)
    return params_df

# -------------------
# ARRHENIUS FIT FROM PER-TEMP B
# -------------------
def fit_arrhenius_from_pertemp(params_df):
    df = params_df.sort_values("temp_C").copy()

    T_K = df["temp_C"].to_numpy(float) + 273.15
    invT = 1.0 / T_K

    B = df["B_um2_per_hr"].to_numpy(float)
    BA = df["BA_um_per_hr"].to_numpy(float)

    if np.any(B <= 0) or np.any(BA <= 0):
        raise ValueError("B and B/A must be positive for Arrhenius fitting")

    m_B, c_B = np.polyfit(invT, np.log(B), 1)
    E_B = -m_B * R
    B0 = np.exp(c_B)

    m_BA, c_BA = np.polyfit(invT, np.log(BA), 1)
    E_BA = -m_BA * R
    BA0 = np.exp(c_BA)

    arrh_df = pd.DataFrame([{
        "B0_um2_per_hr": B0,
        "E_B_J_per_mol": E_B,
        "BA0_um_per_hr": BA0,
        "E_BA_J_per_mol": E_BA
    }])

    return arrh_df

def predict_growth_nm_arrh(T_C, time_min, arrh_df, A_fixed_um=A_FIXED_UM, tau_hr=TAU_FIXED_HR):
    B0 = float(arrh_df.iloc[0]["B0_um2_per_hr"])
    E_B = float(arrh_df.iloc[0]["E_B_J_per_mol"])

    T_K = float(T_C) + 273.15
    t_hr = float(time_min) / 60.0

    B = arrhenius_k(T_K, B0, E_B)
    x_um = deal_grove_x(A_fixed_um, B, t_hr, tau_hr=tau_hr)
    return 1000.0 * x_um, B

# -------------------
# PREDICTIONS
# -------------------
def add_predictions(df, arrh_df):
    rows = []
    for _, row in df.iterrows():
        x_pred_nm, B_interp = predict_growth_nm_arrh(
            row["temp_C"], row["oxidation_time_min"], arrh_df,
            A_fixed_um=A_FIXED_UM, tau_hr=TAU_FIXED_HR
        )
        resid_nm = row["growth_nm"] - x_pred_nm

        out = row.to_dict()
        out["A_um_fit"] = A_FIXED_UM
        out["tau_hr_fit"] = TAU_FIXED_HR
        out["B_interp_um2_per_hr"] = B_interp
        out["pred_growth_nm_fit"] = x_pred_nm
        out["residual_nm_fit"] = resid_nm
        rows.append(out)

    return pd.DataFrame(rows)

def metrics(df, label):
    err = df["residual_nm_fit"].to_numpy(float)
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    bias = np.mean(err)

    print(f"\n=== {label} METRICS ===")
    print(f"n    = {len(df)}")
    print(f"MAE  = {mae:.3f} nm")
    print(f"RMSE = {rmse:.3f} nm")
    print(f"Bias = {bias:.3f} nm")

# -------------------
# PLOTS
# -------------------
def plot_train_val(train_pred, val_pred):
    plt.figure(figsize=(8, 6))

    temps = sorted(pd.concat([train_pred["temp_C"], val_pred["temp_C"]]).unique())

    for T in temps:
        sub_tr = train_pred[train_pred["temp_C"] == T].sort_values("oxidation_time_min")
        sub_va = val_pred[val_pred["temp_C"] == T].sort_values("oxidation_time_min")

        if len(sub_tr) > 0:
            plt.scatter(
                sub_tr["oxidation_time_min"], sub_tr["growth_nm"],
                label=f"Train data {int(T)}C"
            )
            plt.plot(
                sub_tr["oxidation_time_min"], sub_tr["pred_growth_nm_fit"],
                linestyle="-"
            )

        if len(sub_va) > 0:
            plt.scatter(
                sub_va["oxidation_time_min"], sub_va["growth_nm"],
                marker="s", label=f"Val data {int(T)}C"
            )
            plt.plot(
                sub_va["oxidation_time_min"], sub_va["pred_growth_nm_fit"],
                linestyle="--"
            )

    plt.xlabel("Oxidation time (min)")
    plt.ylabel("Growth thickness (nm)")
    plt.title(f"DOOR Baseline, Fixed A = {A_FIXED_UM:.2f} um")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_arrhenius_linearized(params_df):
    df = params_df.sort_values("temp_C").copy()

    # Temperature in Kelvin
    T_K = df["temp_C"].to_numpy(float) + 273.15
    invT = 1.0 / T_K

    # Extract B and B/A
    B = df["B_um2_per_hr"].to_numpy(float)
    BA = df["BA_um_per_hr"].to_numpy(float)

    if np.any(B <= 0) or np.any(BA <= 0):
        raise ValueError("B and B/A must be positive to plot Arrhenius linearization.")

    # Linearized values
    lnB = np.log(B)
    lnBA = np.log(BA)

    # Best-fit lines
    m_B, c_B = np.polyfit(invT, lnB, 1)
    m_BA, c_BA = np.polyfit(invT, lnBA, 1)

    # Smooth x range for plotting fit lines
    invT_line = np.linspace(invT.min() * 0.995, invT.max() * 1.005, 200)
    lnB_line = m_B * invT_line + c_B
    lnBA_line = m_BA * invT_line + c_BA

    # Convert slope/intercept to physical params
    E_B = -m_B * R
    B0 = np.exp(c_B)

    E_BA = -m_BA * R
    BA0 = np.exp(c_BA)

    # R^2 for B
    ss_res_B = np.sum((lnB - (m_B * invT + c_B))**2)
    ss_tot_B = np.sum((lnB - np.mean(lnB))**2)
    r2_B = 1 - ss_res_B / ss_tot_B if ss_tot_B > 0 else np.nan

    # R^2 for B/A
    ss_res_BA = np.sum((lnBA - (m_BA * invT + c_BA))**2)
    ss_tot_BA = np.sum((lnBA - np.mean(lnBA))**2)
    r2_BA = 1 - ss_res_BA / ss_tot_BA if ss_tot_BA > 0 else np.nan

    # -------- Plot ln(B) vs 1/T --------
    plt.figure(figsize=(8, 6))
    plt.scatter(invT, lnB, s=70, label="Per-temp fitted B values")
    plt.plot(
        invT_line,
        lnB_line,
        label=(
            f"Linear fit\n"
            f"ln(B) = {m_B:.3e}(1/T) + {c_B:.3f}\n"
            f"B0 = {B0:.3e} um²/hr\n"
            f"E_B = {E_B/1000:.2f} kJ/mol\n"
            f"R² = {r2_B:.4f}"
        )
    )

    # annotate each point with temp
    for x, y, T_C in zip(invT, lnB, df["temp_C"]):
        plt.annotate(f"{int(T_C)}C", (x, y), textcoords="offset points", xytext=(5, 5))

    plt.xlabel("1 / T (1/K)")
    plt.ylabel("ln(B)")
    plt.title("Arrhenius Linearized Relationship for B")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------- Plot ln(B/A) vs 1/T --------
    plt.figure(figsize=(8, 6))
    plt.scatter(invT, lnBA, s=70, label="Per-temp fitted B/A values")
    plt.plot(
        invT_line,
        lnBA_line,
        label=(
            f"Linear fit\n"
            f"ln(B/A) = {m_BA:.3e}(1/T) + {c_BA:.3f}\n"
            f"BA0 = {BA0:.3e} um/hr\n"
            f"E_BA = {E_BA/1000:.2f} kJ/mol\n"
            f"R² = {r2_BA:.4f}"
        )
    )

    for x, y, T_C in zip(invT, lnBA, df["temp_C"]):
        plt.annotate(f"{int(T_C)}C", (x, y), textcoords="offset points", xytext=(5, 5))

    plt.xlabel("1 / T (1/K)")
    plt.ylabel("ln(B/A)")
    plt.title("Arrhenius Linearized Relationship for B/A")
    plt.legend()
    plt.tight_layout()
    plt.show()
# -------------------
# MAIN
# -------------------
def main():
    df = load_dataset()

    print("\nUsing these DOOR-wafer data points:")
    print(df[["run_id", "temp_C", "oxidation_time_min", "growth_nm"]].to_string(index=False))
    print(f"\nFixed A = {A_FIXED_UM:.6f} um")
    print(f"Fixed tau = {TAU_FIXED_HR:.6f} hr")
    print(f"Held-out validation DOOR run(s): {sorted(list(VAL_RUN_IDS))}")

    train_df, val_df = split_train_val(df)

    print("\nTraining DOOR run_ids:")
    print(sorted(train_df["run_id"].tolist()))

    print("\nValidation DOOR run_ids:")
    print(sorted(val_df["run_id"].tolist()))

    params_df = fit_per_temp_table(train_df)
    arrh_df = fit_arrhenius_from_pertemp(params_df)

    plot_arrhenius_linearized(params_df)

    print("\n=== ARRHENIUS FIT FROM DOOR FIXED-A PER-TEMP PARAMETERS ===")
    print(f"B0   = {float(arrh_df.iloc[0]['B0_um2_per_hr']):.6g} um^2/hr")
    print(f"E_B  = {float(arrh_df.iloc[0]['E_B_J_per_mol'])/1000.0:.3f} kJ/mol")
    print(f"BA0  = {float(arrh_df.iloc[0]['BA0_um_per_hr']):.6g} um/hr")
    print(f"E_BA = {float(arrh_df.iloc[0]['E_BA_J_per_mol'])/1000.0:.3f} kJ/mol")

    train_pred = add_predictions(train_df, arrh_df)
    val_pred = add_predictions(val_df, arrh_df)
    all_pred = add_predictions(df, arrh_df)

    metrics(train_pred, "TRAIN DOOR")
    metrics(val_pred, "VALIDATION DOOR")
    metrics(all_pred, "ALL VALID DOOR")

    params_df.to_csv(OUT_PER_TEMP, index=False)
    params_df[["temp_C", "A_um", "B_um2_per_hr", "tau_hr", "BA_um_per_hr"]].to_csv(
        OUT_PARAM_TABLE, index=False
    )
    arrh_df.to_csv(OUT_ARRH, index=False)
    train_pred.to_csv(OUT_TRAIN_PRED, index=False)
    val_pred.to_csv(OUT_VAL_PRED, index=False)
    all_pred.to_csv(OUT_ALL_VALID_PRED, index=False)

    print(f"\nSaved: {OUT_PER_TEMP}")
    print(f"Saved: {OUT_PARAM_TABLE}")
    print(f"Saved: {OUT_ARRH}")
    print(f"Saved: {OUT_TRAIN_PRED}")
    print(f"Saved: {OUT_VAL_PRED}")
    print(f"Saved: {OUT_ALL_VALID_PRED}")

    plot_train_val(train_pred, val_pred)

if __name__ == "__main__":
    main()