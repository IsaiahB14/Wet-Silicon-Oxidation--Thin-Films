import numpy as np
import pandas as pd

R = 8.314  # J/mol-K

DATA_PATH = "data/osu_gp_ready_dataset.csv"
ARRH_PATH = "data/arrhenius_params_osu_fixedA_door.csv"
OUT_PATH = "maindata/osu_source_residual_dataset.csv"

A_FIXED_UM = 0.20
TAU_FIXED_HR = 0.0
DROP_RUN_IDS = {"R01"}


def deal_grove_x(A_um, B_um2_per_hr, t_hr, tau_hr=0.0):
    t_eff = np.maximum(np.asarray(t_hr, float) + float(tau_hr), 0.0)
    disc = float(A_um)**2 + 4.0 * float(B_um2_per_hr) * t_eff
    disc = np.maximum(disc, 0.0)
    return (-float(A_um) + np.sqrt(disc)) / 2.0


def arrhenius_k(T_K, k0, E):
    return float(k0) * np.exp(-float(E) / (R * float(T_K)))


def predict_growth_nm_arrh(T_C, time_min, arrh_df, A_fixed_um=A_FIXED_UM, tau_hr=TAU_FIXED_HR):
    B0 = float(arrh_df.iloc[0]["B0_um2_per_hr"])
    E_B = float(arrh_df.iloc[0]["E_B_J_per_mol"])

    T_K = float(T_C) + 273.15
    t_hr = float(time_min) / 60.0

    B = arrhenius_k(T_K, B0, E_B)
    x_um = deal_grove_x(A_fixed_um, B, t_hr, tau_hr=tau_hr)
    return 1000.0 * x_um, B


def main():
    df = pd.read_csv(DATA_PATH).copy()
    arrh_df = pd.read_csv(ARRH_PATH)

    df = df[df["valid_final_measurement"] == True].copy()
    df = df[~df["run_id"].isin(DROP_RUN_IDS)].copy()
    df = df[df["position"].str.lower() == "source"].copy()

    rows = []
    for _, row in df.iterrows():
        pred_nm, B_interp = predict_growth_nm_arrh(
            row["temp_C"],
            row["oxidation_time_min"],
            arrh_df
        )

        out = row.to_dict()
        out["door_baseline_pred_nm"] = pred_nm
        out["B_interp_um2_per_hr"] = B_interp
        out["source_residual_nm"] = row["growth_nm"] - pred_nm
        rows.append(out)

    out_df = pd.DataFrame(rows).sort_values(["temp_C", "oxidation_time_min"]).reset_index(drop=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(out_df[[
        "run_id", "temp_C", "oxidation_time_min",
        "growth_nm", "door_baseline_pred_nm", "source_residual_nm"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()