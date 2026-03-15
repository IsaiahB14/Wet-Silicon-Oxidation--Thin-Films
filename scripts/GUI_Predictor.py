import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

R = 8.314  # J/mol-K

ARRH_PATH = "data/arrhenius_params_osu_fixedA_door.csv"
MODEL_PATH = "MLTrainData/gp_source_residual_model.joblib"
SCALER_PATH = "MLTrainData/gp_source_input_scaler.joblib"

A_FIXED_UM = 0.20
TAU_FIXED_HR = 0.0

TEMP_MIN_C = 1000.0
TEMP_MAX_C = 1150.0
TIME_MIN_MIN = 10.0
TIME_MAX_MIN = 45.0


@st.cache_data
def load_arrh():
    return pd.read_csv(ARRH_PATH)


@st.cache_resource
def load_gp_model():
    gp = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return gp, scaler


def deal_grove_x(A_um, B_um2_per_hr, t_hr, tau_hr=0.0):
    t_eff = np.maximum(np.asarray(t_hr, float) + float(tau_hr), 0.0)
    disc = float(A_um) ** 2 + 4.0 * float(B_um2_per_hr) * t_eff
    disc = np.maximum(disc, 0.0)
    return (-float(A_um) + np.sqrt(disc)) / 2.0


def arrhenius_k(T_K, k0, E):
    return float(k0) * np.exp(-float(E) / (R * float(T_K)))


def predict_door_baseline_nm(T_C, time_min, arrh_df, A_fixed_um=A_FIXED_UM, tau_hr=TAU_FIXED_HR):
    B0 = float(arrh_df.iloc[0]["B0_um2_per_hr"])
    E_B = float(arrh_df.iloc[0]["E_B_J_per_mol"])

    T_K = float(T_C) + 273.15
    t_hr = float(time_min) / 60.0

    B = arrhenius_k(T_K, B0, E_B)
    x_um = deal_grove_x(A_fixed_um, B, t_hr, tau_hr=tau_hr)
    return 1000.0 * x_um, B


def predict_hybrid_nm(T_C, time_min, arrh_df, gp, scaler):
    baseline_nm, B_interp = predict_door_baseline_nm(T_C, time_min, arrh_df)

    X = np.array([[float(T_C), float(time_min)]])
    X_scaled = scaler.transform(X)

    gp_resid_nm, gp_std_nm = gp.predict(X_scaled, return_std=True)
    gp_resid_nm = float(gp_resid_nm[0])
    gp_std_nm = float(gp_std_nm[0])

    hybrid_nm = baseline_nm + gp_resid_nm

    return {
        "temp_C": float(T_C),
        "oxidation_time_min": float(time_min),
        "door_baseline_pred_nm": float(baseline_nm),
        "B_interp_um2_per_hr": float(B_interp),
        "gp_pred_residual_nm": gp_resid_nm,
        "gp_pred_std_nm": gp_std_nm,
        "hybrid_pred_nm": float(hybrid_nm),
    }


def build_target_search_df(
    target_nm,
    arrh_df,
    gp,
    scaler,
    temp_step=5.0,
    time_step=0.5,
    max_gp_std_nm=None
):
    temp_grid = np.arange(TEMP_MIN_C, TEMP_MAX_C + 1e-9, temp_step)
    time_grid = np.arange(TIME_MIN_MIN, TIME_MAX_MIN + 1e-9, time_step)

    rows = []
    for T_C in temp_grid:
        for time_min in time_grid:
            pred = predict_hybrid_nm(T_C, time_min, arrh_df, gp, scaler)
            pred["target_thickness_nm"] = float(target_nm)
            pred["abs_error_to_target_nm"] = abs(pred["hybrid_pred_nm"] - float(target_nm))
            rows.append(pred)

    out_df = pd.DataFrame(rows)

    if max_gp_std_nm is not None:
        out_df = out_df[out_df["gp_pred_std_nm"] < float(max_gp_std_nm)].copy()

    out_df = out_df.sort_values(
        ["abs_error_to_target_nm", "oxidation_time_min", "temp_C"]
    ).reset_index(drop=True)

    return out_df


def build_contour_df(arrh_df, gp, scaler, temp_step=2.5, time_step=0.5):
    temp_grid = np.arange(TEMP_MIN_C, TEMP_MAX_C + 1e-9, temp_step)
    time_grid = np.arange(TIME_MIN_MIN, TIME_MAX_MIN + 1e-9, time_step)

    rows = []
    for T_C in temp_grid:
        for time_min in time_grid:
            pred = predict_hybrid_nm(T_C, time_min, arrh_df, gp, scaler)
            rows.append(pred)

    return pd.DataFrame(rows)


def make_contour_plot(contour_df):
    pivot = contour_df.pivot(
        index="oxidation_time_min",
        columns="temp_C",
        values="hybrid_pred_nm"
    )

    X = pivot.columns.to_numpy()
    Y = pivot.index.to_numpy()
    Z = pivot.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, levels=20)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Predicted oxide thickness (nm)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Oxidation time (min)")
    ax.set_title("Hybrid model oxide thickness map")
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="OSU Wet Oxidation Hybrid Model", layout="wide")

    st.title("OSU Wet Oxidation Hybrid Model")
    st.write(
        "Physics baseline: door-side fixed-A Deal-Grove Arrhenius model. "
        "ML correction: Gaussian Process trained on source wafer residuals."
    )

    arrh_df = load_arrh()
    gp, scaler = load_gp_model()

    tab1, tab2, tab3 = st.tabs([
        "Predict Thickness",
        "Target Thickness Search",
        "Contour Map"
    ])

    with tab1:
        st.subheader("Predict source wafer thickness from temperature and time")

        col1, col2 = st.columns(2)
        with col1:
            temp_C = st.number_input(
                "Temperature (°C)",
                min_value=TEMP_MIN_C,
                max_value=TEMP_MAX_C,
                value=1100.0,
                step=5.0
            )
        with col2:
            time_min = st.number_input(
                "Oxidation time (min)",
                min_value=TIME_MIN_MIN,
                max_value=TIME_MAX_MIN,
                value=30.0,
                step=0.5
            )

        if st.button("Predict Thickness"):
            pred = predict_hybrid_nm(temp_C, time_min, arrh_df, gp, scaler)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Door baseline (nm)", f'{pred["door_baseline_pred_nm"]:.2f}')
            c2.metric("GP residual (nm)", f'{pred["gp_pred_residual_nm"]:.2f}')
            c3.metric("Hybrid thickness (nm)", f'{pred["hybrid_pred_nm"]:.2f}')
            c4.metric("GP std (nm)", f'{pred["gp_pred_std_nm"]:.2f}')

            st.write("Detailed prediction")
            st.dataframe(pd.DataFrame([pred]), use_container_width=True)

    with tab2:
        st.subheader("Find process conditions for a target thickness")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            target_nm = st.number_input(
            "Target thickness (nm)",
            min_value=0.0,
            max_value=1000.0,
            value=200.0,
            step=5.0
            )
        with col2:
            temp_step = st.number_input(
                "Temperature grid step (°C)",
                min_value=1.0,
                max_value=25.0,
                value=5.0,
                step=1.0
            )
        with col3:
            time_step = st.number_input(
                "Time grid step (min)",
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1,
                format="%.1f"
            )
        with col4:
            max_gp_std_nm = st.number_input(
                "Max GP std allowed (nm)",
                min_value=0.0,
                max_value=200.0,
                value=30.0,
                step=1.0
            )

        if st.button("Search Best Conditions"):
            results_df = build_target_search_df(
                target_nm,
                arrh_df,
                gp,
                scaler,
                temp_step=temp_step,
                time_step=time_step,
                max_gp_std_nm=max_gp_std_nm
            )

            if results_df.empty:
                st.warning(
                    "No candidate conditions met the uncertainty filter. "
                    "Try increasing the allowed GP std or using a coarser/finer grid."
                )
            else:
                st.write("Top candidate conditions")
                st.dataframe(results_df.head(10), use_container_width=True)

                best = results_df.iloc[0]
                st.success(
                    f'Best match: {best["temp_C"]:.1f} °C, '
                    f'{best["oxidation_time_min"]:.1f} min, '
                    f'predicted thickness = {best["hybrid_pred_nm"]:.2f} nm, '
                    f'error to target = {best["abs_error_to_target_nm"]:.2f} nm, '
                    f'GP std = {best["gp_pred_std_nm"]:.2f} nm'
                )

    with tab3:
        st.subheader("Hybrid model contour map over measured process window")

        col1, col2 = st.columns(2)
        with col1:
            contour_temp_step = st.number_input(
                "Contour temperature step (°C)",
                min_value=1.0,
                max_value=25.0,
                value=2.5,
                step=0.5
            )
        with col2:
            contour_time_step = st.number_input(
                "Contour time step (min)",
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1,
                format="%.1f"
            )

        if st.button("Generate Contour Map"):
            contour_df = build_contour_df(arrh_df, gp, scaler, contour_temp_step, contour_time_step)
            fig = make_contour_plot(contour_df)
            st.pyplot(fig)

            st.write("Contour data preview")
            st.dataframe(contour_df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()