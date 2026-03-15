# Physics-Informed Machine Learning Model for Wet Thermal Oxidation of Silicon

## Project Overview

This project is a proof of concept showing that a Chemical Engineering student with very little coding experience, but a strong interest in AI, machine learning, and semiconductor processing, can build a functional hybrid model that combines physics and machine learning.

The goal of the project is to predict silicon dioxide growth during wet thermal oxidation in a furnace using experimental data collected at Oregon State University. The model predicts oxide thickness from oxidation temperature and time, and it can also help estimate what process conditions may be needed to reach a desired oxide thickness within the experimental range.

This repository represents a working pipeline that combines:

- Deal-Grove oxidation physics
- Arrhenius temperature fitting
- Gaussian Process residual correction
- a simple Streamlit GUI for interactive predictions

A large portion of this project was vibe-coded as a way to explore what is possible with limited programming experience. Even so, the final result is a functional and fitted proof of concept that demonstrates how domain knowledge in chemical engineering can be combined with modern AI tools to make a useful predictive model.

Clearly, this project could become much stronger with more experimental data, broader validation, and further refinement of the model structure.

---

## Working Experimental Range

This model is currently intended only for interpolation within the experimental range used in the dataset:

- **Temperature:** 1000 to 1150 °C
- **Time:** 15 to 45 minutes

Predictions outside of this range should be treated cautiously.

---

## Modeling Logic

## 1. Deal-Grove Oxidation Physics

The starting point for the model is the Deal-Grove oxidation equation:

\[
x^2 + Ax = B(t + \tau)
\]

Where:

- \(x\) = oxide thickness
- \(t\) = oxidation time
- \(A\) = transition parameter between reaction-limited and diffusion-limited behavior
- \(B\) = diffusion-related oxidation constant
- \(\tau\) = time offset

Because the dataset was limited, the parameter **A** was fixed at **0.2 µm** to stabilize the fitting process. The model then solved for **B** at each furnace temperature.

---

## 2. Fitting B at Each Temperature

Door wafer oxidation data was used at four temperatures:

- 1000 °C
- 1050 °C
- 1100 °C
- 1150 °C

For each temperature, the Deal-Grove equation was fit to the oxide thickness vs. time data to estimate an effective **B** value. This produced one fitted diffusion constant for each furnace temperature.

These values are stored in the fitted parameter CSV files and form the bridge between the physical model and the temperature dependence.

---

## 3. Arrhenius Fit for Temperature Dependence

After obtaining the four fitted **B** values, an Arrhenius relationship was used to model how **B** changes with temperature:

\[
B(T) = B_0 e^{-E_B/(RT)}
\]

By fitting:

\[
\ln(B) \text{ vs } 1/T
\]

the code extracts:

- \(B_0\), the prefactor
- \(E_B\), the activation energy

This gives a continuous temperature-dependent function for **B(T)** across the experimental range.

---

## 4. Physics-Based Thickness Prediction

Once the Arrhenius form of **B(T)** is known, it is plugged back into the Deal-Grove equation:

\[
x^2 + Ax = B(T)t
\]

The quadratic is then solved for oxide thickness to generate a baseline physics prediction for any temperature and time pair within the modeled range.

This prediction captures the main physical oxidation trend, but it does not fully account for furnace-specific effects and experimental variation.

---

## 5. Residual Calculation

To improve the baseline model, source wafer measurements were compared against the physics-based predictions:

\[
\text{residual} = x_{\text{measured}} - x_{\text{physics}}
\]

These residuals capture the part of the behavior that the pure physics model does not explain, such as:

- furnace non-uniformity
- position effects
- flow differences
- small systematic measurement deviations

---

## 6. Gaussian Process Residual Modeling

A Gaussian Process was then trained on the residuals as a function of temperature and time.

The Gaussian Process was chosen largely because the dataset was limited. With sparse data, GP models are useful because they can interpolate smoothly and weight nearby points through the kernel structure in a statistically grounded way. In this project, that made it a strong choice for learning furnace-specific corrections without requiring a very large dataset.

The GP does not replace the physics. Instead, it acts as a correction layer on top of the Deal-Grove and Arrhenius-based prediction.

The final hybrid prediction becomes:

\[
x_{\text{final}}(T,t) = x_{\text{physics}}(T,t) + x_{\text{GP}}(T,t)
\]

---

## 7. Streamlit GUI

A simple Streamlit GUI was also created so the user can interact with the model more easily.

The GUI allows a user to enter temperature and time values and receive a predicted oxide thickness from the hybrid model. This was included to make the proof of concept more accessible and to show how the model could eventually be used in a more practical workflow.

---

## Repository File Guide

### `data/`
Contains processed CSV files used throughout the fitting and validation workflow. These include cleaned oxidation datasets, fitted Arrhenius parameters, calibrated Deal-Grove parameters, and model prediction outputs.

Examples include:
- oxidation datasets
- per-temperature fitted parameter tables
- training and validation prediction outputs
- Arrhenius parameter summaries

### `maindata/`
Contains residual-related datasets and Gaussian Process training or validation outputs.

Examples include:
- residual datasets
- GP training predictions
- GP validation predictions

### `MLTrainData/`
Contains saved machine learning artifacts and prediction files used by the Gaussian Process portion of the project.

Examples include:
- trained GP model `.joblib` files
- input scaler `.joblib`
- prediction CSV outputs

### `scripts/`
Contains the Python scripts used to run each stage of the project.

These scripts include:
- fitting the Deal-Grove model
- computing residuals
- training and validating the Gaussian Process
- launching the GUI predictor

---

## What This Project Shows

This project is not meant to claim a perfect or production-ready semiconductor oxidation model.

Instead, it shows that:
- a physics-informed machine learning workflow can be built with limited coding experience
- AI tools can help accelerate technical prototyping
- chemical engineering knowledge can be combined with data-driven modeling to produce a functional predictive tool

With more data, stronger validation, uncertainty analysis, and additional process variables, this model could absolutely be improved further.

---

## Final Note

Thank you for taking the time to look through this repository.

I would be happy to answer any questions about the model logic, the fitting workflow, the Gaussian Process correction, or how the project was built.
