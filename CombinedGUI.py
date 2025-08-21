# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 18:05:29 2025

@author: deeptarka.roy
"""

import pandas as pd
import streamlit as st
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- TITLE ---
st.markdown("<h1 style='font-size: 38px; font-weight: bold;'>Backbone Curve and Damage States of Piers</h1>", unsafe_allow_html=True)

# --- COLUMN TYPE SELECTION ---
st.sidebar.markdown("<h2 style='font-weight:bold; font-size:20px;'>Select Column Type</h2>", unsafe_allow_html=True)

# Selectbox with empty label
column_type = st.sidebar.selectbox(
    "",
    ("Ductile (A.xlsx)", "Non-Ductile (Sn.xlsx)"))

if column_type.startswith("Ductile"):
    file_name = "A.xlsx"
    input_defaults = {"D": 1200.0, "L/D": 6.6, "fc": 45.0, "fyl": 450.0, "fyt": 450.0,
                      "pl": 0.019, "pt": 0.019, "Ny": 0.055}
    app_title = "Ductile Pier"
else:
    file_name = "Sn.xlsx"
    input_defaults = {"D": 700.0, "L/D": 3.5, "fc": 45.0, "fyl": 450.0, "fyt": 450.0,
                      "pl": 0.019, "pt": 0.0019, "Ny": 0.055}
    app_title = "Non-ductile Pier"

# Uniform header size (smaller than title)
def custom_header(text):
    st.markdown(f"<h2 style='font-size: 28px; font-weight: bold;'>{text}</h2>", unsafe_allow_html=True)

custom_header(app_title)

# --- DATA LOADING ---
try:
    df = pd.read_excel(file_name)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

input_cols = ["D","L/D","fc","fyl","fyt","pl","pt","Ny"]
output_cols = ["DS1","DS2","DS3","DS4","F1","F2","F3","F4"]
x, y = df[input_cols], df[output_cols]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- CACHING THE FITTED MODEL ---
@st.cache_resource
def fit_model(_model_obj, x_tr, y_tr):
    _model_obj.fit(x_tr, y_tr)
    return _model_obj

model_obj = XGBRegressor()
model = fit_model(model_obj, x_train, y_train)

# --- SIDEBAR USER INPUT ---
st.sidebar.header("Input Parameters")
def user_input_features(defaults):
    data = {}
    for k, v in defaults.items():
        if k in ("pl", "pt", "Ny"):
            step = 0.0001
            fmt = "%.4f"
        else:
            step = 0.001
            fmt = "%.3f"
        val = st.sidebar.number_input(k, value=float(v), format=fmt, step=step)
        data[k] = round(val, 2)
    return pd.DataFrame(data, index=[0])

Data = user_input_features(input_defaults)
prediction = model.predict(Data)

# ==========================================
# ----------- BOLD, NO-INDEX TABLES --------
# ==========================================
def display_bold_table_styled(df):
    # Drop the row index before styling
    df_reset = df.reset_index(drop=True)

    styled_df = (
        df_reset.style.set_table_styles([
            dict(selector="th", props=[
                ("font-size", "18px"), 
                ("font-weight", "bold"), 
                ("color", "#000000"), 
                ("border", "2px solid black"),
                ("text-align", "center")
            ]),
            dict(selector="td", props=[
                ("font-size", "15px"), 
                ("font-weight", "bold"), 
                ("color", "#000000"), 
                ("border", "2px solid black"),
                ("text-align", "center")
            ]),
            dict(selector="table", props=[("border-collapse", "collapse")])
        ])
    )
    st.table(styled_df)

# ==========================================
# ----------- DISPLAY INPUT ----------------
# ==========================================
custom_header("Specified Input Parameters")

# Show with 2 decimals and labels
Data_disp = Data.copy().applymap(lambda x: f"{x:.2f}")
Data_disp.columns = [
    "D (mm)", "L/D", "fc (MPa)", "fyl (MPa)", "fyt (MPa)", "pl", "pt", "Ny"
]
display_bold_table_styled(Data_disp)

# ==========================================
# ----------- PREDICTED OUTPUT -------------
# ==========================================
custom_header("Predicted Damage States")

P = pd.DataFrame(prediction,
                 columns=["DS1","DS2","DS3","DS4","F1 (kN)","F2 (kN)","F3 (kN)","F4 (kN)"])

P_disp = P[["DS1","DS2","DS3","DS4"]].applymap(lambda x: f"{x:.2f}")
display_bold_table_styled(P_disp)

# ==========================================
# ----------- CURVE PLOTTING ---------------
# ==========================================
custom_header("Predicted Backbone Curve")

a = np.insert(P[["DS1","DS2","DS3","DS4"]].values.flatten(), 0, 0)
b = np.insert(P[["F1 (kN)","F2 (kN)","F3 (kN)","F4 (kN)"]].values.flatten(), 0, 0)

fig, ax = plt.subplots(figsize=(5, 2.8))
ax.plot(a, b, marker="o")

# Range with slight margins
ax.set_xlim(0, max(a) * 1.05)
ax.set_ylim(0, max(b) * 1.05)

# Always place DS labels below curve
for i in range(1, 5):
    ax.annotate(f"DS{i}", (a[i], b[i]), textcoords="offset points",
                xytext=(5,-15) , ha="center", fontsize=8, fontweight="normal")

ax.set_xlabel("Drift Ratio (%)", fontsize=8, fontweight="normal")
ax.set_ylabel("Force (kN)", fontsize=8, fontweight="normal")

ax.tick_params(axis='both', which='major', labelsize=9)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight("normal")

ax.grid(False)
st.pyplot(fig, use_container_width=True)