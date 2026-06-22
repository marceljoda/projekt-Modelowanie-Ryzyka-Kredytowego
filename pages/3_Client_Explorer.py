
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("👤 Eksplorator klientów")

# --------------------
# Dane
# --------------------

model = joblib.load("best_credit_model.pkl")
df = pd.read_csv("data/app_valid_processed.csv")

X = df.drop(columns=["TARGET"])
y = df["TARGET"]

# --------------------
# Wybór klienta
# --------------------

client_id = st.number_input(
    "Wybierz klienta (indeks)",
    min_value=0,
    max_value=len(X)-1,
    value=0,
    step=1
)

client = X.iloc[[client_id]].copy()

# --------------------
# Predykcja
# --------------------

probability = model.predict_proba(client)[0,1]

# --------------------
# Modyfikacja cech klienta (What-If) - 3 pola w jednym wierszu
# --------------------

st.subheader("Modyfikacja cech klienta (What-If)")
col_in1, col_in2, col_in3 = st.columns(3)

with col_in1:
    new_income = st.slider(
        "Mnożnik dochodu",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1
    )

with col_in2:
    new_credit = st.slider(
        "Mnożnik wysokości kredytu",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

with col_in3:
    age_shift = st.slider(
        "Zmiana wieku (w latach)",
        min_value=-20,
        max_value=20,
        value=0
    )

modified_client = client.copy()

# Słownik z parametrami transformacji (IQR clipping + Z-score) wyliczony z danych treningowych
STATS = {
    "AMT_INCOME_TOTAL": {"mean": 162727.786, "std": 73285.162, "lower": -22500.0, "upper": 337500.0},
    "AMT_CREDIT": {"mean": 592698.074, "std": 380435.346, "lower": -537975.0, "upper": 1616625.0},
    "AMT_ANNUITY": {"mean": 26795.941, "std": 13286.719, "lower": -10570.5, "upper": 61681.5},
    "LOAN_TO_GOOD": {"mean": 1.12218, "std": 0.12127, "lower": 0.703, "upper": 1.495},
    "CREDIT_DURATION": {"mean": 21.624, "std": 7.816, "lower": -1.531, "upper": 44.279},
    "ANNUITY_TO_INCOME": {"mean": 0.1784, "std": 0.0859, "lower": -0.0571, "upper": 0.4002}
}

# Odskalowanie do wartości surowych (odpowiadających skali po obcięciu outlierów)
raw_income_orig = client["AMT_INCOME_TOTAL"].values[0] * STATS["AMT_INCOME_TOTAL"]["std"] + STATS["AMT_INCOME_TOTAL"]["mean"]
raw_credit_orig = client["AMT_CREDIT"].values[0] * STATS["AMT_CREDIT"]["std"] + STATS["AMT_CREDIT"]["mean"]
raw_annuity_orig = client["AMT_ANNUITY"].values[0] * STATS["AMT_ANNUITY"]["std"] + STATS["AMT_ANNUITY"]["mean"]
raw_loan_to_good_orig = client["LOAN_TO_GOOD"].values[0] * STATS["LOAN_TO_GOOD"]["std"] + STATS["LOAN_TO_GOOD"]["mean"]

# Zastosowanie zmian wprowadzonych suwakami
new_raw_income = raw_income_orig * new_income
new_raw_credit = raw_credit_orig * new_credit

# Wyliczenie nowych cech inżynieryjnych w skali surowej
new_loan_to_good_val = new_credit * raw_loan_to_good_orig
new_credit_duration_val = new_raw_credit / raw_annuity_orig
new_annuity_to_income_val = raw_annuity_orig / new_raw_income

# Pomocnicza funkcja do obcięcia wartości odstających i wyliczenia Z-score
def clip_and_scale(val, name):
    clipped = np.clip(val, STATS[name]["lower"], STATS[name]["upper"])
    return (clipped - STATS[name]["mean"]) / STATS[name]["std"]

# Zapisanie przeskalowanych wartości do rekordu klienta
modified_client["AMT_INCOME_TOTAL"] = clip_and_scale(new_raw_income, "AMT_INCOME_TOTAL")
modified_client["AMT_CREDIT"] = clip_and_scale(new_raw_credit, "AMT_CREDIT")
modified_client["DAYS_BIRTH"] = modified_client["DAYS_BIRTH"] - age_shift*365
modified_client["LOAN_TO_GOOD"] = clip_and_scale(new_loan_to_good_val, "LOAN_TO_GOOD")
modified_client["CREDIT_DURATION"] = clip_and_scale(new_credit_duration_val, "CREDIT_DURATION")
modified_client["ANNUITY_TO_INCOME"] = clip_and_scale(new_annuity_to_income_val, "ANNUITY_TO_INCOME")

new_probability = model.predict_proba(modified_client)[0,1]

# --------------------
# Wyniki analizy - 3 pola w jednym wierszu (bez powtórzonego prawdopodobieństwa)
# --------------------

st.subheader("Wyniki analizy ryzyka")
col_out1, col_out2, col_out3 = st.columns(3)

with col_out1:
    st.metric(
        "Oryginalne PD",
        f"{100*probability:.2f}%"
    )

with col_out2:
    st.metric(
        "Zmodyfikowane PD",
        f"{100*new_probability:.2f}%",
        delta=f"{100*(new_probability-probability):.2f}%"
    )

real_target = y.iloc[client_id]
target_text = "TAK" if int(real_target) == 1 else "NIE"

with col_out3:
    st.metric(
        "Czy klient zdefaultował?",
        target_text
    )

# --------------------
# Ocena ryzyka (na podstawie zmodyfikowanego PD)
# --------------------

if new_probability < 0.20:
    st.success("🟢 NISKIE RYZYKO")

elif new_probability < 0.50:
    st.warning("🟡 ŚREDNIE RYZYKO")

else:
    st.error("🔴 WYSOKIE RYZYKO")

# --------------------
# Dane klienta
# --------------------

st.subheader("Cechy klienta")

client_display = client.T.reset_index()
client_display.columns = ["Nazwa zmiennej", "Wartość zmiennej"]

def format_val(val):
    if pd.isna(val):
        return ""
    try:
        num_val = float(val)
        if num_val.is_integer():
            return str(int(num_val))
        return f"{num_val:.2f}"
    except (ValueError, TypeError):
        return str(val)

client_display["Wartość zmiennej"] = client_display["Wartość zmiennej"].apply(format_val)

st.dataframe(client_display, use_container_width=True, hide_index=True)
