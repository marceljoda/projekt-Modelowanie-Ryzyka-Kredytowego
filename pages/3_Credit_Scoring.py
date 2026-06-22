
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("💳 Symulator scoringu kredytowego")

# --------------------
# Wczytanie modelu i danych
# --------------------

model = joblib.load("best_credit_model.pkl")
train = pd.read_csv("data/app_train_processed.csv")
X_train = train.drop(columns=["TARGET"])

# --------------------
# Formularz klienta
# --------------------

st.subheader("Dane klienta")

amt_income = st.number_input(
    "Roczny dochód (AMT_INCOME_TOTAL)",
    min_value=1,
    value=150000
)

amt_credit = st.number_input(
    "Kwota kredytu (AMT_CREDIT)",
    min_value=1,
    value=300000
)

amt_annuity = st.number_input(
    "Rata / annuity (AMT_ANNUITY)",
    min_value=1,
    value=25000
)

# Bezpośrednia kontrola wskaźnika Kredyt/Cena (LOAN_TO_GOOD) zamiast ceny towaru
loan_to_good_val = st.slider(
    "Wskaźnik Kredyt/Cena (LOAN_TO_GOOD)",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.05,
)

age_years = st.slider(
    "Wiek (w latach)",
    min_value=18,
    max_value=80,
    value=40
)

children = st.selectbox(
    "Liczba dzieci",
    sorted(X_train["CNT_CHILDREN_grouped"].dropna().unique())
)

own_car_map = {"NIE": 0, "TAK": 1}
own_car_choice = st.selectbox(
    "Posiada samochód?",
    options=list(own_car_map.keys())
)
own_car = own_car_map[own_car_choice]

own_realty_map = {"NIE": 0, "TAK": 1}
own_realty_choice = st.selectbox(
    "Posiada nieruchomość?",
    options=list(own_realty_map.keys())
)
own_realty = own_realty_map[own_realty_choice]

# --------------------
# Budowa rekordu i skalowanie Z-score (po IQR clippingu)
# --------------------

# Bierzemy mediany (które są już przeskalowane) z danych treningowych dla pozostałych zmiennych
new_client = X_train.copy().median(numeric_only=True).to_frame().T

# Dla zmiennych tekstowych podstawiamy najczęstszą wartość (mode)
for col in X_train.select_dtypes(include="object").columns:
    new_client[col] = X_train[col].mode()[0]

# Parametry transformacji (IQR clipping + Z-score) wyliczone z df_train6
STATS = {
    "AMT_INCOME_TOTAL": {"mean": 162727.786, "std": 73285.162, "lower": -22500.0, "upper": 337500.0},
    "AMT_CREDIT": {"mean": 592698.074, "std": 380435.346, "lower": -537975.0, "upper": 1616625.0},
    "AMT_ANNUITY": {"mean": 26795.941, "std": 13286.719, "lower": -10570.5, "upper": 61681.5},
    "LOAN_TO_GOOD": {"mean": 1.12218, "std": 0.12127, "lower": 0.703, "upper": 1.495},
    "CREDIT_DURATION": {"mean": 21.624, "std": 7.816, "lower": -1.531, "upper": 44.279},
    "ANNUITY_TO_INCOME": {"mean": 0.1784, "std": 0.0859, "lower": -0.0571, "upper": 0.4002}
}

# Pomocnicza funkcja do obcięcia wartości odstających i wyliczenia Z-score
def clip_and_scale(val, name):
    clipped = np.clip(val, STATS[name]["lower"], STATS[name]["upper"])
    return (clipped - STATS[name]["mean"]) / STATS[name]["std"]

# Podmiana wartości z formularza po odpowiednim przeskalowaniu Z-score i IQR clippingu
new_client["AMT_INCOME_TOTAL"] = clip_and_scale(amt_income, "AMT_INCOME_TOTAL")
new_client["AMT_CREDIT"] = clip_and_scale(amt_credit, "AMT_CREDIT")
new_client["AMT_ANNUITY"] = clip_and_scale(amt_annuity, "AMT_ANNUITY")

# DAYS_BIRTH jest w dniach surowych
new_client["DAYS_BIRTH"] = -age_years * 365

# Dynamiczne wyliczenie, obcięcie i przeskalowanie cech inżynieryjnych
credit_duration_val = amt_credit / amt_annuity
annuity_to_income_val = amt_annuity / amt_income

new_client["LOAN_TO_GOOD"] = clip_and_scale(loan_to_good_val, "LOAN_TO_GOOD")
new_client["CREDIT_DURATION"] = clip_and_scale(credit_duration_val, "CREDIT_DURATION")
new_client["ANNUITY_TO_INCOME"] = clip_and_scale(annuity_to_income_val, "ANNUITY_TO_INCOME")

# Podmiana cech kategorycznych / binarnych z formularza
new_client["CNT_CHILDREN_grouped"] = children
new_client["FLAG_OWN_CAR"] = own_car
new_client["FLAG_OWN_REALTY"] = own_realty

# --------------------
# Predykcja
# --------------------

if st.button("Oblicz ryzyko"):

    proba = model.predict_proba(new_client)
    probability = proba[0,1]

    st.metric(
        "Prawdopodobieństwo defaultu (PD)",
        f"{100*probability:.2f}%"
    )

    if probability < 0.20:
        st.success("🟢 NISKIE RYZYKO")

    elif probability < 0.50:
        st.warning("🟡 ŚREDNIE RYZYKO")

    else:
        st.error("🔴 WYSOKIE RYZYKO")

    # Dane klienta (formatowanie tabelaryczne)
    st.subheader("Wprowadzone dane i cechy klienta")
    client_display = new_client.T.reset_index()
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
