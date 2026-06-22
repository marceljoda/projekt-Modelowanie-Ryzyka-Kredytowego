
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

st.title("📈 Eksploracyjna analiza danych")

# --- Funkcje do cache'owania danych ---
@st.cache_data
def load_processed_data():
    return pd.read_csv("data/app_train_processed.csv")

@st.cache_data
def load_raw_column(col_name):
    # Wczytujemy tylko tę jedną kolumnę w celu optymalizacji pamięci i prędkości
    return pd.read_csv("data/application_train.csv", usecols=[col_name])

# Słownik transformacji przed / po
MAPPING_BEFORE_AFTER = {
    "EXT_SOURCE_1_imp_income_type_gender": {
        "raw_col": "EXT_SOURCE_1",
        "type": "numeric",
        "desc": "Imputacja medianą w podgrupach (Typ dochodu, Płeć) oraz standaryzacja Z-score (normalizacja do średniej 0 i odchylenia std 1)."
    },
    "EXT_SOURCE_3_imp_education": {
        "raw_col": "EXT_SOURCE_3",
        "type": "numeric",
        "desc": "Imputacja medianą w podgrupach (Wykształcenie) oraz standaryzacja Z-score (normalizacja do średniej 0 i odchylenia std 1)."
    },
    "DAYS_EMPLOYED_imp_income": {
        "raw_col": "DAYS_EMPLOYED",
        "type": "numeric",
        "desc": "Zastąpienie wartości anomalnych (365243 dni, oznaczających bezrobocie lub specyficzny status emerytalny) wartościami brakującymi, imputacja medianą w grupach typu dochodu oraz standaryzacja Z-score (normalizacja do średniej 0 i odchylenia std 1)."
    },
    "CODE_GENDER_imputed": {
        "raw_col": "CODE_GENDER",
        "type": "categorical",
        "desc": "Imputacja rzadkich wartości anomalnych (np. 'XNA' zastąpione przez najczęściej występującą płeć: 'F')."
    },
    "NAME_TYPE_SUITE_grouped": {
        "raw_col": "NAME_TYPE_SUITE",
        "type": "categorical",
        "desc": "Połączenie rzadkich kategorii (udział < 5%) w jedną kategorię zbiorczą 'other'."
    },
    "NAME_HOUSING_TYPE_grouped": {
        "raw_col": "NAME_HOUSING_TYPE",
        "type": "categorical",
        "desc": "Połączenie rzadkich kategorii (udział < 5%) w jedną kategorię zbiorczą 'other'."
    },
    "OCCUPATION_TYPE_grouped": {
        "raw_col": "OCCUPATION_TYPE",
        "type": "categorical",
        "desc": "Połączenie kategorii 'Core staff' oraz braków danych w nową wspólną kategorię 'missing / core staff'."
    },
    "ORGANIZATION_TYPE_grouped": {
        "raw_col": "ORGANIZATION_TYPE",
        "type": "categorical",
        "desc": "Połączenie rzadkich kategorii (udział < 5%) w jedną kategorię zbiorczą 'other'."
    },
    "NAME_INCOME_TYPE_grouped": {
        "raw_col": "NAME_INCOME_TYPE",
        "type": "categorical",
        "desc": "Grupowanie rzadkich kategorii oraz 'State servant' w jedną kategorię 'State_servant'."
    },
    "NAME_FAMILY_STATUS_grouped": {
        "raw_col": "NAME_FAMILY_STATUS",
        "type": "categorical",
        "desc": "Połączenie kategorii: 'Civil marriage' oraz 'Single / not married' w grupę 'Civil marriage / Single / not married', a 'Unknown' oraz 'Married' w grupę 'Married'."
    },
    "CNT_CHILDREN_grouped": {
        "raw_col": "CNT_CHILDREN",
        "type": "categorical",
        "desc": "Grupowanie liczby dzieci: wartości 0 i 1 pozostawiono bez zmian, natomiast wszystkie wartości >= 2 połączono w kategorię '>=2'."
    },
    "OBS_30_CNT_SOCIAL_CIRCLE_grouped": {
        "raw_col": "OBS_30_CNT_SOCIAL_CIRCLE",
        "type": "categorical",
        "desc": "Grupowanie liczby obserwacji w otoczeniu klienta (30 dni): wartości 0 i 1 pozostawiono bez zmian, natomiast wszystkie wartości >= 2 połączono w kategorię '>=2'."
    },
    "OBS_60_CNT_SOCIAL_CIRCLE_grouped": {
        "raw_col": "OBS_60_CNT_SOCIAL_CIRCLE",
        "type": "categorical",
        "desc": "Grupowanie liczby obserwacji w otoczeniu klienta (60 dni): wartości 0 i 1 pozostawiono bez zmian, natomiast wszystkie wartości >= 2 połączono w kategorię '>=2'."
    },
    "AMT_REQ_CREDIT_BUREAU_YEAR_grouped": {
        "raw_col": "AMT_REQ_CREDIT_BUREAU_YEAR",
        "type": "categorical",
        "desc": "Grupowanie liczby zapytań do BIK (rok): wartości 0 i 1 pozostawiono bez zmian, 2 i 3 połączono w '2-3', wartości >= 4 w '>=4', a braki danych oznaczono jako 'missing'."
    },
    "EXT_SOURCE_2": {
        "raw_col": "EXT_SOURCE_2",
        "type": "numeric",
        "desc": "Uzupełnienie braków danych (imputacja ogólną medianą) oraz standaryzacja Z-score (normalizacja do średniej 0 i odchylenia std 1)."
    },
    "AMT_ANNUITY": {
        "raw_col": "AMT_ANNUITY",
        "type": "numeric",
        "desc": "Uzupełnienie braków danych (imputacja ogólną medianą) oraz standaryzacja Z-score (normalizacja do średniej 0 i odchylenia std 1)."
    },
    "DAYS_LAST_PHONE_CHANGE": {
        "raw_col": "DAYS_LAST_PHONE_CHANGE",
        "type": "numeric",
        "desc": "Uzupełnienie braków danych (imputacja ogólną medianą) oraz standaryzacja Z-score (normalizacja do średniej 0 i odchylenia std 1)."
    },
    "edu_higher_academic": {
        "raw_col": "NAME_EDUCATION_TYPE",
        "type": "categorical",
        "desc": "Zmienna binarna: 1 dla wykształcenia 'Higher education' lub 'Academic degree', 0 w pozostałych przypadkach."
    },
    "2_fam_members": {
        "raw_col": "CNT_FAM_MEMBERS",
        "type": "categorical",
        "desc": "Zmienna binarna: 1 jeśli liczba członków rodziny wynosi dokładnie 2, 0 w pozostałych przypadkach."
    },
    "DEF_60_CNT_SOCIAL_CIRCLE_eq_0": {
        "raw_col": "DEF_60_CNT_SOCIAL_CIRCLE",
        "type": "categorical",
        "desc": "Zmienna binarna: 1 jeśli liczba spóźnień w otoczeniu (60 dni) wynosi 0, 0 w pozostałych przypadkach."
    },
    "AMT_CREDIT": {
        "raw_col": "AMT_CREDIT",
        "type": "numeric",
        "desc": "Standaryzacja Z-score (normalizacja do średniej 0 i odchylenia std 1) surowej kwoty kredytu."
    },
    "AMT_INCOME_TOTAL": {
        "raw_col": "AMT_INCOME_TOTAL",
        "type": "numeric",
        "desc": "Standaryzacja Z-score (normalizacja do średniej 0 i odchylenia std 1) surowego całkowitego dochodu."
    },
    "DAYS_BIRTH": {
        "raw_col": "DAYS_BIRTH",
        "type": "numeric",
        "desc": "Brak zmian (wiek w dniach przeniesiony bezpośrednio z danych surowych)."
    },
    "DAYS_REGISTRATION": {
        "raw_col": "DAYS_REGISTRATION",
        "type": "numeric",
        "desc": "Standaryzacja Z-score (normalizacja do średniej 0 i odchylenia std 1) czasu od rejestracji."
    },
    "DAYS_ID_PUBLISH": {
        "raw_col": "DAYS_ID_PUBLISH",
        "type": "numeric",
        "desc": "Brak zmian (czas od wydania dowodu przeniesiony bezpośrednio z danych surowych)."
    },
    "REGION_POPULATION_RELATIVE": {
        "raw_col": "REGION_POPULATION_RELATIVE",
        "type": "numeric",
        "desc": "Standaryzacja Z-score (normalizacja do średniej 0 i odchylenia std 1) względnej populacji regionu."
    },
    "NAME_CONTRACT_TYPE": {
        "raw_col": "NAME_CONTRACT_TYPE",
        "type": "categorical",
        "desc": "Brak zmian (zmienna przeniesiona bezpośrednio z danych surowych)."
    },
    "FLAG_OWN_CAR": {
        "raw_col": "FLAG_OWN_CAR",
        "type": "categorical",
        "desc": "Brak zmian (zmienna przeniesiona bezpośrednio z danych surowych)."
    },
    "FLAG_OWN_REALTY": {
        "raw_col": "FLAG_OWN_REALTY",
        "type": "categorical",
        "desc": "Brak zmian (zmienna przeniesiona bezpośrednio z danych surowych)."
    },
    "FLAG_EMP_PHONE": {
        "raw_col": "FLAG_EMP_PHONE",
        "type": "categorical",
        "desc": "Brak zmian (zmienna przeniesiona bezpośrednio z danych surowych)."
    },
    "FLAG_WORK_PHONE": {
        "raw_col": "FLAG_WORK_PHONE",
        "type": "categorical",
        "desc": "Brak zmian (zmienna przeniesiona bezpośrednio z danych surowych)."
    },
    "FLAG_CONT_MOBILE": {
        "raw_col": "FLAG_CONT_MOBILE",
        "type": "categorical",
        "desc": "Brak zmian (zmienna przeniesiona bezpośrednio z danych surowych)."
    },
    "FLAG_PHONE": {
        "raw_col": "FLAG_PHONE",
        "type": "categorical",
        "desc": "Brak zmian (zmienna przeniesiona bezpośrednio z danych surowych)."
    },
    "FLAG_EMAIL": {
        "raw_col": "FLAG_EMAIL",
        "type": "categorical",
        "desc": "Brak zmian (zmienna przeniesiona bezpośrednio z danych surowych)."
    },
    "REGION_RATING_CLIENT": {
        "raw_col": "REGION_RATING_CLIENT",
        "type": "categorical",
        "desc": "Brak zmian (zmienna przeniesiona bezpośrednio z danych surowych)."
    },
    "REG_REGION_NOT_LIVE_REGION": {
        "raw_col": "REG_REGION_NOT_LIVE_REGION",
        "type": "categorical",
        "desc": "Brak zmian (zmienna przeniesiona bezpośrednio z danych surowych)."
    },
    "REG_REGION_NOT_WORK_REGION": {
        "raw_col": "REG_REGION_NOT_WORK_REGION",
        "type": "categorical",
        "desc": "Brak zmian (zmienna przeniesiona bezpośrednio z danych surowych)."
    },
    "REG_CITY_NOT_LIVE_CITY": {
        "raw_col": "REG_CITY_NOT_LIVE_CITY",
        "type": "categorical",
        "desc": "Brak zmian (zmienna przeniesiona bezpośrednio z danych surowych)."
    },
    "LIVE_CITY_NOT_WORK_CITY": {
        "raw_col": "LIVE_CITY_NOT_WORK_CITY",
        "type": "categorical",
        "desc": "Brak zmian (zmienna przeniesiona bezpośrednio z danych surowych)."
    }
}

# Wczytanie przetworzonych danych
df = load_processed_data()

variables = [c for c in df.columns if c != "TARGET"]

selected = st.selectbox(
    "Wybierz zmienną",
    variables
)

st.write("Wybrana zmienna:")
st.write(selected)

st.write("Typ zmiennej:")
st.write(str(df[selected].dtype))

# Sprawdzenie typu zmiennej i unikalnych wartości (czy kategoryczna lub binarna 0/1)
unique_vals = df[selected].dropna().unique()
is_categorical_or_binary = (not pd.api.types.is_numeric_dtype(df[selected])) or (len(unique_vals) <= 2)

# Wygładzenie wykresu gęstości (KDE bandwidth adjustment) - stała wartość 2.5
bw_adjust = 2.5

# Funkcja pomocnicza do tworzenia gładkiego KDE z wypełnieniem obszaru pod wykresem
def plot_kde(data_list, name_list, color_list=None, bw_adjust=2.5):
    fig = go.Figure()
    for i, (data, name) in enumerate(zip(data_list, name_list)):
        clean_data = data.dropna()
        if len(clean_data) < 2:
            continue

        # Obliczenie gęstości za pomocą scipy stats
        kde = gaussian_kde(clean_data)
        # Zwiększenie wygładzania
        kde.covariance_factor = lambda: kde.factor * bw_adjust
        kde._compute_covariance()

        # Generowanie punktów dla osi X z marginesem
        x_min, x_max = clean_data.min(), clean_data.max()
        margin = (x_max - x_min) * 0.15 if x_max != x_min else 1.0
        x_grid = np.linspace(x_min - margin, x_max + margin, 300)
        y_grid = kde(x_grid)

        # Wybór koloru i przygotowanie przezroczystego wypełnienia
        color = color_list[i] if color_list else None
        fillcolor = f"rgba({color[4:-1]}, 0.2)" if (color and color.startswith("rgb(")) else None

        fig.add_trace(go.Scatter(
            x=x_grid,
            y=y_grid,
            mode="lines",
            name=name,
            line=dict(width=2.5, color=color),
            fill="tozeroy",
            fillcolor=fillcolor
        ))
    return fig

# --- Wykres 1: Rozkład ogólny ---
st.subheader("Gęstość zmiennych")

if is_categorical_or_binary:
    # Udział procentowy dla kategorycznych / binarnych
    cat_df = df[selected].value_counts(normalize=True).reset_index()
    cat_df.columns = [selected, "Share"]
    cat_df = cat_df.sort_values(by=selected)

    fig = px.bar(
        cat_df, 
        x=selected, 
        y="Share", 
        text=cat_df["Share"].apply(lambda x: f"{x*100:.1f}%")
    )
    fig.update_layout(
        yaxis_title="Gęstość",
        xaxis=dict(type='category') # Wymuszenie kategorii na osi X (zapobiega osiom ułamkowym dla 0/1)
    )
else:
    # Wykres gęstości (KDE)
    fig = plot_kde(
        [df[selected]], 
        [selected], 
        color_list=["rgb(31, 119, 180)"], 
        bw_adjust=bw_adjust
    )
    fig.update_layout(
        xaxis_title=selected, 
        yaxis_title="Gęstość",
        showlegend=False
    )

st.plotly_chart(fig, use_container_width=True)


# --- Wykres 2: Rozkład względem TARGET ---
st.subheader("Funkcja gęstości w podziale na klientów defaultujących i nie-defaultujących")

if is_categorical_or_binary:
    # Wykres słupkowy procentowy z podziałem na TARGET
    cat_target = df.groupby([selected, "TARGET"]).size().unstack(fill_value=0)
    cat_target_pct = cat_target.div(cat_target.sum(axis=0), axis=1).reset_index()

    # Przekształcenie do formatu długiego
    cat_target_melted = cat_target_pct.melt(
        id_vars=selected, 
        value_vars=[0, 1], 
        var_name="TARGET", 
        value_name="Share"
    )
    cat_target_melted["TARGET"] = cat_target_melted["TARGET"].map({0: "Brak defaultu (0)", 1: "Default (1)"})

    fig_target = px.bar(
        cat_target_melted, 
        x=selected, 
        y="Share", 
        color="TARGET", 
        barmode="group",
        text=cat_target_melted["Share"].apply(lambda x: f"{x*100:.1f}%")
    )
    fig_target.update_layout(
        yaxis_title="Share",
        xaxis=dict(type='category')
    )
else:
    # Dwie krzywe gęstości (KDE) na jednym wykresie
    group0 = df[df["TARGET"] == 0][selected]
    group1 = df[df["TARGET"] == 1][selected]

    fig_target = plot_kde(
        [group0, group1], 
        ["Brak defaultu (0)", "Default (1)"], 
        color_list=["rgb(44, 160, 44)", "rgb(214, 39, 40)"], # zielony vs czerwony
        bw_adjust=bw_adjust
    )
    fig_target.update_layout(
        xaxis_title=selected, 
        yaxis_title="Density"
    )

st.plotly_chart(fig_target, use_container_width=True)


# --- Wykres 3: Porównanie przed i po transformacji ---
st.write("---")
st.subheader("🔄 Porównanie rozkładów przed i po transformacji")

selected_transform = st.selectbox(
    "Wybierz zmienną do porównania transformacji",
    options=list(MAPPING_BEFORE_AFTER.keys()),
    format_func=lambda x: f"{x} (oryg. {MAPPING_BEFORE_AFTER[x]['raw_col']})"
)

if selected_transform:
    info = MAPPING_BEFORE_AFTER[selected_transform]
    raw_col = info["raw_col"]
    var_type = info["type"]
    desc = info["desc"]

    st.info(f"**Opis transformacji:** {desc}")

    # Wczytanie surowych danych (tylko potrzebnej kolumny z cache)
    df_raw_col = load_raw_column(raw_col)

    if var_type == "numeric":
        col_left, col_right = st.columns(2)
        raw_data = df_raw_col[raw_col].dropna()
        proc_data = df[selected_transform].dropna()

        # Przed transformacją (surowe)
        with col_left:
            st.write("**Przed transformacją (surowe):**")
            if len(raw_data) >= 2:
                kde_raw = gaussian_kde(raw_data)
                kde_raw.covariance_factor = lambda: kde_raw.factor * bw_adjust
                kde_raw._compute_covariance()
                x_min, x_max = raw_data.min(), raw_data.max()
                margin = (x_max - x_min) * 0.15 if x_max != x_min else 1.0
                x_grid = np.linspace(x_min - margin, x_max + margin, 300)
                y_grid = kde_raw(x_grid)

                fig_left = go.Figure()
                fig_left.add_trace(go.Scatter(
                    x=x_grid,
                    y=y_grid,
                    mode="lines",
                    name="Surowe",
                    line=dict(width=2.5, color="rgb(31, 119, 180)"),
                    fill="tozeroy",
                    fillcolor="rgba(31, 119, 180, 0.2)"
                ))
                fig_left.update_layout(
                    xaxis_title=raw_col,
                    yaxis_title="Gęstość",
                    showlegend=False
                )
                st.plotly_chart(fig_left, use_container_width=True)
            else:
                st.warning("Brak danych lub niewystarczająca liczba obserwacji do wyznaczenia gęstości.")

        # Po transformacji (przetworzone)
        with col_right:
            st.write("**Po transformacji (przetworzone):**")
            if len(proc_data) >= 2:
                kde_proc = gaussian_kde(proc_data)
                kde_proc.covariance_factor = lambda: kde_proc.factor * bw_adjust
                kde_proc._compute_covariance()
                x_min, x_max = proc_data.min(), proc_data.max()
                margin = (x_max - x_min) * 0.15 if x_max != x_min else 1.0
                x_grid = np.linspace(x_min - margin, x_max + margin, 300)
                y_grid = kde_proc(x_grid)

                fig_right = go.Figure()
                fig_right.add_trace(go.Scatter(
                    x=x_grid,
                    y=y_grid,
                    mode="lines",
                    name="Przetworzone",
                    line=dict(width=2.5, color="rgb(255, 127, 14)"),
                    fill="tozeroy",
                    fillcolor="rgba(255, 127, 14, 0.2)"
                ))
                fig_right.update_layout(
                    xaxis_title=selected_transform,
                    yaxis_title="Gęstość",
                    showlegend=False
                )
                st.plotly_chart(fig_right, use_container_width=True)
            else:
                st.warning("Brak danych lub niewystarczająca liczba obserwacji do wyznaczenia gęstości.")

    else:
        col_left, col_right = st.columns(2)

        # Przed transformacją (surowe)
        with col_left:
            st.write("**Przed transformacją (surowe):**")
            raw_series = df_raw_col[raw_col].fillna("missing").astype(str)
            cat_raw = raw_series.value_counts(normalize=True).reset_index()
            cat_raw.columns = [raw_col, "Share"]
            cat_raw = cat_raw.sort_values(by=raw_col)

            fig_left = px.bar(
                cat_raw,
                x=raw_col,
                y="Share",
                text=cat_raw["Share"].apply(lambda x: f"{x*100:.1f}%"),
                color_discrete_sequence=["rgb(31, 119, 180)"]
            )
            fig_left.update_layout(
                yaxis_title="Udział",
                xaxis=dict(type='category')
            )
            st.plotly_chart(fig_left, use_container_width=True)

        # Po transformacji (przetworzone)
        with col_right:
            st.write("**Po transformacji (przetworzone):**")
            proc_series = df[selected_transform].fillna("missing").astype(str)
            cat_proc = proc_series.value_counts(normalize=True).reset_index()
            cat_proc.columns = [selected_transform, "Share"]
            cat_proc = cat_proc.sort_values(by=selected_transform)

            fig_right = px.bar(
                cat_proc,
                x=selected_transform,
                y="Share",
                text=cat_proc["Share"].apply(lambda x: f"{x*100:.1f}%"),
                color_discrete_sequence=["rgb(255, 127, 14)"]
            )
            fig_right.update_layout(
                yaxis_title="Udział",
                xaxis=dict(type='category')
            )
            st.plotly_chart(fig_right, use_container_width=True)
