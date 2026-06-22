
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    confusion_matrix
)
import plotly.graph_objects as go
import plotly.express as px

# Konfiguracja cache'owania danych do krzywych ROC
@st.cache_data
def get_roc_data():
    all_models = joblib.load('all_models.pkl')
    train = pd.read_csv('data/app_train_processed.csv')
    test = pd.read_csv('data/app_valid_processed.csv')

    X_train = train.drop(columns=['TARGET'])
    y_train = train['TARGET']
    X_test = test.drop(columns=['TARGET'])
    y_test = test['TARGET']

    roc_data = {
        'train': {},
        'test': {}
    }

    for name, mdl in all_models.items():
        # Train ROC
        p_tr = mdl.predict_proba(X_train)[:, 1]
        fpr_tr, tpr_tr, _ = roc_curve(y_train, p_tr)
        auc_tr = roc_auc_score(y_train, p_tr)
        step_tr = max(1, len(fpr_tr) // 1000)
        roc_data['train'][name] = {
            'fpr': fpr_tr[::step_tr].tolist(),
            'tpr': tpr_tr[::step_tr].tolist(),
            'auc': float(auc_tr)
        }

        # Test ROC
        p_te = mdl.predict_proba(X_test)[:, 1]
        fpr_te, tpr_te, _ = roc_curve(y_test, p_te)
        auc_te = roc_auc_score(y_test, p_te)
        step_te = max(1, len(fpr_te) // 1000)
        roc_data['test'][name] = {
            'fpr': fpr_te[::step_te].tolist(),
            'tpr': tpr_te[::step_te].tolist(),
            'auc': float(auc_te)
        }

    return roc_data

st.title("📊 Podsumowanie i porównanie modeli")

# Wczytanie modelu
model = joblib.load("best_credit_model.pkl")
all_models = joblib.load("all_models.pkl")

# Wczytanie danych
df = pd.read_csv("data/app_valid_processed.csv")
X = df.drop(columns=["TARGET"])
y = df["TARGET"]

# ------------------------------------------------------------------------------
# Sekcja 1: Główny Model i parametry odcięcia
# ------------------------------------------------------------------------------

threshold = st.slider(
    "Próg odcięcia",
    min_value=0.1,
    max_value=0.9,
    value=0.6367,
    step=0.01
)

# Predykcje dla wybranego progu
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# Liczenie metryk
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_prob)

# Podstawowe statystyki zbioru
st.write(f"Liczba obserwacji (zbiór testowy): **{len(df):,}**".replace(",", " "))
st.write(f"Liczba zmiennych wejściowych: **{X.shape[1]}**")

# Wyświetlanie metryk w kolumnach (przetłumaczone na język polski)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("AUC", f"{auc:.4f}")
with col2:
    st.metric("Dokładność", f"{accuracy:.4f}")
with col3:
    st.metric("Precyzja", f"{precision:.4f}")
with col4:
    st.metric("Czułość", f"{recall:.4f}")
with col5:
    st.metric("F1-Score", f"{f1:.4f}")

st.write("---")

# Rozkład target i Confusion Matrix obok siebie w kolumnach
col_dist, col_cm = st.columns(2)

with col_dist:
    st.subheader("Udział klientów defaultujących")
    target_df = y.value_counts(normalize=True).reset_index()
    target_df.columns = ["TARGET", "Udział"]
    fig_target = px.bar(
        target_df,
        x="TARGET",
        y="Udział",
        text=target_df["Udział"].apply(lambda x: f"{x*100:.1f}%"),
        color_discrete_sequence=["#1f77b4"]
    )
    fig_target.update_layout(
        xaxis_title="Default",
        yaxis_title="Udział",
        xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Brak defaultu (0)", "Default (1)"]),
        height=350,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_target, use_container_width=True)

with col_cm:
    st.subheader("Macierz błędu (Confusion Matrix)")
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Brak defaultu (0)", "Default (1)"],
        columns=["Brak defaultu (Pred 0)", "Default (Pred 1)"]
    )
    cm_text = [[f"{val:,}".replace(",", " ") for val in row] for row in cm]

    fig_cm = px.imshow(
        cm_df,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Predykcja", y="Rzeczywistość", color="Liczba")
    )
    fig_cm.update_traces(text=cm_text, texttemplate="%{text}", textfont=dict(size=14))
    fig_cm.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ------------------------------------------------------------------------------
# Sekcja 2: Porównanie Modeli
# ------------------------------------------------------------------------------
st.write("---")
st.subheader("📊 Porównanie wyników modeli (Trening vs Test)")

# Wczytanie prekalkulowanych metryk
df_metrics = pd.read_csv("data/model_comparison_results.csv")

# Tłumaczenie nazw kolumn i filtrowanie (usunięto zysk finansowy i optymalny próg)
df_metrics_pl = df_metrics.rename(columns={
    "Model": "Model",
    "Dataset": "Zbiór danych",
    "AUC": "AUC",
    "Accuracy": "Dokładność",
    "Precision": "Precyzja",
    "Recall": "Czułość",
    "F1": "F1-Score"
})

# Zachowanie tylko wymaganych kolumn
df_metrics_pl = df_metrics_pl[["Model", "Zbiór danych", "AUC", "Dokładność", "Precyzja", "Czułość", "F1-Score"]]

st.dataframe(
    df_metrics_pl.style.format({
        "AUC": "{:.4f}",
        "Dokładność": "{:.4f}",
        "Precyzja": "{:.4f}",
        "Czułość": "{:.4f}",
        "F1-Score": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

# Porównanie krzywych ROC (Trening vs Test)
st.write("### Krzywe ROC (Trening vs Test)")
try:
    roc_data = get_roc_data()
    col_roc_tr, col_roc_te = st.columns(2)

    with col_roc_tr:
        st.write("**Zbiór Treningowy**")
        fig_roc_tr = go.Figure()
        fig_roc_tr.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False
        ))
        for model_name, data in roc_data["train"].items():
            fig_roc_tr.add_trace(go.Scatter(
                x=data["fpr"],
                y=data["tpr"],
                mode="lines",
                name=f"{model_name} (AUC = {data['auc']:.4f})"
            ))
        fig_roc_tr.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(range=[0, 1], constrain="domain"),
            yaxis=dict(scaleanchor="x", scaleratio=1, range=[0, 1], constrain="domain"),
        )
        st.plotly_chart(fig_roc_tr, use_container_width=True)

    with col_roc_te:
        st.write("**Zbiór Testowy**")
        fig_roc_te = go.Figure()
        fig_roc_te.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False
        ))
        for model_name, data in roc_data["test"].items():
            fig_roc_te.add_trace(go.Scatter(
                x=data["fpr"],
                y=data["tpr"],
                mode="lines",
                name=f"{model_name} (AUC = {data['auc']:.4f})"
            ))
        fig_roc_te.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(range=[0, 1], constrain="domain"),
            yaxis=dict(scaleanchor="x", scaleratio=1, range=[0, 1], constrain="domain"),
        )
        st.plotly_chart(fig_roc_te, use_container_width=True)

except Exception as e:
    st.warning(f"Nie udało się załadować danych do krzywych ROC: {e}")
