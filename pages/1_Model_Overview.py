

import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report,make_scorer,roc_curve, precision_score, recall_score, roc_curve, auc
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import plotly.graph_objects as go

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import plotly.express as px

st.title("📊 Podsumowanie modelu")

# Wczytanie modelu
model = joblib.load("best_credit_model.pkl")
all_models = joblib.load("all_models.pkl")

# Wczytanie danych
df = pd.read_csv("data/app_valid_processed.csv")

X = df.drop(columns=["TARGET"])
y = df["TARGET"]

# Predykcje
y_prob = model.predict_proba(X)[:,1]

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Liczba obserwacji, liczba zmiennych
st.write(f"Liczba obserwacji: {len(df)}")
st.write(f"Liczba zmiennych: {X.shape[1]}")


# AUC, Accuracy, Precision, Recall, F1
auc = roc_auc_score(y, y_prob)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("AUC", f"{auc:.4f}")

with col2:
    st.metric("Accuracy", f"{accuracy:.4f}")

with col3:
    st.metric("Precision", f"{precision:.4f}")

with col4:
    st.metric("Recall", f"{recall:.4f}")

with col5:
    st.metric("F1", f"{f1:.4f}")

# wykres niezbalansowania klas
st.subheader("Udział klientów defaultujących")
target_df = (
    y.value_counts(normalize=True)
    .reset_index()
)
target_df.columns = ["TARGET", "Share"]
fig_target = px.bar(
    target_df,
    x="TARGET",
    y="Share",
    text=target_df["Share"].apply(lambda x: f"{x*100:.1f}%")
)
fig_target.update_layout(
    xaxis_title="Default",
    yaxis_title="Udział",
    xaxis=dict(tickmode="array", tickvals=[0, 1])
)
st.plotly_chart(
    fig_target,
    use_container_width=True
)

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

cm_text = [[f"{val:,}".replace(",", " ") for val in row] for row in cm]

fig_cm = px.imshow(
    cm_df,
    aspect="auto",
    title="Confusion Matrix"
)
fig_cm.update_traces(text=cm_text, texttemplate="%{text}")

st.plotly_chart(
    fig_cm,
    use_container_width=True
)

# ROC Curve
# ROC Curve – porównanie modeli
st.subheader("Krzywa ROC")

fig_roc = go.Figure()

fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    line=dict(dash="dash", color="gray"),
    name="Losowy (AUC = 0.5)"
))

for name, mdl in all_models.items():
    y_prob_mdl = mdl.predict_proba(X)[:, 1]
    auc_mdl = roc_auc_score(y, y_prob_mdl)
    fpr_mdl, tpr_mdl, _ = roc_curve(y, y_prob_mdl)

    fig_roc.add_trace(go.Scatter(
        x=fpr_mdl,
        y=tpr_mdl,
        mode="lines",
        name=f"{name} (AUC = {auc_mdl:.4f})"
    ))

fig_roc.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    width=700,
    height=700,
    xaxis=dict(
        range=[0, 1], 
        constrain="domain"
    ),
    yaxis=dict(
        scaleanchor="x", 
        scaleratio=1, 
        range=[0, 1], 
        constrain="domain"
    ),
)


st.plotly_chart(fig_roc)
