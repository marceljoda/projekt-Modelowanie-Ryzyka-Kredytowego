
import streamlit as st

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

def strona_startowa():
    st.title("💳 Credit Risk Dashboard")

    st.markdown("""
Projekt modelowania ryzyka kredytowego.

Autorzy:
- Marcel Joda
- Aleksander Kopera
- Maciej Rak

Dataset: Home Credit Default Risk
""")

pg = st.navigation([
    st.Page(strona_startowa, title="Strona startowa", icon="🏠"),
    st.Page("pages/1_Model_Overview.py", title="Podsumowanie modelu", icon="📊"),
    st.Page("pages/2_Exploratory_Data_Analysis.py", title="Eksploracyjna analiza danych", icon="📈"),
    st.Page("pages/3_Client_Explorer.py", title="Eksplorator klientów", icon="👤"),
    st.Page("pages/3_Credit_Scoring.py", title="Scoring kredytowy", icon="💳"),
])
pg.run()
