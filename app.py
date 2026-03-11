import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Macroeconomic Forecaster", 
    page_icon="🌍", 
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load('gdp_xgboost_model.pkl')

model = load_model()

st.title("🌍 Global GDP Growth Forecaster")
st.markdown("### Interactive Machine Learning Model (XGBoost)")
st.write("Adjust the macroeconomic indicators below to simulate economic scenarios and forecast the resulting GDP Growth Rate.")
st.markdown("---")

st.sidebar.header("Control Panel")
st.sidebar.write("Set the economic parameters:")

country_translation_map = {
    "United States": "USA", "Germany": "DEU", "Japan": "JPN", 
    "United Kingdom": "GBR", "France": "FRA", "Italy": "ITA", "Canada": "CAN",
    "Brazil": "BRA", "Russia": "RUS", "India": "IND", "China": "CHN", "South Africa": "ZAF",
    "Bangladesh": "BGD", "Pakistan": "PAK", "Sri Lanka": "LKA", "Nepal": "NPL"
}

ui_groups = {
    "G7": ["United States", "Germany", "Japan", "United Kingdom", "France", "Italy", "Canada"],
    "BRICS": ["Brazil", "Russia", "India", "China", "South Africa"],
    "South Asia": ["Bangladesh", "Pakistan", "Sri Lanka", "Nepal"]
}

selected_bloc = st.sidebar.selectbox("🌍 Select Economic Bloc", list(ui_groups.keys()))
selected_country_display = st.sidebar.selectbox("📍 Select Target Country", ui_groups[selected_bloc])

actual_model_code = country_translation_map[selected_country_display]

edu_exp = st.sidebar.slider("Education Expenditure", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
health_exp = st.sidebar.slider("Health Expenditure", min_value=0.0, max_value=25.0, value=8.0, step=0.1)
mil_exp = st.sidebar.slider("Military Expenditure", min_value=0.0, max_value=15.0, value=2.0, step=0.1)
infra_exp = st.sidebar.slider("Infrastructure Expenditure", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
fdi = st.sidebar.slider("Foreign Direct Investment", min_value=-10.0, max_value=50.0, value=2.0, step=0.5)
trade = st.sidebar.slider("Trade Openness", min_value=0.0, max_value=200.0, value=50.0, step=1.0)

if st.button("🚀 Forecast GDP Growth", type="primary"):
    
    input_data = pd.DataFrame({
        'Education Expenditure': [edu_exp],
        'Health Expenditure': [health_exp],
        'Military Expenditure': [mil_exp],
        'Infrastructure Expenditure': [infra_exp],
        'Foreign Direct Investment': [fdi],
        'Trade': [trade],
        'country': [actual_model_code]  
    })
    
    prediction = model.predict(input_data)[0]
    
    st.markdown("### 📊 Forecasting Results")
    
    if prediction > 0:
        st.success(f"### Predicted GDP Growth Rate: +{prediction:.2f}%")
    else:
        st.error(f"### Predicted GDP Growth Rate: {prediction:.2f}%")
        
    st.write(f"**Mathematical Scenario:** Based on the macroeconomic indicators set below, the model expects a growth shift of **{prediction:.2f}%** for **{selected_country_display}**, with a validated historical error margin of ±2.11%.")

    st.markdown("---")
    st.markdown("#### 📝 Inputted Feature Summary")
    
    summary_data = {
        "Target Country": selected_country_display,
        "Education Expenditure": f"{edu_exp}",
        "Health Expenditure": f"{health_exp}",
        "Military Expenditure": f"{mil_exp}",
        "Infrastructure Expenditure": f"{infra_exp}",
        "Foreign Direct Investment": f"{fdi}",
        "Trade Openness": f"{trade}"
    }
    
    summary_df = pd.DataFrame(summary_data.items(), columns=["Economic Indicator", "Scenario Value"])
    st.table(summary_df)