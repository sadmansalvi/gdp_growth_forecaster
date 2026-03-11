import streamlit as st
import pandas as pd
import joblib

# 1. Configure the Page settings for your thesis presentation
st.set_page_config(
    page_title="Macroeconomic Forecaster", 
    page_icon="🌍", 
    layout="wide"
)

# 2. Load the Champion Model (Cached so it doesn't reload on every click)
@st.cache_resource
def load_model():
    return joblib.load('gdp_xgboost_model.pkl')

model = load_model()

# 3. Build the Dashboard Header
st.title("🌍 Global GDP Growth Forecaster")
st.markdown("### Interactive Machine Learning Model (XGBoost)")
st.write("Adjust the macroeconomic indicators below to simulate economic scenarios and forecast the resulting GDP Growth Rate.")
st.markdown("---")

# 4. Build the Sidebar Controls (The Sliders)
st.sidebar.header("Control Panel")
st.sidebar.write("Set the economic parameters:")

# 1. The Translation Dictionary (What the User Sees vs. What the Model Needs)
country_translation_map = {
    "United States": "USA", "Germany": "DEU", "Japan": "JPN", 
    "United Kingdom": "GBR", "France": "FRA", "Italy": "ITA", "Canada": "CAN",
    "Brazil": "BRA", "Russia": "RUS", "India": "IND", "China": "CHN", "South Africa": "ZAF",
    "Bangladesh": "BGD", "Pakistan": "PAK", "Sri Lanka": "LKA", "Nepal": "NPL"
}

# 2. The UI Groupings (Using Full Names for the Dropdowns)
ui_groups = {
    "G7": ["United States", "Germany", "Japan", "United Kingdom", "France", "Italy", "Canada"],
    "BRICS": ["Brazil", "Russia", "India", "China", "South Africa"],
    "South Asia": ["Bangladesh", "Pakistan", "Sri Lanka", "Nepal"]
}

# 3. The Interactive Dropdowns (Frontend)
selected_bloc = st.sidebar.selectbox("🌍 Select Economic Bloc", list(ui_groups.keys()))
selected_country_display = st.sidebar.selectbox("📍 Select Target Country", ui_groups[selected_bloc])

# 4. The Magic Translation (Backend)
# This grabs the 3-letter code to feed into your XGBoost pipeline
actual_model_code = country_translation_map[selected_country_display]

# Sliders for the numerical features
edu_exp = st.sidebar.slider("Education Expenditure", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
health_exp = st.sidebar.slider("Health Expenditure", min_value=0.0, max_value=25.0, value=8.0, step=0.1)
mil_exp = st.sidebar.slider("Military Expenditure", min_value=0.0, max_value=15.0, value=2.0, step=0.1)
infra_exp = st.sidebar.slider("Infrastructure Expenditure", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
fdi = st.sidebar.slider("Foreign Direct Investment", min_value=-10.0, max_value=50.0, value=2.0, step=0.5)
trade = st.sidebar.slider("Trade Openness", min_value=0.0, max_value=200.0, value=50.0, step=1.0)

# 5. The Prediction Engine
if st.button("🚀 Forecast GDP Growth", type="primary"):
    
    # Pack the slider inputs into a 1-row matrix with your exact column names
    input_data = pd.DataFrame({
        'Education Expenditure': [edu_exp],
        'Health Expenditure': [health_exp],
        'Military Expenditure': [mil_exp],
        'Infrastructure Expenditure': [infra_exp],
        'Foreign Direct Investment': [fdi],
        'Trade': [trade],
        'country': [actual_model_code]  # <--- Use the translated 3-letter code here!
    })
    
    # Pass the data through the pipeline
    prediction = model.predict(input_data)[0]
    
    # 6. Display the Output beautifully
    st.markdown("### 📊 Forecasting Results")
    
    # Color-code the output: Green for growth, Red for recession
    if prediction > 0:
        st.success(f"### Predicted GDP Growth Rate: +{prediction:.2f}%")
    else:
        st.error(f"### Predicted GDP Growth Rate: {prediction:.2f}%")
        
    st.write(f"**Mathematical Scenario:** Based on the macroeconomic indicators set below, the model expects a growth shift of **{prediction:.2f}%** for **{selected_country_display}**, with a validated historical error margin of ±2.11%.")

    # 7. Print the User Input Summary Table
    st.markdown("---")
    st.markdown("#### 📝 Inputted Feature Summary")
    
    # Create a clean dictionary for display (so they see the full country name, not the 3-letter code)
    summary_data = {
        "Target Country": selected_country_display,
        "Education Expenditure": f"{edu_exp}",
        "Health Expenditure": f"{health_exp}",
        "Military Expenditure": f"{mil_exp}",
        "Infrastructure Expenditure": f"{infra_exp}",
        "Foreign Direct Investment": f"{fdi}",
        "Trade Openness": f"{trade}"
    }
    
    # Convert to a DataFrame and display as a static table
    summary_df = pd.DataFrame(summary_data.items(), columns=["Economic Indicator", "Scenario Value"])
    st.table(summary_df)