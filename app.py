import streamlit as st
import numpy as np
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

def optimize_budget(target_country_code, fdi_val, trade_val, total_budget, min_dict):
    np.random.seed(42) 
    n_scenarios = 5000
    
    total_min = sum(min_dict.values())
    
    if total_min >= total_budget:
        # If minimums exceed the budget, we just use the minimum values
        shares = np.array([[min_dict['edu'], min_dict['health'], min_dict['mil'], min_dict['infra']]])
    else:
        # Calculate discretionary budget
        remaining_budget = total_budget - total_min
        raw_shares = np.random.dirichlet(np.ones(4), size=n_scenarios)
        
        # Add random shares of the remaining budget to the baseline minimums
        edu_s = (raw_shares[:, 0] * remaining_budget) + min_dict['edu']
        health_s = (raw_shares[:, 1] * remaining_budget) + min_dict['health']
        mil_s = (raw_shares[:, 2] * remaining_budget) + min_dict['mil']
        infra_s = (raw_shares[:, 3] * remaining_budget) + min_dict['infra']
        
        shares = np.column_stack([edu_s, health_s, mil_s, infra_s])
    
    opt_df = pd.DataFrame({
        'Education Expenditure': shares[:, 0],
        'Health Expenditure': shares[:, 1],
        'Military Expenditure': shares[:, 2],
        'Infrastructure Expenditure': shares[:, 3],
        'Foreign Direct Investment': [fdi_val] * len(shares),
        'Trade': [trade_val] * len(shares),
        'country': [target_country_code] * len(shares)
    })
    
    preds = model.predict(opt_df)
    best_idx = np.argmax(preds)
    return opt_df.iloc[best_idx], preds[best_idx]

# App UI Header
st.title("🌍 Global GDP Growth Forecaster")
st.markdown("### Interactive Machine Learning Model (XGBoost)")
st.write("Simulate economic scenarios or use the AI Optimizer with custom baseline constraints.")
st.markdown("---")

# Sidebar - Control Panel
st.sidebar.header("Control Panel")

country_translation_map = {
    "United States": "United States", "Germany": "Germany", "Japan": "Japan", 
    "United Kingdom": "United Kingdom", "France": "France", "Italy": "Italy", 
    "Canada": "Canada", "Brazil": "Brazil", "Russia": "Russian Federation", 
    "India": "India", "China": "China", "South Africa": "South Africa",
    "Bangladesh": "Bangladesh", "Pakistan": "Pakistan", "Sri Lanka": "Sri Lanka", "Nepal": "Nepal"
}

ui_groups = {
    "G7": ["United States", "Germany", "Japan", "United Kingdom", "France", "Italy", "Canada"],
    "BRICS": ["Brazil", "Russia", "India", "China", "South Africa"],
    "South Asia": ["Bangladesh", "Pakistan", "Sri Lanka", "Nepal"]
}

selected_bloc = st.sidebar.selectbox("🌍 Select Economic Bloc", list(ui_groups.keys()))
selected_country_display = st.sidebar.selectbox("📍 Select Target Country", ui_groups[selected_bloc])
actual_model_code = country_translation_map[selected_country_display]

# Manual Sliders
st.sidebar.subheader("Manual Indicators")
edu_exp = st.sidebar.slider("Education Expenditure", 0.0, 20.0, 5.0, 0.1)
health_exp = st.sidebar.slider("Health Expenditure", 0.0, 25.0, 8.0, 0.1)
mil_exp = st.sidebar.slider("Military Expenditure", 0.0, 15.0, 2.0, 0.1)
infra_exp = st.sidebar.slider("Infrastructure Expenditure", 0.0, 50.0, 10.0, 0.5)
fdi = st.sidebar.slider("Foreign Direct Investment", -10.0, 50.0, 2.0, 0.5)
trade = st.sidebar.slider("Trade Openness", 0.0, 200.0, 50.0, 1.0)

# Optimizer Constraints in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 AI Policy Optimizer")
budget_limit = st.sidebar.number_input("Total Budget Limit (% of GDP)", 1.0, 100.0, 25.0)

# Individual Min Sliders
st.sidebar.write("**Set Minimum Requirements (%):**")
min_edu = st.sidebar.slider("Min Education", 0.0, 10.0, 2.0, 0.5)
min_health = st.sidebar.slider("Min Health", 0.0, 10.0, 2.0, 0.5)
min_mil = st.sidebar.slider("Min Military", 0.0, 10.0, 1.0, 0.5)
min_infra = st.sidebar.slider("Min Infrastructure", 0.0, 15.0, 3.0, 0.5)

min_constraints = {'edu': min_edu, 'health': min_health, 'mil': min_mil, 'infra': min_infra}

# Main Action Section
col_a, col_b = st.columns(2)

with col_a:
    if st.button("🚀 Forecast Current Scenario", type="primary", use_container_width=True):
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
        
        st.markdown(f"### 📊 Result for {selected_country_display}")
        if prediction > 0:
            st.success(f"## Predicted Growth: +{prediction:.2f}%")
        else:
            st.error(f"## Predicted Growth: {prediction:.2f}%")

with col_b:
    if st.button("🎯 Find Optimal Strategy", use_container_width=True):
        total_min_required = sum(min_constraints.values())
        
        if total_min_required > budget_limit:
            st.error(f"⚠️ **Infeasible Budget!** Your total minimum requirements ({total_min_required:.1f}%) exceed your Total Budget Limit ({budget_limit:.1f}%). Please lower your minimums or increase the budget.")
        else:
            best_params, best_growth = optimize_budget(actual_model_code, fdi, trade, budget_limit, min_constraints)
            st.balloons()
            st.markdown(f"### 🎯 AI Optimization: {selected_country_display}")
            st.info(f"Peak Growth Found: **{best_growth:.2f}%**")
            
            m1, m2 = st.columns(2)
            m1.metric("📚 Education", f"{best_params['Education Expenditure']:.1f}%")
            m1.metric("🏥 Health", f"{best_params['Health Expenditure']:.1f}%")
            m2.metric("🪖 Military", f"{best_params['Military Expenditure']:.1f}%")
            m2.metric("🏗️ Infrastructure", f"{best_params['Infrastructure Expenditure']:.1f}%")
            
            st.success(f"The Model successfully allocated the remaining {(budget_limit - total_min_required):.1f}% of discretionary budget.")
# Summary Table
st.markdown("---")
st.markdown("#### 📝 Current Model Inputs")
summary_df = pd.DataFrame({
    "Indicator": ["Country", "Education", "Health", "Military", "Infrastructure", "FDI", "Trade"],
    "Value": [selected_country_display, f"{edu_exp}%", f"{health_exp}%", f"{mil_exp}%", f"{infra_exp}%", f"{fdi}%", f"{trade}%"]
})
st.table(summary_df)