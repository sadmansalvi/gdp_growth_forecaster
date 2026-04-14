import streamlit as st
import numpy as np
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Prediction of GDP Growth Rate & Budget Optimization",
    layout="wide"
)

# --- STATISTICAL ERROR METRIC ---
# Update this with your exact Test RMSE from Colab if it differs
MODEL_RMSE = 2.1206

# Initialize Session State Variables
if 'manual_result' not in st.session_state:
    st.session_state['manual_result'] = None
if 'optimal_result' not in st.session_state:
    st.session_state['optimal_result'] = None
if 'optimal_error' not in st.session_state:
    st.session_state['optimal_error'] = None

# 2. Optimized Model Loader
@st.cache_resource
def load_model():
    return joblib.load('gdp_xgboost_model.pkl')

model = load_model()

# 3. Helper Function: AI Policy Optimizer
def optimize_budget(target_country_code, fdi_val, trade_val, total_budget_pct, min_dict):
    np.random.seed(42)

    # High density convergence
    n_scenarios = 50000

    total_min_pct = sum(min_dict.values())

    if abs(total_min_pct - total_budget_pct) < 1e-5:
        shares = np.array([[min_dict['edu'], min_dict['health'], min_dict['mil'], min_dict['infra']]])
    else:
        remaining_budget_pct = total_budget_pct - total_min_pct
        raw_shares = np.random.dirichlet(np.ones(4), size=n_scenarios)

        edu_s = (raw_shares[:, 0] * remaining_budget_pct) + min_dict['edu']
        health_s = (raw_shares[:, 1] * remaining_budget_pct) + min_dict['health']
        mil_s = (raw_shares[:, 2] * remaining_budget_pct) + min_dict['mil']
        infra_s = (raw_shares[:, 3] * remaining_budget_pct) + min_dict['infra']

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


# 4. App UI Header
st.title("Forecast GDP Growth Rate of Countries")
st.markdown("### Prediction of GDP Growth Rate of Different Countries and Optimization of Fiscal Budgets Using Machine Learning Models")
st.info(
    "**Disclaimer:** This tool utilizes an XGBoost machine learning model trained exclusively on historical macroeconomic data from 16 specific nations across three economic blocs: the **G7**, **BRICS**, and **South Asia**. "
    "Predictions are probabilistic forecasts, not absolute certainties. Outcomes may deviate from real-world economic conditions due to "
    "unforeseen geopolitical, environmental, or systemic market shifts. Designed strictly for academic simulation and analytical purposes."
)
st.markdown("---")

# 5. Sidebar - Control Panel
st.sidebar.header("Configuration Panel")

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

selected_bloc = st.sidebar.selectbox("Select Economic Bloc", list(ui_groups.keys()))
selected_country_display = st.sidebar.selectbox("Select Target Country", ui_groups[selected_bloc])
actual_model_code = country_translation_map[selected_country_display]

# --- DYNAMIC MACROECONOMIC BOUNDS ---
macro_bounds = {
    "United States": {"fdi": (-2.0, 6.0, 2.0), "trade": (20.0, 35.0, 25.0), "edu": (4.0, 7.0, 5.0), "health": (12.0, 18.0, 15.0), "mil": (3.0, 6.0, 3.5), "infra": (1.0, 4.0, 2.0)},
    "Germany": {"fdi": (-3.0, 10.0, 2.0), "trade": (50.0, 100.0, 80.0), "edu": (4.0, 6.0, 5.0), "health": (9.0, 13.0, 11.0), "mil": (1.0, 2.5, 1.5), "infra": (1.0, 4.0, 2.0)},
    "Japan": {"fdi": (-1.0, 3.0, 1.0), "trade": (20.0, 45.0, 35.0), "edu": (3.0, 5.0, 4.0), "health": (7.0, 12.0, 10.0), "mil": (0.5, 1.5, 1.0), "infra": (3.0, 7.0, 5.0)},
    "United Kingdom": {"fdi": (-5.0, 15.0, 3.0), "trade": (50.0, 75.0, 60.0), "edu": (4.0, 7.0, 5.5), "health": (8.0, 13.0, 10.0), "mil": (1.5, 3.0, 2.0), "infra": (1.0, 4.0, 2.0)},
    "France": {"fdi": (-2.0, 6.0, 2.0), "trade": (40.0, 70.0, 60.0), "edu": (4.0, 6.5, 5.5), "health": (9.0, 13.0, 11.0), "mil": (1.5, 3.0, 2.0), "infra": (1.0, 4.0, 2.0)},
    "Italy": {"fdi": (-2.0, 5.0, 1.5), "trade": (40.0, 70.0, 60.0), "edu": (3.0, 5.5, 4.0), "health": (7.0, 10.0, 9.0), "mil": (1.0, 2.5, 1.5), "infra": (1.0, 4.0, 2.0)},
    "Canada": {"fdi": (-2.0, 8.0, 3.0), "trade": (50.0, 80.0, 65.0), "edu": (4.0, 7.0, 5.5), "health": (9.0, 13.0, 11.0), "mil": (1.0, 2.5, 1.5), "infra": (1.0, 4.0, 2.0)},
    "Brazil": {"fdi": (0.0, 6.0, 3.0), "trade": (15.0, 40.0, 25.0), "edu": (4.0, 7.0, 5.5), "health": (7.0, 11.0, 9.0), "mil": (1.0, 2.5, 1.5), "infra": (1.0, 5.0, 2.5)},
    "Russia": {"fdi": (-3.0, 5.0, 1.0), "trade": (40.0, 70.0, 50.0), "edu": (3.0, 5.5, 4.0), "health": (4.0, 8.0, 5.5), "mil": (2.5, 6.0, 4.0), "infra": (1.0, 5.0, 2.0)},
    "India": {"fdi": (0.0, 4.0, 1.5), "trade": (20.0, 55.0, 40.0), "edu": (2.5, 4.5, 3.5), "health": (1.0, 4.0, 2.5), "mil": (2.0, 4.0, 2.5), "infra": (2.0, 6.0, 4.0)},
    "China": {"fdi": (0.0, 6.0, 2.5), "trade": (30.0, 75.0, 40.0), "edu": (3.0, 5.0, 4.0), "health": (3.0, 7.0, 5.0), "mil": (1.0, 3.0, 1.5), "infra": (4.0, 10.0, 6.0)},
    "South Africa": {"fdi": (-2.0, 6.0, 1.5), "trade": (40.0, 75.0, 60.0), "edu": (5.0, 8.0, 6.5), "health": (7.0, 10.0, 8.5), "mil": (0.5, 2.0, 1.0), "infra": (1.0, 5.0, 2.5)},
    "Bangladesh": {"fdi": (0.0, 3.0, 1.0), "trade": (25.0, 50.0, 35.0), "edu": (1.0, 3.0, 2.0), "health": (1.0, 3.5, 2.5), "mil": (1.0, 2.0, 1.5), "infra": (1.0, 6.0, 3.0)},
    "Pakistan": {"fdi": (0.0, 3.0, 1.0), "trade": (25.0, 40.0, 30.0), "edu": (1.5, 3.5, 2.5), "health": (1.0, 4.0, 2.5), "mil": (2.5, 5.0, 3.5), "infra": (1.0, 5.0, 2.5)},
    "Sri Lanka": {"fdi": (0.0, 3.0, 1.0), "trade": (40.0, 90.0, 50.0), "edu": (1.0, 3.5, 2.0), "health": (2.0, 5.0, 3.5), "mil": (1.5, 4.0, 2.0), "infra": (2.0, 7.0, 4.0)},
    "Nepal": {"fdi": (0.0, 1.5, 0.5), "trade": (30.0, 65.0, 45.0), "edu": (3.0, 6.0, 4.5), "health": (3.0, 6.5, 4.5), "mil": (1.0, 2.5, 1.5), "infra": (2.0, 8.0, 4.0)},
}

b = macro_bounds[selected_country_display]

st.sidebar.markdown("---")
st.sidebar.subheader("Economic & Trade Parameters")
current_gdp = st.sidebar.number_input("Target Country GDP (Billions USD)", min_value=1.0, value=500.0, step=10.0)
st.sidebar.caption("Ranges strictly bounded by historical data.")
fdi = st.sidebar.slider("Foreign Direct Investment (% GDP)", float(b['fdi'][0]), float(b['fdi'][1]), float(b['fdi'][2]), 0.1)
trade = st.sidebar.slider("Trade Openness (% GDP)", float(b['trade'][0]), float(b['trade'][1]), float(b['trade'][2]), 1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Manual Policy Inputs")
edu_exp = st.sidebar.slider("Education (% GDP)", float(b['edu'][0]), float(b['edu'][1]), float(b['edu'][2]), 0.1)
health_exp = st.sidebar.slider("Health (% GDP)", float(b['health'][0]), float(b['health'][1]), float(b['health'][2]), 0.1)
mil_exp = st.sidebar.slider("Military (% GDP)", float(b['mil'][0]), float(b['mil'][1]), float(b['mil'][2]), 0.1)
infra_exp = st.sidebar.slider("Infrastructure (% GDP)", float(b['infra'][0]), float(b['infra'][1]), float(b['infra'][2]), 0.1)

total_manual_pct = edu_exp + health_exp + mil_exp + infra_exp

st.sidebar.markdown("---")
st.sidebar.subheader("Optimization Constraints")
default_budget = current_gdp * (total_manual_pct / 100) if total_manual_pct > 0 else current_gdp * 0.15
budget_usd = st.sidebar.number_input("Total Discretionary Budget (Billions USD)", min_value=0.1,
                                     max_value=float(current_gdp), value=default_budget, step=5.0)
budget_limit_pct = (budget_usd / current_gdp) * 100
st.sidebar.caption(f"Calculated Constraint: {budget_limit_pct:.2f}% of GDP")

if abs(total_manual_pct - budget_limit_pct) > 0.01:
    st.sidebar.warning(
        f"⚠️ **Budget Mismatch:** Your Manual Policy totals **{total_manual_pct:.2f}%** of GDP, "
        f"but your Optimizer is capped at **{budget_limit_pct:.2f}%**. "
        "For a fair mathematical comparison, these should be equal."
    )

st.sidebar.write("Minimum Sectoral Requirements (%):")
min_edu = st.sidebar.slider("Minimum Education", 0.0, float(b['edu'][1]), float(b['edu'][0]), 0.5)
min_health = st.sidebar.slider("Minimum Health", 0.0, float(b['health'][1]), float(b['health'][0]), 0.5)
min_mil = st.sidebar.slider("Minimum Military", 0.0, float(b['mil'][1]), float(b['mil'][0]), 0.5)
min_infra = st.sidebar.slider("Minimum Infrastructure", 0.0, float(b['infra'][1]), float(b['infra'][0]), 0.5)
min_constraints = {'edu': min_edu, 'health': min_health, 'mil': min_mil, 'infra': min_infra}

st.sidebar.markdown("---")
st.sidebar.subheader("👨‍🎓 About the Developer")
st.sidebar.info(
    "**Salim Sadman Salvi** \n"
    "B.Sc. in Pure Mathematics (Final Year)  \n"
    "Jagannath University, Dhaka  \n\n"
    "*Final Year Project: Prediction of GDP Growth Rate of Different Countries and Optimization of Fiscal Budgets Using Machine Learning Models*"
    "\n"
    "🔗 [LinkedIn](https://www.linkedin.com/in/salim-sadman-salvi-345241276/) ｜ 💻 [GitHub](https://github.com/sadmansalvi) ｜ ✉️ [Email](b200302063@math.jnu.ac.bd)"
)

# 6. Action Buttons
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("Execute Baseline Forecast", use_container_width=True):
        input_data = pd.DataFrame({
            'Education Expenditure': [edu_exp], 'Health Expenditure': [health_exp],
            'Military Expenditure': [mil_exp], 'Infrastructure Expenditure': [infra_exp],
            'Foreign Direct Investment': [fdi], 'Trade': [trade], 'country': [actual_model_code]
        })
        pred = model.predict(input_data)[0]
        total_manual_usd = (total_manual_pct / 100) * current_gdp

        st.session_state['manual_result'] = {
            'pred': pred, 'total_pct': total_manual_pct, 'total_usd': total_manual_usd,
            'edu': edu_exp, 'health': health_exp, 'mil': mil_exp, 'infra': infra_exp,
            'country': selected_country_display, 'fdi': fdi, 'trade': trade
        }

with col_btn2:
    if st.button("Execute Policy Optimization", use_container_width=True):
        total_min_required_pct = sum(min_constraints.values())

        if total_min_required_pct > budget_limit_pct:
            st.session_state['optimal_error'] = f"Infeasible Constraint: Total minimums ({total_min_required_pct:.2f}%) exceed the maximum budget limit ({budget_limit_pct:.2f}%)."
            st.session_state['optimal_result'] = None
        else:
            best_params, best_growth = optimize_budget(actual_model_code, fdi, trade, budget_limit_pct, min_constraints)

            st.session_state['optimal_result'] = {
                'pred': best_growth, 'total_pct': budget_limit_pct, 'total_usd': budget_usd,
                'edu': best_params['Education Expenditure'], 'health': best_params['Health Expenditure'],
                'mil': best_params['Military Expenditure'], 'infra': best_params['Infrastructure Expenditure'],
                'country': selected_country_display, 'fdi': fdi, 'trade': trade
            }
            st.session_state['optimal_error'] = None

# 7. Display Results
st.markdown("---")

if st.session_state['optimal_error']:
    st.error(st.session_state['optimal_error'])

col_res1, col_res2 = st.columns(2)

with col_res1:
    st.subheader("Baseline Policy Analysis")
    if st.session_state['manual_result']:
        res = st.session_state['manual_result']

        st.markdown(
            f"**Scenario Overview:** For the economy of **{res['country']}**, this baseline projection assumes a "
            f"global trade openness of **{res['trade']}%** and foreign direct investment at **{res['fdi']}%** of GDP. "
            f"The policy manually allocates **${res['total_usd']:,.1f} Billion** ({res['total_pct']:.2f}% of GDP) across the four core sectors."
        )
        st.caption(f"**Statistical Error Margin (RMSE):** ±{MODEL_RMSE}%")

        if res['pred'] >= 0:
            st.success(f"📈 Projected GDP Growth: {res['pred']:.2f}%")
        else:
            st.error(f"📉 Projected GDP Contraction (Recession): {res['pred']:.2f}%")

        m1, m2 = st.columns(2)
        m1.metric("Education", f"{res['edu']:.2f}%", f"${(res['edu'] / 100) * current_gdp:,.1f} B")
        m1.metric("Health", f"{res['health']:.2f}%", f"${(res['health'] / 100) * current_gdp:,.1f} B")
        m2.metric("Military", f"{res['mil']:.2f}%", f"${(res['mil'] / 100) * current_gdp:,.1f} B")
        m2.metric("Infrastructure", f"{res['infra']:.2f}%", f"${(res['infra'] / 100) * current_gdp:,.1f} B")
    else:
        st.write("Awaiting execution...")

with col_res2:
    st.subheader("Optimized Policy Analysis")
    if st.session_state['optimal_result']:
        opt = st.session_state['optimal_result']

        st.markdown(
            f"**Optimization Overview:** For the economy of **{opt['country']}**, the stochastic optimizer evaluated 50,000 "
            f"budget permutations constrained to a strict maximum of **${opt['total_usd']:,.1f} Billion** ({opt['total_pct']:.2f}% of GDP). "
            f"Assuming a trade openness of **{opt['trade']}%** and FDI at **{opt['fdi']}%**, the algorithm identified the following peak allocation."
        )
        st.caption(f"**Statistical Error Margin (RMSE):** ±{MODEL_RMSE}%")

        if opt['pred'] >= 0:
            st.success(f"📈 Optimized GDP Growth: {opt['pred']:.2f}%")
        else:
            st.error(f"📉 Optimized GDP Contraction (Recession): {opt['pred']:.2f}%")

        o1, o2 = st.columns(2)
        o1.metric("Education", f"{opt['edu']:.2f}%", f"${(opt['edu'] / 100) * current_gdp:,.1f} B")
        o1.metric("Health", f"{opt['health']:.2f}%", f"${(opt['health'] / 100) * current_gdp:,.1f} B")
        o2.metric("Military", f"{opt['mil']:.2f}%", f"${(opt['mil'] / 100) * current_gdp:,.1f} B")
        o2.metric("Infrastructure", f"{opt['infra']:.2f}%", f"${(opt['infra'] / 100) * current_gdp:,.1f} B")
    else:
        st.write("Awaiting execution...")