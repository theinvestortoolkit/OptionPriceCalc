import streamlit as st
import numpy as np
from scipy.stats import norm
import yfinance as yf
import datetime
from dateutil.relativedelta import relativedelta, FR
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CORE BLACK-SCHOLES-MERTON (BSM) FUNCTIONS (Unchanged)
# ==============================================================================

# --- BSM Price Function ---
def black_scholes_price(S, K, T, r, sigma, q=0.0, option_type='call'):
    """Calculates the Black-Scholes-Merton theoretical price."""
    try:
        T_years = T / 365.0
        if T_years <= 0 or sigma < 0:
            return 0.0
            
        if sigma < 1e-6:
            if option_type == 'call':
                price = max(0.0, S * np.exp(-q * T_years) - K * np.exp(-r * T_years))
            else:
                price = max(0.0, K * np.exp(-r * T_years) - S * np.exp(-q * T_years))
            return price
            
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
        d2 = d1 - sigma * np.sqrt(T_years)

        if option_type == 'call':
            price = S * np.exp(-q * T_years) * norm.cdf(d1) - K * np.exp(-r * T_years) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T_years) * norm.cdf(-d2) - S * np.exp(-q * T_years) * norm.cdf(-d1)
        
        return price
    except (ZeroDivisionError, ValueError):
        return 0.0

# --- BSM Greeks Function ---
def black_scholes_greeks(S, K, T, r, sigma, q=0.0, option_type='call'):
    """Calculates the Greeks and Probability ITM."""
    T_years = T / 365.0
    if T_years <= 0 or sigma <= 0:
        return {'Delta': 0.0, 'Gamma': 0.0, 'Theta': 0.0, 'Vega': 0.0, 'Rho': 0.0, 'Prob_ITM': 0.0}
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
    d2 = d1 - sigma * np.sqrt(T_years)
    Nd1_prime = norm.pdf(d1)
    gamma = (np.exp(-q * T_years) * Nd1_prime) / (S * sigma * np.sqrt(T_years))
    vega = S * np.exp(-q * T_years) * Nd1_prime * np.sqrt(T_years) / 100.0
    
    if option_type == 'call':
        delta = np.exp(-q * T_years) * norm.cdf(d1)
        theta_annual = -(S * np.exp(-q * T_years) * Nd1_prime * sigma) / (2 * np.sqrt(T_years)) \
                       - (r - q) * K * np.exp(-r * T_years) * norm.cdf(d2) \
                       + q * S * np.exp(-q * T_years) * norm.cdf(d1)
        theta = theta_annual / 365.0
        rho = K * T_years * np.exp(-r * T_years) * norm.cdf(d2) / 100.0
        prob_itm = norm.cdf(d2)
    else:  # option_type == 'put'
        delta = np.exp(-q * T_years) * (norm.cdf(d1) - 1)
        theta_annual = -(S * np.exp(-q * T_years) * Nd1_prime * sigma) / (2 * np.sqrt(T_years)) \
                       + (r - q) * K * np.exp(-r * T_years) * norm.cdf(-d2) \
                       - q * S * np.exp(-q * T_years) * norm.cdf(-d1)
        theta = theta_annual / 365.0
        rho = -K * T_years * np.exp(-r * T_years) * norm.cdf(-d2) / 100.0
        prob_itm = norm.cdf(-d2)
        
    return {
        'Delta': delta, 'Gamma': gamma, 'Theta': theta, 
        'Vega': vega, 'Rho': rho, 'Prob_ITM': prob_itm
    }

# --- Probability of Touch (POT) Function ---
def calculate_probability_of_touch(S, K, T, r, sigma, q=0.0):
    """Calculates the Probability of the Stock Price touching the Strike K before time T."""
    try:
        T_years = T / 365.0
        if T_years <= 0 or sigma <= 0:
            return 0.0
            
        theta = (r - q) / sigma**2 - 0.5
        x = np.log(S / K) / (sigma * np.sqrt(T_years))
        
        prob_touch_final = norm.cdf(x + theta * np.sqrt(T_years)) \
                           + (K / S)**(2 * theta) * norm.cdf(x - theta * np.sqrt(T_years))
                           
        return np.clip(prob_touch_final, 0.0, 1.0)
        
    except Exception:
        return 1.0 

# --- Implied Volatility Solver ---
def calculate_implied_volatility(S, K, T, r, market_price, q=0.0, option_type='call'):
    """Calculates the Implied Volatility (IV)."""
    def price_difference(sigma):
        if sigma <= 0: sigma = 1e-6 
        return black_scholes_price(S, K, T, r, sigma, q, option_type) - market_price
        
    try:
        implied_volatility = brentq(price_difference, 0.001, 5.0)
        return implied_volatility
    except Exception:
        return 0.20

# --- Implied Time Solver ---
def calculate_implied_time(S, K, sigma, r, market_price, q=0.0, option_type='call'):
    """Calculates the Implied Days to Expiration (T in days)."""
    def price_difference(T_days):
        if T_days <= 0: return market_price
        return black_scholes_price(S, K, T_days, r, sigma, q, option_type) - market_price

    try:
        implied_time = brentq(price_difference, 0.01, 1825)
        return max(0.0, implied_time)
    except Exception:
        return 365
        
# ==============================================================================
# 2. DATA FETCHING (RFR, Stock Price, Option Price for IV)
# ==============================================================================

def find_next_3rd_friday(min_days=30):
    today = datetime.date.today()
    exp_date = today.replace(day=1) + relativedelta(weekday=FR(3))
    if exp_date <= today:
        exp_date = (today.replace(day=1) + relativedelta(months=1)).replace(day=1) + relativedelta(weekday=FR(3))
    while (exp_date - today).days < min_days:
        exp_date = (exp_date.replace(day=1) + relativedelta(months=1)).replace(day=1) + relativedelta(weekday=FR(3))
    return exp_date.strftime('%Y-%m-%d'), exp_date

# --- CORE FETCHING FUNCTION (FIXED IV & DIV YIELD) ---
def get_current_data(ticker_symbol):
    """Fetches stock price, RFR, and calculates initial IV."""
    # Set fallback defaults
    default_iv_decimal = 0.20 
    rfr_percent = 4.0
    price = 100.0
    div_yield_decimal = 0.015
    
    try:
        # 1. Fetch RFR (^IRX)
        rfr_ticker = yf.Ticker('^IRX')
        rfr_info = rfr_ticker.history(period="1d", interval="1m")
        if not rfr_info.empty:
            rfr_percent = rfr_info['Close'].iloc[-1]
        
        # 2. Fetch Stock Data
        ticker = yf.Ticker(ticker_symbol)
        price_info = ticker.info
        price = price_info.get('regularMarketPrice', 100.0)
        
        # FIX: Robust Dividend Yield Fetching
        fetched_div_yield = price_info.get('dividendYield')
        if fetched_div_yield is not None:
            # Ensure it's not a tiny number or a huge percent > 100%
            if fetched_div_yield > 0.001 and fetched_div_yield < 1.0: 
                div_yield_decimal = fetched_div_yield
            elif fetched_div_yield >= 1.0 and fetched_div_yield <= 100.0:
                # If it's reported as a percentage (e.g., 2.5), convert it
                div_yield_decimal = fetched_div_yield / 100.0
        # If fetching fails or is 0, we keep the default of 0.015 (1.5%)
        
        
        # 3. Fetch Option IV directly from the chain (FIXED FOR ROBUSTNESS)
        _, default_exp_date_obj = find_next_3rd_friday(min_days=30)
        default_exp_date_str = default_exp_date_obj.strftime('%Y-%m-%d')
        
        options_data = ticker.option_chain(default_exp_date_str)
        calls = options_data.calls
        
        if not calls.empty:
            # Sort by absolute distance from current price (ATM)
            calls['Strike_Diff'] = np.abs(calls['strike'] - price)
            calls = calls.sort_values(by='Strike_Diff')
            
            # Iterate through the options closest to ATM until we find valid IV data
            for index, option in calls.iterrows():
                implied_vol = option.get('impliedVolatility')
                
                # Use IV if it's available and realistic (e.g., above 1%)
                if implied_vol is not None and implied_vol > 0.01 and implied_vol < 5.0:
                    default_iv_decimal = implied_vol
                    break # Exit loop once a valid IV is found
            
    except Exception:
        # If any fetching fails, the defaults are used
        pass 

    # Return 4 values: Price, Dividend Yield (decimal), Calculated IV (decimal), RFR (percentage)
    return float(price), div_yield_decimal, default_iv_decimal, rfr_percent

# ==============================================================================
# 3. STREAMLIT APPLICATION LAYOUT & LOGIC
# ==============================================================================

# Use a Streamlit function to memoize (cache) data fetching for speed
@st.cache_data
def get_initial_data(ticker):
    """Initial fetch for S, q, sigma, and r."""
    return get_current_data(ticker)

def main():
    st.set_page_config(layout="wide", page_title="Advanced Options Calculator")

    st.title("💰 Advanced Option Calculator")
    st.markdown("---")

    # --- Sidebar Inputs & Ticker Handling ---
    with st.sidebar:
        st.header("1. Core Inputs")
        
        ticker_symbol = st.text_input("Stock Symbol:", "SPY").upper()
        
        # Fetch data (cached) - NOTE THE 4 RETURN VALUES!
        current_price, current_yield_decimal, default_iv_decimal, current_rfr_percent = get_initial_data(ticker_symbol)
        
        # Display/Input Parameters
        S = st.number_input("Underlying Price (S):", value=round(current_price, 2), format="%.2f", disabled=True)
        K = st.number_input("Strike Price (K):", value=round(current_price, 0), format="%.2f") # Default strike is ATM
        
        # Default date calculation
        _, default_date = find_next_3rd_friday(min_days=30)
        expiration_date = st.date_input("Expiration Date:", value=default_date)
        
        # Other inputs
        # RFR NOW USES THE FETCHED VALUE:
        r_percent = st.number_input("Risk-Free Rate (%r):", value=round(current_rfr_percent, 2), format="%.2f")
        
        # IV NOW USES THE CALCULATED IV (or the 20% fallback)
        sigma_percent = st.number_input("Volatility (%Sigma):", value=round(default_iv_decimal * 100, 2), format="%.2f")
        
        # DIV YIELD NOW USES THE FETCHED VALUE:
        q_percent = st.number_input("Dividend Yield (%q):", value=round(current_yield_decimal * 100, 2), format="%.2f")
        option_type = st.radio("Option Type:", ['call', 'put'], horizontal=True)

    # --- Calculations ---
    
    # Convert inputs to decimals and days-to-expiration
    T_delta = expiration_date - datetime.date.today()
    T_days = max(1, T_delta.days)
    r_decimal = r_percent / 100.0
    sigma_decimal = sigma_percent / 100.0
    q_decimal = q_percent / 100.0

    # ---------------------------------------------
    # Main Calculation Section
    # ---------------------------------------------
    st.header("2. Main Calculation")
    
    col1, col2 = st.columns(2)
    
    # Perform main calculation
    price = black_scholes_price(S, K, T_days, r_decimal, sigma_decimal, q_decimal, option_type)
    greeks = black_scholes_greeks(S, K, T_days, r_decimal, sigma_decimal, q_decimal, option_type)
    pot = calculate_probability_of_touch(S, K, T_days, r_decimal, sigma_decimal, q_decimal)

    with col1:
        st.subheader(f"Theoretical Price ({option_type.upper()})")
        st.metric(label="Estimated Price", value=f"${price:.2f}")
        st.info(f"Days to Expiration (DTE): **{T_days}**")

    with col2:
        st.subheader("Greeks & Probabilities")
        st.markdown(f"**📈 Prob. ITM:** {greeks['Prob_ITM'] * 100.0:.2f}%")
        st.markdown(f"**✨ Prob. Touch (POT):** {pot * 100.0:.2f}%")
        st.markdown(f"**Δ Delta:** {greeks['Delta']:.4f}")
        st.markdown(f"**Γ Gamma:** {greeks['Gamma']:.4f}")
        st.markdown(f"**Θ Theta (Daily):** {greeks['Theta']:.4f}")
        st.markdown(f"**𝒱 Vega:** {greeks['Vega']:.4f}")
        st.markdown(f"**ρ Rho:** {greeks['Rho']:.4f}")
        
    st.markdown("---")
    
    # ---------------------------------------------
    # Solver/Simulator Section (Tabs for clean UI)
    # ---------------------------------------------
    
    st.header("3. Option Solvers & Simulators")
    tab1, tab2, tab3 = st.tabs(["IV Solver", "DTE Simulator", "Implied DTE Solver"])

    # --- Tab 1: IV Solver (Price -> IV) ---
    with tab1:
        st.subheader("Solve Implied Volatility (IV)")
        market_price = st.number_input("Market Price (C/P):", min_value=0.01, value=price if price > 0.01 else 0.50, format="%.2f", key='iv_price')
        if st.button("Solve IV"):
            if market_price > 0.01:
                solved_iv_decimal = calculate_implied_volatility(S, K, T_days, r_decimal, market_price, q_decimal, option_type)
                st.success(f"✅ Solved IV: **{solved_iv_decimal * 100.0:.2f}%**")
                st.write(f"*(Enter this value into the Volatility input to see new Greeks)*")
            else:
                 st.warning("Please enter a Market Price > $0.01.")
        
    # --- Tab 2: DTE Simulator (Days -> Price & Greeks) ---
    with tab2:
        st.subheader("Simulate Price at Target DTE")
        target_dte = st.number_input("Target DTE (Days):", min_value=1, value=30, step=1)
        if st.button("Simulate Price at DTE"):
            sim_price = black_scholes_price(S, K, target_dte, r_decimal, sigma_decimal, q_decimal, option_type)
            sim_greeks = black_scholes_greeks(S, K, target_dte, r_decimal, sigma_decimal, q_decimal, option_type)
            sim_pot = calculate_probability_of_touch(S, K, target_dte, r_decimal, sigma_decimal, q_decimal)
            
            sim_date = datetime.date.today() + datetime.timedelta(days=target_dte)
            
            st.success(f"💰 Simulated Price: **${sim_price:.2f}**")
            st.markdown(f"This DTE ({target_dte} days) corresponds to **{sim_date.strftime('%Y-%m-%d')}**.")
            st.markdown(f"**Prob. ITM:** {sim_greeks['Prob_ITM'] * 100.0:.2f}% | **POT:** {sim_pot * 100.0:.2f}%")
            st.markdown(f"**Theta (Daily):** {sim_greeks['Theta']:.4f}")
            
    # --- Tab 3: Implied DTE Solver (Price -> Days) ---
    with tab3:
        st.subheader("Solve Implied DTE")
        implied_dte_price = st.number_input("Price to Infer DTE:", min_value=0.01, value=price if price > 0.01 else 0.50, format="%.2f", key='dte_price')
        if st.button("Solve Implied DTE"):
            if implied_dte_price > 0.01:
                solved_T_days = int(round(calculate_implied_time(S, K, sigma_decimal, r_decimal, implied_dte_price, q_decimal, option_type)))
                
                solved_date = datetime.date.today() + datetime.timedelta(days=solved_T_days)
                st.success(f"⏳ Implied DTE: **{solved_T_days} days**")
                st.write(f"This corresponds to an expiration date of **{solved_date.strftime('%Y-%m-%d')}**.")
                st.write(f"*(Change the main Expiration Date to this value to see full Greeks/Plot.)*")
            else:
                 st.warning("Please enter a Price to Infer DTE > $0.01.")

    # ---------------------------------------------
    # 4. Plotting Section
    # ---------------------------------------------
    st.markdown("---")
    st.header("4. Theoretical Price vs. Underlying")
    
    try:
        S_min = max(0, S * 0.8)
        S_max = S * 1.2
        S_range = np.linspace(S_min, S_max, 100)
        
        prices_over_range = [
            black_scholes_price(s, K, T_days, r_decimal, sigma_decimal, q_decimal, option_type)
            for s in S_range
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(S_range, prices_over_range, label=f'Theoretical Price ({option_type.upper()})', color='darkgreen', linewidth=3)
        ax.axvline(K, color='blue', linestyle='-.', alpha=0.6, label='Strike Price (K)')
        ax.axvline(S, color='red', linestyle='--', alpha=0.6, label='Current Stock Price')
        ax.plot(S, price, 'o', color='red', markersize=8, label=f'Price: ${price:.2f}')
        ax.set_title(f'{option_type.upper()} Price vs. Underlying Price (IV: {sigma_percent:.2f}%)')
        ax.set_xlabel('Underlying Price (S)')
        ax.set_ylabel('Option Theoretical Value ($)')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"An error occurred during plotting: {e}")
    
    st.markdown("---") 

if __name__ == "__main__":
    main()
