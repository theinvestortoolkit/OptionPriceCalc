import streamlit as st
import numpy as np
from scipy.stats import norm
import yfinance as yf
import datetime
from dateutil.relativedelta import relativedelta, FR
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CORE BLACK-SCHOLES-MERTON (BSM) FUNCTIONS
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

# --- Implied Volatility Solver (Unchanged, but now used directly in sidebar) ---
def calculate_implied_volatility(S, K, T, r, market_price, q=0.0, option_type='call'):
    """Calculates the Implied Volatility (IV)."""
    def price_difference(sigma):
        if sigma <= 0: sigma = 1e-6 
        return black_scholes_price(S, K, T, r, sigma, q, option_type) - market_price
        
    try:
        implied_volatility = brentq(price_difference, 0.001, 5.0)
        return implied_volatility
    except Exception:
        return 0.20 # Fallback
    
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

# --- Fetch ALL Market Data for the specified Ticker, Strike, Expiry, and Type ---
def fetch_all_market_data(ticker_symbol, strike, expiration_date_str, option_type):
    """Fetches S, RFR, Q, and the specific option's IV and Price."""
    
    # Fallback defaults
    data = {
        'S': 100.0,
        'r_percent': 4.0,
        'q_percent': 1.5,
        'sigma_decimal': 0.20,
        'market_price': 0.50,
        'error': False
    }
    
    try:
        # 1. Fetch RFR (^IRX)
        rfr_ticker = yf.Ticker('^IRX')
        rfr_info = rfr_ticker.history(period="1d", interval="1m")
        if not rfr_info.empty:
            data['r_percent'] = rfr_info['Close'].iloc[-1]
        
        # 2. Fetch Stock Data (S and Q)
        ticker = yf.Ticker(ticker_symbol)
        price_info = ticker.info
        data['S'] = price_info.get('regularMarketPrice', 100.0)
        
        fetched_div_yield = price_info.get('dividendYield')
        if fetched_div_yield is not None:
            if 0.001 < fetched_div_yield < 1.0: 
                data['q_percent'] = fetched_div_yield * 100.0
            elif 1.0 <= fetched_div_yield <= 100.0:
                data['q_percent'] = fetched_div_yield
        
        # 3. Fetch Option IV and Price
        options_data = ticker.option_chain(expiration_date_str)
        
        contracts = options_data.calls if option_type == 'call' else options_data.puts

        if not contracts.empty:
            # Find the row that matches the strike price exactly
            target_contract = contracts[contracts['strike'] == strike]

            if not target_contract.empty:
                option = target_contract.iloc[0]
                
                # Fetch IV
                implied_vol = option.get('impliedVolatility')
                if implied_vol is not None and implied_vol > 0.01 and implied_vol < 5.0:
                    data['sigma_decimal'] = implied_vol
                    
                # Fetch Price
                bid = option.get('bid')
                ask = option.get('ask')
                
                if bid is not None and ask is not None and bid > 0 and ask > 0:
                    data['market_price'] = (bid + ask) / 2.0
            else:
                 data['error'] = f"Option K=${strike} not found for {expiration_date_str}."
        else:
             data['error'] = f"No option chain found for {expiration_date_str}."
            
    except Exception as e:
        data['error'] = f"Data fetch error: {e}" 

    return data

# ==============================================================================
# 3. STREAMLIT APPLICATION LAYOUT & LOGIC
# ==============================================================================

# Custom function to handle initialization and fetching for the session
def init_or_fetch_data(ticker_symbol, strike, expiration_date_str, option_type, refresh=False):
    if 'data_state' not in st.session_state or refresh:
        # If initializing (first run) or explicitly refreshing
        fetched_data = fetch_all_market_data(ticker_symbol, strike, expiration_date_str, option_type)
        st.session_state.data_state = {
            'S': round(fetched_data['S'], 2),
            'r_percent': round(fetched_data['r_percent'], 2),
            'q_percent': round(fetched_data['q_percent'], 2),
            'sigma_percent': round(fetched_data['sigma_decimal'] * 100, 2),
            'market_price': round(fetched_data['market_price'], 2),
            'error': fetched_data['error']
        }
        # Initialize the dynamic key for Volatility
        if 'sigma_key_version' not in st.session_state:
             st.session_state.sigma_key_version = 0
    return st.session_state.data_state

def main():
    st.set_page_config(layout="wide", page_title="Advanced Options Calculator")

    st.title("💰 Advanced Option Calculator")
    st.markdown("---")
    
    default_ticker = "SPY"
    _, default_date = find_next_3rd_friday(min_days=30)
    default_strike = 450.0 

    # --- Sidebar Inputs & Ticker Handling ---
    with st.sidebar:
        st.header("1. Core Inputs")
        
        ticker_symbol = st.text_input("Stock Symbol:", default_ticker).upper()
        option_type = st.radio("Option Type:", ['call', 'put'], horizontal=True)
        K = st.number_input("Strike Price (K):", value=default_strike, format="%.2f", key='k_input') 
        expiration_date = st.date_input("Expiration Date:", value=default_date, key='exp_date_input')
        expiration_date_str = expiration_date.strftime('%Y-%m-%d')

        # 1A. Get the current state (This fetches initial data if session_state is empty)
        current_data = init_or_fetch_data(ticker_symbol, K, expiration_date_str, option_type, refresh=False)
        
        st.markdown("---")
        st.header("2. Input Parameters")

        # Core BSM Inputs (Pulling from session state)
        S = st.number_input("Underlying Price (S):", value=current_data['S'], format="%.2f", key='s_input')
        r_percent = st.number_input("Risk-free Rate (%r):", value=current_data['r_percent'], format="%.2f", key='r_input')
        
        # FIX: Dynamic key for the Volatility input
        sigma_percent = st.number_input(
            "Volatility (%Sigma):", 
            value=current_data['sigma_percent'], 
            format="%.2f", 
            key=f"sigma_input_{st.session_state.sigma_key_version}" # Key changes on update
        )
        
        q_percent = st.number_input("Dividend Yield (%q):", value=current_data['q_percent'], format="%.2f", key='q_input')
        
        # --- UPDATE BUTTON ---
        if st.button("🔄 Update Input Parameter Data"):
            new_data = fetch_all_market_data(ticker_symbol, K, expiration_date_str, option_type)
            
            # Update the session state with the new fetched data
            st.session_state.data_state = {
                'S': round(new_data['S'], 2),
                'r_percent': round(new_data['r_percent'], 2),
                'q_percent': round(new_data['q_percent'], 2),
                'sigma_percent': round(new_data['sigma_decimal'] * 100, 2),
                'market_price': round(new_data['market_price'], 2),
                'error': new_data['error']
            }
            
            # CRITICAL FIX: Increment the version number to force Streamlit to redraw the sigma input
            st.session_state.sigma_key_version += 1 

            if st.session_state.data_state['error']:
                 st.error(f"Error: {st.session_state.data_state['error']}")
            st.rerun() 

        st.markdown("---")
        st.header("3. Current Market Price")
        
        # Display the fetched market price (Read-only)
        st.metric(
            label=f"Current Market Price ({option_type.upper()})",
            value=f"${current_data['market_price']:.2f}"
        )
        
    # --- Calculations ---
    
    # Convert inputs to decimals and days-to-expiration (always use the values from the input fields)
    T_delta = expiration_date - datetime.date.today()
    T_days = max(1, T_delta.days)
    r_decimal = r_percent / 100.0
    q_decimal = q_percent / 100.0
    sigma_decimal_input = sigma_percent / 100.0 # Use user's input sigma

    # Market price from session state for IV calculation and solver section
    market_price_for_solve = current_data['market_price']

    # Calculate the Implied Volatility based on the fetched Market Price
    solved_iv_decimal = calculate_implied_volatility(
        S, K, T_days, r_decimal, market_price_for_solve, q_decimal, option_type
    )
    
    # Perform the main calculation using the user's input sigma
    price = black_scholes_price(S, K, T_days, r_decimal, sigma_decimal_input, q_decimal, option_type)

    # ---------------------------------------------
    # Main Calculation Section
    ---------------------------------------------
    st.header("2. Main Calculation")
    
    col1, col2 = st.columns(2)
    
    # Perform main calculation (price is already calculated)
    greeks = black_scholes_greeks(S, K, T_days, r_decimal, sigma_decimal_input, q_decimal, option_type)
    pot = calculate_probability_of_touch(S, K, T_days, r_decimal, sigma_decimal_input, q_decimal)

    with col1:
        st.subheader(f"Theoretical Price ({option_type.upper()})")
        # Now explicitly stating that this price uses the user's input Volatility
        st.metric(label="Estimated Price (using Input Volatility)", value=f"${price:.2f}")
        
        # Display the IV solved from the actual market price
        st.metric(
            label="Implied Volatility (Solved from Current Market Price)", 
            value=f"{solved_iv_decimal * 100.0:.2f}%"
        )
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
    ---------------------------------------------
    
    st.header("3. Option Solvers & Simulators")
    tab2, tab3 = st.tabs(["DTE Simulator", "Implied DTE Solver"])

    # --- Tab 2: DTE Simulator (Days -> Price & Greeks) ---
    with tab2:
        st.subheader("Simulate Price at Target DTE")
        target_dte = st.number_input("Target DTE (Days):", min_value=1, value=30, step=1)
        if st.button("Simulate Price at DTE"):
            sim_price = black_scholes_price(S, K, target_dte, r_decimal, sigma_decimal_input, q_decimal, option_type)
            sim_greeks = black_scholes_greeks(S, K, target_dte, r_decimal, sigma_decimal_input, q_decimal, option_type)
            sim_pot = calculate_probability_of_touch(S, K, target_dte, r_decimal, sigma_decimal_input, q_decimal)
            
            sim_date = datetime.date.today() + datetime.timedelta(days=target_dte)
            
            st.success(f"💰 Simulated Price: **${sim_price:.2f}**")
            st.markdown(f"This DTE ({target_dte} days) corresponds to **{sim_date.strftime('%Y-%m-%d')}**.")
            st.markdown(f"**Prob. ITM:** {sim_greeks['Prob_ITM'] * 100.0:.2f}% | **POT:** {sim_pot * 100.0:.2f}%")
            st.markdown(f"**Θ Theta (Daily):** {sim_greeks['Theta']:.4f}")
            
    # --- Tab 3: Implied DTE Solver (Price -> Days) ---
    with tab3:
        st.subheader("Solve Implied DTE")
        # Default value for DTE solver is the fetched market price
        implied_dte_price = st.number_input("Price to Infer DTE:", min_value=0.01, value=market_price_for_solve if market_price_for_solve > 0.01 else 0.50, format="%.2f", key='dte_price')
        if st.button("Solve Implied DTE"):
            if implied_dte_price > 0.01:
                # Use the Solved IV for a realistic implied DTE
                solved_T_days = int(round(calculate_implied_time(S, K, solved_iv_decimal, r_decimal, implied_dte_price, q_decimal, option_type)))
                
                solved_date = datetime.date.today() + datetime.timedelta(days=solved_T_days)
                st.success(f"⏳ Implied DTE: **{solved_T_days} days**")
                st.write(f"This corresponds to an expiration date of **{solved_date.strftime('%Y-%m-%d')}**.")
            else:
                 st.warning("Please enter a Price to Infer DTE > $0.01.")

    # ---------------------------------------------
    # 4. Plotting Section
    ---------------------------------------------
    st.markdown("---")
    st.header("4. Theoretical Price vs. Underlying")
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6)) 
        
        S_min = max(0, S * 0.8)
        S_max = S * 1.2
        S_range = np.linspace(S_min, S_max, 100)
        
        prices_over_range = [
            black_scholes_price(s, K, T_days, r_decimal, sigma_decimal_input, q_decimal, option_type)
            for s in S_range
        ]
        
        ax.plot(S_range, prices_over_range, label=f'Theoretical Price ({option_type.upper()})', color='darkgreen', linewidth=3)
        ax.axvline(K, color='blue', linestyle='-.', alpha=0.6, label='Strike Price (K)')
        ax.axvline(S, color='red', linestyle='--', alpha=0.6, label='Current Stock Price')
        ax.plot(S, price, 'o', color='red', markersize=8, label=f'Price: ${price:.2f}')
        ax.set_title(f'{option_type.upper()} Price vs. Underlying Price (Input IV: {sigma_percent:.2f}%)')
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
