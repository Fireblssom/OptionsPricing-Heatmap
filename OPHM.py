import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm

# Initialize portfolio list (now this holds the stock data as well)
portfolio = []

# Helper functions for Greeks and pricing
def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T == 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def compute_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)) if T > 0 else 0
    d2 = d1 - sigma * np.sqrt(T) if T > 0 else 0
    greeks = {}
    
    if option_type == "call":
        greeks["Delta"] = norm.cdf(d1)
        greeks["Gamma"] = norm.pdf(d1) / (S * sigma * np.sqrt(T)) if T > 0 else 0
        greeks["Theta"] = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        greeks["Vega"] = S * norm.pdf(d1) * np.sqrt(T)
        greeks["Rho"] = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put option
        greeks["Delta"] = norm.cdf(d1) - 1
        greeks["Gamma"] = norm.pdf(d1) / (S * sigma * np.sqrt(T)) if T > 0 else 0
        greeks["Theta"] = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        greeks["Vega"] = S * norm.pdf(d1) * np.sqrt(T)
        greeks["Rho"] = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return greeks

# Sensitivity Analysis: Calculate how Greeks change with strike and maturity
def sensitivity_analysis(S, r, sigma, option_type, strike_range, time_range):
    K_vals = np.linspace(*strike_range, 30)
    T_vals = np.linspace(*time_range, 30)
    
    delta_matrix = np.zeros((len(T_vals), len(K_vals)))
    gamma_matrix = np.zeros((len(T_vals), len(K_vals)))
    vega_matrix = np.zeros((len(T_vals), len(K_vals)))
    
    for i, T in enumerate(T_vals):
        for j, K in enumerate(K_vals):
            greeks = compute_greeks(S, K, T, r, sigma, option_type)
            delta_matrix[i, j] = greeks["Delta"]
            gamma_matrix[i, j] = greeks["Gamma"]
            vega_matrix[i, j] = greeks["Vega"]
    
    return K_vals, T_vals, delta_matrix, gamma_matrix, vega_matrix

# Portfolio Risk Metrics: VaR, CVaR, and Monte Carlo Simulation
def portfolio_risk_metrics(options, S, r, sigma, option_type="call"):
    total_delta = sum([option["quantity"] * compute_greeks(S, option["strike"], option["maturity"], r, sigma, option_type)["Delta"] for option in options])
    total_gamma = sum([option["quantity"] * compute_greeks(S, option["strike"], option["maturity"], r, sigma, option_type)["Gamma"] for option in options])
    
    # VaR & CVaR (simplified)
    portfolio_value = sum([option["quantity"] * black_scholes(S, option["strike"], option["maturity"], r, sigma, option_type) for option in options])
    var = np.percentile(np.random.normal(portfolio_value, portfolio_value * 0.05, 10000), 5)
    cvar = np.mean([x for x in np.random.normal(portfolio_value, portfolio_value * 0.05, 10000) if x <= var])
    
    # Monte Carlo Simulation for risk
    mc_simulations = np.random.normal(portfolio_value, portfolio_value * 0.05, 10000)
    monte_carlo_risk = np.percentile(mc_simulations, 5)
    
    return total_delta, total_gamma, var, cvar, monte_carlo_risk

# Streamlit UI for Sensitivity Analysis
st.title("Options Sensitivity Analysis & Portfolio Risk Metrics")

# Stock Selection
with st.sidebar.expander("Stock Data Inputs", expanded=True):
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", value="AAPL", help="Enter the ticker symbol of the stock you want to analyze.")
    stock_data = yf.Ticker(stock_symbol)
    stock_info = stock_data.history(period="1d")
    
    if stock_info.empty:
        st.error("Invalid stock symbol or data not available.")
    else:
        S = stock_info["Close"].iloc[-1]
        st.write(f"Real-Time Stock Price for {stock_symbol}: ${S:.2f}")
        
        # Add stock to portfolio functionality
        if st.button("Add to Portfolio"):
            # Store the selected stock in the portfolio
            portfolio.append({
                "stock_symbol": stock_symbol,
                "stock_price": S,
                "quantity": 1,  # Default quantity for simplicity
                "strike": S,  # Default strike is the current stock price
                "maturity": 1.0  # Default maturity is 1 year
            })
            st.success(f"{stock_symbol} has been added to your portfolio.")

# Option Inputs
with st.sidebar.expander("Option Inputs", expanded=True):
    r = st.number_input("Risk-Free Rate (r)", value=0.01, help="The rate of return on a risk-free investment (e.g., government bonds).")
    sigma = st.number_input("Volatility (σ)", value=0.2, help="The measure of the asset's price fluctuations over time.")
    option_type = st.selectbox("Option Type", ["call", "put"], help="Select whether the option is a 'call' or a 'put'.")

# Sensitivity Analysis Inputs
with st.sidebar.expander("Sensitivity Analysis Inputs", expanded=True):
    strike_range = st.slider("Strike Price Range (K)", 50, 150, (80, 120), help="The range of strike prices for the options.")
    time_range = st.slider("Maturity Range (T, years)", 0.01, 2.0, (0.1, 1.0), help="The range of time-to-maturity (in years) for the options.")

# Display Portfolio & Option Settings
with st.expander("Your Portfolio", expanded=True):
    st.write("### Portfolio Overview")
    for idx, item in enumerate(portfolio):
        st.write(f"**{item['stock_symbol']}** - Price: ${item['stock_price']:.2f}")
        strike = st.number_input(f"Strike Price for {item['stock_symbol']}", value=item['strike'], key=f"strike_{idx}")
        maturity = st.number_input(f"Maturity for {item['stock_symbol']} (years)", value=item['maturity'], key=f"maturity_{idx}")
        quantity = st.number_input(f"Quantity for {item['stock_symbol']}", value=item['quantity'], key=f"quantity_{idx}")
        
        # Update portfolio with new values
        portfolio[idx]["strike"] = strike
        portfolio[idx]["maturity"] = maturity
        portfolio[idx]["quantity"] = quantity

# Sensitivity Analysis Calculation
K_vals, T_vals, delta_matrix, gamma_matrix, vega_matrix = sensitivity_analysis(S, r, sigma, option_type, strike_range, time_range)

# Display Sensitivity Heatmaps
with st.expander("Greeks Sensitivity Heatmap", expanded=True):
    fig = go.Figure(data=go.Heatmap(z=delta_matrix, x=np.round(K_vals, 2), y=np.round(T_vals, 2), colorscale="YlGnBu", colorbar=dict(title="Delta")))
    fig.update_layout(title="Delta Sensitivity Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure(data=go.Heatmap(z=gamma_matrix, x=np.round(K_vals, 2), y=np.round(T_vals, 2), colorscale="YlGnBu", colorbar=dict(title="Gamma")))
    fig.update_layout(title="Gamma Sensitivity Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure(data=go.Heatmap(z=vega_matrix, x=np.round(K_vals, 2), y=np.round(T_vals, 2), colorscale="YlGnBu", colorbar=dict(title="Vega")))
    fig.update_layout(title="Vega Sensitivity Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# Portfolio Risk Calculation
total_delta, total_gamma, var, cvar, monte_carlo_risk = portfolio_risk_metrics(portfolio, S, r, sigma, option_type)

# Display Portfolio Risk Metrics
with st.expander("Portfolio Risk Metrics"):
    st.write(f"Portfolio Delta: {total_delta}")
    st.write(f"Portfolio Gamma: {total_gamma}")
    st.write(f"Value-at-Risk (VaR): {var}")
    st.write(f"Conditional VaR (CVaR): {cvar}")
    st.write(f"Monte Carlo Risk: {monte_carlo_risk}")
