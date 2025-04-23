import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm

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

# Portfolio Risk Metrics
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

# Initialize session state for portfolio
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

# Streamlit UI for Sensitivity Analysis
st.title("Options Sensitivity Analysis & Portfolio Risk Metrics")

# Stock Selection and Add to Portfolio functionality
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
            # Store the selected stock in the session state portfolio
            st.session_state.portfolio.append({
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

# Display Portfolio & Option Settings for Each Stock
with st.expander("Your Portfolio", expanded=True):
    st.write("### Portfolio Overview")
    for idx, item in enumerate(st.session_state.portfolio):
        st.write(f"**{item['stock_symbol']}** - Price: ${item['stock_price']:.2f}")
        strike = st.number_input(f"Strike Price for {item['stock_symbol']}", value=item['strike'], key=f"strike_{idx}")
        maturity = st.number_input(f"Maturity for {item['stock_symbol']} (years)", value=item['maturity'], key=f"maturity_{idx}")
        quantity = st.number_input(f"Quantity for {item['stock_symbol']}", value=item['quantity'], key=f"quantity_{idx}")
        
        # Update portfolio with new values
        st.session_state.portfolio[idx]["strike"] = strike
        st.session_state.portfolio[idx]["maturity"] = maturity
        st.session_state.portfolio[idx]["quantity"] = quantity

# Portfolio Risk Metrics with Tooltips
with st.expander("Portfolio Risk Metrics"):
    st.write("""
    Portfolio risk metrics help you assess the overall risk exposure of your options portfolio.
    These metrics include:
    - **Delta**: Measures how much the price of the option changes in response to changes in the stock price.
    - **Gamma**: Measures the rate of change of Delta as the stock price changes.
    - **VaR (Value-at-Risk)**: Estimates the potential loss in portfolio value under normal market conditions at a given confidence level.
    - **CVaR (Conditional Value-at-Risk)**: Measures the expected loss assuming that the VaR threshold has been breached.
    - **Monte Carlo Risk**: Simulates portfolio values based on random sampling to estimate the potential risk.

    """)
    total_delta, total_gamma, var, cvar, monte_carlo_risk = portfolio_risk_metrics(st.session_state.portfolio, S, r, sigma, option_type)
    
    st.write(f"Portfolio Delta: {total_delta}")
    st.write(f"Portfolio Gamma: {total_gamma}")
    st.write(f"Value-at-Risk (VaR): {var}")
    st.write(f"Conditional VaR (CVaR): {cvar}")
    st.write(f"Monte Carlo Risk: {monte_carlo_risk}")
    
    # Visualizing Portfolio Risk Metrics
    fig = go.Figure(data=[go.Bar(
        x=["Delta", "Gamma", "VaR", "CVaR", "Monte Carlo Risk"],
        y=[total_delta, total_gamma, var, cvar, monte_carlo_risk],
        marker_color='royalblue'
    )])
    fig.update_layout(
        title="Portfolio Risk Metrics",
        xaxis_title="Risk Metrics",
        yaxis_title="Value",
        template="plotly_dark"
    )
    st.plotly_chart(fig)
