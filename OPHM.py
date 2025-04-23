import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# Humanized descriptions for the Greeks
greek_tooltips = {
    "Delta": (
        "Delta tells you how much the price of the option will move for a small change in the price of the underlying asset. "
        "For calls, it increases as the stock price rises, and for puts, it decreases."
    ),
    "Gamma": (
        "Gamma measures the rate of change of Delta as the underlying asset price changes. "
        "In simple terms, it's like the acceleration of Delta — how much faster Delta changes as the stock price moves."
    ),
    "Theta": (
        "Theta represents time decay, showing how much an option's price decreases as time passes, holding everything else constant. "
        "As expiration nears, an option’s value tends to decline, especially for out-of-the-money options."
    ),
    "Vega": (
        "Vega tells you how sensitive the option’s price is to changes in volatility. "
        "When volatility increases, option prices usually rise, and Vega measures this relationship."
    ),
    "Rho": (
        "Rho shows how much an option's price will change with a change in the risk-free interest rate. "
        "It's most significant for longer-term options, as small interest rate changes have a bigger effect on their value."
    )
}

# Black-Scholes pricing model
def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T == 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Streamlit UI
st.title("Options Pricing Heatmap")

S = st.sidebar.number_input("Spot Price (S)", value=100.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
strike_range = st.sidebar.slider("Strike Price Range (K)", 50, 150, (80, 120))
time_range = st.sidebar.slider("Maturity Range (T, years)", 0.01, 2.0, (0.1, 1.0))
output_metric = st.sidebar.selectbox(
    "Output Metric",
    ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"]
)

# Display selected Greek's description
if output_metric in greek_tooltips:
    st.caption(f"**{output_metric}**: {greek_tooltips[output_metric]}")

# Grid setup for strike prices and maturities
K_vals = np.linspace(*strike_range, 30)
T_vals = np.linspace(*time_range, 30)
Z = np.zeros((len(T_vals), len(K_vals)))

# Compute values for the heatmap based on selected Greek
for i, T in enumerate(T_vals):
    for j, K in enumerate(K_vals):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)) if T > 0 else 0
        d2 = d1 - sigma * np.sqrt(T) if T > 0 else 0

        if output_metric == "Price":
            Z[i, j] = black_scholes(S, K, T, r, sigma, option_type)
        elif output_metric == "Delta":
            Z[i, j] = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
        elif output_metric == "Gamma":
            Z[i, j] = norm.pdf(d1) / (S * sigma * np.sqrt(T)) if T > 0 else 0
        elif output_metric == "Theta":
            first = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) if T > 0 else 0
            second = r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)
            Z[i, j] = first - second if option_type == "call" else first + second
        elif output_metric == "Vega":
            Z[i, j] = S * norm.pdf(d1) * np.sqrt(T) if T > 0 else 0
        elif output_metric == "Rho":
            Z[i, j] = K * T * np.exp(-r * T) * norm.cdf(d2) if option_type == "call" else -K * T * np.exp(-r * T) * norm.cdf(-d2)

# Plot the heatmap using Plotly
fig = go.Figure(data=go.Heatmap(
    z=Z,
    x=np.round(K_vals, 2),
    y=np.round(T_vals, 2),
    colorscale="YlGnBu",
    colorbar=dict(title=output_metric),
    hovertemplate='K: %{x}<br>T: %{y}<br>' + output_metric + ': %{z:.4f}<extra></extra>'
))

# Layout adjustments for the Plotly chart
fig.update_layout(
    title=f"{output_metric} Heatmap",
    xaxis_title="Strike Price (K)",
    yaxis_title="Maturity (T, years)",
    autosize=True,
    margin=dict(l=40, r=40, t=60, b=40)
)

# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)
