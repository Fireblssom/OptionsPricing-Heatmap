import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Descriptions for tooltips
greek_tooltips = {
    "Delta": "Measures sensitivity of option price to changes in the underlying asset price.",
    "Gamma": "Measures rate of change of Delta with respect to the underlying price.",
    "Theta": "Measures sensitivity of the option price to the passage of time (time decay).",
    "Vega": "Measures sensitivity to volatility of the underlying asset.",
    "Rho": "Measures sensitivity to changes in the risk-free interest rate."
}

# Black-Scholes formula
def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T == 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Streamlit app
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

# Tooltip display
if output_metric in greek_tooltips:
    st.caption(f"**{output_metric}**: {greek_tooltips[output_metric]}")

# Create grid
K_vals = np.linspace(*strike_range, 30)
T_vals = np.linspace(*time_range, 30)
Z = np.zeros((len(T_vals), len(K_vals)))

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

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
c = ax.imshow(Z, extent=[K_vals.min(), K_vals.max(), T_vals.min(), T_vals.max()],
              aspect='auto', origin='lower', cmap='viridis')
ax.set_xlabel("Strike Price (K)")
ax.set_ylabel("Maturity (T, years)")
ax.set_title(f"{output_metric} Heatmap")
fig.colorbar(c, ax=ax)
st.pyplot(fig)
