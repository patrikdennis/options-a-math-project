#  -- Import necessary libraries --
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import log, sqrt, exp

# 1. Download last 3 years of daily data
start_date = "2020-12-01"
end_date = "2023-12-01"

# Tickers
ticker_us = "AAPL"         # Apple
ticker_sweden = "VOLV-B.ST"  # Volvo B (Swedish listing)

# Download data from Yahoo! Finance
data_us = yf.download(ticker_us, start=start_date, end=end_date)["Adj Close"]
data_se = yf.download(ticker_sweden, start=start_date, end=end_date)["Adj Close"]

# 2. Compute daily log returns
returns_us = np.log(data_us / data_us.shift(1)).dropna()
returns_se = np.log(data_se / data_se.shift(1)).dropna()

# 2b. Print skewness and kurtosis for raw returns
# Skewness and kurtosis for AAPL
skew_us = returns_us.skew()
kurt_us = returns_us.kurt()  # by default, this is Fisher's definition of kurtosis
print(f"AAPL Returns Skewness: {skew_us:.4f}")
print(f"AAPL Returns Kurtosis: {kurt_us:.4f}\n")

# Skewness and kurtosis for VOLV-B.ST
skew_se = returns_se.skew()
kurt_se = returns_se.kurt()
print(f"VOLV-B.ST Returns Skewness: {skew_se:.4f}")
print(f"VOLV-B.ST Returns Kurtosis: {kurt_se:.4f}\n")

# 3. Plot the returns over time (two vertical subplots)
fig, axes = plt.subplots(nrows=2, figsize=(12, 8), sharex=True)

# AAPL Returns (top)
axes[0].plot(returns_us.index, returns_us, label='AAPL Returns')
axes[0].set_title("Daily Log Returns: AAPL")
axes[0].legend()

# VOLV-B.ST Returns (bottom)
axes[1].plot(returns_se.index, returns_se, color='orange', label='VOLV-B.ST Returns')
axes[1].set_title("Daily Log Returns: VOLV-B.ST")
axes[1].legend()

plt.tight_layout()
plt.show()

# 4. Plot histograms of the daily returns with overlaid normal PDFs
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (rets, label) in enumerate([(returns_us, 'AAPL'), (returns_se, 'VOLV-B.ST')]):
    ax = axes[i]
    ax.hist(rets, bins=50, density=True, alpha=0.6, label='Histogram')
    
    # Overlay normal PDF
    mu = rets.mean()
    sigma = rets.std()
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    ax.plot(x, norm.pdf(x, mu, sigma), 'r', label='Normal PDF')
    ax.set_title(f"{label} Returns Distribution")
    ax.legend()

plt.show()

# 5. Calculate 20-day rolling volatility
window = 20
rolling_vol_us = returns_us.rolling(window).std()  # AAPL
rolling_vol_se = returns_se.rolling(window).std()  # VOLV-B.ST

# Plot the rolling volatilities
plt.figure(figsize=(12, 6))
plt.plot(rolling_vol_us.index, rolling_vol_us, label='AAPL Rolling 20d Vol')
plt.plot(rolling_vol_se.index, rolling_vol_se, label='VOLV-B.ST Rolling 20d Vol')
plt.title("20-Day Rolling Volatility (Daily)")
plt.legend()
plt.show()

# 6. Annualize the rolling volatility and compute the average over the period
annual_factor = np.sqrt(252)
rolling_vol_us_annual = rolling_vol_us * annual_factor
rolling_vol_se_annual = rolling_vol_se * annual_factor

avg_annual_vol_us = rolling_vol_us_annual.mean()
avg_annual_vol_se = rolling_vol_se_annual.mean()
print(f"Average Annualized Vol (AAPL): {avg_annual_vol_us:.2%}")
print(f"Average Annualized Vol (VOLV-B.ST): {avg_annual_vol_se:.2%}\n")

# 7. Compute daily returns scaled by rolling daily volatility and plot
ratio_us = returns_us / rolling_vol_us
ratio_se = returns_se / rolling_vol_se

plt.figure(figsize=(12, 6))
plt.plot(ratio_us.index, ratio_us, label='AAPL Returns / Vol')
plt.plot(ratio_se.index, ratio_se, label='VOLV-B.ST Returns / Vol')
plt.title("Daily Returns Scaled by 20-Day Rolling Volatility")
plt.legend()
plt.show()

# 7b. Print skewness and kurtosis for scaled returns
ratio_us_clean = ratio_us.dropna()
ratio_se_clean = ratio_se.dropna()

skew_ratio_us = ratio_us_clean.skew()
kurt_ratio_us = ratio_us_clean.kurt()
print(f"AAPL Scaled Returns Skewness: {skew_ratio_us:.4f}")
print(f"AAPL Scaled Returns Kurtosis: {kurt_ratio_us:.4f}\n")

skew_ratio_se = ratio_se_clean.skew()
kurt_ratio_se = ratio_se_clean.kurt()
print(f"VOLV-B.ST Scaled Returns Skewness: {skew_ratio_se:.4f}")
print(f"VOLV-B.ST Scaled Returns Kurtosis: {kurt_ratio_se:.4f}\n")

# 8. Plot histogram of scaled returns (returns / daily vol) for each stock
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AAPL scaled returns histogram
mu_us = ratio_us_clean.mean()
sigma_us = ratio_us_clean.std()
axes[0].hist(ratio_us_clean, bins=50, density=True, alpha=0.6, label='Histogram')
x_us = np.linspace(mu_us - 4*sigma_us, mu_us + 4*sigma_us, 200)
axes[0].plot(x_us, norm.pdf(x_us, mu_us, sigma_us), 'r', label='Normal PDF')
axes[0].set_title("AAPL: Returns / Vol Distribution")
axes[0].legend()

# VOLV-B.ST scaled returns histogram
mu_se = ratio_se_clean.mean()
sigma_se = ratio_se_clean.std()
axes[1].hist(ratio_se_clean, bins=50, density=True, alpha=0.6, color='orange', label='Histogram')
x_se = np.linspace(mu_se - 4*sigma_se, mu_se + 4*sigma_se, 200)
axes[1].plot(x_se, norm.pdf(x_se, mu_se, sigma_se), 'r', label='Normal PDF')
axes[1].set_title("VOLV-B.ST: Returns / Vol Distribution")
axes[1].legend()

plt.tight_layout()
plt.show()

# 9. Black-Scholes pricing for hypothetical near-ATM options
def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes for a European call option (no dividends).
    """
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call_price

# Hypothetical option data for AAPL
aapl_option_market_price = 5.00   # e.g. $5.00 premium
aapl_option_strike       = 190    # e.g. $190 strike
aapl_current_spot        = 191    # e.g. $191 current price
time_to_maturity_years   = 0.25   # e.g. 3 months
r_us = 0.05                       # 5% risk-free rate (example)
aapl_vol_est = rolling_vol_us_annual.iloc[-1]  # or use average or a different measure

# Hypothetical option data for VOLV-B.ST
volv_option_market_price = 8.50
volv_option_strike       = 200
volv_current_spot        = 202
r_se = 0.03                       # 3% risk-free rate (example)
volv_vol_est = rolling_vol_se_annual.iloc[-1]

# Compute theoretical BS prices
bs_aapl_price = black_scholes_call(aapl_current_spot,
                                   aapl_option_strike,
                                   time_to_maturity_years,
                                   r_us,
                                   aapl_vol_est)

bs_volv_price = black_scholes_call(volv_current_spot,
                                   volv_option_strike,
                                   time_to_maturity_years,
                                   r_se,
                                   volv_vol_est)

# Print comparison
print(f"Black–Scholes Theoretical Price (AAPL): {bs_aapl_price:.2f}")
print(f"Market Price (AAPL): {aapl_option_market_price:.2f}")
print(f"Difference: {aapl_option_market_price - bs_aapl_price:.2f}\n")

print(f"Black–Scholes Theoretical Price (VOLV-B.ST): {bs_volv_price:.2f}")
print(f"Market Price (VOLV-B.ST): {volv_option_market_price:.2f}")
print(f"Difference: {volv_option_market_price - bs_volv_price:.2f}")
