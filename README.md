# options-a-math-project
Project and full code for the course options and mathematics.


# Project Overview

This repository contains all the code, data, and documentation for our project on **equity returns analysis** and **option pricing** using the Black–Scholes model. Below is a summary of the core steps we followed.

## 1. Data Collection
- **Source**: [Yahoo! Finance](https://finance.yahoo.com/) via the `yfinance` library  
- **Time Frame**: Last three years of daily data  
- **Assets**: Two stocks, one American (e.g., `AAPL`) and one Swedish (e.g., `VOLV-B.ST`)

## 2. Data Preparation and Return Calculations
- **Daily log returns** calculated as:

  $$
  r_t = \ln\left(\frac{P_t}{P_{t-1}}\right),
  $$
  
  where $P_t$ is the adjusted closing price on day $t$.

## 3. Exploratory Plots
1. **Time-Series Plot**  
   - Visualized the daily returns for both stocks on separate subplots.
2. **Histograms**  
   - Compared the empirical return distributions with a fitted normal distribution to highlight phenomena such as fat tails, skewness, and kurtosis.

## 4. Rolling Volatility (20-Day)
- Computed a **20-day rolling standard deviation** of returns to observe how volatility evolves over time (i.e., volatility clustering).
- Plotted the rolling volatility for both stocks on the same figure.

## 5. Annualized Volatility
- Scaled the rolling volatility by \(\sqrt{252}\) to convert from daily to annualized volatility.  
- **Averaged** this annualized volatility over the three-year period to produce a single figure per stock (e.g., ~26–27% for AAPL, ~24–25% for VOLV-B).

## 6. Volatility-Scaled Returns
- Created a new time series by **dividing each day’s return** by its 20-day rolling volatility:

  \[
  \frac{r_t}{\sigma_{d}(t)},
  \]

  to “standardize” fluctuations.
- Plotted both stocks’ scaled returns and their histograms to see if they align more closely with a standard normal distribution.

## 7. Basic Option Pricing via Black–Scholes
- Chose a **near at-the-money (ATM) call option** for each stock, retrieving:
  - Market option premium  
  - Strike price  
  - Time to expiration  
  - Risk-free rate  
- Used the computed annualized volatility estimate as input for the **Black–Scholes** model:

  \[
  C = S_0 \Phi(d_1) - K e^{-r T} \Phi(d_2),
  \]

  where
  \[
  d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2}\right) T}{\sigma \sqrt{T}}, 
  \quad
  d_2 = d_1 - \sigma \sqrt{T}.
  \]

- Compared the **theoretical Black–Scholes price** to the **actual market price**, discussing potential reasons for discrepancies (implied vs. historical volatility, dividends, market microstructure, etc.).

## 8. Skewness and Kurtosis Analysis
- Computed and reported **skewness** and **kurtosis** for:
  1. **Raw daily returns**  
  2. **Volatility-scaled returns**  

Observed whether scaling reduces skewness and kurtosis, noting persistent deviations from normality (e.g., fat tails).

---

## Getting Started

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/YourUsername/YourRepo.git
   cd YourRepo
