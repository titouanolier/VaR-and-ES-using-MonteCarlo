import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Fetch stock data
stock_symbol = 'AAPL'  # Exemple: Apple Inc.
start_date = '2013-01-01'
end_date = '2024-01-01'
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Step 2: Calculate daily returns
stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()

# Step 3: Simulate future prices using Monte Carlo simulation
days_to_simulate = 100  # Nombre de jours à simuler
num_simulations = 1000  # Nombre de simulations

last_price = stock_data['Adj Close'].iloc[-1]
simulated_prices = np.zeros((days_to_simulate, num_simulations))

for i in range(num_simulations):
    daily_returns = np.random.normal(stock_data['Daily Return'].mean(), stock_data['Daily Return'].std(), days_to_simulate)
    price_series = [last_price]
    for ret in daily_returns:
        price_series.append(price_series[-1] * (1 + ret))
    simulated_prices[:, i] = price_series[1:]

# Step 4: Calculate VaR
confidence_level = 0.95  # Niveau de confiance à 95%
sorted_simulated_prices = np.sort(simulated_prices[-1, :])
var_index = int(num_simulations * (1 - confidence_level))
var = sorted_simulated_prices[var_index]

# Step 5: Calculate Expected Shortfall
losses_exceeding_var = simulated_prices[-1, simulated_prices[-1, :] < var]
expected_shortfall = np.mean(losses_exceeding_var)

# Step 6: Visualize results
plt.hist(sorted_simulated_prices, bins=50, density=True, alpha=0.6, color='g')
plt.axvline(x=var, color='r', linestyle='--', linewidth=2, label='VaR at 95% confidence')
plt.xlabel('Price')
plt.ylabel('Density')
plt.title('Monte Carlo Simulation of Stock Prices')
plt.legend()
plt.show()

print(f"Value at Risk (VaR) at {confidence_level*100:.2f}% confidence level: {var:.2f}")
print(f"Expected Shortfall (ES) at {confidence_level*100:.2f}% confidence level: {expected_shortfall:.2f}")
