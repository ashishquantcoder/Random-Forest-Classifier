# Random Forest Classifier
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Fetch Ethereum data from Yahoo Finance
eth_data = yf.download('ETH-USD', start='2010-01-01', end='2023-12-31')

# Feature Engineering
eth_data['Returns'] = eth_data['Close'].pct_change()
eth_data['Price_Rate_Of_Change'] = eth_data['Close'].pct_change(periods=5)
eth_data['Volume_Rate_Of_Change'] = eth_data['Volume'].pct_change(periods=5)

# Drop NaN values
eth_data.dropna(inplace=True)

# Define features and target variable
X = eth_data[['Returns', 'Price_Rate_Of_Change', 'Volume_Rate_Of_Change']]
y = np.where(eth_data['Close'].shift(-1) > eth_data['Close'], 1, 0)  # 1 if price increases, 0 if price decreases

# Split the data into training and testing sets
X_train, X_test = X.loc[:'2022-06-30'], X.loc['2022-09-01':'2023-12-31']
y_train, y_test = y[:len(X_train)], y[len(X_train):len(X_train)+len(X_test)]

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [200],
    'max_depth': [None, 6],
    'min_samples_split': [9],
    'min_samples_leaf': [1]
}

rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the model with the best parameters
best_rf_classifier = RandomForestClassifier(**best_params, random_state=42)
best_rf_classifier.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = best_rf_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Reindex X_test to match the index of eth_data
X_test_reindexed = X_test.reindex(eth_data.index)

# Use the model to predict signals
eth_data['Predicted_Signal'] = best_rf_classifier.predict(scaler.transform(X_test_reindexed))

# Calculate trading metrics on test data
eth_data['Actual_Return'] = eth_data['Close'].pct_change()
eth_data['Strategy_Return'] = eth_data['Actual_Return'] * eth_data['Predicted_Signal'].shift(1)
eth_data['Cumulative_Strategy_Return'] = (1 + eth_data['Strategy_Return']).cumprod()
eth_data['Cumulative_Actual_Return'] = (1 + eth_data['Actual_Return']).cumprod()

# Calculate metrics
percent_drawdown = ((eth_data['Cumulative_Strategy_Return'] - eth_data['Cumulative_Strategy_Return'].cummax()) / eth_data['Cumulative_Strategy_Return'].cummax()).min()
annual_return_rate = (eth_data['Strategy_Return'].mean() * 252)
sharpe_ratio = (annual_return_rate / (eth_data['Strategy_Return'].std() * np.sqrt(252)))

print("Percent Drawdown:", percent_drawdown)
print("Annual Return Rate:", annual_return_rate)
print("Sharpe Ratio:", sharpe_ratio)

# Annual Return
annual_return = eth_data['Strategy_Return'].mean() * 252

# Cumulative Returns
cumulative_return = eth_data['Cumulative_Strategy_Return'][-1]

# Annual Volatility
annual_volatility = eth_data['Strategy_Return'].std() * np.sqrt(252)

# Calmar Ratio
max_drawdown = abs(percent_drawdown)
calmar_ratio = annual_return / max_drawdown

# Stability
stability = annual_return / annual_volatility

# Max Drawdown
max_drawdown_absolute = (eth_data['Cumulative_Strategy_Return'] - eth_data['Cumulative_Strategy_Return'].cummax()).min()

# Omega Ratio
risk_free_rate = 0.03  # Assumed risk-free rate
omega_ratio = (annual_return - risk_free_rate) / abs(max_drawdown_absolute)

# Sortino Ratio
downside_returns = eth_data[eth_data['Strategy_Return'] < 0]['Strategy_Return']
downside_deviation = downside_returns.std() * np.sqrt(252)
sortino_ratio = (annual_return - risk_free_rate) / downside_deviation

# Skew
skewness = eth_data['Strategy_Return'].skew()

# Kurtosis
kurtosis = eth_data['Strategy_Return'].kurtosis()

# Tail Ratio
negative_returns = eth_data[eth_data['Strategy_Return'] < 0]['Strategy_Return']
positive_returns = eth_data[eth_data['Strategy_Return'] > 0]['Strategy_Return']
tail_ratio = abs(negative_returns.mean() / positive_returns.mean())

# Daily Value at Risk (VaR) at 95% confidence level
daily_var = eth_data['Strategy_Return'].quantile(0.05)

# Winning Trades, Losing Trades, Win Rate
winning_trades = (eth_data['Strategy_Return'] > 0).sum()
losing_trades = (eth_data['Strategy_Return'] < 0).sum()
win_rate = winning_trades / (winning_trades + losing_trades)

# Print all calculated metrics
print("Annual Return:", annual_return)
print("Cumulative Returns:", cumulative_return)
print("Annual Volatility:", annual_volatility)
print("Calmar Ratio:", calmar_ratio)
print("Stability:", stability)
print("Omega Ratio:", omega_ratio)
print("Sortino Ratio:", sortino_ratio)
print("Skew:", skewness)
print("Kurtosis:", kurtosis)
print("Tail Ratio:", tail_ratio)
print("Daily Value at Risk (VaR):", daily_var)
print("Winning Trades:", winning_trades)
print("Losing Trades:", losing_trades)
print("Win Rate:", win_rate)

# Plot cumulative returns over the testing data
plt.figure(figsize=(10, 6))
plt.plot(eth_data.loc['2022-09-01':'2023-12-31'].index, eth_data.loc['2022-09-01':'2023-12-31']['Cumulative_Actual_Return'], label='Actual Returns', color='blue')
plt.plot(eth_data.loc['2022-09-01':'2023-12-31'].index, eth_data.loc['2022-09-01':'2023-12-31']['Cumulative_Strategy_Return'], label='Strategy Returns', color='orange')
plt.title('Cumulative Returns (Testing Period)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

# Plot strategy signals over the testing data
plt.figure(figsize=(10, 6))
plt.plot(eth_data.loc['2022-09-01':'2023-12-31'].index, eth_data.loc['2022-09-01':'2023-12-31']['Close'], label='Price', color='black')
buy_signals = eth_data.loc[(eth_data['Predicted_Signal'] == 1) & (eth_data.index >= '2022-09-01') & (eth_data.index <= '2023-12-31')]
plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal')
plt.title('Price and Strategy Signals (Testing Period)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Plot drawdown over the testing data
plt.figure(figsize=(10, 6))
plt.plot(eth_data.loc['2022-09-01':'2023-12-31'].index, ((eth_data.loc['2022-09-01':'2023-12-31']['Cumulative_Strategy_Return'] - eth_data.loc['2022-09-01':'2023-12-31']['Cumulative_Strategy_Return'].cummax()) / eth_data.loc['2022-09-01':'2023-12-31']['Cumulative_Strategy_Return'].cummax()), label='Drawdown', color='red')
plt.title('Drawdown (Testing Period)')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.legend()
plt.grid(True)
plt.show()
