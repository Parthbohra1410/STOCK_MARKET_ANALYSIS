
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

# Download stock data (e.g., Apple)
ticker = 'AAPL'
df = yf.download(ticker, start='2018-01-01', end='2023-12-31')

# Use only Date and Close columns
df = df[['Close']].reset_index()

# Feature engineering: convert date to numerical value
df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)

# Features and target
X = df[['Date_ordinal']]
y = df['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], y, label='Actual Price')
plt.plot(df.loc[X_test.index, 'Date'], y_pred, label='Predicted Price', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.title(f'{ticker} Stock Price Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
