# Random-Forest-Classifier

The macro-level overview of the logic behind the trading strategy adopted in the code.

**Data Collection:** The strategy begins by collecting historical price and volume data for Ethereum (ETH) from Yahoo Finance using the yfinance library. The data spans from the start of 2010 to the end of 2023.

**Feature Engineering:** Several features are derived from the historical price and volume data. These features include:
  Returns: The percentage change in closing price from one day to the next.
  Price Rate of Change: The percentage change in closing price over a specified period (in this case, 5 days).
  Volume Rate of Change: The percentage change in trading volume over a specified period (in this case, 5 days).

**Data Preparation:** NaN values resulting from the feature engineering process are removed, and the features and target variable (signal) are defined. The signal indicates whether the price of Ethereum is expected to increase or decrease based on the closing price of the next day.

**Model Training:** The Random Forest Classifier is selected as the machine learning model for predicting the price direction. The model is trained on the historical data with hyperparameters optimized using GridSearchCV.

**Model Evaluation:** The trained model is evaluated on a separate testing dataset to assess its performance. Metrics such as accuracy and classification report are used to evaluate the model's predictive ability.

**Signal Generation:** The trained model is then used to generate trading signals based on the predicted price direction. A buy signal is generated when the model predicts that the price will increase, and no action is taken otherwise.

**Backtesting:** The strategy is backtested using the testing dataset to assess its performance. Metrics such as cumulative returns, drawdown, annual return rate, Sharpe ratio, and others are calculated to evaluate the strategy's performance.

**Visualization:** Various plots are generated to visualize the strategy's performance, including cumulative returns, price and strategy signals, and drawdown.

The strategy aims to profit from short-term price movements in Ethereum by predicting the direction of price changes using machine learning techniques. It leverages features derived from historical price and volume data to make predictions and generates trading signals based on these predictions. The performance of the strategy is evaluated using various metrics and visualizations to assess its effectiveness and profitability.
