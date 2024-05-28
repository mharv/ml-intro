import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    california = fetch_california_housing()
    X = california.data
    y = california.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    # Mean Squared Error (MSE) is like measuring how far off a bunch of predictions are from 
    # the actual values by squaring each difference (to make sure they're all positive), then 
    # averaging those squared differences. If the predictions are really close to the actual 
    # values, the MSE will be small, but if they're way off, the MSE will be big. It gives us 
    # a way to understand how well a model is doing at making predictions, with smaller values 
    # meaning better performance.
    print("R^2 Score:", r2)
    # The R-squared (R2) score is a measure that tells us how well the regression model fits the data. 
    # It's like a percentage that indicates the proportion of the variance in the dependent variable (target) 
    # that is explained by the independent variables (features) in the model. In simpler terms, R2 score tells us 
    # how much of the variation in the target variable can be explained by the model. A higher R2 score, closer to 1, 
    # indicates that the model explains a larger portion of the variance in the target variable, while a lower R2 score, 
    # closer to 0, indicates that the model does not explain much of the variance and may not be a good fit for the data.


if __name__ == "__main__":
    main()