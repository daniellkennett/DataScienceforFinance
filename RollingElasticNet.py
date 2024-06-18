import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# Load the data
data_df = pd.read_csv('data/CleanData.csv')

# Create the target variable: % Change in Index
data_df['delta_Index:t+1'] = data_df['delta_Index'].shift(-1)
data_df.dropna(inplace=True)

# Define target variable
y = data_df['delta_Index:t+1']
X = data_df.iloc[:, 1:-2]  # Grab all but date, delta_Index, and delta_Index(t+1)
X.drop(['Index', 'Rfree', 'RF'], axis=1, inplace=True)  # Drop repetitive columns

print(f"Columns included in model: {X.columns}\n")

# Initialize parameters for the rolling window
rolling_window = 12  # 12 months

# Initialize lists to store results
predictions = []
actuals = []

# Iterate over rolling windows
for i in range(len(data_df) - rolling_window - 1):
    # Prepare data for current window
    X_train = X.iloc[i:i + rolling_window].values  # Features for training
    y_train = y.iloc[i:i + rolling_window].values  # Target variable for training
    X_test = X.iloc[i + rolling_window:i + rolling_window + 1].values  # Features for testing
    y_test = y.iloc[i + rolling_window:i + rolling_window + 1].values  # Actual target for testing

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set up the parameter grid for ElasticNet
    param_grid = {
        'alpha': [.001, 0.01, .05, 0.1, 1.0],
        'l1_ratio': [.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    }

    # Initialize ElasticNet model
    EN = ElasticNet(random_state=0, max_iter=10000)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(EN, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    best_elastic_net = grid_search.best_estimator_

    # Predict
    y_pred = best_elastic_net.predict(X_test_scaled)

    # Store predictions and actuals
    predictions.append(y_pred[0])
    actuals.append(y_test[0])

    # Print or log results for each window (optional)
    print(f"Window {i+1}: Predicted={y_pred[0]}, Actual={y_test[0]}")

# Calculate R2 score at the end
overall_r2 = r2_score(actuals, predictions)
print(f"Overall R2 score: {overall_r2}")

# Optionally, save predictions, actuals, and scores to a DataFrame or CSV file
results_df = pd.DataFrame({'Actual': actuals, 'Predicted': predictions})
results_df.to_csv('data/Rolling_ElasticNet_Preds.csv', index=False)
print(results_df.head())