import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

# Split the data into training and testing sets
split_index = round(len(data_df.index) * .8)
y_train, y_test = y[:split_index], y[split_index:]
X_train, X_test = X[:split_index], X[split_index:]

print(f"Train date range: {data_df['date'][0]} - {data_df['date'][split_index-1]}\nTest date range: {data_df['date'][split_index]} - {data_df['date'].iloc[-1]}")

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train size: {len(X_train)}\nTest size: {len(X_test)}\n")

# Set up the parameter grid for RandomForestRegressor
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV with RandomForestRegressor
rf = RandomForestRegressor(random_state=0, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2')
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best RandomForestRegressor parameters: {best_params}")

# Train the model with the best parameters
best_rf = grid_search.best_estimator_

# Make predictions
y_pred_train = best_rf.predict(X_train)
y_pred_test = best_rf.predict(X_test)

# Calculate R2 scores
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
print(f"In-Sample R2 is: {round(r2_train, 4)}\nOut-of-Sample R2 is: {round(r2_test, 4)}")

# Save the predictions to the DataFrame and export to CSV
data_df['RandomForest_Pred'] = np.append(y_pred_train, y_pred_test)
data_df.to_csv('data/RandomForest_Preds.csv', index=False)
print(data_df.head())