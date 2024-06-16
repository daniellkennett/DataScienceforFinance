import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

data_df = pd.read_csv('data/CleanData.csv')
# data_df['delta_Index:t-1'] = data_df['delta_Index'].shift(-1)
# I'm choosing to keep the same column name due to simplicity. 
# We are predicting t+1 with current data. We are predeicting the FUTURE
data_df['delta_Index:t+1'] = data_df['delta_Index'].shift(-1)
data_df.dropna(inplace=True)

# Defin target variable: % Change in Index
y = data_df['delta_Index:t+1']
X = data_df.iloc[:,1:-2] # Grab all but date, delta_Index, and delta_Index(t+1)
X.drop(['Index','Rfree', 'RF'], axis=1, inplace=True) # repetitive columns

print(f"Columns included in model: {X.columns}\n")
# Start in sample vs. out of sample split
split_index = round(len(data_df.index)*.8)
y_train, y_test  = y[:split_index], y[split_index:]
X_train, X_test = X[:split_index], X[split_index:]

print(f"Train date range: {data_df['date'][0]} - {data_df['date'][split_index-1]}\nTest date range: {data_df['date'][split_index]} - {data_df['date'].iloc[-1]}")
# NORMALIZE!!!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train size: {len(X_train)}\nTest size: {len(X_test)}\n")

# ElasticNet is an algorithm that should be explored. 
# So I set up a gridsearch to do that
param_grid = {
    'alpha': [.001, 0.01, .05, 0.1, 1.0],
    'l1_ratio': [.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
}

# Initialize GridSearchCV
EN = ElasticNet(random_state=0, max_iter=10000)
grid_search = GridSearchCV(EN, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best EN algo is with the parameters: {best_params}")

# Train the model with the best parameters
best_elastic_net = grid_search.best_estimator_

y_pred_train = best_elastic_net.predict(X_train)
y_pred_test = best_elastic_net.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
print(f"In-Sample R2 is: {round(r2_train,4)}\n Out-of-Sample R2 is: {round(r2_test,4)}")

data_df['ElasticNet_Pred']= np.append(y_pred_train, y_pred_test)
data_df.to_csv('data/ElasticNet_Preds.csv')
print(data_df.head())
