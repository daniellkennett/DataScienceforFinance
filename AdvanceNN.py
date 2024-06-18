import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the data
data_df = pd.read_csv('data/CleanData.csv')

# Create the target variable: % Change in Index
data_df['delta_Index:t+1'] = data_df['delta_Index'].shift(-1)
data_df.dropna(inplace=True)

# Define target variable
y = data_df['delta_Index:t+1'].values
X = data_df.iloc[:, 1:-2].drop(['Index', 'Rfree', 'RF'], axis=1).values  # Grab all but date, delta_Index, and delta_Index(t+1)

print(f"Columns included in model: {data_df.columns[1:-2]}\n")

# Split data into training and testing sets
split_index = round(len(data_df.index) * 0.9)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

print(f"Train date range: {data_df['date'].iloc[0]} - {data_df['date'].iloc[split_index-1]}\nTest date range: {data_df['date'].iloc[split_index]} - {data_df['date'].iloc[-1]}")

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"Train size: {len(X_train)}\nValidation size: {len(X_val)}\nTest size: {len(X_test)}\n")

# Define the enhanced neural network model
class AdvancedNN(nn.Module):
    def __init__(self, input_size):
        super(AdvancedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc5 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        x = self.fc5(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
model = AdvancedNN(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Train the model with validation
num_epochs = 2500
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training phase
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    train_loss = criterion(outputs, y_train_tensor)
    train_loss.backward()
    optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    # Save the best model
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_model = model.state_dict()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# Load the best model
model.load_state_dict(best_model)

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_tensor).numpy()
    y_pred_test = model(X_test_tensor).numpy()

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
print(f"In-Sample R2 is: {round(r2_train, 4)}\nOut-of-Sample R2 is: {round(r2_test, 4)}")

# Save predictions to a DataFrame and CSV file
data_df['NN_Pred'] = np.append(np.append(y_pred_train, y_val), y_pred_test)
data_df.to_csv('data/NN_Preds.csv', index=False)
print(data_df.head())