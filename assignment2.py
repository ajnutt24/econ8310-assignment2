
# Setup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

#############################
#############################

# Load training data 
data = pd.read_csv(
    "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
)

print("Training data shape:", data.shape)
print("Meal distribution:\n", data['meal'].value_counts())

# Define X and Y
data['DateTime'] = pd.to_datetime(data['DateTime'])

data['hour']       = data['DateTime'].dt.hour       
data['day_of_week'] = data['DateTime'].dt.dayofweek

drop_cols = ['id', 'DateTime', 'meal']
Y = data['meal']
X = data.drop(columns=drop_cols)

# Train/validation split
x_train, x_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.1, random_state=42
)

# Define model 
model = XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.1,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

#################################
#################################

# Accuracy check
model.fit(x_train, y_train)
val_pred = model.predict(x_val)
print(f"Validation accuracy: {accuracy_score(y_val, val_pred) * 100:.2f}%")

# Final with all data
modelFit = XGBClassifier(
    n_estimators=14169,
    max_depth=4,
    learning_rate=0.1,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

modelFit.fit(X, Y)

##################################
##################################

# Load test data
test_data = pd.read_csv(
    "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
)

test_data['DateTime'] = pd.to_datetime(test_data['DateTime'])
test_data['hour']        = test_data['DateTime'].dt.hour
test_data['day_of_week'] = test_data['DateTime'].dt.dayofweek

# Define variables
test_drop = [col for col in drop_cols if col in test_data.columns]
X_test = test_data.drop(columns=test_drop)

# Predictions
pred = modelFit.predict(X_test).astype(int).tolist()

print(f"Number of predictions: {len(pred)}")
print(f"Predicted meals (1s): {sum(pred)}")
print(f"Predicted non-meals (0s): {len(pred) - sum(pred)}")
