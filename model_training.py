import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle 
from joblib import dump
from datetime import datetime



#drop all records without "India"
#df = df['country' = "India"]

df = pd.read_csv("phone_repair_data.csv")

df.shape
#df.info
#df.dtypes
#df.isnull().sum()
# Feature: Customer segment
df['customer_segment'] = pd.cut(df['customer_age'], bins=[0, 25, 50, 75, 100], labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])

# Feature: Time since last login
current_date = datetime.now()
df.drop(columns= ['repair_id',  'customer_id', 'customer_name', 'customer_email','repair_start_date','repair_end_date','customer_age','customer_country'], inplace=True)

df['repair_cost']= round(df.repair_cost)
df['cost_per_part']= round(df.cost_per_part)
# Calculate the correlation matrix 
ndf=df.select_dtypes(include=['int64','float64']).copy()
ndf.columns
import seaborn as sns 
import matplotlib.pyplot as plt
correlation_matrix = ndf.corr() # Plot the correlation matrix plt.figure(figsize=(12, 8)) 
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix') 
plt.show()
######################
# encoding all categirical columns into numerical columns
# Feature: Repair complexity index
complexity_weights = {'Screen Replacement': 1, 'Battery Replacement': 1, 'Camera Repair': 2, 'Water Damage Repair': 3, 'Software Issue': 1,
                      'Speaker Replacement': 2, 'Charging Port Repair': 2, 'Microphone Repair': 2, 'Button Repair': 1, 'Antenna Replacement': 2,
                      'Back Cover Replacement': 1, 'Fingerprint Sensor Repair': 3, 'Display Replacement': 2, 'Vibration Motor Repair': 2,
                      'Motherboard Replacement': 4, 'Headphone Jack Repair': 2, 'Proximity Sensor Repair': 2, 'Earpiece Replacement': 2}
df['repair_complexity_index'] = df['repair_type'].map(complexity_weights) * df['parts_used']
#One-hot encoding for many categorical columns
#categorical data
categorical_cols = ['phone_model',  'repair_center','repair_status', 'customer_segment'] 
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()
# apply le on categorical feature columns
df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
#repair_Status completed, inprogress, cancedlled
df.drop(columns= ['repair_type'],  inplace=True)

df= pd.get_dummies(df, columns=['phone_model',  'repair_center', 'repair_status','customer_segment'], drop_first=True)
# Convert boolean columns to numerical (0/1) 
df=df.astype(int)
#df = df.join(sdf)
df.head(5)


# Separate features and target variable
X = df.drop('turnaround_time_days', axis=1)
y = df['turnaround_time_days']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scaler
scaler = StandardScaler()  # You can also use MinMaxScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
mlflow.set_tracking_uri("http://localhost:7000")
mlflow.set_experiment("multi-process")
# Function to log model performance 
def log_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions) 
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("MSE", mse) 
        mlflow.log_param("RMSE", rmse)
        mlflow.log_param("MAE", mae)
        mlflow.log_param("R2", r2) 
        mlflow.sklearn.log_model(model, model_name)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
log_model(rf_model, X_test_scaled, y_test, "RandomForest")
# Train and log Random Forest model 
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train) 
log_model(rf_model, X_test_scaled, y_test, "RandomForest") 
# Save the Random Forest model to both pickle and joblib files 
with open('rf_model.pkl', 'wb') as file: 
    pickle.dump(rf_model, file) 
    dump(rf_model, 'rf_model.joblib')
# Train an XGBoost model
#xgb_model = XGBRegressor(n_estimators=100, random_state=42)
#xgb_model.fit(X_train_scaled, y_train)
#Get feature importances
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
# Get feature importances feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns) # Visualize feature importances
import matplotlib.pyplot as plt 
feature_importances.sort_values().plot(kind='barh')
plt.title('Feature Importances')
plt.show()

# Evaluate the models
rf_predictions = rf_model.predict(X_test_scaled)


# Calculate accuracy (mean squared error as a proxy for model performance)
rf_mse = mean_squared_error(y_test, rf_predictions)


print("Random Forest MSE:", rf_mse)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate the performance metrics for Random Forest
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)



print("Random Forest Performance:")
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
print("MAE:", rf_mae)
print("R^2:", rf_r2)