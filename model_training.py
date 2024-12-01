import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from joblib import dump
from datetime import datetime
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
      
    # Round repair costs
    df['repair_cost'] = round(df.repair_cost)
    #df['cost_per_part'] = round(df.cost_per_part)
    
    # Feature: Customer segment
    df['customer_segment'] = pd.cut(df['customer_age'], bins=[0, 25, 50, 75, 100], labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
    
    # Feature: Repair complexity index
    complexity_weights = {'Screen Replacement': 1, 'Battery Replacement': 1, 'Camera Repair': 2, 'Water Damage Repair': 3, 'Software Issue': 1,
                          'Speaker Replacement': 2, 'Charging Port Repair': 2, 'Microphone Repair': 2, 'Button Repair': 1, 'Antenna Replacement': 2,
                          'Back Cover Replacement': 1, 'Fingerprint Sensor Repair': 3, 'Display Replacement': 2, 'Vibration Motor Repair': 2,
                          'Motherboard Replacement': 4, 'Headphone Jack Repair': 2, 'Proximity Sensor Repair': 2, 'Earpiece Replacement': 2}
    df['repair_complexity_index'] = df['repair_type'].map(complexity_weights) * df['parts_used']
    # Get all numerical columns 
    ndf = df.select_dtypes(include=['number'])
    plot_correlation_matrix(ndf)
    
    # Encode categorical columns
    categorical_cols = ['phone_model', 'repair_center', 'repair_status', 'customer_segment']
    le = LabelEncoder()
    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
    
    df.drop(columns=['repair_type'], inplace=True)
    # Drop columns
    df.drop(columns=['repair_id', 'customer_id', 'customer_name', 'customer_email', 'repair_start_date', 'repair_end_date', 'customer_age', 'customer_country'], inplace=True)
    
    
    # One-hot encoding for categorical columns
    df = pd.get_dummies(df, columns=['phone_model', 'repair_center', 'repair_status', 'customer_segment'], drop_first=True)
    
    # Convert all to int
    df = df.astype(int)
    
    return df

def plot_correlation_matrix(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def log_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(model, model_name)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    log_model(rf_model, X_test, y_test, "RandomForest")
    save_model(rf_model, 'rf_model.pkl', 'rf_model.joblib')
    return rf_model

def save_model(model, pickle_path, joblib_path):
    with open(pickle_path, 'wb') as file:
        pickle.dump(model, file)
    dump(model, joblib_path)
    print(f"Model saved as {pickle_path} and {joblib_path}")

def plot_feature_importances(model, X_train):
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importances.sort_values().plot(kind='barh')
    plt.title('Feature Importances')
    plt.show()

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("multi-process")
    
    df = preprocess_data("phone_repair_data.csv")
    
    
    
    X = df.drop('turnaround_time_days', axis=1)
    y = df['turnaround_time_days']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    rf_model = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    plot_feature_importances(rf_model, X_train)

    rf_predictions = rf_model.predict(X_test_scaled)
    
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

if __name__ == "__main__":
    main()
