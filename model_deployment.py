#Scenario:1 Batch Inference
#Description: In this strategy, the model processes a batch of data at a time. This is useful for scenarios where real-time predictions are not required, and predictions can be made in bulk.

Code Example:
python
import pandas as pd
import joblib

def batch_inference(input_data_path, output_data_path):
    # Load the model
    model = joblib.load('rf_model.joblib')
    
    # Load the input data
    input_data = pd.read_csv(input_data_path)
    
    # Preprocess the data
    # ... (Preprocessing code here)
    
    # Make predictions
    predictions = model.predict(input_data)
    
    # Save the predictions
    output_data = pd.DataFrame(predictions, columns=['PredictedTurnaroundTime'])
    output_data.to_csv(output_data_path, index=False)

# Example usage
batch_inference('input_data.csv', 'predictions.csv')

Scenario 2: Online Inference (REST API)
Description: In this strategy, the model is deployed as a REST API, which can handle real-time requests. This is useful for applications requiring immediate responses, like chatbots or recommendation systems.


from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('rf_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    
    # Preprocess the data
    # ... (Preprocessing code here)
    
    # Make predictions
    predictions = model.predict(df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
