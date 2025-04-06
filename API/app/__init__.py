from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib  # Use joblib instead of pickle
import numpy as np
import os
import pandas as pd  # Import pandas
from werkzeug.utils import secure_filename  # Import for file handling

# Ensure Flask serves only API endpoints
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the VotingClassifier model
model_path = os.path.join('API', 'model', 'saved_models', 'voting_model.pkl')
with open(model_path, 'rb') as model_file:
    model = joblib.load(model_file)  # Use joblib.load

# Load the encoders for 'service' and 'flag'
service_encoder_path = os.path.join('API', 'model', 'saved_models', 'service_encoder.pkl')
with open(service_encoder_path, 'rb') as service_encoder_file:
    service_encoder = joblib.load(service_encoder_file)  # Use joblib.load

flag_encoder_path = os.path.join('API', 'model', 'saved_models', 'flag_encoder.pkl')    
with open(flag_encoder_path, 'rb') as flag_encoder_file:
    flag_encoder = joblib.load(flag_encoder_file)  # Use joblib.load

@app.route('/')
def index():
    # Indicate that the API is running
    return jsonify({'message': 'API is running'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        input_data = request.get_json()
        print(f"Received input data: {input_data}")  # Debugging statement

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode 'service' and 'flag' using the loaded encoders
        input_df['service'] = service_encoder.transform(input_df['service'])
        input_df['flag'] = flag_encoder.transform(input_df['flag'])

        # Prepare features for prediction
        features = input_df[['service', 'flag', 'src_bytes', 'dst_bytes', 'same_srv_rate', 
                             'diff_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
                             'dst_host_diff_srv_rate', 'dst_host_serror_rate']].values

        # Make prediction
        prediction = model.predict(features)
        response = {'prediction': prediction.tolist()}
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        # Check if a file is included in the request
        if 'file' not in request.files:
            print("No file part in the request")  # Debugging log
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        # Check if a file is selected
        if file.filename == '':
            print("No file selected")  # Debugging log
            return jsonify({'error': 'No file selected'}), 400

        # Secure the filename and read the file
        filename = secure_filename(file.filename)
        print(f"Received file: {filename}")  # Debugging log
        file_path = os.path.join(os.getcwd(), filename)
        file.save(file_path)

        # Read the CSV file into a DataFrame
        try:
            input_df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading CSV file: {e}")  # Debugging log
            return jsonify({'error': 'Invalid CSV file format'}), 400

        # Encode 'service' and 'flag' using the loaded encoders
        try:
            input_df['service'] = service_encoder.transform(input_df['service'])
            input_df['flag'] = flag_encoder.transform(input_df['flag'])
        except Exception as e:
            print(f"Error encoding columns: {e}")  # Debugging log
            return jsonify({'error': 'Error encoding columns'}), 400

        # Prepare features for prediction
        try:
            features = input_df[['service', 'flag', 'src_bytes', 'dst_bytes', 'same_srv_rate', 
                                 'diff_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
                                 'dst_host_diff_srv_rate', 'dst_host_serror_rate']].values
        except KeyError as e:
            print(f"Missing required columns: {e}")  # Debugging log
            return jsonify({'error': f'Missing required columns: {e}'}), 400

        # Make predictions
        predictions = model.predict(features)

        # Add predictions to the DataFrame
        input_df['prediction'] = predictions

        # Convert the DataFrame to a dictionary for JSON response
        response = input_df.to_dict(orient='records')

        # Remove the temporary file
        os.remove(file_path)

        return jsonify(response), 200, {'Cache-Control': 'no-store', 'Pragma': 'no-cache'}
    except Exception as e:
        print(f"Unexpected error: {e}")  # Debugging log
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False)
