from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Extract the answers from the JSON data
    input_data = [data['a1'], data['a2']]  # Add more as needed
    # Convert the input data to a numpy array
    input_data = np.array(input_data).reshape(1, -1)
    # Scale the input data
    input_data = scaler.transform(input_data)
    # Make prediction
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
