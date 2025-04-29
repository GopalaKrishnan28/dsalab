from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS to handle Cross-Origin Resource Sharing
import pickle
import numpy as np

app = Flask(__name__)

# Enable CORS for all origins
CORS(app)

# Load your trained model


@app.route('/predict', methods=['POST'])
def predict():
    try:
        with open('XGBoost.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
        data = request.get_json()
        features = [
    float(data['nitrogen']),   # Convert nitrogen to float
    float(data['phosphorus']), # Convert phosphorus to float
    float(data['potassium']),  # Convert potassium to float
    float(data['temperature']),# Convert temperature to float
    float(data['humidity']),   # Convert humidity to float
    float(data['ph']),         # Convert pH to float
    float(data['rainfall'])    # Convert rainfall to float
]

        print(features)
        print("get successfully")
        data = np.array([features])  # Convert the list to a NumPy array
        prediction = model.predict(data)
        print(prediction)
        print(prediction)
        print("predictied successfully")
        return jsonify({'crop': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Bind to 0.0.0.0 to allow access from any IP
    app.run(debug=True, host='0.0.0.0', port=5000)
