import pickle
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Constants
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

FEATURE_NAMES = [
    'Engine rpm', 'Lub oil pressure', 'Fuel pressure', 
    'Coolant pressure', 'lub oil temp', 'Coolant temp'
]

# Load models and scaler globally
model = None
scaler = None

def load_models():
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("Models and Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

load_models()

@app.route('/')
def home():
    """Render the main prediction hub page."""
    return render_template('index.html')



@app.route('/result', methods=['POST'])
def result():
    """Handle form submission and render the result page."""
    if not model or not scaler:
        return render_template('result.html', error="Models not loaded. Please train the models first.")

    try:
        data = request.form
        
        # Prepare data for prediction
        input_data = pd.DataFrame([{
            'Engine rpm': float(data.get('rpm', 0)),
            'Lub oil pressure': float(data.get('lub_oil_pressure', 0)),
            'Fuel pressure': float(data.get('fuel_pressure', 0)),
            'Coolant pressure': float(data.get('coolant_pressure', 0)),
            'lub oil temp': float(data.get('lub_oil_temp', 0)),
            'Coolant temp': float(data.get('coolant_temp', 0))
        }])

        # Scale features
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = int(model.predict(input_scaled)[0])
        prediction_proba = model.predict_proba(input_scaled)[0].tolist()

        return render_template('result.html', prediction=prediction, confidence=prediction_proba)

    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint to predict engine health and lifespan."""
    if not model or not scaler:
        return jsonify({"error": "Models not loaded. Please train the models first."}), 500

    try:
        data = request.json
        
        # Prepare data for prediction
        input_data = pd.DataFrame([{
            'Engine rpm': float(data.get('rpm', 0)),
            'Lub oil pressure': float(data.get('lub_oil_pressure', 0)),
            'Fuel pressure': float(data.get('fuel_pressure', 0)),
            'Coolant pressure': float(data.get('coolant_pressure', 0)),
            'lub oil temp': float(data.get('lub_oil_temp', 0)),
            'Coolant temp': float(data.get('coolant_temp', 0))
        }])

        # Scale features
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = int(model.predict(input_scaled)[0])
        prediction_proba = model.predict_proba(input_scaled)[0].tolist()

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": prediction_proba
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Load models before running the server if not already loaded
    if not model:
        load_models()
    app.run(debug=True, port=5000)
