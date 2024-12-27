from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
try:
    model = joblib.load("failure_prediction_model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: failure_prediction_model.pkl not found. Please ensure the file exists in the project directory.")
    exit()

# Define home route
@app.route("/")
def home():
    return "MAGSOLUTION AI Prediction API is active."

# Define prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse incoming JSON request
        data = request.get_json()
        # Extract features
        temperatura = data.get("temperatura")
        vibracion = data.get("vibracion")
        tiempo_operacion = data.get("tiempo_operacion")
        presion = data.get("presion")

        # Validate inputs
        if None in [temperatura, vibracion, tiempo_operacion, presion]:
            return jsonify({"error": "Missing one or more required parameters: temperatura, vibracion, tiempo_operacion, presion"}), 400

        # Prepare input for the model
        input_features = np.array([[temperatura, vibracion, tiempo_operacion, presion]])

        # Make prediction
        prediction = model.predict(input_features)[0]
        riesgo = "Alto" if prediction == 1 else "Bajo"

        # Return the prediction as JSON
        return jsonify({"riesgo": riesgo})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
