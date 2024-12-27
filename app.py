# MAGSOLUTION AI - COMPLETE SOLUTION FOR MACHINE LEARNING AND SAT MANAGEMENT

## **A. DATA COLLECTION AND SIMULATION**
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from flask import Flask, request, jsonify, render_template

# Step 1: Simulate data collection for machine performance
def generate_simulated_data(filename="machine_data.csv", n_samples=1000):
    np.random.seed(42)
    data = {
        "temperatura": np.random.uniform(100, 600, n_samples),
        "vibracion": np.random.uniform(5, 25, n_samples),
        "tiempo_operacion": np.random.uniform(20, 100, n_samples),
        "presion": np.random.uniform(50, 150, n_samples),
        "fallo": np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Simulated data saved to {filename}")

# Call this function to generate data for training
generate_simulated_data()

## **B. MACHINE LEARNING MODEL TRAINING**
def train_ml_model(data_path="machine_data.csv", model_output="failure_prediction_model.pkl"):
    # Load data
    data = pd.read_csv(data_path)
    X = data.drop(columns=["fallo"])
    y = data["fallo"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save model
    joblib.dump(model, model_output)
    print(f"Model saved to {model_output}")

# Call this function to train and save the model
train_ml_model()

## **C. FLASK API DEVELOPMENT**
app = Flask(__name__)

# Load the trained ML model
MODEL_PATH = "failure_prediction_model.pkl"
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return "<h1>MAGSOLUTION AI Prediction API</h1>"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        features = [
            data["temperatura"],
            data["vibracion"],
            data["tiempo_operacion"],
            data["presion"]
        ]
        prediction = model.predict([features])[0]
        risk = "Alto" if prediction == 1 else "Bajo"
        return jsonify({"riesgo": risk})
    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"})

@app.route("/train", methods=["POST"])
def retrain_model():
    # Retrain the model using new data
    generate_simulated_data("new_data.csv")
    train_ml_model("new_data.csv", MODEL_PATH)
    return jsonify({"message": "Model retrained successfully."})

## **D. USER INTERFACE FOR SAT**
@app.route("/ui")
def user_interface():
    return render_template("index.html")

# HTML Template (index.html)
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAGSOLUTION AI</title>
</head>
<body>
    <h1>MAGSOLUTION SAT - Prediction Portal</h1>
    <form action="/predict" method="post">
        <label>Temperatura:</label><br>
        <input type="number" name="temperatura" required><br>
        <label>Vibración:</label><br>
        <input type="number" name="vibracion" required><br>
        <label>Tiempo de Operación:</label><br>
        <input type="number" name="tiempo_operacion" required><br>
        <label>Presión:</label><br>
        <input type="number" name="presion" required><br><br>
        <button type="submit">Predecir</button>
    </form>
</body>
</html>"""

# Write the template to the templates directory
os.makedirs("templates", exist_ok=True)
with open("templates/index.html", "w") as f:
    f.write(HTML_TEMPLATE)

if __name__ == "__main__":
    with app.app_context():
        from flask_sqlalchemy import SQLAlchemy
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///repuestos.db'
        db = SQLAlchemy(app)
        db.create_all()
    app.run(debug=True)
