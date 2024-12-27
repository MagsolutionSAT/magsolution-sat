from flask import request, jsonify
from app import app, db
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta
from functools import wraps
import numpy as np
import joblib
import os

# Load AI Predictive Model
PREDICTIVE_MODEL_PATH = 'predictive_model.pkl'
if os.path.exists(PREDICTIVE_MODEL_PATH):
    try:
        predictive_model = joblib.load(PREDICTIVE_MODEL_PATH)
    except Exception as e:
        print(f"Failed to load predictive model: {e}")
        predictive_model = None
else:
    predictive_model = None

# JWT Authentication
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.filter_by(id=data['id']).first()
        except Exception as e:
            return jsonify({'message': 'Token is invalid!', 'error': str(e)}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'technician' or 'sat'

class Equipment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    vibration = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), nullable=False)  # 'normal', 'warning', 'critical'
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class CarbonSavings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    material_name = db.Column(db.String(100), nullable=False)
    energy_saved = db.Column(db.Float, nullable=False)  # in kWh
    co2_saved = db.Column(db.Float, nullable=False)  # in kg
    date_recorded = db.Column(db.DateTime, default=datetime.utcnow)

# Routes
@app.route('/')
def home():
    return "Welcome to Magnesitas Navarras AI Innovation!"

# User Authentication
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    if not data or not all(k in data for k in ('username', 'password', 'role')):
        return jsonify({'message': 'Invalid request data'}), 400

    hashed_password = generate_password_hash(data['password'], method='sha256')
    new_user = User(username=data['username'], password=hashed_password, role=data['role'])
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'message': 'User registered successfully'})
    except Exception as e:
        return jsonify({'message': 'Failed to register user', 'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    if not data or not all(k in data for k in ('username', 'password')):
        return jsonify({'message': 'Invalid request data'}), 400

    user = User.query.filter_by(username=data['username']).first()
    if not user or not check_password_hash(user.password, data['password']):
        return jsonify({'message': 'Invalid credentials!'}), 401

    token = jwt.encode({'id': user.id, 'exp': datetime.utcnow() + timedelta(hours=1)}, app.config['SECRET_KEY'], algorithm='HS256')
    return jsonify({'token': token})

# Equipment Management
@app.route('/equipment', methods=['GET'])
@token_required
def get_equipment(current_user):
    equipment = Equipment.query.all()
    output = [{'id': eq.id, 'name': eq.name, 'temperature': eq.temperature, 'vibration': eq.vibration, 'status': eq.status, 'last_updated': eq.last_updated} for eq in equipment]
    return jsonify(output)

@app.route('/equipment/predict', methods=['POST'])
@token_required
def predict_failure(current_user):
    if current_user.role != 'sat':
        return jsonify({'message': 'Access denied!'}), 403

    data = request.json
    if not data or not all(k in data for k in ('temperature', 'vibration')):
        return jsonify({'message': 'Invalid request data'}), 400

    temperature, vibration = data['temperature'], data['vibration']
    if predictive_model:
        features = np.array([[temperature, vibration]])
        prediction = predictive_model.predict(features)[0]
        risk = "high" if prediction > 0.5 else "low"
        return jsonify({'risk': risk, 'probability': round(prediction, 2)})
    return jsonify({'message': 'Predictive model not loaded'}), 500

# Sustainability Insights
@app.route('/carbon_savings', methods=['GET'])
@token_required
def get_carbon_savings(current_user):
    savings = CarbonSavings.query.all()
    output = [{'material_name': record.material_name, 'energy_saved': record.energy_saved, 'co2_saved': record.co2_saved, 'date_recorded': record.date_recorded} for record in savings]
    return jsonify(output)

@app.route('/carbon_savings/add', methods=['POST'])
@token_required
def add_carbon_savings(current_user):
    if current_user.role != 'sat':
        return jsonify({'message': 'Access denied!'}), 403

    data = request.json
    if not data or not all(k in data for k in ('material_name', 'energy_saved', 'co2_saved')):
        return jsonify({'message': 'Invalid request data'}), 400

    new_saving = CarbonSavings(material_name=data['material_name'], energy_saved=data['energy_saved'], co2_saved=data['co2_saved'])
    db.session.add(new_saving)
    db.session.commit()
    return jsonify({'message': 'Carbon savings recorded successfully!'})

# AI Model Training
@app.route('/train_model', methods=['POST'])
@token_required
def train_model(current_user):
    if current_user.role != 'sat':
        return jsonify({'message': 'Access denied!'}), 403

    return jsonify({'message': 'Training endpoint ready. Add model training logic here.'})
