from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime, timedelta
import jwt
import os
import numpy as np
import joblib

# Flask application setup
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:your_password@localhost:3306/magnesitas_navarras'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key_here'

db = SQLAlchemy(app)

# Load AI Model
PREDICTIVE_MODEL_PATH = 'failure_prediction_model.pkl'
if os.path.exists(PREDICTIVE_MODEL_PATH):
    predictive_model = joblib.load(PREDICTIVE_MODEL_PATH)
else:
    predictive_model = None

# JWT Token Decorator
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
    role = db.Column(db.String(20), nullable=False)  # 'technician', 'sat'

class Equipment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    vibration = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), nullable=False)  # 'normal', 'warning', 'critical'
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

# API Routes

@app.route('/')
def home():
    return "Welcome to Magnesitas Navarras AI Innovation!"

# User Registration
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    if not data or not all(k in data for k in ('username', 'password', 'role')):
        return jsonify({'message': 'Invalid data!'}), 400
    hashed_password = generate_password_hash(data['password'], method='sha256')
    new_user = User(username=data['username'], password=hashed_password, role=data['role'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully!'})

# User Login
@app.route('/login', methods=['POST'])
def login():
    data = request.json
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
    result = [{'id': eq.id, 'name': eq.name, 'temperature': eq.temperature, 'vibration': eq.vibration, 'status': eq.status, 'last_updated': eq.last_updated.isoformat()} for eq in equipment]
    return jsonify(result)

@app.route('/equipment/update', methods=['POST'])
@token_required
def update_equipment(current_user):
    if current_user.role != 'technician':
        return jsonify({'message': 'Access denied!'}), 403
    data = request.json
    equipment = Equipment.query.get(data['id'])
    if not equipment:
        return jsonify({'message': 'Equipment not found!'}), 404
    equipment.status = data['status']
    equipment.temperature = data['temperature']
    equipment.vibration = data['vibration']
    equipment.last_updated = datetime.utcnow()
    db.session.commit()
    socketio.emit('equipment_update', {
        'id': equipment.id,
        'name': equipment.name,
        'status': equipment.status,
        'temperature': equipment.temperature,
        'vibration': equipment.vibration,
        'last_updated': equipment.last_updated.isoformat()
    })
    return jsonify({'message': 'Equipment updated successfully!'})

@app.route('/equipment/predict', methods=['POST'])
@token_required
def predict_equipment_failure(current_user):
    if not predictive_model:
        return jsonify({'message': 'Predictive model not loaded!'}), 500
    data = request.json
    features = np.array([[data['temperature'], data['vibration']]])
    prediction = predictive_model.predict(features)[0]
    risk = "high" if prediction > 0.5 else "low"
    return jsonify({'risk': risk, 'probability': float(prediction)})

# WebSocket Event Example
@socketio.on('connect')
def on_connect():
    print('Client connected')

@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')

# Main
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
