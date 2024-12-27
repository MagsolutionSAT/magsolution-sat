from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Repuesto(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(80), nullable=False)
    cantidad = db.Column(db.Integer, nullable=False)
