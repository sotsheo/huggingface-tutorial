from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTExtended
from flask_migrate import Migrate
import os

# Initialize extensions
db = SQLAlchemy()
jwt = JWTExtended()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object('app.config.Config')
    
    # Initialize extensions
    db.init_app(app)
    jwt.init_app(app)
    migrate.init_app(app, db)
    
    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.news import news_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(news_bp, url_prefix='/api')
    
    return app
