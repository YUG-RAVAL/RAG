from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_caching import Cache
from .config import Config, mask_sensitive_string
import logging
import os
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize extensions
db = SQLAlchemy()
jwt = JWTManager()
cache = Cache()

def create_app():
    # Log environment variables (without revealing sensitive values)
    logger.info("Environment variables loaded")
    logger.info(f"OPENAI_API_KEY status: {mask_sensitive_string(os.getenv('OPENAI_API_KEY'))}")
    logger.info(f"FLASK_APP: {os.getenv('FLASK_APP')}")
    logger.info(f"FLASK_ENV: {os.getenv('FLASK_ENV')}")
    
    # Initialize application directories
    Config.init_app()
    logger.info(f"Upload folder: {Config.UPLOAD_FOLDER}")
    logger.info(f"Vector store path: {Config.VECTORSTORE_PATH}")
    
    # Create Flask application
    app = Flask(__name__)
    
    # Configure app settings
    app.config.from_object(Config)
    
    # Enable CORS
    CORS(app)
    
    # Initialize extensions with app
    db.init_app(app)
    jwt.init_app(app)
    cache.init_app(app)
    logger.info("Cache initialized with Redis backend")

    # Import routes after initializing extensions to avoid circular imports
    from .websocket import configure_websocket
    from .routes import main
    from .auth import auth_bp, login_manager, oauth
    from .services.vectorstore import ensure_sonu_collection_exists
    
    # Initialize login manager
    login_manager.init_app(app)
    oauth.init_app(app)
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*")
    configure_websocket(socketio)
    
    # Register blueprints
    app.register_blueprint(main)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    providers_module = importlib.import_module("rag_chatbot.services.providers")
    load_selected_provider = providers_module.load_selected_provider
    providers_module = importlib.import_module("rag_chatbot.models")
    insert_default_data = providers_module.insert_default_data
    # Create database tables
    with app.app_context():
        db.create_all()
        insert_default_data()
        load_selected_provider()
        
        # Ensure sonu collection exists in vectorstore
        logger.info("Checking if sonu collection exists in vectorstore...")
        sonu_exists = ensure_sonu_collection_exists()
        if sonu_exists:
            logger.info("Sonu collection verified successfully")
        else:
            logger.warning("Failed to ensure sonu collection exists")

    logger.info("Application initialized successfully")
    
    return app, socketio