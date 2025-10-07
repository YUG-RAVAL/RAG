import logging
from flask import Blueprint, request, jsonify
from flask_login import LoginManager
from datetime import datetime, timezone
from . import db  # Import db from the main package instead of models
from .models import User
from .config import Config
from authlib.integrations.flask_client import OAuth
from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, jwt_required, get_jwt_identity, get_jwt, set_access_cookies, unset_jwt_cookies
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
auth_bp = Blueprint('auth', __name__)

# Initialize login manager
login_manager = LoginManager()
login_manager.login_view = 'auth.login'

oauth = OAuth()
jwt_manager = JWTManager()

google = oauth.register(
    "google",
    client_id=Config.GOOGLE_CLIENT_ID,
    client_secret=Config.GOOGLE_CLIENT_SECRET,
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    access_token_url="https://oauth2.googleapis.com/token",
    client_kwargs={"scope": "openid email profile"},
)

TOKEN_BLOCKLIST = set()

def is_token_revoked(decoded_jwt):
    """Check if the token is in the blocklist."""
    return decoded_jwt["jti"] in TOKEN_BLOCKLIST

@jwt_manager.token_in_blocklist_loader
def check_if_token_in_blocklist(jwt_header, jwt_payload):
    """JWT Extended method to verify if a token is revoked."""
    jti = jwt_payload.get("jti")
    if jti in TOKEN_BLOCKLIST:
        logger.info(f"Token with JTI {jti} is blacklisted.")
        return True
    return False

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    return User.query.get(int(user_id))


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    token_id = data.get("user_google_token")
    try:
        headers = {
            "Authorization": f"Bearer {token_id}",
        }
        response = requests.get(Config.VERIFY_USER_URL, headers=headers)
        if response.status_code != 200:
            return jsonify({"error": "Failed to verify token with Google"}), 401
        user_info = response.json()
        # Verify Google token
    except Exception as e:
        logger.error(f"Error verifying Google token: {str(e)}")
        return jsonify({"error": "Invalid authentication"}), 401

    if not user_info:
        return jsonify({"error": "Invalid authentication"}), 401

    user = User.query.filter_by(email=user_info["email"]).first()
    if not user:
        user = User(email=user_info["email"], name=user_info["name"], profile_pic=user_info.get("picture"), last_login=datetime.now(timezone.utc))
        db.session.add(user)
        db.session.commit()
    else:
        user.last_login = datetime.now(timezone.utc)
        db.session.commit()
        
    # Create tokens
    access_token = create_access_token(identity=str(user.id), fresh=True)
    refresh_token = create_refresh_token(identity=str(user.id))
    response_store = jsonify({"msg": "login successful"})
    set_access_cookies(response_store, access_token)
    return jsonify({
        "access_token": access_token,
        "refresh_token": refresh_token
    })

@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    jti = get_jwt()["jti"]  # Get token identifier
    TOKEN_BLOCKLIST.add(jti)  # Add token to blocklist
    logger.info(f"jti: {jti}")
    response = jsonify({"message": "Logged out successfully"})
    unset_jwt_cookies(response)
    return response

@auth_bp.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh():
    """Refresh the JWT access token."""
    user_id = get_jwt_identity()
    new_access_token = create_access_token(identity=str(user_id), fresh=False)
    return jsonify({"access_token": new_access_token})

@auth_bp.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    """Example of a protected route requiring JWT."""
    user_id = get_jwt_identity()
    return jsonify({"message": f"Hello User {user_id}, you are authenticated!"})
