from flask_login import UserMixin
import datetime
import enum

# Import the db instance from the main package
from . import db
import logging
from sqlalchemy.exc import SQLAlchemyError

class User(db.Model, UserMixin):
    """User model for authentication and session management"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    name = db.Column(db.String(255), nullable=True)
    profile_pic = db.Column(db.String(255), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    last_login = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<User {self.email}>'
    
    def to_dict(self):
        """Convert user object to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'profile_pic': self.profile_pic,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

class Provider(db.Model):
    __tablename__ = 'providers'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    models = db.relationship('AIModel', backref='provider', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Provider {self.name}>'
    
    def to_dict(self):
        """Convert provider object to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'models_count': len(self.models)
        }

class AIModel(db.Model):
    __tablename__ = 'ai_models'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    provider_id = db.Column(db.Integer, db.ForeignKey('providers.id'), nullable=False)
    is_selected = db.Column(db.Boolean, default=False, nullable=False)
    
    def __repr__(self):
        return f'<AIModel {self.name} ({self.provider.name})>'
    
    def to_dict(self):
        """Convert model object to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'provider_id': self.provider_id,
            'provider_name': self.provider.name if self.provider else None,
            'is_selected': self.is_selected
        }


def insert_default_data():
    # Check if the Provider table is empty
    logger = logging.getLogger(__name__)
    
    try:
        # Check if the Provider table is empty
        if not Provider.query.first():
            logger.info("Initializing database with default providers and models")
            
            # Define default providers
            default_providers = [
                Provider(name='OpenAI'),
                Provider(name='Google'),
                Provider(name='Anthropic')
            ]
            
            # Add all providers in a single operation
            db.session.add_all(default_providers)
            db.session.commit()
            logger.debug("Default providers added successfully")
            
            # Fetch providers by name for setting up models
            openai = Provider.query.filter_by(name='OpenAI').first()
            google = Provider.query.filter_by(name='Google').first()
            anthropic = Provider.query.filter_by(name='Anthropic').first()
            
            # Only one model should be selected by default
            # Ensure we don't have multiple models selected
            default_models = [
                AIModel(name='gpt-4o-mini', provider_id=openai.id, is_selected=False),
                AIModel(name='gemini-2.0-flash', provider_id=google.id, is_selected=False),
                AIModel(name='claude-3-haiku-20240307', provider_id=anthropic.id, is_selected=False),
                AIModel(name='gpt-4o', provider_id=openai.id, is_selected=True)
            ]
            
            # Add all models in a single operation
            db.session.add_all(default_models)
            db.session.commit()
            logger.info("Default data inserted successfully")
        else:
            logger.info("Default data already exists, skipping initialization")
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Database error during default data initialization: {str(e)}")
        raise
    except Exception as e:
        db.session.rollback()
        logger.error(f"Unexpected error during default data initialization: {str(e)}")
        raise


class FeedbackType(enum.Enum):
    """Enum for different types of feedback"""
    POSITIVE = "positive"
    NEGATIVE = "negative"


class Feedback(db.Model):
    """Model for storing user feedback on chat responses"""
    __tablename__ = 'feedback'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    user_query = db.Column(db.Text, nullable=False)
    assistant_response = db.Column(db.Text, nullable=False)
    feedback_type = db.Column(db.Enum(FeedbackType), nullable=False)
    feedback_text = db.Column(db.Text, nullable=True)  # Optional text feedback, required for negative feedback
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    model_name = db.Column(db.String(100), nullable=True)
    provider = db.Column(db.String(50), nullable=True)
    conversation_history = db.Column(db.JSON, nullable=True)  # Last 5 conversation exchanges as JSON

    # Relationship with User model
    user = db.relationship('User', backref=db.backref('feedback', lazy=True))

    def __repr__(self):
        return f'<Feedback {self.id} - {self.feedback_type.value}>'

    def to_dict(self):
        """Convert feedback object to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'user_query': self.user_query,
            'assistant_response': self.assistant_response,
            'feedback_type': self.feedback_type.value,
            'feedback_text': self.feedback_text,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'model_name': self.model_name,
            'provider': self.provider,
            'conversation_history': self.conversation_history
        }