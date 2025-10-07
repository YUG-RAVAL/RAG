import os
import logging
from datetime import timedelta
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mask_sensitive_string(text, visible_chars=4):
    """
    Masks a sensitive string, showing only the first few characters.
    
    Args:
        text (str): The string to mask
        visible_chars (int): Number of characters to leave visible
        
    Returns:
        str: Masked string or indication of absence
    """
    if not text:
        return "missing"
    
    # Always return a generic message instead of the actual value or portion of it
    return "available (value masked)"

class Config:
    load_dotenv()
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # File storage settings
    UPLOAD_FOLDER = os.path.abspath("./uploads")
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB max file size (per file)
    ALLOWED_EXTENSIONS = {'pdf', 'txt', 'csv', 'docx', 'doc', 'zip', 'md', 'markdown'}
    
    # Vector database settings
    VECTORSTORE_PATH = os.path.abspath("./chroma_db")
    COLLECTION = os.getenv('COLLECTION')
    
    # Multiple vector store collections
    KNOWLEDGE_BASE_COLLECTION = os.getenv('KNOWLEDGE_BASE_COLLECTION', 'knowledge_base')
    SAMSUNG_COLLECTION = os.getenv('SAMSUNG_COLLECTION', 'samsung_manuals')
    SONU_COLLECTION = os.getenv('SONU_COLLECTION', 'sonu_qa_v4')
    
    # Default collection (backwards compatibility)
    if not COLLECTION:
        COLLECTION = KNOWLEDGE_BASE_COLLECTION
    
    # Database settings
    DB_USER = os.getenv('DB_USER', 'postgresql')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgresql')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_NAME = os.getenv('DB_NAME', 'user')
    DB_PORT = os.getenv('DB_PORT', '5432')
    
    # Build database URI
    SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Authentication settings
    SECRET_KEY = os.getenv("SECRET_KEY")
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    VERIFY_USER_URL = os.getenv("VERIFY_USER_URL")
    
    # JWT settings
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=1)
    JWT_TOKEN_LOCATION = ["headers", "cookies", "json"]
    JWT_COOKIE_SECURE = os.getenv('JWT_COOKIE_SECURE', 'False').lower() == 'true'
    JWT_COOKIE_CSRF_PROTECT = True
    JWT_CSRF_IN_COOKIES = True
    JWT_COOKIE_SAMESITE = "Lax"
    
    # Chunking settings
    CHUNK_SIZE = 3000
    CHUNK_OVERLAP = 200
    
    # Redis and Caching settings
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', 'redis')  # Default to 'redis' to match Docker setup
    
    # Flask-Caching configuration
    # Try to use Redis but fall back to SimpleCache if Redis is unavailable
    try:
        import redis
        # Test Redis connection
        r = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            socket_timeout=1  # Short timeout for connection test
        )
        r.ping()  # Will raise exception if Redis is unavailable
        
        # Redis is available, use it
        CACHE_TYPE = 'RedisCache'
        CACHE_REDIS_HOST = REDIS_HOST
        CACHE_REDIS_PORT = REDIS_PORT
        CACHE_REDIS_DB = REDIS_DB
        CACHE_REDIS_PASSWORD = REDIS_PASSWORD
        logger.info("Using Redis for caching")
    except:
        # Redis is unavailable, fall back to in-memory cache
        CACHE_TYPE = 'SimpleCache'
        logger.warning("Redis unavailable, falling back to SimpleCache")
    
    CACHE_DEFAULT_TIMEOUT = 3600  # Default cache timeout in seconds (1 hour)
    CACHE_KEY_PREFIX = 'rag_chatbot_'
    
    # Ensure directories exist
    @classmethod
    def init_app(cls):
        """Initialize application directories"""
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.VECTORSTORE_PATH, exist_ok=True)
        
        # Log configuration with masked sensitive information
        logger.info(f"OPENAI_API_KEY status: {mask_sensitive_string(cls.OPENAI_API_KEY)}")
        logger.info(f"UPLOAD_FOLDER: {cls.UPLOAD_FOLDER}")
        logger.info(f"VECTORSTORE_PATH: {cls.VECTORSTORE_PATH}")
        logger.info(f"COLLECTION: {cls.COLLECTION}")
        
        # Log database connection without credentials
        db_uri_masked = f"postgresql://{cls.DB_USER}:****@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
        logger.info(f"Database connection: {db_uri_masked}")
        logger.info(f"Redis cache configured at {cls.REDIS_HOST}:{cls.REDIS_PORT}")
        
        # If API key is missing, try to get it from environment
        if not cls.OPENAI_API_KEY:
            cls.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            logger.info(f"Tried to get OPENAI_API_KEY from environment: {mask_sensitive_string(cls.OPENAI_API_KEY)}")
            
            # If still missing, log a warning
            if not cls.OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY is missing. Some features may not work correctly.")