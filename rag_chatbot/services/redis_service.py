import logging
import json
import redis
from typing import Dict, List, Optional, Any
from ..config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection singleton
_redis_client = None

def init_redis() -> redis.Redis:
    """
    Initialize Redis connection using application configuration
    
    Returns:
        Redis client instance
    """
    global _redis_client
    
    # If Redis client is already initialized, return it
    if _redis_client is not None:
        logger.debug("Redis client already initialized")
        return _redis_client
    
    try:
        # Prepare connection parameters
        connection_params = {
            'host': Config.REDIS_HOST,
            'port': Config.REDIS_PORT,
            'db': Config.REDIS_DB,
            'decode_responses': True
        }
        
        # Add password if configured
        if hasattr(Config, 'REDIS_PASSWORD') and Config.REDIS_PASSWORD:
            logger.info(f"Connecting to Redis with password at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
            connection_params['password'] = Config.REDIS_PASSWORD
        else:
            logger.info(f"Connecting to Redis without password at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        
        # Create Redis client
        _redis_client = redis.Redis(**connection_params)
        # Verify connection
        _redis_client.ping()
        logger.info(f"Redis connection established at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        return _redis_client
            
    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        logger.error("Please check your Redis configuration in .env file or config.py")
        logger.error("Ensure Redis server is running on the configured host and port")
        logger.error(f"Current config - Host: {Config.REDIS_HOST}, Port: {Config.REDIS_PORT}, DB: {Config.REDIS_DB}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error connecting to Redis: {str(e)}")
        return None

def get_redis_client() -> Optional[redis.Redis]:
    """
    Get Redis client instance, initializing it if needed
    
    Returns:
        Redis client instance or None if initialization fails
    """
    global _redis_client
    
    if _redis_client is None:
        logger.info("Redis client not initialized. Initializing now.")
        return init_redis()
    
    # Check if the connection is still alive
    try:
        _redis_client.ping()
        return _redis_client
    except (redis.ConnectionError, redis.ResponseError):
        logger.warning("Redis connection lost. Attempting to reconnect.")
        _redis_client = None
        return init_redis()

def store_chat_history(session_id: str, chat_history: List[Dict[str, Any]]) -> bool:
    """
    Store chat history in Redis
    
    Args:
        session_id: Unique session ID
        chat_history: List of chat messages
        
    Returns:
        True if successful, False otherwise
    """
    client = get_redis_client()
    if not client:
        logger.error("Redis client is not available")
        return False
    
    try:
        # Set chat history with expiration (24 hours = 86400 seconds)
        # This is a fallback in case disconnect event doesn't fire
        key = f"chat:history:{session_id}"
        client.set(key, json.dumps(chat_history), ex=86400)
        logger.info(f"Chat history stored for session {session_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to store chat history: {str(e)}")
        return False

def get_chat_history(session_id: str) -> List[Dict[str, Any]]:
    """
    Get chat history from Redis
    
    Args:
        session_id: Unique session ID
        
    Returns:
        List of chat messages or empty list if not found
    """
    client = get_redis_client()
    if not client:
        logger.error("Redis client is not available")
        return []
    
    try:
        key = f"chat:history:{session_id}"
        chat_data = client.get(key)
        if chat_data:
            return json.loads(chat_data)
        logger.info(f"No chat history found for session {session_id}")
        return []
    except Exception as e:
        logger.error(f"Failed to get chat history: {str(e)}")
        return []

def delete_chat_history(session_id: str) -> bool:
    """
    Delete chat history from Redis
    
    Args:
        session_id: Unique session ID
        
    Returns:
        True if successful, False otherwise
    """
    client = get_redis_client()
    if not client:
        logger.error("Redis client is not available")
        return False
    
    try:
        key = f"chat_history:{session_id}"
        client.delete(key)
        logger.info(f"Chat history deleted for session: {session_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting chat history for session {session_id}: {str(e)}")
        return False

def store_vectorstore_type(session_id: str, vectorstore_type: str) -> bool:
    """
    Store vectorstore type for a session in Redis
    
    Args:
        session_id: Unique session ID
        vectorstore_type: Type of vectorstore ('knowledge_base' or 'sonu')
        
    Returns:
        True if successful, False otherwise
    """
    client = get_redis_client()
    if not client:
        logger.error("Redis client is not available")
        return False
    
    try:
        key = f"vectorstore_type:{session_id}"
        # Store with expiration (24 hours = 86400 seconds)
        client.setex(key, 86400, vectorstore_type)
        logger.info(f"Vectorstore type '{vectorstore_type}' stored for session: {session_id}")
        return True
    except Exception as e:
        logger.error(f"Error storing vectorstore type for session {session_id}: {str(e)}")
        return False

def get_vectorstore_type(session_id: str, default: str = "knowledge_base") -> str:
    """
    Get vectorstore type for a session from Redis
    
    Args:
        session_id: Unique session ID
        default: Default vectorstore type if not found
        
    Returns:
        Vectorstore type as string
    """
    client = get_redis_client()
    if not client:
        logger.error("Redis client is not available")
        return default
    
    try:
        key = f"vectorstore_type:{session_id}"
        vectorstore_type = client.get(key)
        
        if vectorstore_type:
            logger.info(f"Retrieved vectorstore type '{vectorstore_type}' for session: {session_id}")
            return vectorstore_type
        else:
            logger.info(f"No vectorstore type found for session {session_id}, using default: {default}")
            return default
    except Exception as e:
        logger.error(f"Error retrieving vectorstore type for session {session_id}: {str(e)}")
        return default

def delete_vectorstore_type(session_id: str) -> bool:
    """
    Delete vectorstore type for a session from Redis
    
    Args:
        session_id: Unique session ID
        
    Returns:
        True if successful, False otherwise
    """
    client = get_redis_client()
    if not client:
        logger.error("Redis client is not available")
        return False
    
    try:
        key = f"vectorstore_type:{session_id}"
        client.delete(key)
        logger.info(f"Vectorstore type deleted for session: {session_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting vectorstore type for session {session_id}: {str(e)}")
        return False 