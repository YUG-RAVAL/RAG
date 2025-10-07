import logging
import uuid
from flask import request
from flask_socketio import emit, disconnect
from .services.vectorstore import get_vectorstore, get_vectorstore_by_type
from .services.chat import get_answer
from .services.providers import get_selected_provider
from .services.redis_service import (
    init_redis, 
    store_chat_history, 
    get_chat_history, 
    delete_chat_history
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to map socket IDs to session IDs
# Moved outside the function to be globally accessible
socket_sessions = {}

def configure_websocket(socketio):
    # Initialize Redis connection
    init_redis()
    
    @socketio.on('connect', namespace='/chat')
    def handle_connect():
        """
        Handle the connect event
        
        Generate a unique session ID and store it for the socket connection
        """
        session_id = str(uuid.uuid4())
        socket_id = request.sid
        socket_sessions[socket_id] = session_id
        logger.info(f"Client connected. Socket ID: {socket_id}, Session ID: {session_id}")
        emit('session_created', {'session_id': session_id})

    @socketio.on('initialize_chat', namespace='/chat')
    def handle_initialize_chat(data):
        """
        Handle the initialize_chat event
        
        Args:
            data (dict): The data sent with the event
        """
        socket_id = request.sid
        session_id = socket_sessions.get(socket_id)
        
        if not session_id:
            session_id = str(uuid.uuid4())
            socket_sessions[socket_id] = session_id
            logger.info(f"Created new session ID: {session_id} for socket: {socket_id}")
            emit('session_created', {'session_id': session_id})
            
        logger.info(f"Initializing chat for session: {session_id}")
        emit('chat_initialized', {'message': 'Welcome to the chat!', 'session_id': session_id})
        logger.info("Chat initialized")

    @socketio.on('send_message', namespace='/chat')
    def handle_send_message(data):
        """
        Handle the send_message event
        
        Args:
            data (dict): The data sent with the event
        """
        socket_id = request.sid
        session_id = socket_sessions.get(socket_id)
        
        if not session_id:
            error_msg = "Error: No valid session found. Please reconnect."
            logger.error(error_msg)
            emit('receive_message', {'error': error_msg})
            return
            
        user_message = data.get('message', '')
        responseType = data.get('responseType', '')
        vectorstore_type = data.get('vectorstore_type', 'knowledge_base')  # Default to knowledge_base if not provided
        
        # Validate vectorstore_type
        if vectorstore_type.lower() not in ['knowledge_base', 'sonu']:
            logger.warning(f"Invalid vectorstore_type '{vectorstore_type}', defaulting to 'knowledge_base'")
            vectorstore_type = 'knowledge_base'
        else:
            vectorstore_type = vectorstore_type.lower()
            
        logger.info(f"Received message for session {session_id}: {user_message}")
        logger.info(f"Using vectorstore type: {vectorstore_type}")
        
        # Check if vectorstore is initialized for the specified type
        vectorstore = get_vectorstore_by_type(vectorstore_type)
        if vectorstore is None:
            error_msg = f"Error: {vectorstore_type.capitalize()} vectorstore not initialized."
            logger.error(error_msg)
            emit('receive_message', {'error': error_msg})
            return
        
        try:
            selected_provider = get_selected_provider()
            if not selected_provider:
                error_msg = "Error: No provider selected."
                logger.error(error_msg)
                emit('receive_message', {'error': error_msg})
                return
                
            model_name = selected_provider['model_name']
            provider = selected_provider['provider'].lower()
            logger.info(f"Using provider: {selected_provider['provider']}, model: {selected_provider['model_name']}")
            
            # Get chat history from Redis for this session
            chat_history = get_chat_history(session_id)
            
            # Get response using RAG
            logger.info(f"Getting answer for session {session_id}")
            logger.info(f":::::::::::::Using vectorstore type::::::::: {vectorstore_type}")
            response = get_answer(user_message, responseType, chat_history, model_name, provider, vectorstore_type)
            
            # Update chat history
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": response["answer"]})
            
            # Store updated chat history in Redis
            store_chat_history(session_id, chat_history)
            
            logger.info(f"Answer received for session {session_id}: {response['answer'][:50]}...")
            emit('receive_message', {
                'message': response['answer'], 
                'sources': response['sources']
            })
            logger.info(f"Response sent to client for session {session_id}")
        except Exception as e:
            error_msg = f"Error processing your request: {str(e)}"
            logger.error(f"Error in handle_send_message for session {session_id}: {str(e)}")
            emit('receive_message', {'error': error_msg})

    @socketio.on('disconnect', namespace='/chat')
    def handle_disconnect():
        """
        Handle the disconnect event
        
        Remove the session ID from the dictionary and delete chat history
        """
        socket_id = request.sid
        session_id = socket_sessions.pop(socket_id, None)
        
        if session_id:
            # Delete chat history from Redis
            delete_chat_history(session_id)
            logger.info(f"Client disconnected. Session {session_id} data removed.")
        else:
            logger.info(f"Client disconnected. No session found for socket {socket_id}.")


def get_chat_history_by_socket_id(socket_id: str, limit: int = 5):
    """
    Get chat history for a given socket ID, limited to the previous N user-response pairs,
    excluding the current query and its response.
    
    Args:
        socket_id: The WebSocket connection ID
        limit: Number of user-response pairs to return (default: 5, resulting in 10 total messages)
        
    Returns:
        List of chat messages (excluding the current query and response) or empty list if not found
    """
    session_id = socket_sessions.get(socket_id)
    
    if not session_id:
        logger.warning(f"No session found for socket ID: {socket_id}")
        return []
    
    logger.info(f"Retrieving chat history for socket_id: {socket_id}, session_id: {session_id}")
    
    # Get complete chat history from Redis
    chat_history = get_chat_history(session_id)
    
    # Total messages needed = limit * 2 (each pair has user message + assistant response)
    messages_needed = limit * 2
    
    # Skip the last two messages (current query and its response) if they exist
    if len(chat_history) >= 2:
        relevant_history = chat_history[:-2]
    else:
        relevant_history = chat_history
    
    # If we have more messages than needed, return only the last N pairs
    if len(relevant_history) > messages_needed:
        relevant_history = relevant_history[-messages_needed:]
    
    logger.info(f"Retrieved {len(relevant_history)} messages for session: {session_id} (excluding current query and response)")
    return relevant_history
