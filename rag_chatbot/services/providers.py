from ..models import Provider, AIModel
from .. import db, cache
import logging
from sqlalchemy.orm.exc import NoResultFound #for catching no results

# Configure logging
logger = logging.getLogger(__name__)

# Cache key for the selected provider
SELECTED_PROVIDER_CACHE_KEY = 'selected_provider'

def load_selected_provider():
    """Load the selected provider from the database and store it in Redis cache."""
    try:
        selected_model = AIModel.query.filter_by(is_selected=True).first()
        if selected_model:
            provider = Provider.query.get(selected_model.provider_id)
            if provider:
                provider_data = {
                    "provider": provider.name,
                    "model_name": selected_model.name,
                    "provider_id": provider.id,
                    "model_id": selected_model.id
                }
                # Store in Redis cache
                cache.set(SELECTED_PROVIDER_CACHE_KEY, provider_data)
                logger.info(f"Loaded and cached selected provider: {provider.name}, model: {selected_model.name}")
                return provider_data
            else:
                logger.warning("Selected model has no associated provider")
                cache.delete(SELECTED_PROVIDER_CACHE_KEY)
        else:
            logger.warning("No selected model found in the database")
            cache.delete(SELECTED_PROVIDER_CACHE_KEY)
        return None
    except Exception as e:
        logger.error(f"Error loading selected provider: {e}")
        cache.delete(SELECTED_PROVIDER_CACHE_KEY)
        return None

def get_selected_provider():
    """Get the cached selected provider from Redis without querying the database."""
    provider_data = cache.get(SELECTED_PROVIDER_CACHE_KEY)
    
    # If not in cache, attempt to load from database
    if provider_data is None:
        logger.info("Provider not found in cache, loading from database")
        provider_data = load_selected_provider()
    
    return provider_data

def update_selected_provider(provider_name, model_name):
    """Update the selected provider in the database and Redis cache."""
    try:
        # Find provider and model
        provider = Provider.query.filter_by(name=provider_name).first()
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not found")

        model = AIModel.query.filter_by(name=model_name, provider_id=provider.id).first()
        if not model:
            raise ValueError(f"Model '{model_name}' not found for provider '{provider_name}'")

        # Update the database
        # Deselect all models
        AIModel.query.update({AIModel.is_selected: False})
        # Select the specified model
        model.is_selected = True
        db.session.commit()

        # Update the Redis cache
        provider_data = {
            "provider": provider_name,
            "model_name": model_name,
            "provider_id": provider.id,
            "model_id": model.id
        }
        cache.set(SELECTED_PROVIDER_CACHE_KEY, provider_data)
        logger.info(f"Selected provider updated: {provider_name}, model: {model_name}")
        
        return provider_data
    except Exception as e:
        logger.error(f"Error updating selected provider: {e}")
        db.session.rollback()
        # Invalidate cache on error
        cache.delete(SELECTED_PROVIDER_CACHE_KEY)
        raise