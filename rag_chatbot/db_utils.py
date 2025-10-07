"""
Database utility functions to handle common operations with proper context management.
"""
import logging
from contextlib import contextmanager
from flask import current_app
from . import db

logger = logging.getLogger(__name__)

@contextmanager
def session_scope():
    """
    Provide a transactional scope around a series of operations.
    
    This context manager ensures that:
    1. The session is properly managed
    2. Exceptions are properly caught and logged
    3. The session is always properly closed or rolled back
    4. Application context is properly checked
    
    Usage:
        with session_scope() as session:
            obj = Model()
            session.add(obj)
            # No need to call commit - it's done automatically if no exceptions
    """
    session = db.session
    try:
        # Check if we're in application context
        if not current_app:
            raise RuntimeError(
                "Working outside of application context. "
                "Make sure to use this within a Flask route or with app.app_context()."
            )
        yield session
        session.commit()
    except Exception as e:
        logger.exception(f"Error in database transaction: {str(e)}")
        session.rollback()
        raise
    finally:
        session.close()