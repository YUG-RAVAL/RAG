import os
from ..config import Config

def get_uploaded_files():
    """
    Get a list of all files in the upload folder
    
    Returns:
        list: List of filenames in the upload folder
    """
    upload_folder = Config.UPLOAD_FOLDER
    if not os.path.exists(upload_folder):
        return []

    return os.listdir(upload_folder)


def format_file_size(size_in_bytes):
    """
    Format file size in human-readable format
    
    Args:
        size_in_bytes (int): File size in bytes
        
    Returns:
        str: Formatted file size
    """
    # Convert to KB, MB, GB as appropriate
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024 * 1024 * 1024:
        return f"{size_in_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_in_bytes / (1024 * 1024 * 1024):.2f} GB"
