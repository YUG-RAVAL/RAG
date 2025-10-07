import os
import uuid
import datetime
import logging
import zipfile
import tempfile
from flask import Blueprint, request, jsonify, send_from_directory
from flask_jwt_extended import jwt_required, get_jwt_identity
from .services.file_processing import process_file
from .utils.helpers import get_uploaded_files, format_file_size
from .config import Config
from .services.vectorstore import get_vectorstore
from .models import Provider, AIModel, Feedback, FeedbackType
from . import cache, db
from .db_utils import session_scope
from .services.providers import update_selected_provider, get_selected_provider
from sqlalchemy.exc import SQLAlchemyError
from .websocket import get_chat_history_by_socket_id

main = Blueprint("main", __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@main.route("/upload", methods=["POST"])
@jwt_required()
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('file')
    if files[0].filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Use allowed extensions from Config
    allowed_extensions = Config.ALLOWED_EXTENSIONS
    max_file_size = Config.MAX_FILE_SIZE
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

    uploaded_files, file_metadata, rejected_files = [] , [], []

    # Ensure upload directory exists
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    for file in files:
        if file and allowed_file(file.filename):
            # Generate a unique filename to prevent overwriting
            original_filename = file.filename
            file_extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
            if file_extension == 'zip':
                # Handle zip file
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        zip_path = os.path.join(temp_dir, original_filename)
                        file.save(zip_path)

                        if os.path.getsize(zip_path) > max_file_size:
                            rejected_files.append({
                                "filename": original_filename,
                                "reason": f"File exceeds maximum size of {format_file_size(max_file_size)}"
                            })
                            continue

                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)

                        # Process each file in the extracted directory
                        for root, _, extracted_files in os.walk(temp_dir):
                            for extracted_file in extracted_files:
                                extracted_file_path = os.path.join(root, extracted_file)
                                # Process each extracted file
                                process_extracted_file(extracted_file_path, uploaded_files, file_metadata)
                except zipfile.BadZipfile:
                    rejected_files.append({"filename": original_filename, "reason": "Corrupt zip file."})
            else:
                base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename

                # Create a filename with format: original_name_uuid.extension
                unique_id = uuid.uuid4().hex[:8]  # Use shorter UUID for readability
                unique_filename = f"{base_name}_{unique_id}.{file_extension}" if file_extension else f"{base_name}_{unique_id}"

                filepath = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
                file.save(filepath)

                # Get file size and check if it exceeds limit
                file_size = os.path.getsize(filepath)
                if file_size > max_file_size:
                    # Remove the file if it exceeds size limit
                    os.remove(filepath)
                    rejected_files.append({
                        "filename": original_filename,
                        "reason": f"File exceeds maximum size of {format_file_size(max_file_size)}"
                    })
                    continue

                # Get current timestamp
                upload_time = datetime.datetime.now(datetime.timezone.utc)

                # Create metadata
                metadata = {
                    "original_filename": original_filename,
                    "stored_filename": unique_filename,
                    "file_extension": file_extension,
                    "file_size": file_size,
                    "formatted_file_size": format_file_size(file_size),
                    "upload_date": upload_time.strftime("%Y-%m-%d"),
                    "upload_time": upload_time.strftime("%H:%M:%S"),
                    "timestamp": upload_time.timestamp(),
                    "doc_id": unique_id  # For compatibility with vectorstore metadata
                }
                uploaded_files.append(filepath)
                file_metadata.append(metadata)
        else:
            rejected_files.append({
                "filename": file.filename,
                "reason": f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            })

    # If all files were rejected, return an error
    if not uploaded_files and rejected_files:
        return jsonify({
            "error": "All files were rejected",
            "rejected_files": rejected_files
        }), 400

    try:
        # Process files and pass metadata - note the change to handle the tuple return
        added_doc_ids, failed_files = process_file(uploaded_files, file_metadata)

        # Build response based on success/failure
        if failed_files or rejected_files:
            # Some files failed but some succeeded
            if added_doc_ids:
                response = {
                    "message": f"Processed {len(added_doc_ids)} file(s) successfully",
                    "successful_files": len(added_doc_ids)
                }

                if failed_files:
                    response["failed_files"] = failed_files

                if rejected_files:
                    response["rejected_files"] = rejected_files

                return jsonify(response), 207  # 207 Multi-Status
            else:
                # All files failed
                response = {"error": "No files were processed successfully"}

                if failed_files:
                    response["failed_files"] = failed_files

                if rejected_files:
                    response["rejected_files"] = rejected_files

                return jsonify(response), 500
        else:
            # All files succeeded
            return jsonify({
                "message": f"All {len(added_doc_ids)} file(s) processed successfully"
            }), 200
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        return jsonify({"error": str(e)}), 500


def process_extracted_file(file_path, uploaded_files, file_metadata):
    # Determine the file extension
    file_extension = file_path.rsplit('.', 1)[1].lower() if '.' in file_path else ''
    # Check if the file is of an allowed type
    if file_extension in Config.ALLOWED_EXTENSIONS and file_extension != 'zip':
        # Generate a unique filename
        base_name = os.path.basename(file_path).rsplit('.', 1)[0]
        unique_id = uuid.uuid4().hex[:8]
        unique_filename = f"{base_name}_{unique_id}.{file_extension}"

        # Move the file to the upload directory
        final_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        os.rename(file_path, final_path)

        # Get file size
        file_size = os.path.getsize(final_path)

        # Get current timestamp
        upload_time = datetime.datetime.now(datetime.timezone.utc)

        # Create metadata
        metadata = {
            "original_filename": os.path.basename(file_path),
            "stored_filename": unique_filename,
            "file_extension": file_extension,
            "file_size": file_size,
            "formatted_file_size": format_file_size(file_size),
            "upload_date": upload_time.strftime("%Y-%m-%d"),
            "upload_time": upload_time.strftime("%H:%M:%S"),
            "timestamp": upload_time.timestamp(),
            "doc_id": unique_id
        }

        uploaded_files.append(final_path)
        file_metadata.append(metadata)


@main.route("/files", methods=["GET"])
def list_files():
    """
    API to list all uploaded files with metadata
    
    Query Parameters:
    - vectorstore_type: 'knowledge_base' (default) or 'sonu'
                       When 'sonu', includes files from sonu_resources directory
    """
    try:
        # Get vectorstore type from query parameter (defaults to knowledge_base)
        vectorstore_type = request.args.get("vectorstore_type", "knowledge_base").lower()
        
        # First, get files directly from the upload directory (always included)
        files = get_uploaded_files()
        file_metadata_list = []
        stored_file_metadata = {}

        if vectorstore_type == "sonu":
            # For SONU mode, include metadata from SONU collection
            try:
                sonu_vectorstore = get_vectorstore(Config.SONU_COLLECTION)
                if sonu_vectorstore:
                    try:
                        sonu_docs = sonu_vectorstore.get(include=["metadatas"])
                        if sonu_docs and "metadatas" in sonu_docs and sonu_docs["metadatas"]:
                            for metadata in sonu_docs["metadatas"]:
                                doc_id = metadata.get("doc_id")
                                source = metadata.get("source")
                                if not doc_id or not source:
                                    continue

                                # Derive a stored filename for display/grouping
                                stored_filename = metadata.get("stored_filename", source)

                                # Only add if not already collected (avoid duplicates across collections)
                                if stored_filename not in stored_file_metadata:
                                    normalized = dict(metadata)
                                    normalized["stored_filename"] = stored_filename

                                    # Derive a numeric timestamp from processed_date for sorting
                                    if "timestamp" not in normalized:
                                        processed_date = normalized.get("processed_date")
                                        if processed_date:
                                            try:
                                                normalized["timestamp"] = int(datetime.datetime.fromisoformat(processed_date).timestamp())
                                            except Exception:
                                                # Leave timestamp absent if parsing fails
                                                pass

                                    # Infer file_type from filename if missing
                                    if "file_type" not in normalized and "." in stored_filename:
                                        normalized["file_type"] = stored_filename.rsplit(".", 1)[-1].lower()

                                    stored_file_metadata[stored_filename] = normalized
                    except Exception as e:
                        logger.error(f"Error retrieving SONU metadata from vectorstore: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to access SONU collection: {str(e)}")
        else:
            # For knowledge_base mode, include metadata from default vectorstore
            vectorstore = get_vectorstore()
            if vectorstore:
                try:
                    all_docs = vectorstore.get(include=["metadatas"])
                    if all_docs and "metadatas" in all_docs and all_docs["metadatas"]:
                        # Process metadata from vectorstore
                        # Group by doc_id to avoid duplicates (since documents are chunked)
                        for metadata in all_docs["metadatas"]:
                            if "stored_filename" in metadata and "doc_id" in metadata:
                                stored_filename = metadata.get("stored_filename")
                                doc_id = metadata.get("doc_id")
                                
                                # Only add if not already in our list
                                if stored_filename not in stored_file_metadata:
                                    stored_file_metadata[stored_filename] = metadata
                except Exception as e:
                    logger.error(f"Error retrieving metadata from vectorstore: {str(e)}")
                    # Continue with file system metadata only

        # Process files from uploads directory
        for filename in files:
            # Check if we already have metadata from vectorstore
            if filename in stored_file_metadata:
                metadata = stored_file_metadata[filename]
                # Format file size for display if not already formatted
                if "file_size" in metadata and "formatted_file_size" not in metadata:
                    metadata["formatted_file_size"] = format_file_size(metadata["file_size"])
                file_metadata_list.append(metadata)
        
        # Also add files from vectorstore that might not be in uploads folder
        for stored_filename, metadata in stored_file_metadata.items():
            # Only add if not already added from uploads directory
            if not any(f.get("stored_filename") == stored_filename for f in file_metadata_list):
                # Format file size for display if not already formatted
                if "file_size" in metadata and "formatted_file_size" not in metadata:
                    metadata["formatted_file_size"] = format_file_size(metadata["file_size"])
                file_metadata_list.append(metadata)
        
        # Add files directly from sonu_resources directory only if vectorstore_type is 'sonu'
        if vectorstore_type == "sonu":
            sonu_resources_folder = os.path.abspath("./sonu_resources")
            if os.path.exists(sonu_resources_folder):
                try:
                    for filename in os.listdir(sonu_resources_folder):
                        file_path = os.path.join(sonu_resources_folder, filename)
                        if os.path.isfile(file_path):
                            # Check if this file is already in our list (avoid duplicates)
                            if not any(f.get("stored_filename") == filename or f.get("original_filename") == filename for f in file_metadata_list):
                                # Create metadata for SONU resource file
                                file_size = os.path.getsize(file_path)
                                file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
                                metadata = {
                                    "original_filename": filename,
                                    "stored_filename": filename,
                                    "file_extension": file_extension,
                                    "file_size": file_size,
                                    "formatted_file_size": format_file_size(file_size),
                                    "source": "sonu_resources",
                                    "file_type": file_extension,
                                    "timestamp": os.path.getmtime(file_path)  # Use file modification time
                                }
                                file_metadata_list.append(metadata)
                except Exception as e:
                    logger.error(f"Error listing files from sonu_resources: {str(e)}")
        
        # Sort by upload time if available
        file_metadata_list.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return jsonify(file_metadata_list)
    except Exception as e:
        logger.error(f"Error in list_files: {str(e)}")
        return jsonify({"error": str(e)}), 500


@main.route("/files/<filename>", methods=["GET"])
def get_file(filename):
    """
    API to download or open an uploaded file with metadata
    
    Query Parameters:
    - metadata: 'true' to return only metadata, otherwise returns file content
    - vectorstore_type: 'knowledge_base' (default) or 'sonu'
                       When 'sonu', also searches sonu_resources directory
    """
    upload_folder = os.path.abspath(Config.UPLOAD_FOLDER)
    sonu_resources_folder = os.path.abspath("./sonu_resources")
    
    # Check if metadata parameter is provided and is true
    metadata_only = request.args.get("metadata", "").lower() == "true"
    
    # Get vectorstore type from query parameter (defaults to knowledge_base)
    vectorstore_type = request.args.get("vectorstore_type","" ).lower()
    
    # Only check sonu_resources if vectorstore_type is 'sonu'
    if vectorstore_type == "sonu":
        # First, check if file exists directly in sonu_resources directory
        sonu_file_path = os.path.join(sonu_resources_folder, filename)
        if os.path.exists(sonu_file_path):
            if metadata_only:
                # Create metadata for SONU resource file
                file_size = os.path.getsize(sonu_file_path)
                file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
                metadata = {
                    "original_filename": filename,
                    "stored_filename": filename,
                    "file_extension": file_extension,
                    "file_size": file_size,
                    "formatted_file_size": format_file_size(file_size),
                    "source": "sonu_resources",
                    "file_type": file_extension
                }
                return jsonify(metadata)
            else:
                # Serve the file directly from sonu_resources
                return send_from_directory(sonu_resources_folder, filename, conditional=False)
    
    # Try to find it in the appropriate vectorstore based on vectorstore_type
    try:
        vectorstores_to_check = []
        
        if vectorstore_type == "sonu":
            # For SONU vectorstore type, check SONU collection first, then default as fallback
            sonu_vectorstore = get_vectorstore(Config.SONU_COLLECTION)
            if sonu_vectorstore:
                vectorstores_to_check.append(("sonu", sonu_vectorstore))
            # Also check default vectorstore as fallback
            default_vectorstore = get_vectorstore()
            if default_vectorstore:
                vectorstores_to_check.append(("default", default_vectorstore))
        else:
            # For knowledge_base (default), only check the default vectorstore
            default_vectorstore = get_vectorstore()
            if default_vectorstore:
                vectorstores_to_check.append(("default", default_vectorstore))
            
        for store_name, vs in vectorstores_to_check:
            try:
                all_docs = vs.get(include=["metadatas"])
                
                if all_docs and "metadatas" in all_docs:
                    # Helper function to check and return file
                    def check_and_return_file(metadata, stored_filename, is_sonu_store=False):
                        # For SONU files, first check sonu_resources directory
                        if is_sonu_store:
                            source_file = metadata.get("source", "")
                            if source_file:
                                # Extract just the filename from the source path
                                source_filename = os.path.basename(source_file)
                                sonu_path = os.path.join(sonu_resources_folder, source_filename)
                                if os.path.exists(sonu_path):
                                    if metadata_only:
                                        # Add formatted file size for metadata requests
                                        if "file_size" in metadata:
                                            metadata["formatted_file_size"] = format_file_size(metadata["file_size"])
                                        metadata["source"] = "sonu_resources"
                                        return jsonify(metadata)
                                    return send_from_directory(sonu_resources_folder, source_filename, conditional=False)
                        
                        # Default behavior for uploaded files
                        file_path = os.path.join(upload_folder, stored_filename)
                        
                        if metadata_only:
                            # Add formatted file size for metadata requests
                            if "file_size" in metadata:
                                metadata["formatted_file_size"] = format_file_size(metadata["file_size"])
                            return jsonify(metadata)
                        
                        if os.path.exists(file_path):
                            return send_from_directory(upload_folder, stored_filename, conditional=False)
                        else:
                            # File exists in vectorstore but not on disk
                            return jsonify({"error": f"File {filename} exists in database but physical file was removed from uploads folder"}), 404
                    
                    # Search strategies for different filename patterns
                    for metadata in all_docs["metadatas"]:
                        # First, try to find by stored_filename
                        if metadata.get("stored_filename") == filename:
                            return check_and_return_file(metadata, metadata["stored_filename"], store_name == "sonu")
                        
                        # Next, try to find by original_filename
                        if metadata.get("original_filename") == filename and "stored_filename" in metadata:
                            return check_and_return_file(metadata, metadata["stored_filename"], store_name == "sonu")
                        
                        # For SONU files, also try to match against the source filename
                        if store_name == "sonu":
                            source = metadata.get("source", "")
                            if source:
                                source_filename = os.path.basename(source)
                                if source_filename == filename:
                                    return check_and_return_file(metadata, filename, True)
                        
                        # Finally, try to find by source
                        if metadata.get("source") == filename and "stored_filename" in metadata:
                            return check_and_return_file(metadata, metadata["stored_filename"], store_name == "sonu")
                            
            except Exception as e:
                logger.error(f"Error retrieving file from {store_name} vectorstore: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Error retrieving file from vectorstores: {str(e)}")
    
    # If we get here, the file was not found
    return jsonify({"error": "File not found"}), 404


@main.route("/files", methods=["DELETE"])
@jwt_required()
def delete_file():
    """API to delete a file and its embeddings"""
    try:
        filenames = request.json.get("filenames", [])
        if not filenames:
            return jsonify({"error": "No filenames provided for deletion"}), 400

        upload_folder = os.path.abspath(Config.UPLOAD_FOLDER)
        results = {"deleted": 0, "not_found": 0, "errors": 0}

        vectorstore = get_vectorstore()
        if not vectorstore:
            return jsonify({"error": "Vectorstore not available"}), 500

        # Retrieve all metadata in a single query
        all_docs = vectorstore.get(include=["metadatas"])
        metadata_map = {m.get("stored_filename", m.get("original_filename")): m for m in all_docs.get("metadatas", [])}

        delete_queries = []
        for filename in filenames:
            try:
                file_deleted = False
                embeddings_deleted = False
                doc_id = None
                    # Find metadata in vectorstore
                metadata = metadata_map.get(filename)
                if metadata:
                    stored_filename = metadata.get("stored_filename")
                    doc_id = metadata.get("doc_id")

                    if stored_filename:
                        stored_file_path = os.path.join(upload_folder, stored_filename)
                        if os.path.exists(stored_file_path):
                            os.remove(stored_file_path)
                            file_deleted = True

                # Prepare batch deletion queries
                if doc_id:
                    delete_queries.append({"doc_id": doc_id})
                    embeddings_deleted = True
                elif filename in metadata_map:
                    delete_queries.extend([
                        {"source": filename},
                        {"original_filename": filename},
                        {"stored_filename": metadata_map[filename].get("stored_filename")},
                    ])
                    embeddings_deleted = True

                if file_deleted or embeddings_deleted:
                    results["deleted"] += 1
                else:
                    results["not_found"] += 1

            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                results["errors"] += 1

        # Batch delete embeddings
        for query in delete_queries:
            vectorstore.delete(where=query)

        # Construct response message
        messages = []
        if results['deleted']:
            messages.append(f"Successfully deleted {results['deleted']} file(s).")
        if results['not_found']:
            messages.append(f"{results['not_found']} file(s) were not found.")
        if results['errors']:
            messages.append(f"Encountered {results['errors']} error(s) during deletion.")

        return jsonify({"message": " ".join(messages) or "No files were processed."}), 200

    except Exception as e:
        logger.error(f"Unexpected error in delete_files: {str(e)}")
        return jsonify({"error": "An unexpected error occurred."}), 500


@main.route('/providers', methods=['GET'])
def list_providers():
    with session_scope() as session:
            try:
                # First, get all providers
                providers = Provider.query.all()
                result = []

                # For each provider, get its models and structure the response
                for provider in providers:
                    provider_data = {
                        "provider_id": provider.id,
                        "provider_name": provider.name,
                        "models": []
                    }

                    # Get all models for this provider
                    models = AIModel.query.filter_by(provider_id=provider.id).all()

                    # Add each model to the provider's models array
                    for model in models:
                        provider_data["models"].append({
                            "model_id": model.id,
                            "model_name": model.name,
                            "is_selected": model.is_selected
                        })

                    result.append(provider_data)

                logger.info(f"Found {len(result)} providers with their models")
                return jsonify(result)

            except Exception as query_err:
                logger.error(f"Error querying providers: {str(query_err)}")
                return jsonify({
                    "error": f"Database query failed: {str(query_err)}",
                    "message": "Database is accessible but query failed"
                }), 500


@main.route('/providers', methods=['POST'])
def add_provider():
    data = request.json
    added_entries = []
    skipped_entries = []

    try:
        with session_scope() as session:
            for entry in data:
                provider_name = entry['provider']
                model_name = entry['model_name']

                # First, find or create the provider
                provider = Provider.query.filter_by(name=provider_name).first()
                if not provider:
                    provider = Provider(name=provider_name)
                    session.add(provider)
                    session.flush()  # Flush to get the provider ID

                # Check if model already exists for this provider
                existing_model = AIModel.query.filter_by(
                    name=model_name,
                    provider_id=provider.id
                ).first()

                if existing_model:
                    skipped_entries.append({"provider": provider_name, "model_name": model_name})
                    continue

                # Create new model and associate with provider
                new_model = AIModel(name=model_name, provider_id=provider.id)
                session.add(new_model)
                added_entries.append({"provider": provider_name, "model_name": model_name})

        # Set default selected model if none exists (in a separate transaction)
        with session_scope() as session:
            if not AIModel.query.filter_by(is_selected=True).first():
                # Find OpenAI provider
                openai_provider = Provider.query.filter_by(name="openai").first()
                if openai_provider:
                    # Find GPT-4o model or create it if it doesn't exist
                    default_model = AIModel.query.filter_by(
                        provider_id=openai_provider.id,
                        name="gpt-4o"
                    ).first()

                    if default_model:
                        default_model.is_selected = True

        return jsonify({
            "message": "Provider(s) and model(s) processed successfully",
            "added": added_entries,
            "skipped": skipped_entries
        }), 201

    except Exception as e:
        # Handle any exceptions that occurred during the process
        logger.error(f"Error adding provider: {str(e)}")
        return jsonify({"error": f"Failed to process providers: {str(e)}"}), 500


@main.route('/models/<int:model_id>', methods=['DELETE'])
def delete_model(model_id: int):
    """
    Delete an AI model by its ID and handle related cleanup operations.
    
    This endpoint deletes a model and performs cleanup operations including:
    1. Checking if the model was selected and selecting another one if needed
    2. Deleting the parent provider if this was its last model
    3. Updating the provider cache if necessary
    """
    try:
        with session_scope() as session:
            model = AIModel.query.get(model_id)
            if not model:
                return jsonify({"message": "Model not found"}), 404

            provider_id = model.provider_id
            model_name = model.name
            
            # Check if this model is currently selected
            is_model_selected = model.is_selected
            
            # Delete the model
            session.delete(model)
            
            # Check if this was the last model for the provider
            remaining_models = AIModel.query.filter_by(provider_id=provider_id).count()
            if remaining_models == 0:
                # Delete provider if it has no more models
                provider = Provider.query.get(provider_id)
                if provider:
                    provider_name = provider.name
                    session.delete(provider)
                    logger.info(f"Deleted provider '{provider_name}' as it had no remaining models")
            elif is_model_selected and remaining_models > 0:
                # If the deleted model was selected, select another model from the same provider
                another_model = AIModel.query.filter_by(provider_id=provider_id).first()
                if another_model:
                    another_model.is_selected = True
                    # Update the Redis cache
                    try:
                        provider = Provider.query.get(provider_id)
                        provider_data = update_selected_provider(provider.name, another_model.name)
                        logger.info(f"Auto-selected model '{another_model.name}' after deleting selected model '{model_name}'")
                        logger.info(f"Provider cache updated: {provider_data}")
                    except Exception as e:
                        logger.warning(f"Could not update provider cache: {str(e)}")

        return jsonify({"message": "Model deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        return jsonify({"error": f"Failed to delete model: {str(e)}"}), 500


@main.route('/providers/<int:provider_id>', methods=['DELETE'])
def delete_provider(provider_id: int):
    """
    Delete a provider and all its associated models.
    
    This endpoint deletes a provider and all its models, leveraging the cascade
    relationship. If one of the deleted models was selected, it automatically
    selects another model from a different provider.
    """
    try:
        with session_scope() as session:
            provider = Provider.query.get(provider_id)
            if not provider:
                return jsonify({"message": "Provider not found"}), 404
                
            provider_name = provider.name
            
            # Check if any models from this provider are selected
            has_selected_model = AIModel.query.filter_by(provider_id=provider_id, is_selected=True).first() is not None
            
            # Get all models associated with this provider
            models = AIModel.query.filter_by(provider_id=provider_id).all()
            model_count = len(models)
            model_names = [model.name for model in models]
            
            logger.info(f"Deleting provider '{provider_name}' with {model_count} models: {', '.join(model_names)}")
            
            # Delete the provider - this should cascade delete all models due to the relationship definition
            session.delete(provider)
            
            # If a model from this provider was selected, select another model from a different provider if available
            if has_selected_model:
                # Find another model from a different provider
                another_model = AIModel.query.filter(AIModel.provider_id != provider_id).first()
                if another_model:
                    another_model.is_selected = True
                    # Update the Redis cache
                    try:
                        another_provider = Provider.query.get(another_model.provider_id)
                        provider_data = update_selected_provider(another_provider.name, another_model.name)
                        logger.info(f"Auto-selected model '{another_model.name}' from provider '{another_provider.name}' " 
                                   f"after deleting provider '{provider_name}'")
                        logger.info(f"Provider cache updated: {provider_data}")
                    except Exception as e:
                        logger.warning(f"Could not update provider cache: {str(e)}")
                else:
                    logger.warning("No alternative models available to select after provider deletion")

                    cache.delete('selected_provider')
                    logger.info("Provider cache cleared (no models available)")

        return jsonify({
            "message": f"Provider '{provider_name}' deleted successfully along with {model_count} model(s)"
        })
    except Exception as e:
        logger.error(f"Error deleting provider: {str(e)}")
        return jsonify({"error": f"Failed to delete provider: {str(e)}"}), 500


@main.route('/model/<int:model_id>/select', methods=['POST'])
def select_model(model_id: int):
    try:
        with session_scope() as session:
            # Find model using get_or_404
            model = AIModel.query.get_or_404(model_id)
            
            # Get provider object
            provider = Provider.query.get(model.provider_id)
            if not provider:
                return jsonify({"message": "Provider not found for this model"}), 500
            
            # Get model and provider names as strings
            model_name = model.name
            provider_name = provider.name
            
            # Deselect all models
            AIModel.query.update({AIModel.is_selected: False})

            # Select the specified model
            model.is_selected = True

        # Update the Redis cache
        try:
            from .services.providers import update_selected_provider
            provider_data = update_selected_provider(provider_name, model_name)
            logger.info(f"Provider cache updated: {provider_data}")
        except ImportError:
            logger.warning("Could not update provider cache - service not available")

        return jsonify({
            "message": f"Model '{model_name}' from provider '{provider_name}' selected"
        })
    except Exception as e:
        logger.error(f"Error selecting model: {str(e)}")
        return jsonify({"error": f"Failed to select model: {str(e)}"}), 500


@main.route('/model/selected', methods=['GET'])
def get_selected_model():
    try:
        # Try to get from the service cache first
        cached_provider = get_selected_provider()

        if cached_provider:
            return jsonify({
                "provider": cached_provider["provider"],
                "model_name": cached_provider["model_name"],
                "provider_id": cached_provider.get("provider_id"),
                "model_id": cached_provider.get("model_id")
            })

        # If not in cache, query the database directly
        selected_model = AIModel.query.filter_by(is_selected=True).first()
        if not selected_model:
            return jsonify({"message": "No model selected"}), 404

        provider = Provider.query.get(selected_model.provider_id)
        if not provider:
            return jsonify({"message": "Selected model has no valid provider"}), 500

        # Return the result from database
        return jsonify({
            "provider": provider.name,
            "model_name": selected_model.name,
            "provider_id": provider.id,
            "model_id": selected_model.id
        })
    except Exception as e:
        logger.error(f"Error retrieving selected model: {str(e)}")
        return jsonify({"error": f"Failed to retrieve selected model: {str(e)}"}), 500


@main.route('/feedback/stats', methods=['GET'])
@jwt_required()
def get_feedback_stats():
    """
    API endpoint to get statistics about the feedback.
    
    Query parameters:
    - user_id: Filter stats by user ID (optional)
    - start_date: Filter by created_at >= start_date (format: YYYY-MM-DD) (optional)
    - end_date: Filter by created_at <= end_date (format: YYYY-MM-DD) (optional)
    
    Returns:
    - Total count
    - Positive feedback count
    - Negative feedback count (all negative feedback requires feedback_text)
    - Unique users count
    
    Note: All negative feedback requires feedback_text to explain the reason.
    """
    try:
        # Get query parameters
        user_id = request.args.get('user_id')
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        # Base query
        query = Feedback.query
        
        # Apply user filter if provided
        if user_id:
            query = query.filter(Feedback.user_id == user_id)
        
        # Apply date filters if provided
        if start_date_str:
            try:
                start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
                query = query.filter(Feedback.created_at >= start_date)
            except ValueError:
                return jsonify({"error": "Invalid start_date format. Use YYYY-MM-DD."}), 400
                
        if end_date_str:
            try:
                # Set end date to end of day
                end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
                end_date = end_date.replace(hour=23, minute=59, second=59, tzinfo=datetime.timezone.utc)
                query = query.filter(Feedback.created_at <= end_date)
            except ValueError:
                return jsonify({"error": "Invalid end_date format. Use YYYY-MM-DD."}), 400
        
        # Get total count
        total_count = query.count()
        
        # Get positive feedback count
        positive_count = query.filter(Feedback.feedback_type == FeedbackType.POSITIVE).count()
        
        # Get negative feedback count (all should have feedback_text)
        negative_count = query.filter(Feedback.feedback_type == FeedbackType.NEGATIVE).count()
        
        # For validation: count negative feedback with feedback_text
        # negative_with_feedback_count = query.filter(
        #     Feedback.feedback_type == FeedbackType.NEGATIVE,
        #     Feedback.feedback_text.isnot(None)
        # ).count()
        
        # Get unique users count (excluding anonymous)
        unique_users_count = db.session.query(Feedback.user_id).filter(
            Feedback.user_id.isnot(None),
            *[Feedback.user_id == user_id] if user_id else []
        ).distinct().count()
        
        # Build response
        stats = {
            "total_count": total_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "unique_users_count": unique_users_count,
            "positive_feedback_ratio": round(positive_count / total_count, 2) if total_count > 0 else 0,
            "note": "All negative feedback requires feedback_text to explain the reason"
        }
        
        # Add filter information if provided
        if user_id:
            stats["user_id"] = user_id
            
        if start_date_str:
            stats["start_date"] = start_date_str
            
        if end_date_str:
            stats["end_date"] = end_date_str
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error retrieving feedback statistics: {str(e)}")
        return jsonify({"error": f"Failed to retrieve feedback statistics: {str(e)}"}), 500


@main.route('/feedback', methods=['POST'])
@jwt_required(optional=True)
def submit_feedback():
    """
    API endpoint to submit feedback for a chat response.
    
    Accepts:
    - user_query: The original user query (required)
    - assistant_response: The assistant's response (required)
    - feedback_type: String value ('positive' or 'negative') (required)
    - feedback_text: Text feedback explaining why (required for negative feedback only)
    - model_name: Name of the model that generated the response (optional)
    - provider: Name of the provider (e.g., "OpenAI") (optional)
    - conversation_history: List of previous messages (required for negative feedback if socket_id not provided)
    - socket_id: The WebSocket connection ID to retrieve chat history (optional)
    
    Returns:
    - Success message with feedback ID or error
    """
    try:
        # Get current user if authenticated
        current_user_id = get_jwt_identity()
        
        # Get JSON data
        data = request.json
        
        # Validate required fields
        required_fields = ['user_query', 'assistant_response']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Support both feedback_type and is_helpful for backward compatibility
        feedback_type = data.get('feedback_type')

        
        # Ensure we have either feedback_type or is_helpful
        if feedback_type is None:
            return jsonify({"error": "Missing required field: either feedback_type must be provided"}), 400

        
        # Get feedback text - prioritize feedback_text but support feedback for backward compatibility
        feedback_text = data.get('feedback_text', data.get('feedback'))
            
        # For negative feedback, feedback_text is required
        if feedback_type == 'negative' and (not feedback_text or feedback_text.strip() == ''):
            return jsonify({"error": "Feedback text is required for negative feedback. Please explain why the response wasn't helpful."}), 400
        
        # For positive feedback, feedback_text is optional
        feedback_text = feedback_text if feedback_type == 'negative' else None
        
        # Get conversation history from:
        # 1. socket_id if provided
        # 2. conversation_history in request data
        conversation_history = None
        
        # Try to get conversation history from socket_id if provided
        socket_id = data.get('socket_id')
        if socket_id and feedback_type == 'negative':
            try:
                # Get last 5 messages from chat history using socket_id
                conversation_history = get_chat_history_by_socket_id(socket_id, 10)
                
                if conversation_history:
                    logger.info(f"Retrieved conversation history from socket_id: {socket_id}, found {len(conversation_history)} messages")
                else:
                    logger.warning(f"No conversation history found for socket_id: {socket_id}")
            except Exception as e:
                logger.error(f"Error retrieving conversation history from socket_id: {str(e)}")

        if feedback_type == 'negative' and not conversation_history:
            conversation_history = [
                {"role": "user", "content": data['user_query']},
                {"role": "assistant", "content": data['assistant_response']}
            ]
            logger.info("Created minimal conversation history for first response feedback")
        
        # Convert string feedback_type to enum
        enum_feedback_type = FeedbackType.POSITIVE if feedback_type == 'positive' else FeedbackType.NEGATIVE
        selected_provider = get_selected_provider()
        model_name = selected_provider.get('model_name')
        provider = selected_provider.get('provider').lower()
        # Create new feedback entry
        feedback = Feedback(
            user_id=current_user_id,
            user_query=data['user_query'],
            assistant_response=data['assistant_response'],
            feedback_type=enum_feedback_type,
            feedback_text=feedback_text,
            model_name=model_name,
            provider=provider,
            conversation_history=conversation_history
        )
        
        # Save to database
        with session_scope() as session:
            session.add(feedback)
            session.flush()  # To get the ID
            feedback_id = feedback.id
        
        # Log the feedback
        logger.info(f"Feedback received - ID: {feedback_id}, Type: {feedback_type}")
        if feedback_text:
            logger.info(f"Feedback text: {feedback_text}")
        
        return jsonify({
            "success": True,
            "message": "Feedback submitted successfully",
            "feedback_id": feedback_id,
            "has_conversation_history": conversation_history is not None
        })
        
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Database error while submitting feedback: {str(e)}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return jsonify({"error": f"Failed to submit feedback: {str(e)}"}), 500


@main.route('/feedback', methods=['GET'])
@jwt_required()
def get_feedback():
    """
    API endpoint to retrieve feedback entries.
    
    Query parameters:
    - limit: Maximum number of entries to return (default: 100)
    - offset: Offset for pagination (default: 0)
    - feedback_type: Filter by feedback_type value (optional, 'positive' or 'negative')
      If not provided, returns all feedback types
    - user_id: Filter by user ID (optional, admin only)
    - start_date: Filter by created_at >= start_date (format: YYYY-MM-DD) (optional)
    - end_date: Filter by created_at <= end_date (format: YYYY-MM-DD) (optional)
    
    Returns:
    - List of feedback entries
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        feedback_type = request.args.get('feedback_type')
        user_id = request.args.get('user_id')
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        # Build query
        query = Feedback.query
        
        # Apply feedback_type filter only if provided
        if feedback_type:
            if feedback_type.lower() == 'positive':
                query = query.filter(Feedback.feedback_type == FeedbackType.POSITIVE)
            elif feedback_type.lower() == 'negative':
                query = query.filter(Feedback.feedback_type == FeedbackType.NEGATIVE)
            elif feedback_type.lower() != 'all':  # 'all' is treated as no filter
                # Invalid value for feedback_type parameter
                return jsonify({"error": "Invalid value for 'feedback_type' parameter. Must be 'positive', 'negative', or 'all'"}), 400
            # If feedback_type is 'all' or not specified, don't apply any filter - return all feedback types
        
        if user_id:
            # Check if current user is admin or querying their own feedback
            current_user_id = get_jwt_identity()
            if current_user_id and (current_user_id == user_id):
                query = query.filter(Feedback.user_id == user_id)
            else:
                # In a real system, you'd check if user is admin here
                # For now, we'll allow it but in production you might restrict this
                query = query.filter(Feedback.user_id == user_id)
        
        # Apply date filters if provided
        if start_date_str:
            try:
                start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
                query = query.filter(Feedback.created_at >= start_date)
            except ValueError:
                return jsonify({"error": "Invalid start_date format. Use YYYY-MM-DD."}), 400
                
        if end_date_str:
            try:
                # Set end date to end of day
                end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
                end_date = end_date.replace(hour=23, minute=59, second=59, tzinfo=datetime.timezone.utc)
                query = query.filter(Feedback.created_at <= end_date)
            except ValueError:
                return jsonify({"error": "Invalid end_date format. Use YYYY-MM-DD."}), 400
        
        # Get total count before applying limit and offset
        total_count = query.count()
        
        # Apply pagination and sorting
        query = query.order_by(Feedback.created_at.desc()).limit(limit).offset(offset)
        
        # Execute query
        feedback_entries = query.all()
        
        # Convert to dictionaries
        feedback_list = [entry.to_dict() for entry in feedback_entries]
        
        return jsonify({
            "feedback": feedback_list,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "feedback_type": feedback_type or "all"
        })
        
    except Exception as e:
        logger.error(f"Error retrieving feedback: {str(e)}")
        return jsonify({"error": f"Failed to retrieve feedback: {str(e)}"}), 500

@main.route('/feedback/bulk-delete', methods=['DELETE'])
@jwt_required()
def bulk_delete_feedback():
    """
    API endpoint to delete multiple feedback entries by their IDs.
    Uses DELETE method with JSON body containing the IDs to delete.
    Any authenticated user can delete any feedback entries.
    
    Required fields:
    - feedback_ids: List of feedback IDs to delete
    
    Returns:
    - Counts of deleted, not found, and error entries
    - HTTP 200 on success, 400 on validation error, 500 on server error
    """
    try:
        # Get JSON data
        data = request.json
        
        # Validate required fields
        if not data or 'feedback_ids' not in data:
            return jsonify({"error": "Missing required field: feedback_ids"}), 400
            
        if not isinstance(data['feedback_ids'], list):
            return jsonify({"error": "feedback_ids must be a list of feedback IDs"}), 400
            
        if len(data['feedback_ids']) == 0:
            return jsonify({"error": "Empty list of feedback IDs provided"}), 400
            
        # Get the list of feedback IDs to delete
        feedback_ids = data['feedback_ids']
        
        # Create a response object to track results
        result = {
            "total_processed": len(feedback_ids),
            "deleted_count": 0,
            "not_found_count": 0,
            "error_count": 0,
            "errors": [],  # To store specific error messages
            "not_found_ids": []  # To track which IDs don't exist
        }
        
        with session_scope() as session:
            # First, get all feedback entries that match the IDs
            feedback_entries = session.query(Feedback).filter(
                Feedback.id.in_(feedback_ids)
            ).all()
            
            # Create a map for faster lookups
            feedback_map = {feedback.id: feedback for feedback in feedback_entries}
            
            # Track which IDs were not found
            for feedback_id in feedback_ids:
                if feedback_id not in feedback_map:
                    result["not_found_count"] += 1
                    result["not_found_ids"].append(feedback_id)
            
            # Process each found feedback entry - any authenticated user can delete any feedback
            for feedback_id, feedback in feedback_map.items():
                try:
                    # Delete the feedback (no authorization check as per requirements)
                    session.delete(feedback)
                    result["deleted_count"] += 1
                except Exception as e:
                    error_msg = f"Error deleting feedback ID {feedback_id}: {str(e)}"
                    logger.error(error_msg)
                    result["error_count"] += 1
                    result["errors"].append(error_msg)
        
        # Determine if the operation was successful
        result["success"] = result["deleted_count"] > 0
        
        # Build a human-readable message
        messages = []
        if result['deleted_count']:
            messages.append(f"Successfully deleted {result['deleted_count']} feedback entries.")
        if result['not_found_count']:
            messages.append(f"{result['not_found_count']} feedback entries were not found.")
        if result['error_count']:
            messages.append(f"Encountered {result['error_count']} errors during deletion.")
        
        result["message"] = " ".join(messages) or "No feedback entries were processed."
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in bulk delete operation: {str(e)}")
        return jsonify({"error": f"Failed to delete feedback: {str(e)}"}), 500

@main.route('/set-vectorstore-type', methods=['POST'])
@jwt_required()
def set_vectorstore_type():
    """
    API endpoint to set the vectorstore type for the current session.
    
    Expected JSON payload:
    {
        "vectorstore_type": "knowledge_base" | "sonu"
    }
    
    Returns:
    - Success message with the set vectorstore type
    """
    try:
        data = request.json
        
        if not data or 'vectorstore_type' not in data:
            return jsonify({"error": "Missing required field: vectorstore_type"}), 400
            
        vectorstore_type = data['vectorstore_type'].lower()
        
        # Validate vectorstore type
        if vectorstore_type not in ['knowledge_base', 'sonu']:
            return jsonify({"error": "Invalid vectorstore_type. Must be 'knowledge_base' or 'sonu'"}), 400
            
        # Store the vectorstore type in the session or cache
        # For now, we'll use a simple approach - the frontend should send this with each message
        # In a more sophisticated setup, you might want to store this in Redis or session
        
        logger.info(f"Vectorstore type set to: {vectorstore_type}")
        
        return jsonify({
            "message": f"Vectorstore type set to {vectorstore_type}",
            "vectorstore_type": vectorstore_type
        })
        
    except Exception as e:
        logger.error(f"Error setting vectorstore type: {str(e)}")
        return jsonify({"error": f"Failed to set vectorstore type: {str(e)}"}), 500

@main.route('/get-vectorstore-type', methods=['GET'])
@jwt_required()
def get_vectorstore_type():
    """
    API endpoint to get the current vectorstore type.
    
    Returns:
    - Current vectorstore type (defaults to 'knowledge_base')
    """
    try:
        # For now, return the default
        # In a more sophisticated setup, you might retrieve this from session/cache
        vectorstore_type = "knowledge_base"  # Default
        
        return jsonify({
            "vectorstore_type": vectorstore_type
        })
        
    except Exception as e:
        logger.error(f"Error getting vectorstore type: {str(e)}")
        return jsonify({"error": f"Failed to get vectorstore type: {str(e)}"}), 500

@main.route('/feedback/<int:feedback_id>', methods=['GET'])
@jwt_required()
def get_feedback_detail(feedback_id):
    """
    API endpoint to retrieve details of a specific feedback entry.
    
    Returns:
    - Detailed feedback information
    """
    try:
        # Get the feedback entry
        feedback = Feedback.query.get(feedback_id)
        
        if not feedback:
            return jsonify({"error": "Feedback not found"}), 404
            
        # Convert to dictionary
        feedback_data = feedback.to_dict()
        
        return jsonify(feedback_data)
        
    except Exception as e:
        logger.error(f"Error retrieving feedback detail: {str(e)}")
        return jsonify({"error": f"Failed to retrieve feedback detail: {str(e)}"}), 500



