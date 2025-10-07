import os
import logging
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from ..config import Config

# Global vectorstore instances
vectorstore_instances = {}

def get_vectorstore(collection_name=None):
    """
    Get or initialize the vector store
    
    Args:
        collection_name: The name of the collection to use. If None, uses default Samsung collection.
    
    Returns:
        Chroma: The vector store instance
    """
    global vectorstore_instances
    
    # Determine collection name
    if collection_name is None:
        collection_name = Config.KNOWLEDGE_BASE_COLLECTION
    
    try:
        if collection_name not in vectorstore_instances:
            # Log API key status (without revealing the key)
            api_key_status = "available" if (hasattr(Config, 'OPENAI_API_KEY') and Config.OPENAI_API_KEY) else "missing"
            logging.info(f"Initializing vector store for collection '{collection_name}'. OpenAI API key status: {api_key_status}")
            
            # Check if the vectorstore directory exists
            if os.path.exists(Config.VECTORSTORE_PATH):
                logging.info(f"Vector store directory exists at {Config.VECTORSTORE_PATH}")
                # Initialize with embedding function if available
                if hasattr(Config, 'OPENAI_API_KEY') and Config.OPENAI_API_KEY:
                    embeddings = OpenAIEmbeddings(api_key=Config.OPENAI_API_KEY)
                    vectorstore_instances[collection_name] = Chroma(
                        persist_directory=Config.VECTORSTORE_PATH, 
                        embedding_function=embeddings,
                        collection_name=collection_name
                    )
                    logging.info(f"Vector store initialized with embedding function for collection '{collection_name}'")
                else:
                    # Fallback to environment variable directly if Config doesn't have it
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key:
                        logging.info("Using OpenAI API key from environment variable")
                        embeddings = OpenAIEmbeddings(api_key=api_key)
                        vectorstore_instances[collection_name] = Chroma(
                            persist_directory=Config.VECTORSTORE_PATH, 
                            embedding_function=embeddings,
                            collection_name=collection_name
                        )
                        logging.info(f"Vector store initialized with embedding function from environment for collection '{collection_name}'")
                    else:
                        logging.error("OpenAI API key is missing. Cannot initialize vector store with embedding function.")
                        raise ValueError("OpenAI API key is required for initializing the vector store")
            else:
                # Create directory if it doesn't exist
                logging.info(f"Creating vector store directory at {Config.VECTORSTORE_PATH}")
                os.makedirs(os.path.dirname(Config.VECTORSTORE_PATH), exist_ok=True)
                
                # Initialize with embedding function
                if hasattr(Config, 'OPENAI_API_KEY') and Config.OPENAI_API_KEY:
                    embeddings = OpenAIEmbeddings(api_key=Config.OPENAI_API_KEY)
                    vectorstore_instances[collection_name] = Chroma(
                        persist_directory=Config.VECTORSTORE_PATH, 
                        embedding_function=embeddings,
                        collection_name=collection_name
                    )
                    logging.info(f"New vector store initialized with embedding function for collection '{collection_name}'")
                else:
                    # Fallback to environment variable directly if Config doesn't have it
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key:
                        logging.info("Using OpenAI API key from environment variable")
                        embeddings = OpenAIEmbeddings(api_key=api_key)
                        vectorstore_instances[collection_name] = Chroma(
                            persist_directory=Config.VECTORSTORE_PATH, 
                            embedding_function=embeddings,
                            collection_name=collection_name
                        )
                        logging.info(f"New vector store initialized with embedding function from environment for collection '{collection_name}'")
                    else:
                        logging.error("OpenAI API key is missing. Cannot initialize vector store with embedding function.")
                        raise ValueError("OpenAI API key is required for initializing the vector store")
        
        return vectorstore_instances[collection_name]
    except Exception as e:
        logging.error(f"Error initializing vector store for collection '{collection_name}': {str(e)}")
        # Return None in case of error
        return e

def get_vectorstore_by_type(vectorstore_type="knowledge_base"):
    """
    Get vectorstore by type (knowledge_base, samsung, or sonu)
    
    Args:
        vectorstore_type: Either "knowledge_base", "samsung", or "sonu"
    
    Returns:
        Chroma: The vector store instance
    """
    if vectorstore_type.lower() == "sonu":
        return get_vectorstore(Config.SONU_COLLECTION)
    elif vectorstore_type.lower() == "samsung":
        return get_vectorstore(Config.SAMSUNG_COLLECTION)
    else:  # Default to knowledge_base
        return get_vectorstore(Config.KNOWLEDGE_BASE_COLLECTION)

def reset_vectorstore(collection_name=None):
    """
    Reset the global vectorstore instance(s)
    
    Args:
        collection_name: If provided, reset only that collection. If None, reset all.
    """
    global vectorstore_instances
    if collection_name:
        if collection_name in vectorstore_instances:
            del vectorstore_instances[collection_name]
    else:
        vectorstore_instances = {}

def ensure_sonu_collection_exists():
    """
    Ensures that the 'sonu' collection exists in the vectorstore.
    If it doesn't exist, creates it and populates it with data from Qsv1.txt.
    
    This function is designed to be called on application startup to handle
    production deployments where the 'sonu' collection might not exist.
    
    Returns:
        bool: True if collection exists or was successfully created, False otherwise
    """
    global vectorstore_instances
    
    try:
        # Get the sonu collection name from config
        sonu_collection_name = Config.SONU_COLLECTION
        
        # Check if we can get OpenAI API key
        api_key = None
        if hasattr(Config, 'OPENAI_API_KEY') and Config.OPENAI_API_KEY:
            api_key = Config.OPENAI_API_KEY
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            logging.error("OpenAI API key is missing. Cannot ensure sonu collection exists.")
            return False
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Create vectorstore directory if it doesn't exist
        os.makedirs(Config.VECTORSTORE_PATH, exist_ok=True)
        
        # Try to connect to existing sonu collection
        try:
            sonu_vectorstore = Chroma(
                persist_directory=Config.VECTORSTORE_PATH,
                embedding_function=embeddings,
                collection_name=sonu_collection_name
            )
            
            # Check if collection has any documents
            # If collection exists but is empty, we'll still populate it
            collection_count = sonu_vectorstore._collection.count()
            
            if collection_count > 0:
                logging.info(f"Sonu collection '{sonu_collection_name}' already exists with {collection_count} documents")
                # Update global instance
                vectorstore_instances[sonu_collection_name] = sonu_vectorstore
                return True
            else:
                logging.info(f"Sonu collection '{sonu_collection_name}' exists but is empty. Populating with Qsv1.txt data...")
        except Exception as e:
            logging.info(f"Sonu collection '{sonu_collection_name}' does not exist or cannot be accessed. Creating new collection...")
            # Create new collection
            sonu_vectorstore = Chroma(
                persist_directory=Config.VECTORSTORE_PATH,
                embedding_function=embeddings,
                collection_name=sonu_collection_name
            )
        
        # Path to Qsv1.txt file (relative to project root)
        qsv1_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Qsv1.txt")
        
        if not os.path.exists(qsv1_file_path):
            logging.error(f"Qsv1.txt file not found at {qsv1_file_path}")
            return False
        
        logging.info(f"Loading and processing Qsv1.txt from {qsv1_file_path}")
        
        # Load the Qsv1.txt file
        loader = TextLoader(qsv1_file_path)
        documents = loader.load()
        
        # Initialize text splitter with config settings
        text_splitter = CharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Split documents into chunks
        split_docs = text_splitter.split_documents(documents)
        
        if not split_docs:
            logging.error("No content could be extracted from Qsv1.txt")
            return False
        
        # Prepare data for vectorstore
        texts = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(split_docs):
            texts.append(doc.page_content)
            
            # Create metadata for each chunk
            metadata = {
                "source": "Qsv1.txt",
                "collection": "sonu",
                "chunk_index": i,
                "total_chunks": len(split_docs)
            }
            
            # Add any existing metadata from the document
            if hasattr(doc, 'metadata') and doc.metadata:
                metadata.update(doc.metadata)
            
            metadatas.append(metadata)
            ids.append(f"qsv1_chunk_{i}")
        
        # Add texts to the sonu vectorstore
        sonu_vectorstore.add_texts(
            texts=texts,
            ids=ids,
            metadatas=metadatas
        )
        
        logging.info(f"Successfully added {len(texts)} chunks from Qsv1.txt to sonu collection '{sonu_collection_name}'")
        
        # Update global vectorstore instances
        vectorstore_instances[sonu_collection_name] = sonu_vectorstore
        
        return True
        
    except Exception as e:
        logging.error(f"Error ensuring sonu collection exists: {str(e)}")
        return False

def check_collection_exists(collection_name):
    """
    Check if a specific collection exists and has documents.
    
    Args:
        collection_name: Name of the collection to check
        
    Returns:
        dict: Status information about the collection
    """
    try:
        # Get API key
        api_key = None
        if hasattr(Config, 'OPENAI_API_KEY') and Config.OPENAI_API_KEY:
            api_key = Config.OPENAI_API_KEY
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return {"exists": False, "error": "OpenAI API key is missing"}
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Try to connect to the collection
        vectorstore = Chroma(
            persist_directory=Config.VECTORSTORE_PATH,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        
        # Get collection count
        count = vectorstore._collection.count()
        
        return {
            "exists": True,
            "document_count": count,
            "collection_name": collection_name,
            "path": Config.VECTORSTORE_PATH
        }
        
    except Exception as e:
        return {
            "exists": False,
            "error": str(e),
            "collection_name": collection_name
        }
