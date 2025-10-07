import logging
import os.path
import uuid
import re
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
from ..config import Config

def extract_source_url_from_markdown_section(content):
    """
    Extract source URL from markdown section content.
    
    Args:
        content (str): The content of a markdown section
        
    Returns:
        str or None: The extracted URL or None if not found
    """
    # Look for **Source URL:** pattern followed by a URL
    url_pattern = r'\*\*Source URL:\*\*\s*(https?://[^\s\n]+)'
    url_match = re.search(url_pattern, content)
    
    if url_match:
        return url_match.group(1)
    
    return None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_document_loader(file_path):
    if file_path.endswith('.pdf'):
        return PyMuPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        return TextLoader(file_path)
    elif file_path.endswith('.md') or file_path.endswith('.markdown'):
        return TextLoader(file_path)
    elif file_path.endswith('.docx' or '.doc'):
        return UnstructuredWordDocumentLoader(file_path)
    elif file_path.endswith('.csv'):
        return CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type for file: {file_path}")

def process_file(file_paths, file_metadata=None):
    """
    Process files and store them in the vector database
    
    Args:
        file_paths (list): List of file paths to process
        file_metadata (list, optional): List of metadata dictionaries for each file
    
    Returns:
        list: List of document IDs added to the vector store
    """
    embeddings = OpenAIEmbeddings(api_key=Config.OPENAI_API_KEY)
    text_splitter = CharacterTextSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
    added_doc_ids = []
    failed_files = [] 

    # Initialize or load existing vector store
    if os.path.exists(Config.VECTORSTORE_PATH):
        vector_store = Chroma(
            persist_directory=Config.VECTORSTORE_PATH,
            embedding_function=embeddings,
            collection_name=Config.COLLECTION
        )
    else:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(Config.VECTORSTORE_PATH), exist_ok=True)
        vector_store = Chroma(
            persist_directory=Config.VECTORSTORE_PATH,
            embedding_function=embeddings,
            collection_name=Config.COLLECTION
        )

    # Create a metadata lookup dictionary if metadata is provided
    metadata_lookup = {}
    if file_metadata:
        for meta in file_metadata:
            if "file_path" in meta:
                metadata_lookup[meta["file_path"]] = meta

    # Process each file
    for i, file_path in enumerate(file_paths):
        try:
            # Load the document
            loader = get_document_loader(file_path)
            documents = loader.load()
            
            # Split documents into chunks
            # For markdown files, first split by headers to preserve section context
            is_markdown = file_path.endswith('.md') or file_path.endswith('.markdown')
            if is_markdown and documents:
                headers_to_split_on = [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
                markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                md_header_splits = markdown_splitter.split_text(documents[0].page_content)

                # Attach section-level source_url (if any) so sub-splits inherit it
                split_docs = []
                for section_doc in md_header_splits:
                    section_source_url = extract_source_url_from_markdown_section(section_doc.page_content)
                    if section_source_url:
                        if not hasattr(section_doc, 'metadata') or section_doc.metadata is None:
                            section_doc.metadata = {}
                        section_doc.metadata["source_url"] = section_source_url

                    # Further split large sections; metadata will be propagated
                    sub_splits = text_splitter.split_documents([section_doc])
                    split_docs.extend(sub_splits)
            else:
                split_docs = text_splitter.split_documents(documents)
            
            # Generate unique ID for this file
            doc_id = str(uuid.uuid4())
            
            # Get file metadata if available
            file_meta = metadata_lookup.get(file_path, {})
            if not file_meta and file_metadata and i < len(file_metadata):
                file_meta = file_metadata[i]
            
            # Add doc_id to metadata
            file_meta["doc_id"] = doc_id
            
            # Extract texts and prepare metadata for each chunk
            texts = []
            metadatas = []
            ids = []
            
            for j, doc in enumerate(split_docs):
                # Add the text content
                texts.append(doc.page_content)
                
                # Create metadata for this chunk
                chunk_meta = file_meta.copy()
                chunk_meta.update({
                    "source": os.path.basename(file_path),
                })
                
                # For markdown files, ensure each chunk has a source_url
                if is_markdown:
                    existing_source_url = None
                    if hasattr(doc, 'metadata') and doc.metadata:
                        existing_source_url = doc.metadata.get("source_url")
                    if existing_source_url:
                        chunk_meta["source_url"] = existing_source_url
                    else:
                        # Fallback: attempt extraction from the chunk itself
                        source_url = extract_source_url_from_markdown_section(doc.page_content)
                        if source_url:
                            chunk_meta["source_url"] = source_url
                            logger.info(f"Extracted source URL for chunk {j}: {source_url}")
                
                # If the document has metadata, merge it with our metadata
                if hasattr(doc, 'metadata') and doc.metadata:
                    for key, value in doc.metadata.items():
                        if key not in chunk_meta:
                            chunk_meta[key] = value
                
                metadatas.append(chunk_meta)
                ids.append(f"{doc_id}_{j}")
            
            # Add texts to the vector store
            if texts:
                vector_store.add_texts(
                    texts=texts,
                    ids=ids,
                    metadatas=metadatas
                )
                
                added_doc_ids.append(doc_id)
                logger.info(f"Added document with ID {doc_id} and {len(texts)} chunks to vector store")
            else:
                logger.warning(f"Warning: No text content extracted from {file_path}")

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            failed_file_info = {
                "error": str(e),
                "original_filename": os.path.basename(file_path)
            }
            
            # Add any available metadata
            if file_metadata and i < len(file_metadata):
                for key, value in file_metadata[i].items():
                    if key not in failed_file_info:
                        failed_file_info[key] = value
            
            failed_files.append(failed_file_info)
            
                        # Delete the file from uploads directory if it exists
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted corrupted/unprocessable file: {file_path}")
            except Exception as delete_error:
                logger.error(f"Failed to delete file {file_path}: {str(delete_error)}")
        
    return added_doc_ids, failed_files


def process_qa_markdown(file_path, collection_name=None):
    """
    Process Q&A markdown file with specialized chunking strategy for SONU knowledge base
    
    Args:
        file_path (str): Path to the markdown file
        collection_name (str, optional): Collection name to store documents in
    
    Returns:
        list: List of document IDs added to the vector store
    """
    if collection_name is None:
        collection_name = Config.SONU_COLLECTION
    
    embeddings = OpenAIEmbeddings(api_key=Config.OPENAI_API_KEY)
    added_doc_ids = []
    
    try:
        # Read the markdown file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Initialize vector store
        if os.path.exists(Config.VECTORSTORE_PATH):
            vector_store = Chroma(
                persist_directory=Config.VECTORSTORE_PATH,
                embedding_function=embeddings,
                collection_name=collection_name
            )
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(Config.VECTORSTORE_PATH), exist_ok=True)
            vector_store = Chroma(
                persist_directory=Config.VECTORSTORE_PATH,
                embedding_function=embeddings,
                collection_name=collection_name
            )
        
        # Define headers to split by (markdown sections)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        # Initialize markdown header text splitter
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(content)
        
        # Process each section
        documents = []
        for i, doc in enumerate(md_header_splits):
            # Extract Q&A pairs from each section
            qa_pairs = extract_qa_from_section(doc.page_content, doc.metadata)
            
            # If we found Q&A pairs, use them; otherwise use the whole section
            if qa_pairs:
                documents.extend(qa_pairs)
            else:
                # Add section-level metadata
                doc.metadata.update({
                    "source": file_path,
                    "collection": collection_name,
                    "type": "markdown_section",
                    "section_index": i
                })
                
                # Extract source URL for markdown sections
                source_url = extract_source_url_from_markdown_section(doc.page_content)
                if source_url:
                    doc.metadata["source_url"] = source_url
                
                documents.append(doc)
        
        # Add documents to vector store
        for i, doc in enumerate(documents):
            try:
                doc_id = str(uuid.uuid4())
                doc.metadata["doc_id"] = doc_id
                
                vector_store.add_documents([doc], ids=[doc_id])
                added_doc_ids.append(doc_id)
                logger.info(f"Added document {i+1}/{len(documents)} to collection '{collection_name}'")
                
            except Exception as e:
                logger.error(f"Failed to add document {i+1}: {str(e)}")
        
        logger.info(f"Successfully processed {len(added_doc_ids)} documents from {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to process Q&A markdown file {file_path}: {str(e)}")
        raise
    
    return added_doc_ids


def extract_qa_from_section(content, section_metadata):
    """
    Extract question-answer pairs from a markdown section
    
    Args:
        content (str): Section content
        section_metadata (dict): Metadata from the section
    
    Returns:
        list: List of Document objects for Q&A pairs
    """
    qa_documents = []
    
    # Split by ### headers (questions)
    question_pattern = r'###\s+(.+?)\n(.*?)(?=###|\Z)'
    matches = re.findall(question_pattern, content, re.DOTALL)
    
    for i, (question, answer) in enumerate(matches):
        question = question.strip()
        answer = answer.strip()
        
        if question and answer:
            # Create combined Q&A text for better retrieval
            combined_text = f"Question: {question}\n\nAnswer: {answer}"
            
            # Create metadata
            qa_metadata = {
                **section_metadata,
                "question": question,
                "answer": answer,
                "type": "qa_pair",
                "qa_index": i
            }
            
    # Extract source URL for Q&A pairs; fallback to section metadata if missing
        source_url = extract_source_url_from_markdown_section(combined_text)
        if source_url:
            qa_metadata["source_url"] = source_url
        elif section_metadata and section_metadata.get("source_url"):
            qa_metadata["source_url"] = section_metadata["source_url"]
            
            doc = Document(
                page_content=combined_text,
                metadata=qa_metadata
            )
            qa_documents.append(doc)
    
    return qa_documents

