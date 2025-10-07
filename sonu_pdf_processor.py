#!/usr/bin/env python3
"""
SONU Content Processor Script

This script processes all PDF files and markdown files in the sonu_resources directory
and adds them to the SONU vectorDB collection for RAG chatbot retrieval.

Usage:
    python3 sonu_pdf_processor.py

Requirements:
    - Core dependencies: langchain, langchain-chroma, langchain-openai, langchain-community, PyMuPDF
    - OpenAI API key must be set in OPENAI_API_KEY environment variable
    - sonu_resources directory must contain PDF and/or markdown files
"""

import os
import sys
import logging
import uuid
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sonu_pdf_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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

class SonuContentProcessor:
    """Processes PDF files and markdown files and adds them to SONU vectorDB"""
    
    def __init__(self):
        """Initialize the processor with configuration"""
        # Configuration - standalone without Flask dependencies
        self.project_root = Path(__file__).parent
        self.sonu_resources_dir = self.project_root / "sonu_resources"
        self.vectorstore_path = self.project_root / "chroma_db"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.sonu_collection = "sonu_qa_v4"  
        
        # Processing settings
        self.chunk_size = 3000
        self.chunk_overlap = 200
        
        # Results tracking
        self.processed_files = []
        self.failed_files = []
        self.total_chunks_added = 0
        
        # Validate configuration
        self._validate_config()
        
        # Initialize text splitter
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        logger.info("SONU Content Processor initialized")
    
    def _validate_config(self):
        """Validate required configuration"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY environment variable.")
        
        if not self.sonu_resources_dir.exists():
            raise FileNotFoundError(f"SONU resources directory not found: {self.sonu_resources_dir}")
        
        # Create vectorstore directory if it doesn't exist
        self.vectorstore_path.mkdir(exist_ok=True)
        
        logger.info(f"Configuration validated. SONU resources directory: {self.sonu_resources_dir}")
        logger.info(f"Vectorstore path: {self.vectorstore_path}")
        logger.info(f"Collection name: {self.sonu_collection}")
    
    def get_content_files(self):
        """Get all content files (PDF and markdown) from sonu_resources directory"""
        pdf_files = list(self.sonu_resources_dir.glob("*.pdf"))
        md_files = list(self.sonu_resources_dir.glob("*.md"))
        all_files = pdf_files + md_files
        
        logger.info(f"Found {len(all_files)} content files in {self.sonu_resources_dir}")
        logger.info(f"  - {len(pdf_files)} PDF files")
        logger.info(f"  - {len(md_files)} Markdown files")
        
        for content_file in all_files:
            logger.info(f"  - {content_file.name}")
        
        return all_files
    
    def process_single_file(self, file_path, vectorstore):
        """
        Process a single content file (PDF or markdown) and add it to vectorstore
        
        Args:
            file_path (Path): Path to the content file
            vectorstore: Chroma vectorstore instance
            
        Returns:
            dict: Processing results
        """
        try:
            file_extension = file_path.suffix.lower()
            logger.info(f"Processing {file_extension.upper()} file: {file_path.name}")
            
            # Load the document based on file type
            if file_extension == '.pdf':
                loader = PyMuPDFLoader(str(file_path))
                documents = loader.load()
                use_markdown_splitter = False
            elif file_extension == '.md':
                loader = TextLoader(str(file_path), encoding='utf-8')
                documents = loader.load()
                use_markdown_splitter = True
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_extension}",
                    "chunks_added": 0
                }
            
            if not documents:
                logger.warning(f"No content extracted from {file_path.name}")
                return {
                    "success": False,
                    "error": "No content extracted",
                    "chunks_added": 0
                }
            
            # Split documents into chunks based on file type
            if use_markdown_splitter:
                # Use specialized markdown splitter for better structure preservation
                headers_to_split_on = [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
                markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                
                # First split by headers and attach section-level source_url so sub-splits inherit it
                md_header_splits = markdown_splitter.split_text(documents[0].page_content)
                
                # Then further split large sections if needed
                split_docs = []
                for header_doc in md_header_splits:
                    section_source_url = extract_source_url_from_markdown_section(header_doc.page_content)
                    if section_source_url:
                        if not hasattr(header_doc, 'metadata') or header_doc.metadata is None:
                            header_doc.metadata = {}
                        header_doc.metadata["source_url"] = section_source_url
                    if len(header_doc.page_content) > self.chunk_size:
                        # Split large sections further; metadata is propagated by the splitter
                        sub_splits = self.text_splitter.split_documents([header_doc])
                        split_docs.extend(sub_splits)
                    else:
                        split_docs.append(header_doc)
            else:
                # Use regular text splitter for PDFs
                split_docs = self.text_splitter.split_documents(documents)
            
            if not split_docs:
                logger.warning(f"No chunks created from {file_path.name}")
                return {
                    "success": False,
                    "error": "No chunks created",
                    "chunks_added": 0
                }
            
            # Generate unique document ID
            doc_id = str(uuid.uuid4())
            
            # Prepare texts, metadata, and IDs for vectorstore
            texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(split_docs):
                texts.append(doc.page_content)
                
                # Create comprehensive metadata
                metadata = {
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "collection": "sonu",
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(split_docs),
                    "processed_date": datetime.now().isoformat(),
                    "file_type": file_extension[1:],  # Remove the dot
                    "processor": "sonu_content_processor"
                }
                
                # For markdown files, ensure each chunk has a source_url
                if file_extension == '.md':
                    existing_source_url = None
                    if hasattr(doc, 'metadata') and doc.metadata:
                        existing_source_url = doc.metadata.get("source_url")
                    if existing_source_url:
                        metadata["source_url"] = existing_source_url
                    else:
                        source_url = extract_source_url_from_markdown_section(doc.page_content)
                        if source_url:
                            metadata["source_url"] = source_url
                            logger.info(f"Extracted source URL for chunk {i}: {source_url}")
                
                # Add any existing metadata from the document
                if hasattr(doc, 'metadata') and doc.metadata:
                    for key, value in doc.metadata.items():
                        if key not in metadata:
                            metadata[key] = value
                
                metadatas.append(metadata)
                ids.append(f"{doc_id}_chunk_{i}")
            
            # Add texts to vectorstore
            vectorstore.add_texts(
                texts=texts,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(texts)} chunks from {file_path.name} to SONU vectorstore")
            
            return {
                "success": True,
                "chunks_added": len(texts),
                "doc_id": doc_id
            }
            
        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": str(e),
                "chunks_added": 0
            }
    
    def process_all_content(self):
        """Process all content files (PDF and markdown) in the sonu_resources directory"""
        logger.info("Starting SONU content processing...")
        
        # Get all content files
        content_files = self.get_content_files()
        
        if not content_files:
            logger.warning("No content files found to process")
            return
        
        # Get SONU vectorstore
        logger.info("Initializing SONU vectorstore...")
        try:
            embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
            vectorstore = Chroma(
                persist_directory=str(self.vectorstore_path),
                embedding_function=embeddings,
                collection_name=self.sonu_collection
            )
            logger.info("SONU vectorstore initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SONU vectorstore: {str(e)}")
            logger.error(f"Chroma error type: {type(e).__name__}")
            
            # Try to delete the corrupted collection and recreate
            try:
                logger.info(f"Attempting to delete and recreate collection '{self.sonu_collection}'")
                import shutil
                collection_path = self.vectorstore_path / self.sonu_collection
                if collection_path.exists():
                    shutil.rmtree(collection_path)
                    logger.info(f"Deleted corrupted collection directory: {collection_path}")
                
                # Create new collection
                vectorstore = Chroma(
                    persist_directory=str(self.vectorstore_path),
                    embedding_function=embeddings,
                    collection_name=self.sonu_collection
                )
                logger.info(f"Successfully recreated vector store for collection '{self.sonu_collection}'")
            except Exception as recreate_error:
                logger.error(f"Failed to recreate collection '{self.sonu_collection}': {str(recreate_error)}")
                raise
        
        # Process each content file
        for content_path in content_files:
            result = self.process_single_file(content_path, vectorstore)
            
            if result["success"]:
                self.processed_files.append({
                    "file": content_path.name,
                    "chunks_added": result["chunks_added"],
                    "doc_id": result["doc_id"]
                })
                self.total_chunks_added += result["chunks_added"]
            else:
                self.failed_files.append({
                    "file": content_path.name,
                    "error": result["error"]
                })
        
        # Log summary
        self._log_processing_summary()
    
    def _log_processing_summary(self):
        """Log processing summary"""
        logger.info("=" * 60)
        logger.info("SONU CONTENT PROCESSING SUMMARY")
        logger.info("=" * 60)
        all_files = self.get_content_files()
        
        # Show file breakdown by type
        for content_file in all_files:
            file_type = content_file.suffix.upper()[1:] if content_file.suffix else "UNKNOWN"
            logger.info(f"  - {content_file.name} ({file_type})")
        
        logger.info(f"Total content files found: {len(all_files)}")
        logger.info(f"Successfully processed: {len(self.processed_files)}")
        logger.info(f"Failed to process: {len(self.failed_files)}")
        logger.info(f"Total chunks added to vectorstore: {self.total_chunks_added}")
        
        if self.processed_files:
            logger.info("\nSuccessfully processed files:")
            for file_info in self.processed_files:
                logger.info(f"  ✓ {file_info['file']} ({file_info['chunks_added']} chunks)")
        
        if self.failed_files:
            logger.info("\nFailed files:")
            for file_info in self.failed_files:
                logger.info(f"  ✗ {file_info['file']}: {file_info['error']}")
        
        logger.info("=" * 60)
    
    def verify_vectorstore_content(self):
        """Verify that content was added to the vectorstore"""
        try:
            embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
            vectorstore = Chroma(
                persist_directory=str(self.vectorstore_path),
                embedding_function=embeddings,
                collection_name=self.sonu_collection
            )
            collection_count = vectorstore._collection.count()
            logger.info(f"SONU vectorstore now contains {collection_count} total documents")
            
            # Try a sample query to test retrieval
            if collection_count > 0:
                sample_results = vectorstore.similarity_search("SONU", k=3)
                logger.info(f"Sample retrieval test returned {len(sample_results)} results")
                
                if sample_results:
                    logger.info("Sample document metadata:")
                    for i, doc in enumerate(sample_results[:2]):  # Show first 2 results
                        logger.info(f"  Document {i+1}: {doc.metadata.get('source', 'Unknown source')}")
            
        except Exception as e:
            logger.error(f"Error verifying vectorstore content: {str(e)}")
            logger.error(f"Verification error type: {type(e).__name__}")

def main():
    """Main execution function"""
    try:
        # Initialize processor
        processor = SonuContentProcessor()
        
        # Process all content files
        processor.process_all_content()
        
        # Verify results
        processor.verify_vectorstore_content()
        
        # Check processing results
        total_files = len(processor.get_content_files())
        successful_files = len(processor.processed_files)
        failed_files = len(processor.failed_files)
        
        if successful_files > 0:
            logger.info(f"✅ Processing completed! {successful_files}/{total_files} files processed successfully.")
            logger.info(f"📊 Total chunks added: {processor.total_chunks_added}")
            
            if failed_files > 0:
                logger.info(f"⚠️  {failed_files} files had no extractable content (this is normal for some files)")
            
            sys.exit(0)
        else:
            logger.error("❌ No files were processed successfully!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
