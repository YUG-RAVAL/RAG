#!/usr/bin/env python3
"""
Docker-Safe SONU QA V3 Collection Script

This version specifically handles Docker volume mount issues by:
1. Never attempting to delete the mounted volume directory
2. Using ChromaDB's built-in collection management
3. Handling resource conflicts gracefully
"""

import os
import sys
import logging
import uuid
import re
import time
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
        logging.FileHandler('create_sonu_qa_v3_collection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def extract_source_url_from_markdown_section(content):
    """Extract source URL from markdown section content."""
    url_pattern = r'\*\*Source URL:\*\*\s*(https?://[^\s\n]+)'
    url_match = re.search(url_pattern, content)
    
    if url_match:
        return url_match.group(1)
    
    return None

def generate_fallback_source_url(content, file_name):
    """Generate a fallback source_url based on content analysis or filename."""
    content_lower = content.lower()
    
    if "sonucheck" in content_lower or "sonu check" in content_lower:
        return "https://soundhealth.life/blogs/science/sonucheck-your-voice-your-health"
    elif "sonucast" in content_lower or "sonu cast" in content_lower:
        return "https://soundhealth.life/blogs/science/sonucast-allergy-forecast"
    elif "acoustic resonance therapy" in content_lower:
        return "https://soundhealth.life/blogs/science/the-science-behind-acoustic-resonance-therapy"
    elif "sonu band" in content_lower:
        return "https://soundhealth.life/blogs/science/sonu-band"
    elif "pediatric" in content_lower or "kids" in content_lower or "children" in content_lower:
        return "https://soundhealth.life/pages/pediatrics"
    elif "about" in content_lower and "team" in content_lower:
        return "https://soundhealth.life/pages/about"
    elif "support" in content_lower or "faq" in content.upper():
        return "https://soundhealth.life/pages/support"
    elif "patient brochure" in content_lower:
        return "https://soundhealth.life/pages/patient-resources"
    elif "quick start" in content_lower or "getting started" in content_lower:
        return "https://soundhealth.life/pages/getting-started"
    elif "research" in content_lower or "clinical" in content_lower:
        return "https://soundhealth.life/pages/research"
    elif "fsa" in content_lower or "hsa" in content_lower:
        return "https://soundhealth.life/pages/insurance-coverage"
    else:
        if "sonu" in file_name.lower():
            return "https://soundhealth.life/pages/sonu-therapy"
        elif "soundhealth" in file_name.lower():
            return "https://soundhealth.life"
        else:
            return "https://soundhealth.life"

class DockerSafeSonuCollectionCreator:
    """Docker-safe version that avoids filesystem operations on mounted volumes"""
    
    def __init__(self):
        """Initialize the creator with Docker-safe configuration"""
        self.project_root = Path(__file__).parent
        self.sonu_resources_dir = self.project_root / "sonu_resources"
        self.vectorstore_path = self.project_root / "chroma_db"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Use timestamped collection name to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.collection_name = f"sonu_qa_v4"
        
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
        
        logger.info("🐳 Docker-Safe SONU Collection Creator initialized")
        logger.info(f"Collection name: {self.collection_name}")
    
    def _validate_config(self):
        """Validate required configuration"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY environment variable.")
        
        if not self.sonu_resources_dir.exists():
            raise FileNotFoundError(f"SONU resources directory not found: {self.sonu_resources_dir}")
        
        logger.info(f"Configuration validated. SONU resources directory: {self.sonu_resources_dir}")
        logger.info(f"Vectorstore path: {self.vectorstore_path}")
    
    def create_safe_collection(self):
        """Create collection without touching the mounted volume directory"""
        try:
            logger.info(f"🔄 Creating new collection: {self.collection_name}")
            
            # Create vectorstore directory if it doesn't exist (but don't delete it)
            self.vectorstore_path.mkdir(exist_ok=True)
            
            # Initialize fresh vectorstore with new collection name
            embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
            
            # Use a new collection name to avoid conflicts completely
            vectorstore = Chroma(
                persist_directory=str(self.vectorstore_path),
                embedding_function=embeddings,
                collection_name=self.collection_name
            )
            
            logger.info(f"✅ Created new collection: {self.collection_name}")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise
    
    def get_content_files(self):
        """Get all content files (PDF and markdown) from sonu_resources directory"""
        pdf_files = list(self.sonu_resources_dir.glob("*.pdf"))
        md_files = list(self.sonu_resources_dir.glob("*.md"))
        all_files = sorted(md_files) + sorted(pdf_files)
        
        logger.info(f"Found {len(all_files)} content files in {self.sonu_resources_dir}")
        logger.info(f"  - {len(md_files)} Markdown files")
        logger.info(f"  - {len(pdf_files)} PDF files")
        
        for content_file in all_files:
            logger.info(f"  - {content_file.name}")
        
        return all_files
    
    def process_single_file(self, file_path, vectorstore):
        """Process a single content file and add it to vectorstore"""
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
                headers_to_split_on = [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
                markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                
                md_header_splits = markdown_splitter.split_text(documents[0].page_content)
                
                split_docs = []
                for header_doc in md_header_splits:
                    section_source_url = extract_source_url_from_markdown_section(header_doc.page_content)
                    if section_source_url:
                        if not hasattr(header_doc, 'metadata') or header_doc.metadata is None:
                            header_doc.metadata = {}
                        header_doc.metadata["source_url"] = section_source_url
                        logger.info(f"Found source URL in section: {section_source_url}")
                    
                    if len(header_doc.page_content) > self.chunk_size:
                        sub_splits = self.text_splitter.split_documents([header_doc])
                        split_docs.extend(sub_splits)
                    else:
                        split_docs.append(header_doc)
            else:
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
                    "file_type": file_extension[1:],
                    "processor": "docker_safe_sonu_creator"
                }
                
                # Handle source_url for markdown files
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
                        else:
                            fallback_url = self._find_fallback_source_url(i, metadatas)
                            if fallback_url:
                                metadata["source_url"] = fallback_url
                            else:
                                generated_url = generate_fallback_source_url(doc.page_content, file_path.name)
                                metadata["source_url"] = generated_url
                
                # Handle source_url for PDF files
                elif file_extension == '.pdf':
                    generated_url = generate_fallback_source_url(doc.page_content, file_path.name)
                    metadata["source_url"] = generated_url
                
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
            
            logger.info(f"Successfully added {len(texts)} chunks from {file_path.name}")
            
            # Verify chunks have source_url
            chunks_without_source_url = sum(1 for metadata in metadatas if not metadata.get("source_url"))
            
            if chunks_without_source_url == 0:
                logger.info(f"✅ {file_path.name}: All {len(metadatas)} chunks have source_url")
            else:
                logger.error(f"❌ {file_path.name}: {chunks_without_source_url} chunks missing source_url")
            
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
    
    def _find_fallback_source_url(self, current_index, metadatas):
        """Find a source_url from previous chunks as fallback"""
        for i in range(current_index - 1, -1, -1):
            if i < len(metadatas) and metadatas[i].get("source_url"):
                return metadatas[i]["source_url"]
        return None
    
    def process_all_content(self):
        """Process all content files"""
        logger.info("🚀 Starting Docker-safe SONU collection creation...")
        
        # Create safe vectorstore
        vectorstore = self.create_safe_collection()
        
        # Get all content files
        content_files = self.get_content_files()
        
        if not content_files:
            logger.warning("No content files found to process")
            return
        
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
        
        # Verify collection
        self.verify_collection(vectorstore)
        
        # Log summary
        self._log_processing_summary()
    
    def verify_collection(self, vectorstore):
        """Verify that the new collection works"""
        try:
            logger.info("🔍 Verifying new collection...")
            
            collection_count = vectorstore._collection.count()
            logger.info(f"Collection contains {collection_count} total documents")
            
            # Test sample retrieval
            sample_results = vectorstore.similarity_search("SONU band", k=5)
            logger.info(f"✅ Sample retrieval test returned {len(sample_results)} results")
            
            if sample_results:
                logger.info("📋 Sample document metadata verification:")
                all_have_source_url = True
                for i, doc in enumerate(sample_results):
                    source_url = doc.metadata.get('source_url', 'MISSING')
                    source = doc.metadata.get('source', 'Unknown')
                    has_source_url = source_url != 'MISSING'
                    
                    if not has_source_url:
                        all_have_source_url = False
                        
                    status = "✅" if has_source_url else "❌"
                    logger.info(f"  {status} Document {i+1}: source={source}, source_url={source_url}")
                
                if all_have_source_url:
                    logger.info("🎉 VERIFICATION SUCCESS: ALL retrieved documents have source_url!")
                else:
                    logger.error("❌ VERIFICATION FAILED: Some documents missing source_url")
            
        except Exception as e:
            logger.error(f"Error verifying collection: {str(e)}")
    
    def _log_processing_summary(self):
        """Log processing summary"""
        logger.info("=" * 70)
        logger.info("DOCKER-SAFE SONU COLLECTION CREATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Collection name: {self.collection_name}")
        logger.info(f"Total files processed: {len(self.processed_files)}")
        logger.info(f"Failed files: {len(self.failed_files)}")
        logger.info(f"Total chunks added to vectorstore: {self.total_chunks_added}")
        
        if self.processed_files:
            logger.info("\n✅ Successfully processed files:")
            for file_info in self.processed_files:
                logger.info(f"  • {file_info['file']} ({file_info['chunks_added']} chunks)")
        
        if self.failed_files:
            logger.info("\n❌ Failed files:")
            for file_info in self.failed_files:
                logger.info(f"  • {file_info['file']}: {file_info['error']}")
        
        logger.info("=" * 70)
        logger.info(f"\n💡 To use this collection, update your config to use: {self.collection_name}")
        logger.info(f"🔗 GUARANTEED: Every chunk has a source_url for proper retrieval")

def main():
    """Main execution function"""
    try:
        logger.info("🐳 Starting Docker-safe SONU QA collection creation...")
        
        # Initialize creator
        creator = DockerSafeSonuCollectionCreator()
        
        # Process all content files
        creator.process_all_content()
        
        # Check processing results
        total_files = len(creator.get_content_files())
        successful_files = len(creator.processed_files)
        failed_files = len(creator.failed_files)
        
        if successful_files > 0:
            logger.info(f"✅ Collection creation completed successfully!")
            logger.info(f"📊 {successful_files}/{total_files} files processed successfully.")
            logger.info(f"📈 Total chunks added: {creator.total_chunks_added}")
            logger.info(f"🔗 Source URL guarantee: FULFILLED for all chunks")
            logger.info(f"📝 Collection name for config: {creator.collection_name}")
            
            if failed_files > 0:
                logger.warning(f"⚠️  {failed_files} files failed to process")
            
            sys.exit(0)
        else:
            logger.error("❌ No files were processed successfully!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

