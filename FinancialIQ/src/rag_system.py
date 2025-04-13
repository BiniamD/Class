from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_vertexai import VertexAI
from langchain.embeddings.base import Embeddings
import pandas as pd
import warnings
from cryptography.utils import CryptographyDeprecationWarning
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.schema import Document
from dataclasses import dataclass
import shutil
from datetime import datetime
import traceback
import sys
import json

# Suppress specific deprecation warnings
warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning, 
                      message='.*ARC4 has been moved to cryptography.hazmat.decrepit.*')

from .document_processor import EnhancedSECFilingProcessor
from .cache_manager import CacheManager
from .logger import FinancialIQLogger

@dataclass
class VectorStoreConfig:
    """Configuration for vector store creation and management."""
    save_path: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    create_backup: bool = True
    max_retries: int = 3
    backup_retention: int = 3  # Number of backups to keep
    validate_store: bool = True
    compression: bool = False  # Whether to compress the vector store
    index_type: str = "L2"  # Index type for FAISS (L2, IP, etc.)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.save_path:
            self.save_path = str(Path(self.save_path))
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.backup_retention < 0:
            raise ValueError("backup_retention must be non-negative")

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper around sentence_transformers to make it compatible with LangChain."""
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using sentence_transformers."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using sentence_transformers."""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

class RAGSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", config: Optional[VectorStoreConfig] = None):
        self.logger = FinancialIQLogger()
        self.doc_processor = EnhancedSECFilingProcessor()
        self.cache_manager = CacheManager()
        self.config = config or VectorStoreConfig()
        
        # Initialize embedding model with error handling
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            # Create LangChain compatible embeddings
            self.embeddings = SentenceTransformerEmbeddings(self.embedding_model)
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}", "INITIALIZATION")
            raise
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )
        
        self.llm = VertexAI(
            model_name="text-bison@002",
            max_output_tokens=1024,
            temperature=0.1,
            top_p=0.8,
            top_k=40,
        )
        self.vector_store = None

    def _create_backup(self, save_path: str) -> None:
        """Create a backup of the existing vector store if it exists."""
        if not os.path.exists(save_path):
            return
            
        backup_dir = os.path.join(os.path.dirname(save_path), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"vector_store_{timestamp}")
        
        # Copy the existing store to backup
        shutil.copytree(save_path, backup_path)
        
        # Clean up old backups if exceeding retention
        backups = sorted([d for d in os.listdir(backup_dir) if d.startswith("vector_store_")])
        if len(backups) > self.config.backup_retention:
            for old_backup in backups[:-self.config.backup_retention]:
                shutil.rmtree(os.path.join(backup_dir, old_backup))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_vector_store(self, processed_docs: List[Dict[str, Any]], config: Optional[VectorStoreConfig] = None):
        """Create a vector store from processed documents with retry logic."""
        # Use provided config or default
        current_config = config or self.config

        # --- Input Validation ---
        if not isinstance(processed_docs, list):
            self.logger.error("Invalid input: processed_docs must be a list.", "VECTOR_STORE")
            raise TypeError("processed_docs must be a list.")
        if not all(isinstance(doc, dict) for doc in processed_docs):
            self.logger.error("Invalid input: all items in processed_docs must be dictionaries.", "VECTOR_STORE")
            raise TypeError("All items in processed_docs must be dictionaries.")
        self.logger.error(f"Validated basic structure of processed_docs (List[Dict]). Count: {len(processed_docs)}", "VECTOR_STORE_DEBUG")
        # --- End Input Validation ---

        try:
            texts = []
            metadatas = []

            # Log input validation
            self.logger.error(f"Starting vector store creation with {len(processed_docs)} documents", "VECTOR_STORE_DEBUG")

            for idx, doc in enumerate(processed_docs):
                # --- More Granular Validation (Example) ---
                if not isinstance(doc.get("metadata"), dict):
                     self.logger.error(f"Document {idx} has invalid or missing metadata.", "VECTOR_STORE_WARNING")
                     # Decide how to handle: skip doc, use default metadata, or raise error
                     # continue # Example: skip this document

                # Extract text from various sections (Add type checks if needed)
                doc_texts = []
                metadata_dict = doc.get("metadata", {}) # Use .get for safety

                # Add metadata text
                doc_texts.extend([
                    str(metadata_dict.get("company_name", "")), # Ensure string conversion
                    str(metadata_dict.get("filing_date", "")),
                    str(metadata_dict.get("form_type", ""))
                ])

                # Add tables
                if "tables" in doc and isinstance(doc["tables"], list):
                    for table in doc["tables"]:
                        if isinstance(table, pd.DataFrame):
                            doc_texts.append(table.to_string())
                        # else: log unexpected table type?

                # Add risk factors
                if "risk_factors" in doc and isinstance(doc["risk_factors"], dict):
                     for factor in doc["risk_factors"].values():
                         if isinstance(factor, str):
                             doc_texts.append(factor)
                         # else: log unexpected risk factor type?

                # Add financial metrics
                if "financial_metrics" in doc and isinstance(doc["financial_metrics"], dict):
                    for metric in doc["financial_metrics"].values():
                        if isinstance(metric, (str, int, float, np.number)): # Allow numpy numbers
                            doc_texts.append(str(metric)) # Ensure string conversion
                        # else: log unexpected metric type?

                # Join all text and split into chunks
                full_text = "\n".join(filter(None, doc_texts))
                if full_text:
                    chunks = self.text_splitter.split_text(full_text)

                    # Create metadata for each chunk
                    chunk_metadata = {
                        "source": str(metadata_dict.get("company_name", "Unknown")),
                        "filing_date": str(metadata_dict.get("filing_date", "Unknown")),
                        "form_type": str(metadata_dict.get("form_type", "Unknown")),
                    }

                    texts.extend(chunks)
                    metadatas.extend([chunk_metadata] * len(chunks))

                    # Log chunk creation for first document
                    if idx == 0:
                        self.logger.error(
                            f"First document chunks: Count={len(chunks)}, "
                            f"First chunk type={type(chunks[0]).__name__}, "
                            f"First metadata type={type(chunk_metadata).__name__}",
                            "VECTOR_STORE_DEBUG"
                        )

            if not texts:
                self.logger.error("No valid text content found in processed documents", "VECTOR_STORE")
                return None

            # Log text and metadata stats
            self.logger.error(
                f"Prepared {len(texts)} total chunks with metadata. "
                f"Text types: {set(type(t).__name__ for t in texts)}, "
                f"Metadata types: {set(type(m).__name__ for m in metadatas)}",
                "VECTOR_STORE_DEBUG"
            )

            # Create documents with metadata
            documents = []
            for idx, (text, metadata) in enumerate(zip(texts, metadatas)):
                try:
                    # Log the raw input
                    self.logger.error(f"Processing document {idx}:", "VECTOR_STORE_DEBUG")
                    self.logger.error(f"Text type: {type(text).__name__}, Text length: {len(str(text))}", "VECTOR_STORE_DEBUG")
                    self.logger.error(f"Metadata type: {type(metadata).__name__}, Metadata keys: {list(metadata.keys())}", "VECTOR_STORE_DEBUG")

                    # Ensure text is a string and not empty
                    if not isinstance(text, str):
                        text = str(text)
                    if not text.strip():
                        self.logger.error(f"Skipping empty document {idx}", "VECTOR_STORE_DEBUG")
                        continue

                    # Split text into chunks using RecursiveCharacterTextSplitter
                    try:
                        chunks = self.text_splitter.split_text(text)
                        self.logger.error(f"Split document {idx} into {len(chunks)} chunks", "VECTOR_STORE_DEBUG")
                        
                        # Log chunk statistics
                        if chunks:
                            self.logger.error(
                                f"Chunk statistics for document {idx}: "
                                f"Min length: {min(len(c) for c in chunks)}, "
                                f"Max length: {max(len(c) for c in chunks)}, "
                                f"Avg length: {sum(len(c) for c in chunks)/len(chunks):.2f}",
                                "VECTOR_STORE_DEBUG"
                            )
                    except Exception as e:
                        self.logger.error(f"Error splitting text for document {idx}: {str(e)}", "VECTOR_STORE_DEBUG")
                        continue

                    # Ensure all metadata values are strings and not empty
                    clean_metadata = {}
                    for key, value in metadata.items():
                        if value is None:
                            clean_metadata[key] = ""
                        elif isinstance(value, (str, int, float, bool)):
                            clean_metadata[key] = str(value)
                        elif isinstance(value, (list, dict)):
                            # Convert lists and dicts to JSON strings
                            try:
                                clean_metadata[key] = json.dumps(value)
                            except:
                                clean_metadata[key] = str(value)
                        else:
                            clean_metadata[key] = str(value)

                    # Create documents for each chunk
                    for chunk_idx, chunk in enumerate(chunks):
                        try:
                            # Log chunk details
                            self.logger.error(
                                f"Processing chunk {chunk_idx} of document {idx}: "
                                f"Length: {len(chunk)}, "
                                f"Type: {type(chunk).__name__}",
                                "VECTOR_STORE_DEBUG"
                            )

                            # Create document with cleaned content and metadata
                            doc = Document(
                                page_content=chunk,
                                metadata={**clean_metadata, "chunk_index": str(chunk_idx)}
                            )
                            documents.append(doc)
                        except Exception as e:
                            self.logger.error(f"Error creating document for chunk {chunk_idx} of document {idx}: {str(e)}", "VECTOR_STORE_DEBUG")
                            continue

                except Exception as e:
                    self.logger.error(f"Error processing document {idx}: {str(e)}", "VECTOR_STORE_DEBUG")
                    continue

            if not documents:
                self.logger.error("No valid documents were created", "VECTOR_STORE")
                return None

            self.logger.error(f"Successfully created {len(documents)} documents", "VECTOR_STORE_DEBUG")

            # Validate documents before creating vector store
            for idx, doc in enumerate(documents):
                try:
                    if not isinstance(doc.page_content, str):
                        raise TypeError(f"Document {idx} content must be string, got {type(doc.page_content)}")
                    if not isinstance(doc.metadata, dict):
                        raise TypeError(f"Document {idx} metadata must be dict, got {type(doc.metadata)}")
                    for key, value in doc.metadata.items():
                        if not isinstance(value, str):
                            raise TypeError(f"Document {idx} metadata value for key '{key}' must be string, got {type(value)}")
                except Exception as e:
                    self.logger.error(f"Validation error for document {idx}: {str(e)}", "VECTOR_STORE_DEBUG")
                    raise

            # Create vector store with validation
            try:
                self.logger.error("Starting FAISS.from_documents...", "VECTOR_STORE_DEBUG")
                self.logger.error(f"Number of documents: {len(documents)}", "VECTOR_STORE_DEBUG")
                self.logger.error(f"First document content type: {type(documents[0].page_content).__name__}", "VECTOR_STORE_DEBUG")
                self.logger.error(f"First document metadata type: {type(documents[0].metadata).__name__}", "VECTOR_STORE_DEBUG")

                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                self.logger.error("FAISS.from_documents completed successfully", "VECTOR_STORE_DEBUG")

                # Validate vector store if configured
                if current_config.validate_store:
                    if not self.vector_store.index.ntotal == len(documents):
                        raise ValueError("Vector store size mismatch with input documents")
                    self.logger.error("Vector store validation successful.", "VECTOR_STORE_DEBUG")

                # Save vector store if path is provided
                if current_config.save_path:
                    # Log the initial type and value
                    self.logger.error(f"Initial current_config.save_path: {current_config.save_path} (type: {type(current_config.save_path).__name__})", "VECTOR_STORE_DEBUG")

                    # Ensure save_path is not a dictionary (already validated in __post_init__, but double-check)
                    if isinstance(current_config.save_path, dict):
                        self.logger.error("Error: current_config.save_path is a dictionary.", "VECTOR_STORE")
                        raise TypeError("save_path cannot be a dictionary. Please provide a string path.")

                    # Convert to string (handles Path objects as well)
                    save_path = str(current_config.save_path)
                    self.logger.error(f"Converted save_path to string: {save_path} (type: {type(save_path).__name__})", "VECTOR_STORE_DEBUG")

                    # Ensure the directory exists (FAISS saves to a directory)
                    try:
                        self.logger.error(f"Calling os.makedirs with path: {save_path} (type: {type(save_path).__name__})", "VECTOR_STORE_DEBUG")
                        os.makedirs(save_path, exist_ok=True)
                        self.logger.error(f"Ensured directory exists: {save_path}", "VECTOR_STORE_DEBUG")
                    except Exception as e:
                        self.logger.error(f"Error creating directory {save_path}: {str(e)}", "VECTOR_STORE")
                        self.logger.error(f"Path type at makedirs failure: {type(save_path).__name__}", "VECTOR_STORE_DEBUG")
                        raise

                    # Create backup if configured
                    if current_config.create_backup:
                        self.logger.error(f"Calling _create_backup with path: {save_path} (type: {type(save_path).__name__})", "VECTOR_STORE_DEBUG")
                        self._create_backup(save_path)

                    # Save the vector store using the string path
                    self.logger.error(f"Calling vector_store.save_local with path: {save_path} (type: {type(save_path).__name__})", "VECTOR_STORE_DEBUG")
                    try:
                        self.vector_store.save_local(save_path)
                        self.logger.error(f"Vector store successfully saved to {save_path}", "VECTOR_STORE")
                    except Exception as e:
                         self.logger.error(f"Error during save_local: {str(e)}", "VECTOR_STORE")
                         self.logger.error(f"Path type at save_local failure: {type(save_path).__name__}", "VECTOR_STORE_DEBUG")
                         raise

                return self.vector_store

            except Exception as e:
                self.logger.error(f"Error during FAISS index creation or saving: {str(e)}", "VECTOR_STORE")
                # --- Log the full traceback ---
                self.logger.error(f"Traceback:\n{traceback.format_exc()}", "VECTOR_STORE_DEBUG")
                # --- End log traceback ---
                raise # Re-raise the exception after logging

        except Exception as e:
            self.logger.error(f"Critical Error in vector store creation process: {str(e)}", "VECTOR_STORE")
            # --- Log the full traceback ---
            # Temporarily log traceback with VECTOR_STORE tag to ensure visibility
            # self.logger.error(f"Traceback:\n{traceback.format_exc()}", "VECTOR_STORE") # Comment out logger version
            print("--- TRACEBACK START ---", file=sys.stderr) # Add print to stderr
            traceback.print_exc(file=sys.stderr) # Print traceback to stderr
            print("--- TRACEBACK END ---", file=sys.stderr)
            # --- End log traceback ---
            raise # Re-raise the exception after logging

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def load_vector_store(self, load_path: str):
        """Load a previously saved vector store with retry logic."""
        try:
            self.vector_store = FAISS.load_local(load_path)
            # Validate loaded vector store
            if not hasattr(self.vector_store, 'index'):
                raise ValueError("Invalid vector store format")
        except Exception as e:
            self.logger.error(f"Error loading vector store from {load_path}: {str(e)}", "VECTOR_STORE")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        """Query the RAG system with a question and retry logic."""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Please process documents first.")
            
            # Validate vector store before querying
            if not hasattr(self.vector_store, 'index') or self.vector_store.index.ntotal == 0:
                raise ValueError("Vector store is empty or invalid")
                
            # Create retrieval chain
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            # Get response
            response = qa_chain({"query": question})
            
            return {
                "answer": response["result"],
                "sources": [doc.metadata for doc in response["source_documents"]],
                "question": question
            }
            
        except Exception as e:
            self.logger.error(f"Error querying RAG system: {str(e)}", "QUERY")
            return {
                "error": str(e),
                "question": question
            } 