"""
RAG System Implementation
Handles document processing, vector storage, and retrieval
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import numpy as np
from src.logger import FinancialIQLogger

@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    save_path: str = "data/vector_store"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    create_backup: bool = True
    validate_store: bool = True

class RAGSystem:
    """RAG system for document processing and retrieval"""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        config: Optional[VectorStoreConfig] = None
    ):
        self.logger = FinancialIQLogger("logs")
        self.config = config or VectorStoreConfig()
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Initialize vector store
        self.vector_store = None
        
    def create_vector_store(self, documents: List[Dict[str, Any]]) -> Optional[FAISS]:
        """Create vector store from processed documents"""
        try:
            # Convert documents to langchain format
            docs = []
            for doc in documents:
                # Extract text chunks
                chunks = self.text_splitter.split_text(doc.get("content", ""))
                
                # Create documents with metadata
                for chunk in chunks:
                    docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": doc.get("source", "unknown"),
                            "page": doc.get("page", 0),
                            "metadata": doc.get("metadata", {})
                        }
                    ))
            
            if not docs:
                self.logger.error("No documents to create vector store")
                return None
            
            # Create vector store
            self.vector_store = FAISS.from_documents(
                documents=docs,
                embedding=self.embeddings
            )
            
            # Save vector store
            if self.config.save_path:
                os.makedirs(self.config.save_path, exist_ok=True)
                self.vector_store.save_local(self.config.save_path)
                self.logger.info(f"Vector store saved to {self.config.save_path}")
            
            return self.vector_store
            
        except Exception as e:
            self.logger.error(f"Error creating vector store: {str(e)}")
            return None
    
    def load_vector_store(self, path: Optional[str] = None) -> bool:
        """Load vector store from disk"""
        try:
            load_path = path or self.config.save_path
            if not os.path.exists(load_path):
                self.logger.error(f"Vector store path does not exist: {load_path}")
                return False
            
            self.vector_store = FAISS.load_local(
                load_path,
                self.embeddings
            )
            self.logger.info(f"Vector store loaded from {load_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search vector store for relevant documents"""
        if not self.vector_store:
            self.logger.error("Vector store not initialized")
            return []
        
        try:
            # Search for relevant documents
            docs = self.vector_store.similarity_search_with_score(
                query,
                k=k
            )
            
            # Format results
            results = []
            for doc, score in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching vector store: {str(e)}")
            return []