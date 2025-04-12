from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_vertexai import VertexAI

from .document_processor import EnhancedSECFilingProcessor
from .cache_manager import CacheManager
from .logger import FinancialIQLogger

class RAGSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = FinancialIQLogger()
        self.doc_processor = EnhancedSECFilingProcessor()
        self.cache_manager = CacheManager()
        self.embedding_model = SentenceTransformer(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
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
        
    def process_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple SEC filing documents and prepare them for RAG."""
        processed_docs = []
        for file_path in file_paths:
            try:
                # Check cache first
                cached_doc = self.cache_manager.get_document_cache(file_path)
                if cached_doc:
                    self.logger.info(f"Using cached document for {file_path}")
                    processed_docs.append(cached_doc)
                    continue
                
                # Process document if not in cache
                doc_result = self.doc_processor.process_document(file_path)
                self.cache_manager.set_document_cache(file_path, doc_result)
                processed_docs.append(doc_result)
                
            except Exception as e:
                self.logger.error(f"Error processing document {file_path}: {str(e)}")
                continue
                
        return processed_docs
        
    def create_vector_store(self, processed_docs: List[Dict[str, Any]], save_path: Optional[str] = None):
        """Create a vector store from processed documents."""
        try:
            texts = []
            metadatas = []
            
            for doc in processed_docs:
                # Extract text from various sections
                doc_texts = [
                    doc.get("metadata", {}).get("company_name", ""),
                    *[table.to_string() for table in doc.get("tables", [])],
                    *doc.get("risk_factors", {}).values(),
                    *[str(metric) for metric in doc.get("financial_metrics", {}).values()],
                ]
                
                # Split texts into chunks
                chunks = self.text_splitter.split_text("\n".join(doc_texts))
                
                # Create metadata for each chunk
                chunk_metadata = [{
                    "source": doc.get("metadata", {}).get("company_name", "Unknown"),
                    "filing_date": doc.get("metadata", {}).get("filing_date", "Unknown"),
                    "form_type": doc.get("metadata", {}).get("form_type", "Unknown"),
                } for _ in chunks]
                
                texts.extend(chunks)
                metadatas.extend(chunk_metadata)
            
            # Create embeddings and vector store
            embeddings = [self.embedding_model.encode(text) for text in texts]
            self.vector_store = FAISS.from_embeddings(
                embeddings=embeddings,
                texts=texts,
                metadatas=metadatas
            )
            
            if save_path:
                self.vector_store.save_local(save_path)
                
        except Exception as e:
            self.logger.error(f"Error creating vector store: {str(e)}")
            raise
            
    def load_vector_store(self, load_path: str):
        """Load a previously saved vector store."""
        try:
            self.vector_store = FAISS.load_local(load_path)
        except Exception as e:
            self.logger.error(f"Error loading vector store from {load_path}: {str(e)}")
            raise
            
    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Please process documents first.")
                
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
            self.logger.error(f"Error querying RAG system: {str(e)}")
            return {
                "error": str(e),
                "question": question
            } 