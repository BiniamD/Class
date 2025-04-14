"""
FinancialIQ: SEC Filing RAG System
Core implementation for document processing and question answering
"""

import os
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import pandas as pd
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from google.cloud import storage
from google.cloud import aiplatform
from dotenv import load_dotenv
from pathlib import Path
import faiss
import numpy as np
from langchain.prompts import PromptTemplate
from sec_filing_metadata import SECFilingMetadata
from cache_manager import CacheManager
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

class SECFilingProcessor:
    """Handles processing of SEC filing PDFs"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract metadata from SEC filing text"""
        # Basic metadata extraction - can be enhanced with more sophisticated parsing
        metadata = {
            "source": filename,
            "processed_date": datetime.now().strftime("%Y-%m-%d"),
            "form_type": self._extract_form_type(text),
            "company_name": self._extract_company_name(text),
            "filing_date": self._extract_filing_date(text)
        }
        return metadata
    
    def _extract_form_type(self, text: str) -> str:
        """Extract form type from filing text"""
        form_patterns = {
            "10-K": r"Form\s+10-K",
            "10-Q": r"Form\s+10-Q",
            "8-K": r"Form\s+8-K",
            "DEF 14A": r"DEF\s+14A",
            "S-1": r"Form\s+S-1"
        }
        
        for form_type, pattern in form_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return form_type
        return "Unknown"
    
    def _extract_company_name(self, text: str) -> str:
        """Extract company name from filing text"""
        # Look for common company name patterns
        patterns = [
            r"Registrant:\s*([^\n]+)",
            r"Company Name:\s*([^\n]+)",
            r"Name of Registrant:\s*([^\n]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "Unknown Company"
    
    def _extract_filing_date(self, text: str) -> str:
        """Extract filing date from text"""
        date_patterns = [
            r"Filing Date:\s*([^\n]+)",
            r"Date of Report:\s*([^\n]+)",
            r"Filing Date of Report:\s*([^\n]+)"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return datetime.now().strftime("%Y-%m-%d")
    
    def process_document(self, pdf_path: str) -> List[Document]:
        """Process a single PDF document into chunks with metadata"""
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Extract metadata
        metadata = self.extract_metadata(text, os.path.basename(pdf_path))
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk_id": i,
                    "page": i + 1  # Approximate page number
                }
            )
            documents.append(doc)
        
        return documents

class FinancialIQSystem:
    """Main system class for SEC filing analysis"""
    
    def __init__(self, project_id: str, location: str, metadata_csv_path: str):
        self.project_id = project_id
        self.location = location
        self.metadata_csv_path = metadata_csv_path
        self.vector_store = None
        self.embeddings = None
        self.processor = SECFilingProcessor()
        
    def setup_system(self, load_existing: bool = False):
        """Setup the system with vector store and embeddings"""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
            
            # Load or create vector store
            vector_store_path = "data/vector_store"
            if load_existing and os.path.exists(vector_store_path):
                self.vector_store = FAISS.load_local(
                    vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Allow deserialization since we trust our own data
                )
                print("Loaded existing vector store")
            else:
                raise ValueError("Vector store not found. Please run initialization script first.")
                
            return True
        except Exception as e:
            print(f"Error setting up system: {str(e)}")
            raise
            
    def search_documents(self, query: str, k: int = 5):
        """Search for relevant documents"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Please run setup_system first.")
            
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            raise

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF file and return extracted information"""
        try:
            print(f"Processing PDF: {pdf_path}")
            
            # Process the PDF using the SECFilingProcessor
            result = self.processor.process_document(pdf_path)
            
            if result:
                self.processed_files[pdf_path] = datetime.now().isoformat()
                self._save_processed_files()
            
            return result
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def process_pdfs_from_gcs(self) -> List[Document]:
        """Process PDFs from Google Cloud Storage"""
        print(f"Processing PDFs from GCS bucket: {self.bucket_name}")
        
        bucket = self.storage_client.bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=self.pdf_folder)
        
        documents = []
        for blob in blobs:
            if blob.name.lower().endswith('.pdf'):
                file_hash = self._get_file_hash(blob.name)
                
                # Skip if already processed
                if file_hash in self.processed_files:
                    print(f"Skipping already processed PDF: {blob.name}")
                    continue
                
                print(f"Processing new PDF: {blob.name}")
                # Download PDF to temporary file
                temp_path = f"/tmp/{os.path.basename(blob.name)}"
                blob.download_to_filename(temp_path)
                
                # Process PDF
                docs = self.processor.process_document(temp_path)
                documents.extend(docs)
                
                # Mark as processed
                self.processed_files[file_hash] = blob.name
                self._save_processed_files()
                
                # Clean up
                os.remove(temp_path)
        
        print(f"Processed {len(documents)} new document chunks")
        return documents
    
    def process_pdf_from_gcs(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file from Google Cloud Storage"""
        print(f"Processing PDF from GCS: {pdf_path}")
        
        file_hash = self._get_file_hash(pdf_path)
        
        # Skip if already processed
        if file_hash in self.processed_files:
            print(f"Skipping already processed PDF: {pdf_path}")
            return None
        
        # Download PDF to temporary location
        temp_path = f"/tmp/{os.path.basename(pdf_path)}"
        blob = self.storage_client.bucket(self.bucket_name).blob(pdf_path)
        blob.download_to_filename(temp_path)
        
        # Process the PDF
        result = self.processor.process_document(temp_path)
        
        # Mark as processed
        self.processed_files[file_hash] = pdf_path
        self._save_processed_files()
        
        # Clean up temporary file
        os.remove(temp_path)
        
        print(f"PDF processing completed: {pdf_path}")
        return result
    
    def setup_vector_store(self, pdf_paths: List[str]) -> None:
        """Set up vector store with processed documents"""
        print(f"Setting up vector store with {len(pdf_paths)} PDFs")
        
        documents = []
        for pdf_path in pdf_paths:
            result = self.process_pdf_from_gcs(pdf_path)
            if result:
                documents.append(result)
        
        if not documents:
            print("No new documents to process")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        texts = []
        for doc in documents:
            chunks = text_splitter.split_text(doc['text'])
            for chunk in chunks:
                texts.append(chunk)
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(texts, self.embeddings)
        else:
            self.vector_store.add_texts(texts)
        
        # Set up QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True,
            verbose=True
        )
        
        print("Vector store setup completed")
    
    def _setup_qa_chain(self):
        """Set up the question-answering chain"""
        prompt_template = """
        You are FinancialIQ, an intelligent assistant for financial analysts.
        Answer the following question about SEC filings based solely on the retrieved documents.
        Be precise, clear, and use proper financial terminology.
        If the documents don't contain enough information to answer confidently, acknowledge the limitations.
        Cite the specific SEC forms, companies, and filing dates used in your answer.

        Context: {context}

        Question: {question}

        Answer:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question about SEC filings"""
        if not self.qa_chain:
            raise Exception("QA chain not initialized. Call setup_system() first.")
        
        result = self.qa_chain({"query": question})
        
        # Format sources
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "company": doc.metadata.get("company_name", "Unknown"),
                "form_type": doc.metadata.get("form_type", "Unknown"),
                "filing_date": doc.metadata.get("filing_date", "Unknown"),
                "page": doc.metadata.get("page_number", "Unknown"),
                "source": doc.metadata.get("section", "Unknown")
            })
        
        return {
            "answer": result["result"],
            "sources": sources
        }

    def get_financial_metrics(self, company: str) -> Dict[str, Any]:
        """Get financial metrics for a specific company"""
        # Query the vector store for financial statements
        docs = self.vector_store.similarity_search(
            f"financial statements {company}",
            k=5
        )
        
        # Extract metrics from documents
        metrics = {}
        for doc in docs:
            if doc.metadata.get("company_name") == company:
                # Extract metrics using regex or other methods
                # This is a simplified example
                text = doc.page_content
                metrics.update(self._extract_metrics_from_text(text))
        
        return metrics

    def _extract_metrics_from_text(self, text: str) -> Dict[str, float]:
        """Extract financial metrics from text"""
        metrics = {}
        
        # Revenue
        revenue_match = re.search(r"Total Revenue\s*\$?\s*([\d,]+\.?\d*)", text)
        if revenue_match:
            metrics["revenue"] = float(revenue_match.group(1).replace(",", ""))
        
        # Net Income
        net_income_match = re.search(r"Net Income\s*\$?\s*([\d,]+\.?\d*)", text)
        if net_income_match:
            metrics["net_income"] = float(net_income_match.group(1).replace(",", ""))
        
        # EPS
        eps_match = re.search(r"Earnings Per Share\s*\$?\s*([\d,]+\.?\d*)", text)
        if eps_match:
            metrics["eps"] = float(eps_match.group(1).replace(",", ""))
        
        return metrics

    def _initialize_embeddings(self):
        """Initialize Vertex AI embeddings"""
        self.embeddings = VertexAIEmbeddings(
            model_name="textembedding-gecko@001",
            project=self.project_id,
            location=self.location
        )

    def _initialize_llm(self):
        """Initialize Gemini 2.0 flash model"""
        self.llm = VertexAI(
            model_name="gemini-1.5-flash-001",
            project=self.project_id,
            location=self.location,
            max_output_tokens=1024,
            temperature=0.1
        )

    def process_query(self, query: str, chat_history: List[tuple] = None) -> Dict[str, Any]:
        """Process a user query and return response with sources"""
        print(f"Processing query: {query}")
        
        if not self.qa_chain:
            error_msg = "Vector store not initialized. Call setup_vector_store first."
            print(f"Error: {error_msg}")
            raise ValueError(error_msg)
        
        if chat_history is None:
            chat_history = []
        
        result = self.qa_chain({"question": query, "chat_history": chat_history})
        
        # Get metadata for source documents
        source_metadata = []
        for doc in result['source_documents']:
            # Extract accession number from document metadata if available
            accession_no = doc.metadata.get('accession_no')
            if accession_no:
                metadata = self.metadata_handler.get_filing_metadata(accession_no)
                if metadata:
                    source_metadata.append(metadata)
        
        print(f"Query processed with {len(source_metadata)} sources")
        
        return {
            'answer': result['answer'],
            'sources': source_metadata
        }
    
    def get_metadata_statistics(self) -> Dict[str, Any]:
        """Get statistics about the available filings"""
        print("Retrieving metadata statistics")
        return self.metadata_handler.get_filing_statistics()
    
    def get_companies(self) -> List[str]:
        """Get list of available companies"""
        print("Retrieving list of companies")
        return self.metadata_handler.get_companies()
    
    def get_form_types(self) -> List[str]:
        """Get list of available form types"""
        print("Retrieving list of form types")
        return self.metadata_handler.get_form_types()
    
    def get_recent_filings(self, days: int = 30) -> pd.DataFrame:
        """Get recent filings"""
        print(f"Retrieving filings from the last {days} days")
        return self.metadata_handler.get_recent_filings(days)

    def add_to_vector_store(self, document):
        """Add a document to the vector store"""
        try:
            # Handle both dictionary and Document inputs
            if isinstance(document, dict):
                doc = Document(
                    page_content=document.get('text', ''),
                    metadata=document.get('metadata', {})
                )
            else:
                doc = document
            
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents([doc], self.embeddings)
            else:
                self.vector_store.add_documents([doc])
            
            # Save vector store
            self.vector_store.save_local("vector_store")
            
            print("Document added to vector store")
        except Exception as e:
            print(f"Error adding document to vector store: {str(e)}")
            raise 