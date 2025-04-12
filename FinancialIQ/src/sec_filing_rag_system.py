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
from google.cloud import logging as cloud_logging
from dotenv import load_dotenv
from pathlib import Path
import faiss
import numpy as np
from langchain.prompts import PromptTemplate
from src.sec_filing_metadata import SECFilingMetadata
from src.cache_manager import CacheManager
from src.logger import FinancialIQLogger

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
    
    def __init__(
        self,
        project_id: str,
        location: str,
        bucket_name: Optional[str] = None,
        pdf_folder: Optional[str] = None,
        metadata_csv_path: Optional[str] = None
    ):
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.pdf_folder = pdf_folder
        
        # Initialize processing tracking
        self.processed_files_path = "processed_files.json"
        self.processed_files = self._load_processed_files()
        
        self.processor = SECFilingProcessor()
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        
        # Initialize GCP clients
        self.storage_client = storage.Client(project=project_id)
        aiplatform.init(project=project_id, location=location)
        
        # Initialize logging
        logging_client = cloud_logging.Client(project=project_id)
        self.logger = logging_client.logger('financialiq-system')
        self.logger.log_struct(
            {"message": "FinancialIQSystem initialized"},
            severity="INFO"
        )
        
        # Initialize metadata handler
        self.metadata_handler = SECFilingMetadata(metadata_csv_path) if metadata_csv_path else None
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_llm()
    
    def _load_processed_files(self) -> Dict[str, str]:
        """Load the record of processed files"""
        if os.path.exists(self.processed_files_path):
            with open(self.processed_files_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_processed_files(self) -> None:
        """Save the record of processed files"""
        with open(self.processed_files_path, 'w') as f:
            json.dump(self.processed_files, f)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for a file based on its metadata"""
        blob = self.storage_client.bucket(self.bucket_name).blob(file_path)
        return f"{blob.name}_{blob.updated}"
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF file and return extracted information"""
        try:
            self.logger.log_struct(
                {"message": f"Processing PDF: {pdf_path}"},
                severity="INFO"
            )
            
            # Process the PDF using the SECFilingProcessor
            result = self.processor.process_document(pdf_path)
            
            if result:
                self.processed_files[pdf_path] = datetime.now().isoformat()
                self._save_processed_files()
            
            return result
            
        except Exception as e:
            self.logger.log_struct(
                {
                    "message": f"Error processing PDF {pdf_path}",
                    "error": str(e)
                },
                severity="ERROR"
            )
            raise
    
    def process_pdfs_from_gcs(self) -> List[Document]:
        """Process PDFs from Google Cloud Storage"""
        self.logger.log_struct(
            {"message": f"Processing PDFs from GCS bucket: {self.bucket_name}"},
            severity="INFO"
        )
        
        bucket = self.storage_client.bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=self.pdf_folder)
        
        documents = []
        for blob in blobs:
            if blob.name.lower().endswith('.pdf'):
                file_hash = self._get_file_hash(blob.name)
                
                # Skip if already processed
                if file_hash in self.processed_files:
                    self.logger.log_struct(
                        {"message": f"Skipping already processed PDF: {blob.name}"},
                        severity="INFO"
                    )
                    continue
                
                self.logger.log_struct(
                    {"message": f"Processing new PDF: {blob.name}"},
                    severity="INFO"
                )
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
        
        self.logger.log_struct(
            {"message": f"Processed {len(documents)} new document chunks"},
            severity="INFO"
        )
        return documents
    
    def process_pdf_from_gcs(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file from Google Cloud Storage"""
        self.logger.log_struct(
            {"message": f"Processing PDF from GCS: {pdf_path}"},
            severity="INFO"
        )
        
        file_hash = self._get_file_hash(pdf_path)
        
        # Skip if already processed
        if file_hash in self.processed_files:
            self.logger.log_struct(
                {"message": f"Skipping already processed PDF: {pdf_path}"},
                severity="INFO"
            )
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
        
        self.logger.log_struct(
            {"message": f"PDF processing completed: {pdf_path}"},
            severity="INFO"
        )
        return result
    
    def setup_vector_store(self, pdf_paths: List[str]) -> None:
        """Set up vector store with processed documents"""
        self.logger.log_struct(
            {"message": f"Setting up vector store with {len(pdf_paths)} PDFs"},
            severity="INFO"
        )
        
        documents = []
        for pdf_path in pdf_paths:
            result = self.process_pdf_from_gcs(pdf_path)
            if result:
                documents.append(result)
        
        if not documents:
            self.logger.log_struct(
                {"message": "No new documents to process"},
                severity="INFO"
            )
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
        
        self.logger.log_struct(
            {"message": "Vector store setup completed"},
            severity="INFO"
        )
    
    def setup_system(self, load_existing: bool = False):
        """Set up the RAG system"""
        if load_existing and self._load_existing_vector_store():
            print("Loaded existing vector store")
        else:
            self._process_documents()
            self._create_vector_store()
        
        self._setup_qa_chain()
    
    def _load_existing_vector_store(self) -> bool:
        """Load existing vector store if available"""
        vector_store_path = Path("vector_store")
        if vector_store_path.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(vector_store_path),
                    self.embeddings
                )
                return True
            except Exception as e:
                print(f"Error loading vector store: {e}")
        return False

    def _process_documents(self):
        """Process documents and create embeddings"""
        if self.bucket_name and self.pdf_folder:
            self._process_cloud_documents()
        else:
            self._process_local_documents()

    def _process_cloud_documents(self):
        """Process documents from Google Cloud Storage"""
        from google.cloud import storage
        
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(self.bucket_name)
        
        for blob in bucket.list_blobs(prefix=self.pdf_folder):
            if blob.name.endswith('.pdf'):
                # Download PDF
                local_path = Path("temp") / Path(blob.name).name
                local_path.parent.mkdir(exist_ok=True)
                blob.download_to_filename(str(local_path))
                
                # Process PDF
                self.processor.process_document(local_path)
                
                # Clean up
                local_path.unlink()

    def _process_local_documents(self):
        """Process documents from local directory"""
        documents_dir = Path("documents")
        for pdf_path in documents_dir.glob("*.pdf"):
            self.processor.process_document(pdf_path)

    def _create_vector_store(self):
        """Create vector store from processed documents"""
        documents = []
        for chunk in self.processor.chunks:
            doc = Document(
                page_content=chunk.text,
                metadata=chunk.metadata
            )
            documents.append(doc)
        
        self.vector_store = FAISS.from_documents(
            documents,
            self.embeddings
        )
        
        # Save vector store
        self.vector_store.save_local("vector_store")

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
        self.logger.log_struct(
            {"message": f"Processing query: {query}"},
            severity="INFO"
        )
        
        if not self.qa_chain:
            error_msg = "Vector store not initialized. Call setup_vector_store first."
            self.logger.log_struct(
                {"message": error_msg},
                severity="ERROR"
            )
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
        
        self.logger.log_struct(
            {"message": f"Query processed with {len(source_metadata)} sources"},
            severity="INFO"
        )
        
        return {
            'answer': result['answer'],
            'sources': source_metadata
        }
    
    def get_metadata_statistics(self) -> Dict[str, Any]:
        """Get statistics about the available filings"""
        self.logger.log_struct(
            {"message": "Retrieving metadata statistics"},
            severity="INFO"
        )
        return self.metadata_handler.get_filing_statistics()
    
    def get_companies(self) -> List[str]:
        """Get list of available companies"""
        self.logger.log_struct(
            {"message": "Retrieving list of companies"},
            severity="INFO"
        )
        return self.metadata_handler.get_companies()
    
    def get_form_types(self) -> List[str]:
        """Get list of available form types"""
        self.logger.log_struct(
            {"message": "Retrieving list of form types"},
            severity="INFO"
        )
        return self.metadata_handler.get_form_types()
    
    def get_recent_filings(self, days: int = 30) -> pd.DataFrame:
        """Get recent filings"""
        self.logger.log_struct(
            {"message": f"Retrieving filings from the last {days} days"},
            severity="INFO"
        )
        return self.metadata_handler.get_recent_filings(days) 