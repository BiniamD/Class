"""
FinancialIQ: A Streamlit UI for SEC Filing Q&A System
ADTA 5770: Final Project
"""

import os
import re
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from google.cloud import storage
import concurrent.futures

# Set page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="FinancialIQ - SEC Filing Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import SEC filing processor and RAG system
from src.sec_filing_rag_system import FinancialIQSystem
from src.visualization import FinancialVisualizer
from src.logger import FinancialIQLogger
from src.cache_manager import CacheManager
from src.document_processor import EnhancedSECFilingProcessor

# Styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E3A8A;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #1E3A8A;
    margin-bottom: 1rem;
}
.source-box {
    background-color: #F3F4F6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #EFF6FF;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #3B82F6;
}
.footnote {
    font-size: 0.8rem;
    color: #6B7280;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'financial_iq' not in st.session_state:
    st.session_state.financial_iq = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = FinancialVisualizer()
if 'companies' not in st.session_state:
    st.session_state.companies = []
if 'form_types' not in st.session_state:
    st.session_state.form_types = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = None
if 'current_sources' not in st.session_state:
    st.session_state.current_sources = None
if 'financial_metrics' not in st.session_state:
    st.session_state.financial_metrics = None

class FinancialIQApp:
    def __init__(self):
        """Initialize the FinancialIQ application"""
        # Get configuration from environment
        self.project_id = "adta5760nlp"  # Set project ID directly
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.local_dir = os.getenv("LOCAL_DIR", "documents")
        
        # Create necessary directories
        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Set metadata path in Google Cloud Storage
        self.bucket_name = "adta5770-docs"
        self.metadata_blob = "documents/sec_filings.csv"
        self.pdfs_blob = "documents/sec_filings_pdf"
        self.local_metadata_path = os.path.join("data", "metadata.csv")
        
        # Initialize components
        self.logger = FinancialIQLogger()
        self.cache_manager = CacheManager()
        
        # Initialize storage client
        self.storage_client = storage.Client(project=self.project_id)
        
        # Check for existing vector store in GCS
        try:
            vector_store_blob = self.storage_client.bucket(self.bucket_name).blob("vector_store/vector_store.faiss")
            if vector_store_blob.exists():
                # Download vector store
                vector_store_path = "data/vector_store"
                os.makedirs(vector_store_path, exist_ok=True)
                vector_store_blob.download_to_filename(os.path.join(vector_store_path, "vector_store.faiss"))
                self.logger.info("Loaded existing vector store from GCS")
            
            # Check for existing processed files in GCS
            processed_files_blob = self.storage_client.bucket(self.bucket_name).blob("processed_files.json")
            if processed_files_blob.exists():
                processed_files_blob.download_to_filename("processed_files.json")
                self.logger.info("Loaded existing processed files from GCS")
        except Exception as e:
            self.logger.error(f"Error loading existing data from GCS: {str(e)}")
        
        # Initialize RAG system with local metadata path
        self.rag_system = FinancialIQSystem(
            project_id=self.project_id,
            location=self.location,
            metadata_csv_path=self.local_metadata_path
        )
        
        # Initialize prompt store
        self.prompt_store = {
            "Financial Analysis": [
                {
                    "name": "Revenue Growth Analysis",
                    "prompt": "Analyze the company's revenue growth trends over the past 3 years. Include key drivers and potential risks.",
                    "category": "Financial Metrics"
                },
                {
                    "name": "Risk Factor Analysis",
                    "prompt": "Identify and analyze the top 5 risk factors mentioned in the filing. Assess their potential impact.",
                    "category": "Risk Analysis"
                },
                {
                    "name": "Executive Compensation",
                    "prompt": "Analyze the executive compensation structure and compare it with industry standards.",
                    "category": "Governance"
                }
            ],
            "Industry Analysis": [
                {
                    "name": "Market Position",
                    "prompt": "Evaluate the company's market position and competitive advantages.",
                    "category": "Market Analysis"
                },
                {
                    "name": "Industry Trends",
                    "prompt": "Identify key industry trends and their potential impact on the company.",
                    "category": "Market Analysis"
                }
            ],
            "Regulatory Compliance": [
                {
                    "name": "Compliance Status",
                    "prompt": "Assess the company's compliance with SEC regulations and reporting requirements.",
                    "category": "Compliance"
                }
            ]
        }
        
        try:
            # Initialize storage client
            self.storage_client = storage.Client(project=self.project_id)
            
            # Download metadata file if it doesn't exist locally
            if not os.path.exists(self.local_metadata_path):
                self.logger.info("Downloading metadata file from Google Cloud Storage...")
                bucket = self.storage_client.bucket(self.bucket_name)
                blob = bucket.blob(self.metadata_blob)
                blob.download_to_filename(self.local_metadata_path)
                self.logger.info("Metadata file downloaded successfully")
            
            # Load metadata into memory
            self.metadata_df = pd.read_csv(self.local_metadata_path)
            self.logger.info(f"Loaded metadata for {len(self.metadata_df)} filings")
            
            # Initialize visualizer
            self.visualizer = FinancialVisualizer()
            
            # Initialize vector store if exists
            vector_store_path = "data/vector_store"
            if os.path.exists(vector_store_path):
                self.rag_system._load_existing_vector_store()
                self.logger.info("Vector store loaded successfully")
                
            self.logger.info("FinancialIQApp initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing FinancialIQApp: {str(e)}")
            raise
    
    def get_companies(self) -> List[str]:
        """Get list of unique companies from metadata"""
        return sorted(self.metadata_df['companyName'].unique().tolist())
    
    def get_form_types(self) -> List[str]:
        """Get list of unique form types from metadata"""
        return sorted(self.metadata_df['formType'].unique().tolist())
    
    def get_filings_by_company(self, company: str) -> pd.DataFrame:
        """Get filings for a specific company"""
        return self.metadata_df[self.metadata_df['companyName'] == company]
    
    def get_filings_by_form_type(self, form_type: str) -> pd.DataFrame:
        """Get filings of a specific form type"""
        return self.metadata_df[self.metadata_df['formType'] == form_type]
    
    def get_filing_url(self, accession_no: str) -> str:
        """Get filing URL for a specific accession number"""
        filing = self.metadata_df[self.metadata_df['accessionNo'] == accession_no]
        if not filing.empty:
            return filing.iloc[0]['filing_url']
        return None

    def process_documents(self, uploaded_files=None):
        """Process documents from either uploaded files or GCS"""
        try:
            if uploaded_files:
                self.logger.info(f"Processing {len(uploaded_files)} uploaded files")
                documents = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                error_container = st.empty()
                
                # Process uploaded files in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_file = {
                        executor.submit(self._process_uploaded_file, file): file 
                        for file in uploaded_files
                    }
                    
                    for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                        file = future_to_file[future]
                        try:
                            doc = future.result()
                            if doc:
                                documents.append(doc)
                                progress_bar.progress((i + 1) / len(uploaded_files))
                                status_text.text(f"Processed {i + 1}/{len(uploaded_files)} files")
                            else:
                                error_msg = f"Failed to process: {file.name} - Document processing returned no content"
                                self.logger.error(error_msg)
                                error_container.error(error_msg)
                        except Exception as e:
                            error_msg = f"Error processing uploaded file {file.name}: {str(e)}"
                            self.logger.error(error_msg)
                            error_container.error(error_msg)
                            continue
                
                status_text.text("Processing complete!")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
            else:
                # Process documents from Google Cloud Storage
                self.logger.info(f"Processing documents from GCS: {self.bucket_name}/{self.pdfs_blob}")
                try:
                    # Initialize document processor with enhanced parallel processing
                    processor = EnhancedSECFilingProcessor(
                        project_id=self.project_id,
                        bucket_name=self.bucket_name,
                        max_workers=4,
                        batch_size=10
                    )
                    
                    # Process documents in parallel with progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    error_container = st.empty()
                    log_viewer = st.expander("View Processing Logs", expanded=False)
                    
                    # Process documents in batches
                    documents = processor.process_gcs_directory(self.pdfs_blob)
                    
                    # Update progress
                    progress_bar.progress(1.0)
                    status_text.text(f"Processed {len(documents)} documents")
                    
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    error_msg = f"Error accessing GCS: {str(e)}"
                    self.logger.error(error_msg)
                    st.error(error_msg)
                    return None
            
            if not documents:
                error_msg = "No documents were successfully processed"
                self.logger.error(error_msg)
                st.error(error_msg)
                return None
            
            # Create vector store with parallel processing
            self.logger.info(f"Creating vector store with {len(documents)} documents")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Creating vector store...")
            
            try:
                # Process documents in parallel
                processed_docs = []
                processed_files = []
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_doc = {executor.submit(self.rag_system.process_pdf, doc): doc for doc in documents}
                    for i, future in enumerate(concurrent.futures.as_completed(future_to_doc)):
                        doc_path = future_to_doc[future]
                        try:
                            doc = future.result()
                            if doc:
                                processed_docs.append(doc)
                                processed_files.append(os.path.basename(doc_path))
                                self.logger.info(f"Successfully processed {os.path.basename(doc_path)}")
                                progress_bar.progress((i + 1) / len(documents))
                                status_text.text(f"Processed {i + 1}/{len(documents)} documents")
                        except Exception as e:
                            self.logger.error(f"Error processing {os.path.basename(doc_path)}: {str(e)}")
                            continue
                
                if not processed_docs:
                    error_msg = "No documents were successfully processed"
                    self.logger.error(error_msg)
                    st.error(error_msg)
                    return None
                
                # Create vector store from processed documents
                self.vector_store = self.rag_system.create_vector_store(processed_docs)
                
                # Save vector store locally
                vector_store_path = "data/vector_store"
                self.vector_store.save_local(vector_store_path)
                
                # Upload vector store to GCS
                vector_store_blob = self.storage_client.bucket(self.bucket_name).blob("vector_store/vector_store.faiss")
                vector_store_blob.upload_from_filename(os.path.join(vector_store_path, "vector_store.faiss"))
                
                # Upload processed files to GCS
                processed_files_blob = self.storage_client.bucket(self.bucket_name).blob("processed_files.json")
                processed_files_blob.upload_from_filename("processed_files.json")
                
                progress_bar.progress(1.0)
                status_text.text("Vector store created and persisted successfully!")
                
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                self.logger.info("Vector store created and persisted successfully")
                return processed_docs
                
            except Exception as e:
                error_msg = f"Error in vector store creation: {str(e)}"
                self.logger.error(error_msg)
                st.error(error_msg)
                progress_bar.empty()
                status_text.empty()
                return None
            
        except Exception as e:
            error_msg = f"Unexpected error in process_documents: {str(e)}"
            self.logger.error(error_msg)
            st.error(error_msg)
            return None

    def _process_uploaded_file(self, file):
        """Process a single uploaded file"""
        try:
            # Save uploaded file temporarily
            temp_path = os.path.join(self.local_dir, file.name)
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())
            
            # Process the PDF
            self.logger.info(f"Processing uploaded file: {file.name}")
            doc = self.rag_system.process_pdf(temp_path)
            
            # Clean up
            os.remove(temp_path)
            
            return doc
        except Exception as e:
            self.logger.error(f"Error processing uploaded file {file.name}: {str(e)}")
            return None

    def query_documents(self, query):
        """Query the RAG system."""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Processing query
            status_text.text("Processing query...")
            progress_bar.progress(0.3)
            
            # Step 2: Searching vector store
            status_text.text("Searching relevant documents...")
            progress_bar.progress(0.6)
            
            # Step 3: Generating answer
            status_text.text("Generating answer...")
            progress_bar.progress(0.9)
            
            result = self.rag_system.answer_question(query)
            
            # Complete
            progress_bar.progress(1.0)
            status_text.text("Query processed successfully!")
            
            time.sleep(0.5)  # Show completion message briefly
            progress_bar.empty()
            status_text.empty()
            
            self.logger.info(f"Query processed successfully: {query}")
            return result
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {"error": str(e), "question": query}

    def create_visualizations(self, processed_docs):
        """Create visualizations from processed documents."""
        try:
            figures = []
            for doc in processed_docs:
                # Financial trends
                if "tables" in doc and len(doc["tables"]) > 0:
                    fig = self.visualizer.create_financial_trends(doc["tables"][0])
                    figures.append(("Financial Trends", fig))
                
                # Risk factors
                if "risk_factors" in doc:
                    fig = self.visualizer.create_risk_factor_heatmap(doc["risk_factors"])
                    figures.append(("Risk Factor Analysis", fig))
                
                # Financial metrics
                if "financial_metrics" in doc:
                    fig = self.visualizer.create_financial_metrics_comparison(doc["financial_metrics"])
                    figures.append(("Financial Metrics", fig))
            
            self.logger.info(f"Created {len(figures)} visualizations")
            return figures
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            return []

    def check_system_status(self):
        """Check if the system is ready to use."""
        try:
            status = {
                "vector_store_ready": False,
                "documents_processed": 0,
                "rag_system_ready": False,
                "metadata_loaded": False,
                "gcs_documents_available": False
            }
            
            # Check vector store
            vector_store_path = "data/vector_store"
            if os.path.exists(vector_store_path):
                status["vector_store_ready"] = True
                # Count processed documents
                status["documents_processed"] = len(os.listdir(vector_store_path))
            
            # Check RAG system
            if hasattr(self.rag_system, 'vector_store') and self.rag_system.vector_store is not None:
                status["rag_system_ready"] = True
            
            # Check metadata
            if hasattr(self, 'metadata_df') and not self.metadata_df.empty:
                status["metadata_loaded"] = True
                status["total_filings"] = len(self.metadata_df)
            
            # Check Google Cloud Storage for documents
            try:
                bucket = self.storage_client.bucket(self.bucket_name)
                blobs = list(bucket.list_blobs(prefix=self.pdfs_blob))
                status["gcs_documents_available"] = len(blobs) > 0
                status["gcs_document_count"] = len(blobs)
            except Exception as e:
                self.logger.error(f"Error checking GCS documents: {str(e)}")
                status["gcs_documents_available"] = False
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error checking system status: {str(e)}")
            return {
                "vector_store_ready": False,
                "documents_processed": 0,
                "rag_system_ready": False,
                "metadata_loaded": False,
                "gcs_documents_available": False,
                "error": str(e)
            }

    def display_financial_visualizations(self, company=None):
        """Display financial visualizations for a company"""
        st.markdown("### Financial Visualizations")
        
        # Get company selection if not provided
        if company is None:
            companies = self.get_companies()
            company = st.selectbox("Select Company", companies)
        
        if company:
            # Show progress container
            progress_container = st.container()
            with progress_container:
                st.markdown("#### Processing Status")
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Get company's filings
            status_text.text("Retrieving company filings...")
            progress_bar.progress(0.2)
            company_filings = self.get_filings_by_company(company)
            
            if not company_filings.empty:
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Revenue Trends", "Risk Factors", "Executive Compensation"])
                
                with tab1:
                    # Revenue trends visualization
                    status_text.text("Processing revenue data...")
                    progress_bar.progress(0.4)
                    revenue_data = self.visualizer.extract_revenue_data(company_filings)
                    if revenue_data is not None:
                        status_text.text("Creating revenue chart...")
                        progress_bar.progress(0.6)
                        fig = self.visualizer.create_revenue_chart(revenue_data)
                        st.plotly_chart(fig, use_container_width=True, key="revenue_chart")
                    else:
                        st.info("No revenue data available for visualization")
                
                with tab2:
                    # Risk factors visualization
                    status_text.text("Processing risk factors...")
                    progress_bar.progress(0.7)
                    risk_data = self.visualizer.extract_risk_factors(company_filings)
                    if risk_data is not None:
                        status_text.text("Creating risk factors chart...")
                        progress_bar.progress(0.8)
                        fig = self.visualizer.create_risk_factors_chart(risk_data)
                        st.plotly_chart(fig, use_container_width=True, key="risk_factors_chart")
                    else:
                        st.info("No risk factor data available for visualization")
                
                with tab3:
                    # Executive compensation visualization
                    status_text.text("Processing executive compensation...")
                    progress_bar.progress(0.9)
                    comp_data = self.visualizer.extract_executive_compensation(company_filings)
                    if comp_data is not None:
                        status_text.text("Creating compensation chart...")
                        progress_bar.progress(1.0)
                        fig = self.visualizer.create_compensation_chart(comp_data)
                        st.plotly_chart(fig, use_container_width=True, key="compensation_chart")
                    else:
                        st.info("No executive compensation data available for visualization")
                
                # Show completion message
                status_text.text("Visualization complete!")
                time.sleep(1)  # Show completion message briefly
                progress_bar.empty()
                status_text.empty()
                
                # Show summary statistics
                st.markdown("#### Summary Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Filings", len(company_filings))
                with col2:
                    st.metric("Form Types", len(company_filings['formType'].unique()))
                with col3:
                    # Convert string dates to datetime objects
                    min_date = pd.to_datetime(company_filings['filedAt'].min())
                    max_date = pd.to_datetime(company_filings['filedAt'].max())
                    st.metric("Date Range", 
                             f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
            else:
                st.warning(f"No filings found for {company}")
        else:
            st.info("Please select a company to view financial visualizations")

    def run(self):
        """Run the FinancialIQ application"""
        # Display header
        st.markdown('<h1 class="main-header">FinancialIQ</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">SEC Filing Analysis & Q&A System</p>', unsafe_allow_html=True)
        
        # Configure sidebar
        with st.sidebar:
            st.markdown("## Navigation")
            page = st.radio(
                "Select Page",
                ["Document Processing", "Query Interface", "Visualizations", "System Status"]
            )
            
            # Add system status in sidebar
            st.markdown("## System Status")
            status = self.check_system_status()
            
            if status["vector_store_ready"]:
                st.success("✓ Vector Store Ready")
            else:
                st.error("✗ Vector Store Not Ready")
                
            if status["rag_system_ready"]:
                st.success("✓ RAG System Ready")
            else:
                st.error("✗ RAG System Not Ready")
                
            if status["metadata_loaded"]:
                st.success(f"✓ Metadata Loaded ({status['total_filings']} filings)")
            else:
                st.error("✗ Metadata Not Loaded")
                
            if status["gcs_documents_available"]:
                st.success(f"✓ GCS Documents Available ({status['gcs_document_count']} files)")
            else:
                st.error("✗ GCS Documents Not Available")
        
        # Display system flow diagram
        st.markdown("### System Architecture")
        st.markdown("""
        ```mermaid
        graph TD
            subgraph "Frontend"
                A[Streamlit UI] -->|User Input| B[FinancialIQApp]
                B -->|Display| A
            end
            
            subgraph "Backend Services"
                B -->|Process| C[Document Processor<br>PyPDF2, pdfplumber]
                B -->|Query| D[RAG System<br>LangChain, FAISS]
                C -->|Extract| E[Metadata<br>Pandas]
                C -->|Process| F[Vector Store<br>FAISS]
                D -->|Search| F
                D -->|Generate| G[LLM<br>Google Vertex AI]
                G -->|Answer| B
            end
            
            subgraph "Storage"
                H[Google Cloud Storage<br>gcsfs] -->|Download| C
                I[Local Storage<br>File System] -->|Cache| B
                J[Logging System<br>Cloud Logging] -->|Monitor| B
                J -->|Monitor| C
                J -->|Monitor| D
            end
            
            subgraph "APIs & Tools"
                K[Google Cloud APIs<br>Storage, Logging]
                L[Plotly<br>Visualization]
                M[Pandas<br>Data Analysis]
                N[Streamlit<br>Web Framework]
            end
        ```
        """)
        
        # Display system status
        status = self.check_system_status()
        st.markdown("### System Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Vector Store",
                "Ready" if status["vector_store_ready"] else "Not Ready",
                f"{status['documents_processed']} documents"
            )
        
        with col2:
            st.metric(
                "RAG System",
                "Ready" if status["rag_system_ready"] else "Not Ready",
                "Initialized" if status["rag_system_ready"] else "Not Initialized"
            )
        
        with col3:
            st.metric(
                "Metadata",
                "Loaded" if status["metadata_loaded"] else "Not Loaded",
                f"{status['total_filings']} filings"
            )
            
        with col4:
            st.metric(
                "GCS Documents",
                "Available" if status["gcs_documents_available"] else "Not Available",
                f"{status['gcs_document_count']} files"
            )

        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Document Processing", "Prompt Store", "Query Interface"])
        
        with tab1:
            # Document processing section
            st.header("Document Processing")
            
            # System readiness and action buttons
            if status["vector_store_ready"] and status["rag_system_ready"]:
                st.success("System Ready for Queries")
            else:
                st.warning("System Not Ready - Process Documents First")
                
                # Add prominent button to process documents from GCS
                if status["gcs_documents_available"]:
                    st.markdown("### Process Documents")
                    st.info(
                        f"""
                        Documents are available in Google Cloud Storage:
                        - Bucket: {self.bucket_name}
                        - Path: {self.pdfs_blob}
                        - Count: {status.get('gcs_document_count', 0)} files
                        
                        Click the button below to process these documents and make the system ready for queries.
                        """
                    )
                    
                    if st.button("🚀 Process Documents from Google Cloud Storage", type="primary"):
                        with st.spinner("Processing documents from Google Cloud Storage..."):
                            try:
                                processed_docs = self.process_documents()
                                if processed_docs:
                                    st.success(f"✅ Successfully processed {len(processed_docs)} documents")
                                    # Refresh status
                                    status = self.check_system_status()
                                    st.experimental_rerun()
                                else:
                                    st.error("❌ Failed to process documents. Check the logs for details.")
                            except Exception as e:
                                st.error(f"❌ Error processing documents: {str(e)}")
                                self.logger.error(f"Error in document processing: {str(e)}")
                else:
                    st.error(
                        f"""
                        ❌ No documents found in Google Cloud Storage:
                        - Bucket: {self.bucket_name}
                        - Path: {self.pdfs_blob}
                        
                        Please ensure the documents are uploaded to the correct location.
                        """
                    )
            
            # File upload section
            st.header("Upload SEC Filings")
            uploaded_files = st.file_uploader("Choose SEC filing PDFs", type="pdf", accept_multiple_files=True)
            
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    processed_docs = self.process_documents(uploaded_files)
                    if processed_docs:
                        st.success(f"✅ Successfully processed {len(processed_docs)} documents")
                    else:
                        st.error("❌ Failed to process documents. Check the logs for details.")
        
        with tab2:
            # Prompt store section
            st.header("Prompt Store")
            
            # Display prompt categories
            selected_category = st.selectbox(
                "Select Category",
                list(self.prompt_store.keys())
            )
            
            # Display prompts for selected category
            prompts = self.prompt_store[selected_category]
            for prompt in prompts:
                with st.expander(f"{prompt['name']} ({prompt['category']})"):
                    st.write(prompt['prompt'])
                    if st.button("Use This Prompt", key=f"use_{prompt['name']}"):
                        st.session_state.current_prompt = prompt['prompt']
                        st.experimental_rerun()
        
        with tab3:
            # Query interface section
            st.header("Query Interface")
            
            # Company and form type filters
            col1, col2 = st.columns(2)
            with col1:
                selected_company = st.selectbox(
                    "Select Company",
                    ["All"] + sorted(self.metadata_df['companyName'].unique().tolist())
                )
            with col2:
                selected_form_type = st.selectbox(
                    "Select Form Type",
                    ["All"] + sorted(self.metadata_df['formType'].unique().tolist())
                )
            
            # Query input
            query = st.text_area(
                "Enter your question",
                value=st.session_state.get('current_prompt', ''),
                height=100
            )
            
            if query:
                if not status["rag_system_ready"]:
                    st.error("RAG system not ready. Please process documents first.")
                else:
                    with st.spinner("Processing query..."):
                        result = self.query_documents(query)
                        
                        if "error" in result:
                            st.error(result["error"])
        else:
                            st.subheader("Answer")
                            st.write(result["answer"])
                            
                            if "sources" in result:
                                st.subheader("Sources")
                                for source in result["sources"]:
                                    st.write(f"- {source['source']} ({source.get('filing_date', 'N/A')})")
        
        # Add log viewer section
        st.markdown("### System Logs")
        log_viewer = st.expander("View Detailed System Logs", expanded=False)
        with log_viewer:
            try:
                # Read the latest log file
                log_files = sorted([f for f in os.listdir("logs") if f.startswith("financialiq_")])
                if log_files:
                    latest_log = log_files[-1]
                    with open(os.path.join("logs", latest_log), "r") as f:
                        log_content = f.read()
                    
                    # Display logs with syntax highlighting
                    st.code(log_content, language="json")
            except Exception as e:
                st.error(f"Error reading log files: {str(e)}")

def create_required_directories():
    """Create all required directories for the application"""
    directories = ["data", "logs", "cache", "documents"]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            st.success(f"Created directory: {directory}")
        except Exception as e:
            st.error(f"Error creating directory {directory}: {str(e)}")
            raise

def initialize_system(project_id: str, location: str, local_dir: str = "documents"):
    """Initialize the FinancialIQ system"""
    try:
        # Ensure local_dir is not None
        if local_dir is None:
            local_dir = "documents"
        
        # Create required directories
        create_required_directories()
        
        # Set metadata path
        metadata_path = os.path.join(local_dir, "metadata.csv")
        
        # Initialize the system
            st.session_state.financial_iq = FinancialIQSystem(
                project_id=project_id,
                location=location,
            metadata_csv_path=metadata_path
            )
        
        st.success("System initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        raise

def extract_metadata():
    """Extract metadata from processed documents"""
    # This would normally come from your documents
    # For demonstration, we'll use sample data
    st.session_state.companies = [
        "Apple Inc.",
        "Microsoft Corporation",
        "Amazon.com, Inc.",
        "Tesla, Inc.",
        "Alphabet Inc."
    ]
    
    st.session_state.form_types = [
        "10-K",
        "10-Q",
        "8-K",
        "DEF 14A",
        "S-1"
    ]
    
    # Sample financial metrics data
    st.session_state.financial_metrics = pd.DataFrame({
        "Company": ["Apple Inc.", "Apple Inc.", "Apple Inc.", "Microsoft Corporation", "Microsoft Corporation", "Microsoft Corporation"],
        "Year": [2022, 2023, 2024, 2022, 2023, 2024],
        "Revenue (Billions)": [365.8, 383.3, 394.3, 198.3, 211.9, 226.1],
        "Net Income (Billions)": [99.8, 97.0, 111.4, 72.7, 74.5, 82.9],
        "EPS": [6.11, 6.14, 7.35, 9.65, 10.03, 11.25]
    })

def display_answer():
    """Display the current answer and sources"""
    if st.session_state.current_answer:
        st.markdown(f"<div>{st.session_state.current_answer}</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='sub-header'>Sources</div>", unsafe_allow_html=True)
        
        for source in st.session_state.current_sources:
            st.markdown(
                f"""<div class='source-box'>
                    <strong>{source['company']}</strong> - {source['form_type']} ({source['filing_date']})<br>
                    Page: {source['page']} | Source: {source['source']}
                </div>""",
                unsafe_allow_html=True
            )

def display_financial_visualizations(company=None):
    """Display financial visualizations"""
    if st.session_state.financial_metrics is None:
        return
    
    df = st.session_state.financial_metrics
    
    if company and company != "All":
        df = df[df["Company"] == company]
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Financial Trends",
        "Financial Ratios",
        "Risk Analysis",
        "Executive Compensation"
    ])
    
    with tab1:
        # Financial Trends
        metrics = ["Revenue (Billions)", "Net Income (Billions)", "EPS"]
        fig_trends = st.session_state.visualizer.create_financial_trends(
            df.rename(columns={"Year": "date"}),
            metrics,
            "Financial Performance Trends"
        )
        st.plotly_chart(fig_trends, use_container_width=True)
    
    with tab2:
        # Financial Ratios
        ratios = ["EPS"]
        fig_ratios = st.session_state.visualizer.create_financial_ratios(
            df.rename(columns={"Year": "date"}),
            ratios,
            "Financial Ratios"
        )
        st.plotly_chart(fig_ratios, use_container_width=True)
    
    with tab3:
        # Risk Analysis
        # Sample risk factor data
        risk_data = pd.DataFrame({
            "Company": ["Apple Inc.", "Microsoft Corporation", "Amazon.com, Inc."],
            "Market Risk": [0.8, 0.6, 0.7],
            "Operational Risk": [0.5, 0.4, 0.6],
            "Regulatory Risk": [0.3, 0.4, 0.5]
        }).set_index("Company")
        
        fig_risk = st.session_state.visualizer.create_risk_factor_heatmap(
            risk_data,
            "Risk Factor Analysis"
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with tab4:
        # Executive Compensation
        # Sample compensation data
        comp_data = pd.DataFrame({
            "executive": ["CEO", "CFO", "CTO"],
            "salary": [1000000, 800000, 750000],
            "bonus": [2000000, 1500000, 1200000],
            "stock_awards": [5000000, 3000000, 2500000],
            "option_awards": [3000000, 2000000, 1500000],
            "other": [500000, 400000, 300000]
        })
        
        fig_comp = st.session_state.visualizer.create_executive_compensation(
            comp_data,
            "Executive Compensation Breakdown"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

def main():
    """Main application function"""
    try:
        # Get configuration
        project_id = "adta5760nlp"  # Set project ID directly
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        local_dir = os.getenv("LOCAL_DIR", "documents")
        
        # Ensure local_dir is not None
        if local_dir is None:
            local_dir = "documents"
        
        # Initialize system
        if "financial_iq" not in st.session_state:
            initialize_system(project_id, location, local_dir)
        
        # Initialize app if not already done
        if "app" not in st.session_state:
            st.session_state.app = FinancialIQApp()
        
        # Run the app
        st.session_state.app.run()
        
    except Exception as e:
        st.error(f"Error in main application: {str(e)}")
        raise

if __name__ == "__main__":
    main() 