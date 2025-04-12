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
            
            # Initialize RAG system with local metadata path
            self.rag_system = FinancialIQSystem(
                project_id=self.project_id,
                location=self.location,
                metadata_csv_path=self.local_metadata_path
            )
            
            # Initialize visualizer
            self.visualizer = FinancialVisualizer()
            
            # Initialize vector store if exists
            vector_store_path = "data/vector_store"
            if os.path.exists(vector_store_path):
                self.rag_system.load_vector_store(vector_store_path)
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
        """Process documents from either uploaded files or Google Cloud Storage"""
        try:
            if uploaded_files:
                self.logger.info(f"Processing {len(uploaded_files)} uploaded files")
                documents = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {file.name}")
                        progress_bar.progress((i + 0.5) / len(uploaded_files))
                        
                        # Save uploaded file temporarily
                        temp_path = os.path.join(self.local_dir, file.name)
                        with open(temp_path, "wb") as f:
                            f.write(file.getvalue())
                        
                        # Process the PDF
                        self.logger.info(f"Processing uploaded file: {file.name}")
                        doc = self.rag_system.process_pdf(temp_path)
                        if doc:
                            documents.append(doc)
                            self.logger.info(f"Successfully processed: {file.name}")
                        else:
                            self.logger.error(f"Failed to process: {file.name}")
                        
                        # Clean up
                        os.remove(temp_path)
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                    except Exception as e:
                        self.logger.error(f"Error processing uploaded file {file.name}: {str(e)}")
                        continue
                
                status_text.text("Processing complete!")
                time.sleep(1)  # Show completion message briefly
                progress_bar.empty()
                status_text.empty()
                
            else:
                # Process documents from Google Cloud Storage
                self.logger.info(f"Processing documents from GCS: {self.bucket_name}/{self.pdfs_blob}")
                try:
                    # List PDFs in the specified directory
                    blobs = self.storage_client.list_blobs(
                        self.bucket_name,
                        prefix=self.pdfs_blob
                    )
                    pdf_blobs = [blob for blob in blobs if blob.name.endswith('.pdf')]
                    self.logger.info(f"Found {len(pdf_blobs)} PDF files in GCS")
                    
                    if not pdf_blobs:
                        self.logger.error(f"No PDF files found in GCS path: {self.pdfs_blob}")
                        return None
                    
                    documents = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, blob in enumerate(pdf_blobs):
                        try:
                            status_text.text(f"Processing file {i+1}/{len(pdf_blobs)}: {os.path.basename(blob.name)}")
                            progress_bar.progress((i + 0.5) / len(pdf_blobs))
                            
                            # Download PDF temporarily
                            temp_path = os.path.join(self.local_dir, os.path.basename(blob.name))
                            self.logger.info(f"Downloading {blob.name} to {temp_path}")
                            blob.download_to_filename(temp_path)
                            
                            # Process the PDF
                            self.logger.info(f"Processing GCS file: {blob.name}")
                            doc = self.rag_system.process_pdf(temp_path)
                            if doc:
                                documents.append(doc)
                                self.logger.info(f"Successfully processed: {blob.name}")
                            else:
                                self.logger.error(f"Failed to process: {blob.name}")
                            
                            # Clean up
                            os.remove(temp_path)
                            progress_bar.progress((i + 1) / len(pdf_blobs))
                            
                        except Exception as e:
                            self.logger.error(f"Error processing GCS file {blob.name}: {str(e)}")
                            continue
                    
                    status_text.text("Processing complete!")
                    time.sleep(1)  # Show completion message briefly
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    self.logger.error(f"Error accessing GCS: {str(e)}")
                    return None
            
            if not documents:
                self.logger.error("No documents were successfully processed")
                return None
            
            # Create vector store with progress
            self.logger.info(f"Creating vector store with {len(documents)} documents")
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Creating vector store...")
                
                self.vector_store = self.rag_system.create_vector_store(documents)
                progress_bar.progress(1.0)
                status_text.text("Vector store created successfully!")
                
                time.sleep(1)  # Show completion message briefly
                progress_bar.empty()
                status_text.empty()
                
                self.logger.info("Vector store created successfully")
                return documents
            except Exception as e:
                self.logger.error(f"Error creating vector store: {str(e)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Unexpected error in process_documents: {str(e)}")
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

    def run(self):
        """Run the FinancialIQ application"""
        # Display header
        st.markdown('<h1 class="main-header">FinancialIQ</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">SEC Filing Analysis & Q&A System</p>', unsafe_allow_html=True)
        
        # Display system flow diagram
        st.markdown("### System Architecture")
        st.markdown("""
        ```mermaid
        graph TD
            A[User Interface] -->|Upload/Query| B[FinancialIQApp]
            B -->|Process| C[Document Processor]
            B -->|Query| D[RAG System]
            C -->|Extract| E[Metadata]
            C -->|Process| F[Vector Store]
            D -->|Search| F
            D -->|Generate| G[LLM]
            G -->|Answer| B
            H[Google Cloud Storage] -->|Download| C
            I[Local Storage] -->|Cache| B
            J[Logging System] -->|Monitor| B
            J -->|Monitor| C
            J -->|Monitor| D
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
                    st.success(f"Successfully processed {len(processed_docs)} documents")
                    
                    # Create and display visualizations
                    figures = self.create_visualizations(processed_docs)
                    if figures:
                        st.header("Document Analysis")
                        cols = st.columns(len(figures))
                        for i, (title, fig) in enumerate(figures):
                            with cols[i]:
                                st.subheader(title)
                                st.plotly_chart(fig, use_container_width=True)
        
        # Query section
        st.header("Ask Questions")
        query = st.text_input("Enter your question about the SEC filings")
        
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
        
        # Display metadata filters
        st.sidebar.header("Filters")
        companies = self.get_companies()
        form_types = self.get_form_types()
        
        selected_company = st.sidebar.selectbox("Company", ["All"] + companies)
        selected_form_type = st.sidebar.selectbox("Form Type", ["All"] + form_types)
        
        # Display filtered filings
        if selected_company != "All" or selected_form_type != "All":
            filtered_filings = self.metadata_df
            if selected_company != "All":
                filtered_filings = filtered_filings[filtered_filings['companyName'] == selected_company]
            if selected_form_type != "All":
                filtered_filings = filtered_filings[filtered_filings['formType'] == selected_form_type]
            
            st.header("Filtered Filings")
            st.dataframe(filtered_filings[['companyName', 'formType', 'filedAt']])
        
    def display_financial_visualizations(self, company=None):
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
            fig_trends = self.visualizer.create_financial_trends(
                df.rename(columns={"Year": "date"}),
                metrics,
                "Financial Performance Trends"
            )
            st.plotly_chart(fig_trends, use_container_width=True)
        
        with tab2:
            # Financial Ratios
            ratios = ["EPS"]
            fig_ratios = self.visualizer.create_financial_ratios(
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
            
            fig_risk = self.visualizer.create_risk_factor_heatmap(
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
            
            fig_comp = self.visualizer.create_executive_compensation(
                comp_data,
                "Executive Compensation Breakdown"
            )
            st.plotly_chart(fig_comp, use_container_width=True)

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

# Initialize app if not already done
if "app" not in st.session_state:
    st.session_state.app = FinancialIQApp()

# Main content
st.markdown("<div class='main-header'>FinancialIQ: SEC Filing Analysis</div>", unsafe_allow_html=True)

st.markdown(
    """
    Ask questions about SEC filings to extract insights about companies, financial performance,
    risk factors, and more. The system analyzes 10-K, 10-Q, 8-K, and proxy statement documents.
    """
)

# Display tabs
tab1, tab2 = st.tabs(["Document Analysis", "Financial Analysis"])

with tab1:
    # File upload section
    st.header("Upload SEC Filings")
    uploaded_files = st.file_uploader(
        "Choose SEC filing PDFs",
        type="pdf",
        accept_multiple_files=True,
        key="document_uploader"
    )

    if uploaded_files:
        with st.spinner("Processing documents..."):
            processed_docs = st.session_state.app.process_documents(uploaded_files)
            if processed_docs:
                st.success(f"Successfully processed {len(processed_docs)} documents")
                
                # Create and display visualizations
                figures = st.session_state.app.create_visualizations(processed_docs)
                if figures:
                    st.header("Document Analysis")
                    cols = st.columns(len(figures))
                    for i, (title, fig) in enumerate(figures):
                        with cols[i]:
                            st.subheader(title)
                            st.plotly_chart(fig, use_container_width=True, key=f"doc_viz_{i}")

    # Query section
    st.header("Ask Questions")
    query = st.text_input("Enter your question about the SEC filings", key="query_input")

    if query:
        with st.spinner("Processing query..."):
            result = st.session_state.app.query_documents(query)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.subheader("Answer")
                st.write(result["answer"])
                
                if "sources" in result:
                    st.subheader("Sources")
                    for i, source in enumerate(result["sources"]):
                        st.write(f"- {source['source']} ({source.get('filing_date', 'N/A')})", key=f"source_{i}")

with tab2:
    st.session_state.app.display_financial_visualizations()

# Sidebar
st.sidebar.markdown("<div class='main-header'>FinancialIQ</div>", unsafe_allow_html=True)
st.sidebar.markdown("SEC Filing Analysis System")

# System initialization
st.sidebar.markdown("## System Setup")
project_id = st.sidebar.text_input("GCP Project ID", value="adta5760nlp", key="project_id_input")
location = st.sidebar.text_input("GCP Location", value="us-central1", key="location_input")

init_option = st.sidebar.radio("Data Source", ["Google Cloud Storage", "Local Directory"], key="data_source_radio")
local_dir = None

if init_option == "Local Directory":
    local_dir = st.sidebar.text_input("Local PDF Directory", value="./pdfs", key="local_dir_input")

if st.sidebar.button("Initialize System", key="init_button"):
    initialize_system(project_id, location, local_dir)

# Filters
st.sidebar.markdown("## Filters")
company_filter = st.sidebar.selectbox("Company", ["All"] + st.session_state.companies, key="company_filter")
form_filter = st.sidebar.selectbox("Form Type", ["All"] + st.session_state.form_types, key="form_filter")

# Query history
st.sidebar.markdown("## Query History")
for i, q in enumerate(st.session_state.query_history[-5:]):
    st.sidebar.text(f"{q['timestamp']}: {q['query'][:30]}...", key=f"query_history_{i}")

# About
st.sidebar.markdown("## About")
st.sidebar.info(
    """
    FinancialIQ is a RAG-based Q&A system for SEC filings.
    Created as part of ADTA 5770: Generative AI with Large Language Models.
    """
)

# Footer
st.markdown(
    """
    <div class='footnote'>
    Data sourced from SEC EDGAR database. This application is for educational purposes only and
    should not be used as financial advice.
    </div>
    """,
    unsafe_allow_html=True
)

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
        
        # Run the app
        st.session_state.app = FinancialIQApp()
        st.session_state.app.run()
        
    except Exception as e:
        st.error(f"Error in main application: {str(e)}")
        raise

if __name__ == "__main__":
    main() 