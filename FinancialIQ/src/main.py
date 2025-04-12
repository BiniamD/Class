"""
Main application file for FinancialIQ
Integrates all components and provides the entry point
"""

import os
import time
from typing import Dict, Any, List
import streamlit as st
from dotenv import load_dotenv

from document_processor import EnhancedSECFilingProcessor
from cache_manager import CacheManager
from logger import FinancialIQLogger
from sec_filing_rag_system import FinancialIQSystem
from visualization import FinancialVisualizer

class FinancialIQApp:
    """Main application class for FinancialIQ"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self.logger = FinancialIQLogger()
        self.cache_manager = CacheManager()
        self.document_processor = EnhancedSECFilingProcessor()
        self.visualizer = FinancialVisualizer()
        
        # Initialize RAG system
        self.financial_iq = None
        
        # Track processing times
        self.processing_times = {
            "document_processing": [],
            "query_processing": [],
            "visualization": []
        }

    def initialize_system(self, project_id: str, location: str, local_dir: str = None) -> None:
        """Initialize the FinancialIQ system"""
        start_time = time.time()
        
        try:
            self.logger.info("Initializing FinancialIQ system...")
            
            # Initialize RAG system
            self.financial_iq = FinancialIQSystem(
                project_id=project_id,
                location=location,
                bucket_name=os.getenv("BUCKET_NAME", "adta5770-docs-folder"),
                pdf_folder=os.getenv("PDF_FOLDER", "documents/pdfs")
            )
            
            if local_dir:
                # Process local documents
                self.process_local_documents(local_dir)
            else:
                # Setup cloud system
                self.financial_iq.setup_system(load_existing=True)
            
            duration = time.time() - start_time
            self.processing_times["system_initialization"] = duration
            self.logger.log_performance("system_initialization", duration)
            
            self.logger.info("System initialized successfully")
            
        except Exception as e:
            self.logger.log_error(e, "system_initialization")
            raise

    def process_local_documents(self, directory: str) -> None:
        """Process documents from local directory"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing documents from {directory}")
            
            # Get PDF files
            pdf_files = [
                os.path.join(directory, f) 
                for f in os.listdir(directory) 
                if f.lower().endswith('.pdf')
            ]
            
            # Process each document
            for pdf_file in pdf_files:
                # Check cache first
                cached_result = self.cache_manager.get_document_cache(pdf_file)
                
                if cached_result:
                    self.logger.info(f"Using cached result for {pdf_file}")
                    result = cached_result
                else:
                    # Process document
                    result = self.document_processor.process_document(pdf_file)
                    self.cache_manager.set_document_cache(pdf_file, result)
                    self.logger.log_document_processing(pdf_file, result)
                
                # Add to vector store
                self.financial_iq.add_to_vector_store(result)
            
            duration = time.time() - start_time
            self.processing_times["document_processing"].append(duration)
            self.logger.log_performance("document_processing", duration)
            
        except Exception as e:
            self.logger.log_error(e, "document_processing")
            raise

    def process_query(self, query: str, company_filter: str = None, form_filter: str = None) -> Dict[str, Any]:
        """Process a user query"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Apply filters
            if company_filter not in ["All", None]:
                query = f"For {company_filter}: {query}"
            if form_filter not in ["All", None]:
                query = f"In {form_filter} filings: {query}"
            
            # Get answer
            result = self.financial_iq.answer_question(query)
            
            duration = time.time() - start_time
            self.processing_times["query_processing"].append(duration)
            self.logger.log_query(query, duration, len(result.get("sources", [])))
            
            return result
            
        except Exception as e:
            self.logger.log_error(e, "query_processing")
            raise

    def get_visualizations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualizations for the data"""
        start_time = time.time()
        
        try:
            self.logger.info("Generating visualizations")
            
            visualizations = {}
            
            # Financial trends
            if "financial_metrics" in data:
                metrics = ["revenue", "net_income", "eps"]
                visualizations["trends"] = self.visualizer.create_financial_trends(
                    data["financial_metrics"],
                    metrics,
                    "Financial Performance Trends"
                )
            
            # Risk factors
            if "risk_factors" in data:
                visualizations["risk"] = self.visualizer.create_risk_factor_heatmap(
                    data["risk_factors"],
                    "Risk Factor Analysis"
                )
            
            # Executive compensation
            if "executive_compensation" in data:
                visualizations["compensation"] = self.visualizer.create_executive_compensation(
                    data["executive_compensation"],
                    "Executive Compensation Breakdown"
                )
            
            duration = time.time() - start_time
            self.processing_times["visualization"].append(duration)
            self.logger.log_performance("visualization", duration)
            
            return visualizations
            
        except Exception as e:
            self.logger.log_error(e, "visualization")
            raise

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        metrics = {}
        
        for operation, times in self.processing_times.items():
            if times:
                metrics[f"{operation}_avg"] = sum(times) / len(times)
                metrics[f"{operation}_max"] = max(times)
                metrics[f"{operation}_min"] = min(times)
        
        return metrics

def main():
    """Main entry point for the application"""
    # Initialize the app
    app = FinancialIQApp()
    
    # Set up Streamlit page
    st.set_page_config(
        page_title="FinancialIQ - SEC Filing Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    st.sidebar.markdown("# FinancialIQ")
    st.sidebar.markdown("SEC Filing Analysis System")
    
    # System initialization
    st.sidebar.markdown("## System Setup")
    project_id = st.sidebar.text_input("GCP Project ID", value=os.getenv("GOOGLE_CLOUD_PROJECT", "financial-iq-project"))
    location = st.sidebar.text_input("GCP Location", value=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
    
    init_option = st.sidebar.radio("Data Source", ["Google Cloud Storage", "Local Directory"])
    local_dir = None
    
    if init_option == "Local Directory":
        local_dir = st.sidebar.text_input("Local PDF Directory", value="./pdfs")
    
    if st.sidebar.button("Initialize System"):
        with st.spinner("Initializing system..."):
            app.initialize_system(project_id, location, local_dir)
            st.success("System initialized successfully!")
    
    # Main content
    st.markdown("# FinancialIQ: SEC Filing Analysis")
    
    if app.financial_iq is None:
        st.warning("Please initialize the system first")
        return
    
    # Query interface
    query = st.text_input("Ask a question about SEC filings:")
    company_filter = st.selectbox("Filter by company:", ["All"] + app.financial_iq.get_companies())
    form_filter = st.selectbox("Filter by form type:", ["All"] + app.financial_iq.get_form_types())
    
    if query:
        with st.spinner("Processing query..."):
            result = app.process_query(query, company_filter, form_filter)
            
            # Display answer
            st.markdown("### Answer")
            st.markdown(result["answer"])
            
            # Display sources
            st.markdown("### Sources")
            for source in result["sources"]:
                st.markdown(f"- {source['company']} ({source['form_type']}) - {source['filing_date']}")
            
            # Generate and display visualizations
            st.markdown("### Visualizations")
            visualizations = app.get_visualizations(result)
            
            for title, fig in visualizations.items():
                st.plotly_chart(fig, use_container_width=True)
            
            # Display performance metrics
            st.markdown("### Performance Metrics")
            metrics = app.get_performance_metrics()
            st.json(metrics)

if __name__ == "__main__":
    main() 