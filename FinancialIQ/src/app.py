"""
FinancialIQ: A Streamlit UI for SEC Filing Q&A System
ADTA 5770: Final Project
"""

import logging
import os
import sys
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
import asyncio
import nest_asyncio

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting app.py execution")

# --- Log Key Library Versions ---
try:
    import streamlit as st
    logging.info(f"Streamlit version: {st.__version__}")
except ImportError:
    logging.error("Streamlit not found.")
try:
    import torch
    logging.info(f"Torch version: {torch.__version__}")
except ImportError:
    logging.info("Torch not directly imported here, likely via dependencies.")
except Exception as e:
    logging.warning(f"Could not get torch version: {e}")
# ------------------------

# --- Add Project Root to sys.path ---
current_file_path = os.path.abspath(__file__)
src_dir_path = os.path.dirname(current_file_path)
project_root_dir = os.path.dirname(src_dir_path)

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
    print(f"Added {project_root_dir} to sys.path")
logging.info("Project root added to sys.path")
# -------------------------------------

# Set page configuration - MUST BE FIRST STREAMLIT COMMAND
logging.info("Setting page config")
st.set_page_config(
    page_title="FinancialIQ - SEC Filing Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
logging.info("Page config set")

# Import visualization components
logging.info("Importing FinancialVisualizer")
try:
    from src.visualization import FinancialVisualizer
    logging.info("Successfully imported FinancialVisualizer")
except ModuleNotFoundError as e:
    logging.error(f"Failed to import FinancialVisualizer: {e}")
    raise

logging.info("Importing CacheManager and EnhancedSECFilingProcessor")
from src.cache_manager import CacheManager
from src.document_processor import EnhancedSECFilingProcessor
logging.info("Imports complete")

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

# --- Advanced Filtering Setup ---
# Try to load sec_filings.csv for filter options if metadata.csv is missing
filings_df = None
try:
    filings_df = pd.read_csv("../documents/sec_filings.csv")
    filings_df['filedAt'] = pd.to_datetime(filings_df['filedAt'], errors='coerce')
    available_companies = sorted(filings_df['companyName'].dropna().unique().tolist())
    available_form_types = sorted(filings_df['formType'].dropna().unique().tolist())
    min_date = filings_df['filedAt'].min()
    max_date = filings_df['filedAt'].max()
except Exception as e:
    filings_df = None
    available_companies = []
    available_form_types = []
    min_date = None
    max_date = None

class FinancialIQApp:
    """Streamlit application for FinancialIQ"""
    
    def __init__(self):
        logging.info("FinancialIQApp.__init__ started")
        self.system = None
        self.initialize_system()
        logging.info("FinancialIQApp.__init__ finished")
        
    def initialize_system(self):
        """Initialize the FinancialIQ system"""
        logging.info("FinancialIQApp.initialize_system started")
        try:
            # Check if vector store exists
            vector_store_path = os.path.join(project_root_dir, "data", "vector_store")
            logging.info(f"Checking for vector store at: {vector_store_path}")
            if not os.path.exists(vector_store_path):
                logging.error("Vector store not found.")
                st.error("Vector store not found. Please run test_processing_pipeline.py first to process documents and create the vector store.")
                return
            logging.info("Vector store found.")
                
            # Check if metadata exists
            metadata_csv_path = os.path.join(project_root_dir, "documents", "metadata.csv")
            logging.info(f"Checking for metadata at: {metadata_csv_path}")
            if not os.path.exists(metadata_csv_path):
                logging.error("Metadata file not found.")
                st.error("Metadata file not found. Please run test_processing_pipeline.py first to process documents.")
                return
            logging.info("Metadata file found.")
                
            # Initialize system with existing resources
            logging.info("Initializing FinancialIQSystem")
            # Import here to ensure logging happens first
            from sec_filing_rag_system import FinancialIQSystem
            self.system = FinancialIQSystem(
                project_id=os.getenv("PROJECT_ID", "adta5760nlp"),
                location=os.getenv("LOCATION", "us-central1"),
                metadata_csv_path=metadata_csv_path
            )
            logging.info("FinancialIQSystem instantiated")
            
            # Load existing vector store
            logging.info("Setting up system (loading vector store)")
            self.system.setup_system(load_existing=True)
            logging.info("System setup complete (vector store loaded)")
            
            st.success("System initialized successfully!")
            
            # Load metadata into session state
            logging.info("Loading metadata into session state")
            if os.path.exists(metadata_csv_path):
                metadata_df = pd.read_csv(metadata_csv_path)
                st.session_state.companies = sorted(metadata_df['companyName'].unique().tolist())
                st.session_state.form_types = sorted(metadata_df['formType'].unique().tolist())
                logging.info("Metadata loaded into session state")
            else:
                logging.warning("Metadata file not found during session state loading?")
            
        except Exception as e:
            logging.error(f"Failed to initialize system: {str(e)}", exc_info=True)
            st.error(f"Failed to initialize system: {str(e)}")
            st.error("Please ensure you have run test_processing_pipeline.py first.")
        finally:
            logging.info("FinancialIQApp.initialize_system finished")
            
    def run(self):
        """Run the Streamlit application"""
        logging.info("FinancialIQApp.run started")
        st.title("FinancialIQ - SEC Filing Analysis")
        
        if self.system is None or self.system.vector_store is None:
            logging.warning("System not properly initialized in run method.")
            st.error("System not properly initialized. Please run test_processing_pipeline.py first.")
            return
            
        # --- Enhanced Filtering UI ---
        col1, col2, col3 = st.columns(3)
        with col1:
            company_filter = st.selectbox(
                "Filter by Company",
                ["All"] + (available_companies if available_companies else st.session_state.companies)
            )
        with col2:
            form_filter = st.selectbox(
                "Filter by Form Type",
                ["All"] + (available_form_types if available_form_types else st.session_state.form_types)
            )
        with col3:
            if min_date is not None and max_date is not None:
                date_range = st.date_input(
                    "Filing Date Range",
                    value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date()
                )
            else:
                date_range = None
        
        # Add search functionality
        query = st.text_input("Enter your question about SEC filings:")
        if query:
            logging.info(f"Processing query: {query}")
            try:
                # Apply filters to query if selected
                filtered_query = query
                if company_filter != "All":
                    filtered_query = f"For {company_filter}: {query}"
                if form_filter != "All":
                    filtered_query = f"In {form_filter} filings: {filtered_query}"
                
                # --- Filter filings_df for visualization and search context ---
                filtered_filings = filings_df.copy() if filings_df is not None else None
                if filtered_filings is not None:
                    if company_filter != "All":
                        filtered_filings = filtered_filings[filtered_filings['companyName'] == company_filter]
                    if form_filter != "All":
                        filtered_filings = filtered_filings[filtered_filings['formType'] == form_filter]
                    if date_range is not None:
                        start_date, end_date = date_range
                        filtered_filings = filtered_filings[(filtered_filings['filedAt'] >= pd.to_datetime(start_date)) & (filtered_filings['filedAt'] <= pd.to_datetime(end_date))]
                
                logging.info(f"Searching documents with query: {filtered_query}")
                results = self.system.search_documents(filtered_query, k=5)
                logging.info(f"Found {len(results)} results")
                
                # Display results in a more organized way
                st.markdown("### Search Results")
                for i, doc in enumerate(results, 1):
                    with st.expander(f"Result {i} - {doc.metadata.get('companyName', 'Unknown')} ({doc.metadata.get('formType', 'Unknown')})"):
                        st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                        st.markdown(f"**Filing Date:** {doc.metadata.get('filedAt', 'Unknown')}")
                        st.markdown("**Content:**")
                        st.markdown(doc.page_content)

                # --- Advanced Visualization: Filings per Form Type (Filtered) ---
                st.markdown("### Filings per Form Type (Filtered Data)")
                try:
                    if filtered_filings is not None and not filtered_filings.empty:
                        form_counts = filtered_filings['formType'].value_counts().reset_index()
                        form_counts.columns = ['Form Type', 'Count']
                        fig = px.bar(form_counts, x='Form Type', y='Count', title='Number of Filings per Form Type (Filtered)', color='Form Type')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No filings available for the selected filters.")
                except Exception as e:
                    st.info(f"Could not load filings data for visualization: {e}")
                
            except Exception as e:
                logging.error(f"Error searching documents: {str(e)}", exc_info=True)
                st.error(f"Error searching documents: {str(e)}")
        logging.info("FinancialIQApp.run finished")

def main():
    """Main function to run the FinancialIQ application"""
    logging.info("main() started")
    try:
        # Create and run the application
        logging.info("Creating FinancialIQApp instance")
        app = FinancialIQApp()
        logging.info("Running FinancialIQApp")
        app.run()
        
    except Exception as e:
        logging.error(f"Error running application in main(): {str(e)}", exc_info=True)
        st.error(f"Error running application: {str(e)}")
        # We might not want to raise here in production, but useful for debugging
        # raise
    finally:
        logging.info("main() finished")

if __name__ == "__main__":
    logging.info("__main__ block started")
    main()
    logging.info("__main__ block finished") 