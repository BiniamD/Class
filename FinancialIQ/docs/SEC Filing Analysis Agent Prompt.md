# SEC Filing Analysis Agent Prompt

# SEC Filing Analysis Agent Prompt

## Task Overview

Create a RAG-based Q&A system for analyzing SEC filings with the following capabilities:

1. PDF stored in Google cloud Documents 
2. Extract structured information from these filings
3. Build a vector database for semantic search
4. Provide natural language responses to queries about the filings
5. Visualize key financial data and trends

## Context and Background

We have 100 PDF documents containing SEC filings from various companies. These documents need processing and analysis to create an intelligent query system that enables users to ask questions about financial data, corporate governance, risk factors, and other regulatory information.

## System Components

### 1. Document Processing

- Extract text from PDF SEC filings
- Identify form type, company name, ticker symbol, and filing date
- Recognize document structure (sections like Risk Factors, MD&A, Financial Statements)
- Chunk documents into smaller segments with appropriate metadata

### 2. Information Extraction

- Extract structured data such as:
    - Key financial metrics (revenue, net income, EPS, etc.)
    - Risk factors and their categories
    - Executive compensation details
    - Corporate governance information
    - Competitive landscape mentions

### 3. Vector Database Creation

- Generate embeddings for document chunks
- Store embeddings in a vector database (FAISS)
- Create efficient retrieval mechanisms

### 4. RAG Implementation

- Implement retrieval component to find relevant document chunks
- Use Gemini 2.0 flash as the generation model
- Create effective prompting strategies for financial/SEC context
- Ensure answers cite specific sources within documents

### 5. User Interface

- Create a simple interface for querying the system
- Display responses with source citations
- Include options to filter by company, form type, and date range
- Visualize key financial metrics when appropriate

## Technical Requirements

- Use Google Cloud Platform (GCP) Vertex AI services
- Implement with Python and Google Colab
- Utilize LangChain for RAG framework
- Store documents in GCP Cloud Storage
- Performance considerations:
    - Optimize for both speed and accuracy
    - Handle large documents efficiently
    - Ensure financial data precision

## Expected Queries

The system should handle queries like:

- "What were Apple's main risk factors in their latest 10-K?"
- "How has Microsoft's revenue changed over the last three years?"
- "Summarize the executive compensation for Amazon in their proxy statement"
- "What financial metrics does Tesla highlight in their earnings releases?"
- "Compare the risk factors related to supply chain across technology companies"
- "Show me trends in revenue and profit margins for tech companies over time"

## Implementation Steps

1. Set up Google Cloud environment and services
2. Develop document processing pipeline
3. Create information extraction components
4. Build vector embedding and storage system
5. Implement RAG query system
6. Develop simple user interface
7. Test with sample queries
8. Optimize performance
9. Document system architecture and usage

## Evaluation Criteria

- Accuracy of responses
- Relevance of retrieved information
- Citation of specific sources
- Processing speed and efficiency
- User interface simplicity and effectiveness
- Handling of complex financial concepts
- Quality of data visualizations

## Challenges to Address

- Handling tables and financial data in PDFs
- Managing document structure variations across companies
- Ensuring financial calculation accuracy
- Managing context length for large documents
- Providing nuanced answers for complex financial questions

## Deliverables

1. Complete Python codebase for the SEC filing RAG system
2. Documentation of system architecture and components
3. User guide for interacting with the system
4. Performance analysis and optimization notes
5. Sample queries and responses demonstrating system capabilities

# SEC Filing RAG System Implementation Plan

## 1. Project Setup

### 1.1 Environment Configuration

```python
# Install required packages
!pip install google-cloud-storage
!pip install google-cloud-aiplatform
!pip install langchain langchain-google-vertexai faiss-cpu
!pip install PyPDF2 unstructured
!pip install streamlit plotly pandas matplotlib

```

### 1.2 Google Cloud Authentication

```python
# Set up GCP authentication
import os
from google.colab import auth
auth.authenticate_user()

```

### 1.3 Project Configuration

```python
# Configuration variables
PROJECT_ID = "your-project-id"  # Replace with your GCP project ID
LOCATION = "us-central1"
BUCKET_NAME = "adta5770-docs-folder"
PDF_FOLDER = "documents/pdfs"
EMBEDDING_MODEL_NAME = "text-embedding-004"
LLM_MODEL_NAME = "gemini-1.5-flash-001"

```

## 2. Document Processing System

### 2.1 SEC Filing Processor Class

Create a class that handles the SEC filing processing pipeline:

- Extract text from PDFs
- Identify form type, company, and filing date
- Extract document sections
- Parse financial data

### 2.2 Document Chunking Strategy

Implement document chunking with appropriate metadata preservation:

- Split documents into meaningful chunks (sections or paragraphs)
- Maintain metadata including source, form type, company, etc.
- Preserve section information

## 3. Vector Database Creation

### 3.1 Embedding Generation

Generate embeddings for document chunks using Google's text embedding model:

```python
from langchain_google_vertexai.embeddings import VertexAIEmbeddings

embeddings = VertexAIEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    project=PROJECT_ID,
    location=LOCATION
)

```

### 3.2 Vector Store Implementation

Create and populate the vector database:

```python
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(documents, embeddings)

```

### 3.3 Persistence Strategy

Save and load the vector database:

```python
# Save
vector_store.save_local("./vector_store")

# Load
vector_store = FAISS.load_local("./vector_store", embeddings)

```

## 4. RAG System Implementation

### 4.1 Retriever Configuration

Set up the document retriever component:

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

```

### 4.2 LLM Configuration

Configure the Gemini model:

```python
from langchain_google_vertexai import VertexAI

llm = VertexAI(
    model_name=LLM_MODEL_NAME,
    project=PROJECT_ID,
    location=LOCATION,
    max_output_tokens=1024,
    temperature=0.1
)

```

### 4.3 RAG Chain Setup

Create the question-answering chain:

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

```

### 4.4 Financial Prompt Engineering

Design specialized prompts for financial and SEC contexts:

```python
def format_financial_prompt(query: str) -> str:
    prompt = f"""You are FinancialIQ, an intelligent assistant for financial analysts.
    Answer the following question about SEC filings based solely on the retrieved documents.
    Be precise, clear, and use proper financial terminology.
    If the documents don't contain enough information to answer confidently, acknowledge the limitations.
    Cite the specific SEC forms, companies, and filing dates used in your answer.

    Question about SEC filings: {query}
    """
    return prompt

```

## 5. Financial Analysis Components

### 5.1 Financial Metrics Extraction

Extract key financial metrics from financial statements:

- Revenue, Net Income, EPS, etc.
- Create structured data for visualization

### 5.2 Financial Ratio Calculation

Calculate important financial ratios:

- PE Ratio, ROE, Debt-to-Equity, etc.
- Compare across companies or time periods

### 5.3 Trend Analysis

Implement time-series analysis for financial metrics:

- Year-over-year or quarter-over-quarter changes
- Visualization of trends

## 6. User Interface

### 6.1 Streamlit Application

Create a web interface with Streamlit:

```python
import streamlit as st

st.title("FinancialIQ: SEC Filing Analysis System")

query = st.text_input("Ask a question about SEC filings:")
if query:
    result = financial_iq.answer_question(query)
    st.write(result["answer"])

    st.subheader("Sources:")
    for source in result["sources"]:
        st.write(f"- {source['company']} {source['form_type']} ({source['filing_date']}), Page {source['page']}")

```

### 6.2 Filtering Options

Add company and document type filters:

```python
companies = ["All"] + list(set([doc.metadata["company_name"] for doc in documents]))
selected_company = st.selectbox("Filter by company:", companies)

form_types = ["All", "10-K", "10-Q", "8-K", "DEF 14A"]
selected_form = st.selectbox("Filter by form type:", form_types)

```

### 6.3 Visualization Components

Create financial visualizations:

```python
import plotly.express as px

def create_financial_chart(data, companies, metric):
    fig = px.line(data, x="Year", y=metric, color="Company",
                  title=f"{metric} Over Time")
    st.plotly_chart(fig)

```

## 7. Testing and Optimization

### 7.1 Sample Query Testing

Test with various financial queries:

- Simple factual queries
- Complex analytical questions
- Comparison queries
- Time-series questions

### 7.2 Performance Optimization

Optimize for better performance:

- Fine-tune chunk size and overlap
- Adjust retrieval parameters
- Consider hybrid search approaches

### 7.3 Quality Evaluation

Evaluate response quality:

- Accuracy of financial information
- Relevance of retrieved documents
- Citation quality

## 8. Deployment Strategy

### 8.1 Local Deployment

Run the Streamlit application locally:

```bash
streamlit run app.py

```

### 8.2 Cloud Deployment Options

Consider deploying to GCP:

- Google Cloud Run
- App Engine
- Compute Engine

### 8.3 CI/CD Considerations

Set up continuous integration for ongoing improvements:

- Code repository structure
- Testing pipelines
- Deployment automation

## 9. Documentation and Knowledge Transfer

### 9.1 System Documentation

Document the system architecture:

- Component diagrams
- Data flow
- API documentation

### 9.2 User Guide

Create a user guide for financial analysts:

- How to ask effective questions
- Understanding system responses
- Interpreting visualizations

### 9.3 Code Documentation

Ensure code is well-documented:

- Function and class docstrings
- Clear variable naming
- Comments for complex operations

## 10. Future Enhancements

### 10.1 Multi-document Analysis

Enable cross-document comparison:

- Compare multiple companies
- Track changes across quarters/years

### 10.2 Advanced Visualization

Add more sophisticated visualizations:

- Interactive dashboards
- Comparative analysis tools
- Custom financial charting

### 10.3 Integration Options

Consider integration with other systems:

- Financial data APIs
- Real-time market data
- Automated report generation

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

# Import SEC filing processor and RAG system

# In a real application, you would import from your module

# For demonstration, we'll include simplified versions here

from sec_filing_rag_system import FinancialIQSystem, SECFilingProcessor

# Set page configuration

st.set_page_config(
page_title="FinancialIQ - SEC Filing Analysis",
page_icon="📊",
layout="wide",
initial_sidebar_state="expanded"
)

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

def initialize_system(project_id, location, local_dir=None):
"""Initialize the FinancialIQ system"""
with st.spinner("Initializing SEC Filing analysis system..."):
if local_dir:
# For local directory processing
st.session_state.financial_iq = FinancialIQSystem(
project_id=project_id,
location=location,
bucket_name="",  # Not used for local processing
pdf_folder=""    # Not used for local processing
)

```
        # Get PDF files
        pdf_files = [os.path.join(local_dir, f) for f in os.listdir(local_dir) if f.lower().endswith('.pdf')]

        # Process PDFs
        documents = st.session_state.financial_iq.process_pdfs(pdf_files)

        # Create vector store
        st.session_state.financial_iq.create_vector_store(documents)

        # Initialize LLM and QA chain
        st.session_state.financial_iq.initialize_llm()
        st.session_state.financial_iq.setup_qa_chain()
    else:
        # For Google Cloud Storage processing
        st.session_state.financial_iq = FinancialIQSystem(
            project_id=project_id,
            location=location,
            bucket_name="adta5770-docs-folder",
            pdf_folder="documents/pdfs"
        )
        st.session_state.financial_iq.setup_system(load_existing=True)

    # Extract companies and form types for filtering
    extract_metadata()

    st.success("System initialized successfully!")

```

def extract_metadata():
"""Extract metadata from processed documents"""
# This would normally come from your documents
# For demonstration, we'll use sample data
st.session_state.companies = [
"Apple Inc.",
"Microsoft Corporation",
"[Amazon.com](http://amazon.com/), Inc.",
"Tesla, Inc.",
"Alphabet Inc."
]

```
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

```

def process_query(query, company_filter=None, form_filter=None):
"""Process a user query with optional filters"""
if st.session_state.financial_iq is None:
st.error("System not initialized. Please initialize first.")
return

```
if company_filter not in ["All", None]:
    query = f"For {company_filter}: {query}"

if form_filter not in ["All", None]:
    query = f"In {form_filter} filings: {query}"

with st.spinner("Processing query..."):
    result = st.session_state.financial_iq.answer_question(query)

    # Add to query history
    st.session_state.query_history.append({
        "query": query,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Store current answer and sources
    st.session_state.current_answer = result["answer"]
    st.session_state.current_sources = result["sources"]

    return result

```

def display_answer():
"""Display the current answer and sources"""
if st.session_state.current_answer:
st.markdown(f"<div>{st.session_state.current_answer}</div>", unsafe_allow_html=True)

```
    st.markdown("<div class='sub-header'>Sources</div>", unsafe_allow_html=True)

    for source in st.session_state.current_sources:
        st.markdown(
            f"""<div class='source-box'>
                <strong>{source['company']}</strong> - {source['form_type']} ({source['filing_date']})<br>
                Page: {source['page']} | Source: {source['source']}
            </div>""",
            unsafe_allow_html=True
        )

```

def display_financial_metrics(company=None):
"""Display financial metrics and visualizations"""
if st.session_state.financial_metrics is None:
return

```
df = st.session_state.financial_metrics

if company and company != "All":
    df = df[df["Company"] == company]

# Revenue chart
st.markdown("<div class='sub-header'>Revenue Trends</div>", unsafe_allow_html=True)
fig_revenue = px.line(df, x="Year", y="Revenue (Billions)", color="Company",
                    markers=True, line_shape="linear",
                    title="Revenue Over Time (Billions USD)")
fig_revenue.update_layout(
    xaxis_title="Year",
    yaxis_title="Revenue (Billions USD)",
    legend_title="Company",
    plot_bgcolor="white"
)
st.plotly_chart(fig_revenue, use_container_width=True)

# Net Income chart
st.markdown("<div class='sub-header'>Net Income Trends</div>", unsafe_allow_html=True)
fig_income = px.line(df, x="Year", y="Net Income (Billions)", color="Company",
                   markers=True, line_shape="linear",
                   title="Net Income Over Time (Billions USD)")
fig_income.update_layout(
    xaxis_title="Year",
    yaxis_title="Net Income (Billions USD)",
    legend_title="Company",
    plot_bgcolor="white"
)
st.plotly_chart(fig_income, use_container_width=True)

# EPS chart
st.markdown("<div class='sub-header'>Earnings Per Share (EPS) Trends</div>", unsafe_allow_html=True)
fig_eps = px.line(df, x="Year", y="EPS", color="Company",
                markers=True, line_shape="linear",
                title="Earnings Per Share Over Time")
fig_eps.update_layout(
    xaxis_title="Year",
    yaxis_title="EPS (USD)",
    legend_title="Company",
    plot_bgcolor="white"
)
st.plotly_chart(fig_eps, use_container_width=True)

```

# Sidebar

st.sidebar.markdown("<div class='main-header'>FinancialIQ</div>", unsafe_allow_html=True)
st.sidebar.markdown("SEC Filing Analysis System")

# System initialization

st.sidebar.markdown("## System Setup")
project_id = st.sidebar.text_input("GCP Project ID", value="financial-iq-project")
location = st.sidebar.text_input("GCP Location", value="us-central1")

init_option = st.sidebar.radio("Data Source", ["Google Cloud Storage", "Local Directory"])
local_dir = None

if init_option == "Local Directory":
local_dir = st.sidebar.text_input("Local PDF Directory", value="./pdfs")

if st.sidebar.button("Initialize System"):
initialize_system(project_id, location, local_dir)

# Filters

st.sidebar.markdown("## Filters")
company_filter = st.sidebar.selectbox("Company", ["All"] + st.session_state.companies)
form_filter = st.sidebar.selectbox("Form Type", ["All"] + st.session_state.form_types)

# Query history

st.sidebar.markdown("## Query History")
for i, q in enumerate(st.session_state.query_history[-5:]):
st.sidebar.text(f"{q['timestamp']}: {q['query'][:30]}...")

# About

st.sidebar.markdown("## About")
[st.sidebar.info](http://st.sidebar.info/)(
"""
FinancialIQ is a RAG-based Q&A system for SEC filings.
Created as part of ADTA 5770: Generative AI with Large Language Models.
"""
)

# Main content

st.markdown("<div class='main-header'>FinancialIQ: SEC Filing Analysis</div>", unsafe_allow_html=True)

st.markdown(
"""
Ask questions about SEC filings to extract insights about companies, financial performance,
risk factors, and more. The system analyzes 10-K, 10-Q, 8-K, and proxy statement documents.
"""
)

# Query input

query = st.text_input("Ask a question about SEC filings:")
analyze_button = st.button("Analyze")

if analyze_button and query:
result = process_query(query, company_filter, form_filter)

# Display tabs

tab1, tab2 = st.tabs(["Answer", "Financial Metrics"])

with tab1:
display_answer()

with tab2:
display_financial_metrics(company_filter)

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

# Run the app with: streamlit run [app.py](http://app.py/)