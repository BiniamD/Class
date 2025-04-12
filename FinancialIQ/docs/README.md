# FinancialIQ System

A system for processing and analyzing SEC filings using RAG (Retrieval-Augmented Generation) and Google Cloud services.

## Setup Instructions

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env` file:
```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GCS_BUCKET_NAME=your-bucket-name
PDF_FOLDER=sec_filings
METADATA_CSV_PATH=documents/documents_sec_filings.csv
```

4. Make sure you have:
   - Google Cloud project set up
   - Service account with necessary permissions
   - Google Cloud credentials configured

## Running the System

1. Process and analyze SEC filings:
```bash
python run_setup.py
```

2. Run the web interface:
```bash
streamlit run src/app.py
```

## Project Structure

```
FinancialIQ/
├── requirements.txt    # Project dependencies
├── .env               # Environment variables
├── documents/         # SEC filing documents and metadata
├── src/              # Source code
│   ├── app.py        # Streamlit web interface
│   ├── sec_filing_rag_system.py  # Core RAG system
│   ├── sec_filing_processor.py   # PDF processing
│   └── sec_filing_metadata.py    # Metadata handling
└── venv/             # Virtual environment
```

## Features

- PDF processing and text extraction
- Metadata extraction from SEC filings
- Vector store for semantic search
- Question answering using LLMs
- Document retrieval with source citations
- Web interface for easy interaction

## Dependencies

See `requirements.txt` for the complete list of dependencies. 