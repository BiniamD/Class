# FinancialIQ: SEC Filing Analysis System

A RAG-based Q&A system for analyzing SEC filings using Google Cloud Platform and Vertex AI.

## Features

- Process and analyze SEC filings (10-K, 10-Q, 8-K, etc.)
- Extract structured information from PDF documents
- Semantic search across filing contents
- Natural language question answering
- Financial metrics visualization
- Support for both local and cloud storage

## Prerequisites

- Python 3.9+
- Google Cloud Platform account
- Vertex AI API enabled
- Google Cloud Storage bucket for document storage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-iq.git
cd financial-iq
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Google Cloud authentication:
```bash
gcloud auth application-default login
```

## Configuration

1. Create a `.env` file in the project root:
```
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
BUCKET_NAME=your-bucket-name
PDF_FOLDER=documents/pdfs
```

2. Update the configuration in `src/app.py` with your GCP project details.

## Usage

1. Start the Streamlit application:
```bash
streamlit run src/app.py
```

2. Access the web interface at `http://localhost:8501`

3. Initialize the system:
   - Choose between local PDF directory or Google Cloud Storage
   - Enter your GCP project ID and location
   - Click "Initialize System"

4. Ask questions about SEC filings:
   - Use the query input field
   - Filter by company and form type
   - View answers with source citations
   - Explore financial metrics visualizations

## Project Structure

```
financial-iq/
├── src/
│   ├── app.py              # Streamlit UI
│   └── sec_filing_rag_system.py  # Core RAG implementation
├── data/                   # Local data storage
├── config/                 # Configuration files
├── tests/                  # Test files
├── docs/                   # Documentation
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Features in Detail

### Document Processing
- PDF text extraction
- Metadata extraction (company, form type, filing date)
- Document chunking with overlap
- Structured data extraction

### RAG Implementation
- Vector embeddings using Vertex AI
- FAISS vector store for efficient retrieval
- Gemini 1.5 Flash for generation
- Context-aware prompting

### Financial Analysis
- Key metrics extraction
- Trend analysis
- Comparative analysis
- Visualization capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ADTA 5770: Generative AI with Large Language Models
- Google Cloud Platform
- LangChain
- Streamlit 