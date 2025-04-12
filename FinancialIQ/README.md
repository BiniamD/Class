# FinancialIQ

An intelligent system for analyzing SEC filings using advanced NLP and RAG (Retrieval-Augmented Generation) techniques.

## Features

- **Document Processing**: Advanced extraction of tables, metadata, risk factors, and financial metrics from SEC filings
- **Intelligent Querying**: Natural language question answering about SEC filings using RAG
- **Interactive Visualizations**: Dynamic visualization of financial trends, risk factors, and metrics
- **Performance Optimization**: Built-in caching system for faster processing
- **Comprehensive Logging**: Detailed logging system for monitoring and debugging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FinancialIQ.git
cd FinancialIQ
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

## Usage

1. Start the application:
```bash
streamlit run src/app.py
```

2. Upload SEC filings:
   - Use the file upload section to select PDF files of SEC filings
   - The system will process them automatically

3. Ask questions:
   - Enter natural language questions about the uploaded filings
   - View answers with source attribution
   - Explore interactive visualizations

## Project Structure

```
FinancialIQ/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── rag_system.py       # RAG implementation
│   ├── document_processor.py # Enhanced document processing
│   ├── visualization.py    # Data visualization
│   ├── cache_manager.py    # Caching system
│   └── logger.py          # Logging system
├── tests/
│   ├── test_rag_system.py
│   ├── test_document_processor.py
│   ├── test_cache_manager.py
│   └── test_visualization.py
├── data/
│   └── vector_store/      # Persistent vector store
├── docs/                  # Documentation
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

The project follows PEP 8 guidelines. Format code using:
```bash
black src/ tests/
isort src/ tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit
- Uses LangChain for RAG implementation
- Powered by Google Cloud Platform 