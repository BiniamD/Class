# FinancialIQ Development Guide

## Development Environment Setup

1. **Prerequisites**
   - Python 3.8 or higher
   - Google Cloud account with Vertex AI access
   - Git

2. **Local Setup**
   ```bash
   # Clone the repository
   git clone [repository-url]
   cd FinancialIQ

   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt

   # Set up environment variables
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Google Cloud Setup**
   - Create a new project
   - Enable Vertex AI API
   - Create service account with necessary permissions
   - Download service account key
   - Set up API key for Gemini 2.0 flash

## Development Workflow

1. **Branch Strategy**
   - `main`: Production code
   - `develop`: Development branch
   - `feature/*`: New features
   - `bugfix/*`: Bug fixes
   - `release/*`: Release preparation

2. **Code Style**
   - Follow PEP 8 guidelines
   - Use type hints
   - Document all functions and classes
   - Write unit tests for new features

3. **Testing**
   ```bash
   # Run all tests
   pytest

   # Run specific test file
   pytest tests/test_sec_filing_processor.py

   # Run with coverage
   pytest --cov=src tests/
   ```

4. **Documentation**
   - Update README.md for major changes
   - Document new features in docs/
   - Update CHANGELOG.md for releases
   - Keep API documentation current

## Common Tasks

1. **Adding New Dependencies**
   ```bash
   pip install new-package
   pip freeze > requirements.txt
   ```

2. **Running the Application**
   ```bash
   # Development mode
   streamlit run src/app.py

   # Production mode
   python run_setup.py
   ```

3. **Processing New Documents**
   ```bash
   # Place PDFs in documents/ directory
   python src/sec_filing_processor.py
   ```

## Troubleshooting

1. **Common Issues**
   - API key errors: Check .env file
   - Memory issues: Adjust chunk size
   - Performance issues: Check vector store

2. **Debugging**
   - Enable debug logging
   - Check error logs
   - Use debug mode in Streamlit

## Contributing

1. **Pull Request Process**
   - Create feature branch
   - Write tests
   - Update documentation
   - Submit PR to develop

2. **Code Review**
   - Ensure tests pass
   - Check code style
   - Verify documentation
   - Test functionality

## Deployment

1. **Local Deployment**
   ```bash
   python run_setup.py
   streamlit run src/app.py
   ```

2. **Cloud Deployment**
   - Set up GCP project
   - Configure Cloud Run
   - Deploy using Cloud Build

## Monitoring

1. **Logging**
   - Check logs/ directory
   - Monitor error rates
   - Track performance metrics

2. **Performance**
   - Monitor response times
   - Check memory usage
   - Track API usage 