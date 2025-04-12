import pytest
import os
from pathlib import Path
from dotenv import load_dotenv

# Load test environment variables
load_dotenv()

@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory"""
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def sample_pdf_path(test_data_dir):
    """Return the path to a sample PDF file"""
    return test_data_dir / "sample_sec_filing.pdf"

@pytest.fixture(scope="session")
def sample_metadata_path(test_data_dir):
    """Return the path to sample metadata CSV"""
    return test_data_dir / "sample_metadata.csv"

@pytest.fixture(scope="session")
def vector_store_path():
    """Return the path to the vector store directory"""
    return Path("vector_store/test")

@pytest.fixture(scope="session")
def test_config():
    """Return test configuration dictionary"""
    return {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "max_documents": 10,
        "test_mode": True
    }

@pytest.fixture(scope="session")
def mock_gcp_credentials(monkeypatch):
    """Mock GCP credentials for testing"""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "test_credentials.json")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

@pytest.fixture(scope="function")
def clean_test_dirs(vector_store_path):
    """Clean up test directories before and after tests"""
    # Clean up before test
    if vector_store_path.exists():
        for file in vector_store_path.glob("*"):
            file.unlink()
    
    yield
    
    # Clean up after test
    if vector_store_path.exists():
        for file in vector_store_path.glob("*"):
            file.unlink() 