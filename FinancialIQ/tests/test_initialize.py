"""
Tests for the FinancialIQ initialization script
"""

import os
import pytest
from unittest.mock import Mock, patch
from src.initialize import FinancialIQInitializer

@pytest.fixture
def initializer():
    """Create a FinancialIQInitializer instance for testing"""
    return FinancialIQInitializer(log_dir="test_logs")

@pytest.fixture
def mock_logger():
    """Create a mock logger"""
    with patch('src.initialize.FinancialIQLogger') as mock:
        yield mock

@pytest.fixture
def mock_cache_manager():
    """Create a mock cache manager"""
    with patch('src.initialize.CacheManager') as mock:
        yield mock

@pytest.fixture
def mock_document_processor():
    """Create a mock document processor"""
    with patch('src.initialize.EnhancedSECFilingProcessor') as mock:
        yield mock

@pytest.fixture
def mock_financial_iq():
    """Create a mock FinancialIQSystem"""
    with patch('src.initialize.FinancialIQSystem') as mock:
        yield mock

def test_setup_environment(initializer, mock_logger):
    """Test environment setup"""
    # Mock os.makedirs
    with patch('os.makedirs') as mock_makedirs:
        initializer.setup_environment()
        
        # Verify directories were created
        mock_makedirs.assert_any_call("logs", exist_ok=True)
        mock_makedirs.assert_any_call("cache", exist_ok=True)
        mock_makedirs.assert_any_call("documents/pdfs", exist_ok=True)
        
        # Verify logging
        mock_logger.return_value.info.assert_called_with("Environment setup completed")

def test_initialize_system_cloud(initializer, mock_financial_iq):
    """Test system initialization with cloud storage"""
    # Mock environment variables
    with patch.dict('os.environ', {
        'BUCKET_NAME': 'test-bucket',
        'PDF_FOLDER': 'test-folder'
    }):
        initializer.initialize_system(
            project_id="test-project",
            location="test-location"
        )
        
        # Verify FinancialIQSystem was initialized correctly
        mock_financial_iq.assert_called_once_with(
            project_id="test-project",
            location="test-location",
            bucket_name="test-bucket",
            pdf_folder="test-folder"
        )
        
        # Verify setup_system was called
        mock_financial_iq.return_value.setup_system.assert_called_once_with(load_existing=True)

def test_initialize_system_local(initializer, mock_financial_iq):
    """Test system initialization with local documents"""
    # Mock process_local_documents
    with patch.object(initializer, 'process_local_documents') as mock_process:
        initializer.initialize_system(
            project_id="test-project",
            location="test-location",
            local_dir="test-dir"
        )
        
        # Verify process_local_documents was called
        mock_process.assert_called_once_with("test-dir")

def test_process_local_documents(initializer, mock_financial_iq, mock_cache_manager):
    """Test processing of local documents"""
    # Mock os.listdir and document processing
    with patch('os.listdir', return_value=['test1.pdf', 'test2.pdf']), \
         patch('os.path.join', side_effect=lambda *args: '/'.join(args)), \
         patch.object(initializer.document_processor, 'process_document') as mock_process:
        
        # Setup mock cache manager
        mock_cache_manager.return_value.get_document_cache.return_value = None
        
        # Process documents
        initializer.financial_iq = mock_financial_iq
        initializer.process_local_documents("test-dir")
        
        # Verify document processing
        assert mock_process.call_count == 2
        mock_financial_iq.return_value.add_to_vector_store.call_count == 2

def test_verify_setup_success(initializer, mock_financial_iq):
    """Test successful setup verification"""
    # Setup mock FinancialIQSystem
    mock_financial_iq.return_value.vector_store = Mock()
    mock_financial_iq.return_value.llm = Mock()
    initializer.financial_iq = mock_financial_iq.return_value
    
    # Verify setup
    assert initializer.verify_setup() is True

def test_verify_setup_failure(initializer, mock_financial_iq):
    """Test failed setup verification"""
    # Setup mock FinancialIQSystem with missing components
    mock_financial_iq.return_value.vector_store = None
    mock_financial_iq.return_value.llm = Mock()
    initializer.financial_iq = mock_financial_iq.return_value
    
    # Verify setup
    assert initializer.verify_setup() is False

def test_main_success():
    """Test successful main execution"""
    with patch('src.initialize.FinancialIQInitializer') as mock_initializer, \
         patch.dict('os.environ', {'GOOGLE_CLOUD_PROJECT': 'test-project'}):
        
        # Setup mock initializer
        mock_instance = mock_initializer.return_value
        mock_instance.verify_setup.return_value = True
        
        # Import and run main
        from src.initialize import main
        main()
        
        # Verify initialization steps
        mock_instance.setup_environment.assert_called_once()
        mock_instance.initialize_system.assert_called_once_with(
            project_id="test-project",
            location="us-central1"
        )
        mock_instance.verify_setup.assert_called_once()

def test_main_failure():
    """Test failed main execution"""
    with patch('src.initialize.FinancialIQInitializer') as mock_initializer, \
         patch('sys.exit') as mock_exit:
        
        # Setup mock initializer to raise exception
        mock_instance = mock_initializer.return_value
        mock_instance.setup_environment.side_effect = Exception("Test error")
        
        # Import and run main
        from src.initialize import main
        main()
        
        # Verify system exit was called
        mock_exit.assert_called_once_with(1) 