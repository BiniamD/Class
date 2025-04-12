"""
Tests for the Cache Manager
"""

import os
import json
import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.cache_manager import CacheManager

@pytest.fixture
def cache_manager(tmp_path):
    """Create a cache manager with a temporary directory"""
    return CacheManager(cache_dir=str(tmp_path))

@pytest.fixture
def sample_document_data():
    """Sample document processing results"""
    return {
        "metadata": {
            "company_name": "Test Company",
            "form_type": "10-K",
            "filing_date": "2023-12-31"
        },
        "financial_statements": {
            "balance_sheet": pd.DataFrame({
                "Assets": [100, 200],
                "Liabilities": [50, 100]
            })
        },
        "financial_metrics": {
            "revenue": 1000,
            "net_income": 100
        }
    }

def test_cache_initialization(cache_manager):
    """Test cache manager initialization"""
    assert os.path.exists(cache_manager.document_cache)
    assert os.path.exists(cache_manager.embedding_cache)

def test_document_caching(cache_manager, sample_document_data):
    """Test document caching functionality"""
    file_path = "test_document.pdf"
    
    # Test setting cache
    cache_manager.set_document_cache(file_path, sample_document_data)
    cache_path = cache_manager._get_cache_path(file_path, "document")
    assert os.path.exists(cache_path)
    
    # Test getting cache
    cached_data = cache_manager.get_document_cache(file_path)
    assert cached_data is not None
    assert cached_data["metadata"] == sample_document_data["metadata"]
    assert cached_data["financial_metrics"] == sample_document_data["financial_metrics"]
    
    # Test DataFrame conversion
    assert isinstance(pd.DataFrame(cached_data["financial_statements"]["balance_sheet"]), pd.DataFrame)

def test_embedding_caching(cache_manager):
    """Test embedding caching functionality"""
    text = "Test text for embedding"
    embedding = [0.1, 0.2, 0.3]
    
    # Test setting cache
    cache_manager.set_embedding_cache(text, embedding)
    cache_path = cache_manager._get_cache_path(text, "embedding")
    assert os.path.exists(cache_path)
    
    # Test getting cache
    cached_embedding = cache_manager.get_embedding_cache(text)
    assert cached_embedding == embedding

def test_cache_expiration(cache_manager, sample_document_data):
    """Test cache expiration"""
    file_path = "expired_document.pdf"
    
    # Set cache
    cache_manager.set_document_cache(file_path, sample_document_data)
    
    # Modify cache file timestamp to make it expired
    cache_path = cache_manager._get_cache_path(file_path, "document")
    expired_time = datetime.now() - timedelta(days=31)
    os.utime(cache_path, (expired_time.timestamp(), expired_time.timestamp()))
    
    # Test getting expired cache
    cached_data = cache_manager.get_document_cache(file_path)
    assert cached_data is None

def test_cache_clearance(cache_manager, sample_document_data):
    """Test cache clearance functionality"""
    # Create some cache files
    file_paths = ["doc1.pdf", "doc2.pdf"]
    for path in file_paths:
        cache_manager.set_document_cache(path, sample_document_data)
    
    # Test clearing expired cache
    cache_manager.clear_expired_cache()
    for path in file_paths:
        cache_path = cache_manager._get_cache_path(path, "document")
        assert os.path.exists(cache_path)  # Should still exist as not expired
    
    # Test clearing all cache
    cache_manager.clear_all_cache()
    for path in file_paths:
        cache_path = cache_manager._get_cache_path(path, "document")
        assert not os.path.exists(cache_path)

def test_error_handling(cache_manager):
    """Test error handling in cache operations"""
    # Test with invalid file path
    invalid_path = "invalid/path/document.pdf"
    assert cache_manager.get_document_cache(invalid_path) is None
    
    # Test with invalid data
    cache_manager.set_document_cache("test.pdf", {"invalid": object()})
    assert cache_manager.get_document_cache("test.pdf") is None 