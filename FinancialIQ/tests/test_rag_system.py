import pytest
import os
import tempfile
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path
from src.cache_manager import CacheManager
from src.logger import FinancialIQLogger

from src.rag_system import RAGSystem

@pytest.fixture
def mock_logger():
    return Mock(spec=FinancialIQLogger)

@pytest.fixture
def mock_cache_manager():
    return Mock(spec=CacheManager)

@pytest.fixture
def sample_processed_doc():
    return {
        'metadata': {
            'company_name': 'Test Corp',
            'form_type': '10-K',
            'filing_date': '2024-01-01'
        },
        'tables': [pd.DataFrame({'Revenue': [100, 200], 'Year': [2022, 2023]})],
        'risk_factors': {
            'Market Risk': ['Competition risk', 'Economic risk'],
            'Operational Risk': ['Supply chain risk']
        },
        'financial_metrics': {
            'revenue': 1000000,
            'net_income': 500000
        }
    }

@pytest.fixture
def rag_system(mock_logger, mock_cache_manager):
    with patch('src.rag_system.create_vector_store') as mock_create_store:
        system = RAGSystem(
            logger=mock_logger,
            cache_manager=mock_cache_manager,
            vector_store_path='test_vector_store'
        )
        yield system

def test_init(rag_system):
    """Test RAG system initialization"""
    assert rag_system.logger is not None
    assert rag_system.cache_manager is not None
    assert rag_system.vector_store_path == 'test_vector_store'

def test_process_documents_with_cache(rag_system, sample_processed_doc, mock_cache_manager):
    """Test document processing with cache hit"""
    mock_cache_manager.get_document_cache.return_value = sample_processed_doc
    
    result = rag_system.process_documents(['test.pdf'])
    
    assert result == {'test.pdf': sample_processed_doc}
    mock_cache_manager.get_document_cache.assert_called_once_with('test.pdf')
    rag_system.logger.info.assert_called_with('Document processed from cache: test.pdf')

def test_process_documents_without_cache(rag_system, sample_processed_doc, mock_cache_manager):
    """Test document processing with cache miss"""
    mock_cache_manager.get_document_cache.return_value = None
    
    with patch('src.rag_system.process_document') as mock_process:
        mock_process.return_value = sample_processed_doc
        result = rag_system.process_documents(['test.pdf'])
        
        assert result == {'test.pdf': sample_processed_doc}
        mock_process.assert_called_once_with('test.pdf')
        mock_cache_manager.set_document_cache.assert_called_once_with('test.pdf', sample_processed_doc)

def test_create_vector_store(rag_system, sample_processed_doc):
    """Test vector store creation"""
    with patch('src.rag_system.create_embeddings') as mock_create_embeddings:
        mock_create_embeddings.return_value = {'text': [0.1, 0.2, 0.3]}
        
        rag_system.create_vector_store({'test.pdf': sample_processed_doc})
        
        mock_create_embeddings.assert_called_once()
        rag_system.logger.info.assert_called_with('Vector store created successfully')

def test_load_vector_store(rag_system):
    """Test vector store loading"""
    with patch('src.rag_system.load_vector_store') as mock_load:
        mock_load.return_value = Mock()
        
        rag_system.load_vector_store()
        
        mock_load.assert_called_once_with(rag_system.vector_store_path)
        rag_system.logger.info.assert_called_with('Vector store loaded successfully')

def test_query(rag_system):
    """Test querying functionality"""
    mock_response = {
        'answer': 'Test answer',
        'sources': ['test.pdf'],
        'confidence': 0.95
    }
    
    with patch('src.rag_system.query_vector_store') as mock_query:
        mock_query.return_value = mock_response
        
        result = rag_system.query('What is the revenue?')
        
        assert result == mock_response
        mock_query.assert_called_once_with('What is the revenue?', rag_system.vector_store)
        rag_system.logger.info.assert_called_with('Query processed successfully')

def test_query_no_vector_store(rag_system):
    """Test querying without initialized vector store"""
    rag_system.vector_store = None
    
    with pytest.raises(ValueError) as exc_info:
        rag_system.query('What is the revenue?')
    
    assert str(exc_info.value) == 'Vector store not initialized'
    rag_system.logger.error.assert_called_with('Query failed: Vector store not initialized')

def test_error_handling(rag_system):
    """Test error handling during document processing"""
    with patch('src.rag_system.process_document') as mock_process:
        mock_process.side_effect = Exception('Processing error')
        
        with pytest.raises(Exception) as exc_info:
            rag_system.process_documents(['test.pdf'])
        
        assert str(exc_info.value) == 'Processing error'
        rag_system.logger.error.assert_called_with('Error processing document test.pdf: Processing error')

def test_cache_expiration(rag_system, mock_cache_manager):
    """Test handling of expired cache"""
    mock_cache_manager.get_document_cache.return_value = None
    mock_cache_manager.is_cache_expired.return_value = True
    
    with patch('src.rag_system.process_document') as mock_process:
        rag_system.process_documents(['test.pdf'])
        
        mock_cache_manager.clear_expired_cache.assert_called_once()
        rag_system.logger.info.assert_any_call('Cleared expired cache entries') 