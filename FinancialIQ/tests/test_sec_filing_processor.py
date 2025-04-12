import pytest
from pathlib import Path
from src.sec_filing_processor import SECFilingProcessor

def test_processor_initialization(test_config):
    """Test SECFilingProcessor initialization"""
    processor = SECFilingProcessor(**test_config)
    assert processor.chunk_size == test_config["chunk_size"]
    assert processor.chunk_overlap == test_config["chunk_overlap"]

def test_pdf_processing(sample_pdf_path, test_config, clean_test_dirs):
    """Test PDF processing functionality"""
    processor = SECFilingProcessor(**test_config)
    result = processor.process_pdf(sample_pdf_path)
    
    assert result is not None
    assert "text" in result
    assert "metadata" in result
    assert len(result["text"]) > 0

def test_metadata_extraction(sample_pdf_path, test_config):
    """Test metadata extraction from PDF"""
    processor = SECFilingProcessor(**test_config)
    metadata = processor.extract_metadata(sample_pdf_path)
    
    assert "company_name" in metadata
    assert "filing_date" in metadata
    assert "form_type" in metadata
    assert "cik" in metadata

def test_document_chunking(sample_pdf_path, test_config):
    """Test document chunking functionality"""
    processor = SECFilingProcessor(**test_config)
    chunks = processor.chunk_document(sample_pdf_path)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk["text"]) <= test_config["chunk_size"]
        assert "metadata" in chunk

def test_financial_data_extraction(sample_pdf_path, test_config):
    """Test financial data extraction"""
    processor = SECFilingProcessor(**test_config)
    financial_data = processor.extract_financial_data(sample_pdf_path)
    
    assert "revenue" in financial_data
    assert "net_income" in financial_data
    assert "eps" in financial_data
    assert "assets" in financial_data
    assert "liabilities" in financial_data

def test_error_handling(test_config):
    """Test error handling for invalid files"""
    processor = SECFilingProcessor(**test_config)
    with pytest.raises(Exception):
        processor.process_pdf(Path("nonexistent.pdf"))

def test_performance(test_config, sample_pdf_path):
    """Test processing performance"""
    import time
    processor = SECFilingProcessor(**test_config)
    
    start_time = time.time()
    processor.process_pdf(sample_pdf_path)
    processing_time = time.time() - start_time
    
    # Ensure processing time is reasonable (adjust threshold as needed)
    assert processing_time < 30  # seconds

def test_output_format(sample_pdf_path, test_config):
    """Test output format consistency"""
    processor = SECFilingProcessor(**test_config)
    result = processor.process_pdf(sample_pdf_path)
    
    # Check required fields
    required_fields = ["text", "metadata", "financial_data", "chunks"]
    for field in required_fields:
        assert field in result
    
    # Check metadata fields
    required_metadata = ["company_name", "filing_date", "form_type", "cik"]
    for field in required_metadata:
        assert field in result["metadata"] 