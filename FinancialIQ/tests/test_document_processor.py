"""
Tests for the Enhanced Document Processor
"""

import os
import pytest
from src.document_processor import EnhancedSECFilingProcessor
import pandas as pd

@pytest.fixture
def processor():
    return EnhancedSECFilingProcessor()

@pytest.fixture
def sample_pdf_path():
    # This would be a path to a sample SEC filing PDF
    # For testing, we'll use a mock path
    return "tests/sample_filings/sample_10k.pdf"

def test_extract_metadata(processor, sample_pdf_path):
    """Test metadata extraction from SEC filing"""
    metadata = processor.extract_metadata(sample_pdf_path)
    
    assert isinstance(metadata, dict)
    assert "company_name" in metadata
    assert "form_type" in metadata
    assert "filing_date" in metadata
    assert "cik" in metadata
    assert "period_end" in metadata
    assert "fiscal_year" in metadata

def test_extract_tables(processor, sample_pdf_path):
    """Test table extraction from PDF"""
    tables = processor.extract_tables(sample_pdf_path)
    
    assert isinstance(tables, list)
    if tables:  # If tables were found
        assert all(isinstance(table, pd.DataFrame) for table in tables)

def test_identify_financial_statements(processor, sample_pdf_path):
    """Test financial statement identification"""
    tables = processor.extract_tables(sample_pdf_path)
    statements = processor.identify_financial_statements(tables)
    
    assert isinstance(statements, dict)
    for statement_type, table in statements.items():
        assert statement_type in processor.financial_patterns
        assert isinstance(table, pd.DataFrame)

def test_extract_financial_metrics(processor, sample_pdf_path):
    """Test financial metrics extraction"""
    tables = processor.extract_tables(sample_pdf_path)
    statements = processor.identify_financial_statements(tables)
    metrics = processor.extract_financial_metrics(statements)
    
    assert isinstance(metrics, dict)
    for metric_name, value in metrics.items():
        assert metric_name in processor.metric_patterns
        assert isinstance(value, (int, float))

def test_extract_risk_factors(processor, sample_pdf_path):
    """Test risk factor extraction and categorization"""
    risk_factors = processor.extract_risk_factors(sample_pdf_path)
    
    assert isinstance(risk_factors, dict)
    for category, factors in risk_factors.items():
        assert category in processor.risk_categories
        assert isinstance(factors, list)
        if factors:  # If risk factors were found
            assert all(isinstance(factor, str) for factor in factors)

def test_extract_executive_compensation(processor, sample_pdf_path):
    """Test executive compensation extraction"""
    compensation = processor.extract_executive_compensation(sample_pdf_path)
    
    assert isinstance(compensation, dict)
    assert "executives" in compensation
    assert "summary" in compensation
    assert isinstance(compensation["executives"], list)
    assert isinstance(compensation["summary"], dict)

def test_process_document(processor, sample_pdf_path):
    """Test complete document processing"""
    result = processor.process_document(sample_pdf_path)
    
    assert isinstance(result, dict)
    assert "metadata" in result
    assert "financial_statements" in result
    assert "financial_metrics" in result
    assert "risk_factors" in result
    assert "executive_compensation" in result

def test_process_directory(processor, tmp_path):
    """Test processing of multiple documents in a directory"""
    # Create a temporary directory with sample PDFs
    sample_dir = tmp_path / "sample_filings"
    sample_dir.mkdir()
    
    # Create some empty PDF files for testing
    for i in range(3):
        (sample_dir / f"filing_{i}.pdf").touch()
    
    results = processor.process_directory(str(sample_dir))
    
    assert isinstance(results, list)
    assert len(results) == 3
    for result in results:
        assert isinstance(result, dict)
        assert all(key in result for key in [
            "metadata",
            "financial_statements",
            "financial_metrics",
            "risk_factors",
            "executive_compensation"
        ])

def test_error_handling(processor):
    """Test error handling with invalid input"""
    # Test with non-existent file
    result = processor.process_document("non_existent.pdf")
    assert isinstance(result, dict)
    assert all(not result[key] for key in result)
    
    # Test with invalid PDF
    with pytest.raises(Exception):
        processor.extract_tables("tests/sample_filings/invalid.pdf") 