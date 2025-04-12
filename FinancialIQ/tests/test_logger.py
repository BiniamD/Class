import os
import logging
import pytest
from datetime import datetime
from src.logger import FinancialIQLogger

@pytest.fixture
def logger(tmp_path):
    """Create a logger instance with a temporary log directory"""
    log_dir = str(tmp_path / "logs")
    return FinancialIQLogger(log_dir)

def test_logger_initialization(logger, tmp_path):
    """Test logger initialization"""
    assert os.path.exists(str(tmp_path / "logs"))
    assert isinstance(logger.logger, logging.Logger)
    assert logger.logger.name == "FinancialIQ"
    assert logger.logger.level == logging.INFO

def test_info_logging(logger, tmp_path):
    """Test info level logging"""
    test_message = "Test info message"
    logger.info(test_message)
    
    log_file = str(tmp_path / "logs" / f"financialiq_{datetime.now().strftime('%Y%m%d')}.log")
    assert os.path.exists(log_file)
    
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert test_message in log_content
        assert "INFO" in log_content

def test_error_logging(logger, tmp_path):
    """Test error level logging"""
    test_message = "Test error message"
    logger.error(test_message)
    
    log_file = str(tmp_path / "logs" / f"financialiq_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert test_message in log_content
        assert "ERROR" in log_content

def test_warning_logging(logger, tmp_path):
    """Test warning level logging"""
    test_message = "Test warning message"
    logger.warning(test_message)
    
    log_file = str(tmp_path / "logs" / f"financialiq_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert test_message in log_content
        assert "WARNING" in log_content

def test_debug_logging(logger, tmp_path):
    """Test debug level logging"""
    test_message = "Test debug message"
    logger.debug(test_message)
    
    log_file = str(tmp_path / "logs" / f"financialiq_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert test_message in log_content
        assert "DEBUG" in log_content

def test_critical_logging(logger, tmp_path):
    """Test critical level logging"""
    test_message = "Test critical message"
    logger.critical(test_message)
    
    log_file = str(tmp_path / "logs" / f"financialiq_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert test_message in log_content
        assert "CRITICAL" in log_content

def test_performance_logging(logger, tmp_path):
    """Test performance logging"""
    operation = "test_operation"
    duration = 1.23
    extra_data = {"key": "value"}
    
    logger.log_performance(operation, duration, extra_data)
    
    log_file = str(tmp_path / "logs" / f"financialiq_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert operation in log_content
        assert str(duration) in log_content
        assert "Performance" in log_content

def test_document_processing_logging(logger, tmp_path):
    """Test document processing logging"""
    file_path = "test.pdf"
    result = {"metadata": {"key": "value"}, "metrics": [1, 2, 3]}
    
    logger.log_document_processing(file_path, result)
    
    log_file = str(tmp_path / "logs" / f"financialiq_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert file_path in log_content
        assert "Processed document" in log_content

def test_query_logging(logger, tmp_path):
    """Test query logging"""
    query = "test query"
    response = "test response"
    sources = ["source1", "source2"]
    
    logger.log_query(query, response, sources)
    
    log_file = str(tmp_path / "logs" / f"financialiq_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert query in log_content
        assert response in log_content
        assert "Query" in log_content

def test_extra_data_logging(logger, tmp_path):
    """Test logging with extra data"""
    test_message = "Test message with extra data"
    extra_data = {"key1": "value1", "key2": 123}
    
    logger.info(test_message, extra_data)
    
    log_file = str(tmp_path / "logs" / f"financialiq_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert test_message in log_content
        assert "key1" in log_content
        assert "value1" in log_content 