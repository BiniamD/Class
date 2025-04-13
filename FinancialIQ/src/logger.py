"""
Logging system for FinancialIQ
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from pythonjsonlogger import jsonlogger

class FinancialIQLogger:
    """Logger class for FinancialIQ application"""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the logger
        
        Args:
            log_dir: Directory to store log files (default: "logs")
        """
        self.logger = logging.getLogger("FinancialIQ")
        self.logger.setLevel(logging.ERROR)
        
        # Prevent duplicate log messages
        self.logger.propagate = False
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        log_file = os.path.join(log_dir, f"financialiq_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)
        
        # Create formatter with proper format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(operation_type)s] %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message - will be ignored due to ERROR level setting
        
        Args:
            message: Message to log
            extra: Additional data to include in log
        """
        # Don't log INFO messages since we're only logging ERROR and above
        pass
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message - will be ignored due to ERROR level setting
        
        Args:
            message: Message to log
            extra: Additional data to include in log
        """
        # Don't log WARNING messages since we're only logging ERROR and above
        pass
    
    def error(self, message: str, operation_type: str = "UNKNOWN"):
        """Log an error message with operation type."""
        self.logger.error(message, extra={"operation_type": operation_type})
    
    def critical(self, message: str, operation_type: str = "UNKNOWN"):
        """Log a critical message with operation type."""
        self.logger.critical(message, extra={"operation_type": operation_type})
    
    def log_error(self, error: Exception, context: str = "", operation_type: str = "UNKNOWN"):
        """Log an error with context and operation type."""
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.error(error_msg, operation_type)
    
    def log_performance(self, operation: str, duration: float, operation_type: str = "PERFORMANCE"):
        """Log performance metrics."""
        self.error(f"{operation} completed in {duration:.2f} seconds", operation_type)
    
    def log_document_processing(self, doc_id: str, status: str, details: str = "", operation_type: str = "DOCUMENT_PROCESSING"):
        """Log document processing status."""
        self.error(f"Document {doc_id}: {status} - {details}", operation_type)
    
    def log_query(self, query: str, results_count: int, operation_type: str = "QUERY"):
        """Log query execution."""
        self.error(f"Query: {query} - Found {results_count} results", operation_type)
    
    def log_vector_store(self, operation: str, details: str = "", operation_type: str = "VECTOR_STORE"):
        """Log vector store operations."""
        self.error(f"{operation} - {details}", operation_type)
    
    def log_table_extraction(self, doc_id: str, table_count: int, success: bool, details: str = "", operation_type: str = "TABLE_EXTRACTION"):
        """Log table extraction results."""
        status = "successful" if success else "failed"
        self.error(f"Document {doc_id}: Extracted {table_count} tables ({status}) - {details}", operation_type) 