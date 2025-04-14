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
    
    _instance = None
    
    @classmethod
    def get_logger(cls, name: str = "FinancialIQ") -> 'FinancialIQLogger':
        """Get or create a logger instance
        
        Args:
            name: Name of the logger (default: "FinancialIQ")
            
        Returns:
            FinancialIQLogger instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the logger with a file handler."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger("FinancialIQ")
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create file handler
        log_file = os.path.join(log_dir, f"financialiq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log an info message."""
        self.logger.info(message, extra=extra or {})
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log an error message."""
        self.logger.error(message, extra=extra or {})
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message."""
        self.logger.warning(message, extra=extra or {})
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message."""
        self.logger.debug(message, extra=extra or {})
    
    def critical(self, message: str, operation_type: str = "UNKNOWN"):
        """Log a critical message with operation type."""
        self.logger.critical(message, extra={"operation_type": operation_type})
    
    def log_error(self, error: Exception, context: str = "", operation_type: str = "UNKNOWN"):
        """Log an error with context and operation type."""
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.error(error_msg, operation_type)
    
    def log_performance(self, operation: str, duration: float, operation_type: str = "PERFORMANCE"):
        """Log performance metrics."""
        self.info(f"{operation} completed in {duration:.2f} seconds", extra={"operation_type": operation_type})
    
    def log_document_processing(self, doc_id: str, status: str, details: str = "", operation_type: str = "DOCUMENT_PROCESSING"):
        """Log document processing status."""
        self.info(f"Document {doc_id}: {status} - {details}", extra={"operation_type": operation_type})
    
    def log_query(self, query: str, results_count: int, operation_type: str = "QUERY"):
        """Log query execution."""
        self.info(f"Query: {query} - Found {results_count} results", extra={"operation_type": operation_type})
    
    def log_vector_store(self, operation: str, details: str = "", operation_type: str = "VECTOR_STORE"):
        """Log vector store operations."""
        self.info(f"{operation} - {details}", extra={"operation_type": operation_type})
    
    def log_table_extraction(self, doc_id: str, table_count: int, success: bool, details: str = "", operation_type: str = "TABLE_EXTRACTION"):
        """Log table extraction results."""
        status = "successful" if success else "failed"
        self.info(f"Document {doc_id}: Extracted {table_count} tables ({status}) - {details}", extra={"operation_type": operation_type}) 