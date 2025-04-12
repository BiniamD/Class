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
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger("FinancialIQ")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.log_dir / f"financialiq_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create JSON formatter
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message
        
        Args:
            message: Message to log
            extra: Additional data to include in log
        """
        self.logger.info(message, extra=extra or {})
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message
        
        Args:
            message: Message to log
            extra: Additional data to include in log
        """
        self.logger.error(message, extra=extra or {})
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message
        
        Args:
            message: Message to log
            extra: Additional data to include in log
        """
        self.logger.warning(message, extra=extra or {})
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message
        
        Args:
            message: Message to log
            extra: Additional data to include in log
        """
        self.logger.debug(message, extra=extra or {})
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message
        
        Args:
            message: Message to log
            extra: Additional data to include in log
        """
        self.logger.critical(message, extra=extra or {})
    
    def log_performance(self, operation: str, duration: float, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log performance metrics
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            extra: Additional data to include in log
        """
        log_data = {
            "operation": operation,
            "duration": duration,
            **(extra or {})
        }
        self.logger.info(f"Performance: {operation}", extra=log_data)
    
    def log_document_processing(self, file_path: str, result: Dict[str, Any]) -> None:
        """Log document processing results
        
        Args:
            file_path: Path to the processed document
            result: Processing results
        """
        log_data = {
            "file_path": file_path,
            "result": result
        }
        self.logger.info(f"Document processed: {file_path}", extra=log_data)
    
    def log_query(self, query: str, response: str, sources: List[str]) -> None:
        """Log query and response
        
        Args:
            query: User query
            response: System response
            sources: List of source documents
        """
        log_data = {
            "query": query,
            "response": response,
            "sources": sources
        }
        self.logger.info(f"Query processed: {query}", extra=log_data) 