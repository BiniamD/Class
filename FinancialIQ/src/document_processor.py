"""
Enhanced Document Processor for SEC Filings
Handles structured data extraction, table processing, and metadata identification
"""

import os
import re
from typing import Dict, List, Any, Optional, Generator, Tuple, Union
import pandas as pd
from PyPDF2 import PdfReader
import pdfplumber
import camelot
from datetime import datetime
import json
import concurrent.futures
from io import BytesIO
from src.logger import FinancialIQLogger
from functools import lru_cache
from google.api_core import retry
from google.api_core.exceptions import GoogleAPIError
from google.cloud import storage
import tempfile
import time

class EnhancedSECFilingProcessor:
    """Enhanced processor for SEC filings with improved table and structured data extraction"""
    
    def __init__(self, project_id: Optional[str] = None, bucket_name: Optional[str] = None, log_dir: str = "logs", max_workers: int = 4, batch_size: int = 10):
        self.logger = FinancialIQLogger(log_dir)
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        if project_id and bucket_name:
            self.gcs_client = storage.Client(project=project_id)
            self.bucket = self.gcs_client.bucket(bucket_name)
            self.logger.log_vector_store("GCS client initialized", "Local processing mode", "INITIALIZATION")
        else:
            self.gcs_client = None
            self.bucket = None
            self.logger.log_vector_store("GCS client not initialized", "Local processing mode", "INITIALIZATION")
        
        # LLM context length settings
        self.max_context_length = 4096  # Default context length
        self.chunk_overlap = 200  # Overlap between chunks
        self.min_chunk_size = 500  # Minimum chunk size
        
        self.form_types = {
            "10-K": "Annual Report",
            "10-Q": "Quarterly Report",
            "8-K": "Current Report",
            "DEF 14A": "Proxy Statement",
            "S-1": "Registration Statement"
        }
        
        # Common financial statement patterns
        self.financial_patterns = {
            "balance_sheet": r"(CONSOLIDATED\s+BALANCE\s+SHEETS?|BALANCE\s+SHEETS?)",
            "income_statement": r"(CONSOLIDATED\s+STATEMENTS?\s+OF\s+INCOME|INCOME\s+STATEMENTS?)",
            "cash_flow": r"(CONSOLIDATED\s+STATEMENTS?\s+OF\s+CASH\s+FLOWS?|CASH\s+FLOW\s+STATEMENTS?)",
            "equity": r"(CONSOLIDATED\s+STATEMENTS?\s+OF\s+STOCKHOLDERS'\s+EQUITY|EQUITY\s+STATEMENTS?)"
        }
        
        # Common financial metrics patterns
        self.metric_patterns = {
            "revenue": r"(Total\s+Revenue|Net\s+Sales|Revenue)",
            "net_income": r"(Net\s+Income|Net\s+Earnings)",
            "eps": r"(Earnings\s+Per\s+Share|EPS)",
            "gross_margin": r"(Gross\s+Profit|Gross\s+Margin)",
            "operating_income": r"(Operating\s+Income|Operating\s+Earnings)",
            "total_assets": r"(Total\s+Assets)",
            "total_liabilities": r"(Total\s+Liabilities)",
            "total_equity": r"(Total\s+Stockholders'\s+Equity|Total\s+Equity)"
        }
        
        # Risk factor categories
        self.risk_categories = {
            "market": ["market", "competition", "industry", "economic"],
            "operational": ["operations", "supply chain", "manufacturing", "production"],
            "regulatory": ["regulation", "compliance", "legal", "government"],
            "financial": ["financial", "liquidity", "debt", "capital"],
            "technology": ["technology", "innovation", "intellectual property", "patent"],
            "environmental": ["environment", "climate", "sustainability", "ESG"]
        }

        # Financial table patterns
        self.financial_table_patterns = {
            "balance_sheet": [
                r"CONSOLIDATED\s+BALANCE\s+SHEETS?",
                r"BALANCE\s+SHEETS?",
                r"STATEMENTS?\s+OF\s+FINANCIAL\s+POSITION",
                r"ASSETS\s+AND\s+LIABILITIES",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+ASSETS\s+AND\s+LIABILITIES",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+FINANCIAL\s+CONDITION",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+FINANCIAL\s+POSITION"
            ],
            "income_statement": [
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+INCOME",
                r"INCOME\s+STATEMENTS?",
                r"STATEMENTS?\s+OF\s+OPERATIONS",
                r"STATEMENTS?\s+OF\s+EARNINGS",
                r"REVENUES?\s+AND\s+EXPENSES?",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+COMPREHENSIVE\s+INCOME",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+OPERATIONS",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+EARNINGS",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+LOSS"
            ],
            "cash_flow": [
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+CASH\s+FLOWS?",
                r"CASH\s+FLOW\s+STATEMENTS?",
                r"STATEMENTS?\s+OF\s+CASH\s+FLOWS?",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+CASH\s+AND\s+CASH\s+EQUIVALENTS",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+CASH\s+AND\s+SHORT-TERM\s+INVESTMENTS"
            ],
            "equity": [
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+STOCKHOLDERS'\s+EQUITY",
                r"EQUITY\s+STATEMENTS?",
                r"STATEMENTS?\s+OF\s+CHANGES\s+IN\s+EQUITY",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+SHAREHOLDERS'\s+EQUITY",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+CHANGES\s+IN\s+STOCKHOLDERS'\s+EQUITY",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+CHANGES\s+IN\s+SHAREHOLDERS'\s+EQUITY"
            ],
            "segment": [
                r"SEGMENT\s+INFORMATION",
                r"BUSINESS\s+SEGMENTS?",
                r"OPERATING\s+SEGMENTS?",
                r"REPORTABLE\s+SEGMENTS?",
                r"SEGMENT\s+RESULTS",
                r"SEGMENT\s+OPERATING\s+RESULTS"
            ],
            "supplementary": [
                r"SUPPLEMENTARY\s+INFORMATION",
                r"SUPPLEMENTAL\s+INFORMATION",
                r"ADDITIONAL\s+INFORMATION",
                r"OTHER\s+INFORMATION",
                r"NOTES\s+TO\s+FINANCIAL\s+STATEMENTS"
            ]
        }
        
        # Table validation metrics
        self.table_validation_metrics = {
            "min_rows": 3,  # Minimum number of rows (including header)
            "min_cols": 2,  # Minimum number of columns
            "numeric_cols_ratio": 0.3,  # Minimum ratio of numeric columns
            "header_confidence": 0.7,  # Minimum confidence for header detection
            "row_consistency": 0.8,  # Minimum consistency ratio for row structure
            "financial_term_ratio": 0.4,  # Minimum ratio of financial terms in headers
            "numeric_value_ratio": 0.5,  # Minimum ratio of numeric values in numeric columns
            "date_column_confidence": 0.6  # Minimum confidence for date column detection
        }

    @retry.Retry(
        predicate=retry.if_exception_type(GoogleAPIError),
        initial=1.0,
        maximum=60.0,
        multiplier=2.0,
        deadline=300.0
    )
    def get_blob_content(self, blob_name: str, chunk_size: int = 1024*1024) -> Generator[bytes, None, None]:
        """Stream blob content in chunks to manage memory"""
        try:
            blob = self.bucket.blob(blob_name)
            if not blob.exists():
                self.logger.error(f"Blob {blob_name} does not exist")
                return
            
            with blob.open("rb") as f:
                while chunk := f.read(chunk_size):
                    yield chunk
                    
        except Exception as e:
            self.logger.log_error(e, "blob_content_streaming", {"blob_name": blob_name})
            raise

    def validate_pdf_content(self, content: bytes) -> bool:
        """Validate PDF content before processing"""
        try:
            # Check if content is a valid PDF
            BytesIO(content).seek(0)
            PdfReader(BytesIO(content))
            return True
        except Exception as e:
            self.logger.log_error(e, "pdf_validation")
            return False

    @lru_cache(maxsize=100)
    def get_cached_metadata(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """Get cached metadata for a blob if available"""
        try:
            cache_blob = self.bucket.blob(f"cache/metadata/{blob_name}.json")
            if cache_blob.exists():
                return json.loads(cache_blob.download_as_text())
            return None
        except Exception as e:
            self.logger.log_error(e, "metadata_cache_retrieval", {"blob_name": blob_name})
            return None

    def process_gcs_pdf(self, blob_name: str) -> Dict[str, Any]:
        """Process a single PDF file from GCS with detailed logging"""
        try:
            self.logger.info(f"Starting end-to-end processing of file: {blob_name}")
            
            # Step 1: Check cache
            self.logger.info("Step 1: Checking cache")
            cached_result = self.get_cached_metadata(blob_name)
            if cached_result:
                self.logger.info(f"Found cached result for {blob_name}")
                return cached_result
            
            # Step 2: Download and validate PDF
            self.logger.info("Step 2: Downloading and validating PDF")
            pdf_content = b""
            for chunk in self.get_blob_content(blob_name):
                pdf_content += chunk
            
            if not self.validate_pdf_content(pdf_content):
                self.logger.error(f"Invalid PDF content for {blob_name}")
                return {}
            
            # Step 3: Process PDF content
            self.logger.info("Step 3: Processing PDF content")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.process_pdf_content, pdf_content, blob_name)
                try:
                    result = future.result(timeout=300)  # 5-minute timeout
                except concurrent.futures.TimeoutError:
                    self.logger.error(f"Timeout processing PDF: {blob_name}")
                    return {}
            
            if not result:
                self.logger.error(f"No result generated for {blob_name}")
                return {}
            
            # Step 4: Validate result structure
            self.logger.info("Step 4: Validating result structure")
            if "data" not in result or "vector_content" not in result:
                self.logger.error(f"Invalid result structure for {blob_name}")
                return {}
            
            # Step 5: Store results
            self.logger.info("Step 5: Storing results")
            try:
                # Store processed data
                result_blob = self.bucket.blob(f"processed/{blob_name}.json")
                result_blob.upload_from_string(
                    json.dumps(result["data"]),
                    content_type="application/json"
                )
                self.logger.info(f"Stored processed data for {blob_name}")
                
                # Store vector content
                vector_blob = self.bucket.blob(f"vector_store/{blob_name}.txt")
                vector_blob.upload_from_string(
                    result["vector_content"],
                    content_type="text/plain"
                )
                self.logger.info(f"Stored vector content for {blob_name}")
                
                # Cache metadata
                cache_blob = self.bucket.blob(f"cache/metadata/{blob_name}.json")
                cache_blob.upload_from_string(
                    json.dumps(result["data"]),
                    content_type="application/json"
                )
                self.logger.info(f"Cached metadata for {blob_name}")
                
            except Exception as e:
                self.logger.error(f"Error storing results for {blob_name}: {str(e)}")
                return {}
            
            # Step 6: Log processing summary
            self.logger.info("Step 6: Logging processing summary")
            try:
                summary = {
                    "file": blob_name,
                    "processing_time": datetime.now().isoformat(),
                    "metadata": result["data"].get("metadata", {}),
                    "financial_statements": list(result["data"].get("financial_statements", {}).keys()),
                    "metrics_count": len(result["data"].get("metrics", {})),
                    "risk_factors_count": sum(len(factors) for factors in result["data"].get("risk_factors", {}).values()),
                    "executives_count": len(result["data"].get("executive_compensation", {}).get("executives", [])),
                    "vector_content_length": len(result["vector_content"])
                }
                self.logger.info(f"Processing summary: {json.dumps(summary, indent=2)}")
            except Exception as e:
                self.logger.error(f"Error generating summary for {blob_name}: {str(e)}")
            
            self.logger.info(f"Successfully completed end-to-end processing of {blob_name}")
            return result["data"]
            
        except Exception as e:
            self.logger.log_error(e, "gcs_pdf_processing", {"blob_name": blob_name})
            return {}

    def _chunk_pdf_content(self, reader: PdfReader) -> List[Tuple[int, str, List[pd.DataFrame]]]:
        """Split PDF content into chunks suitable for LLM processing with integrated table extraction"""
        chunks = []
        current_chunk = []
        current_length = 0
        current_page = 0
        current_tables = []
        
        try:
            for page_num, page in enumerate(reader.pages):
                # Extract tables first
                page_tables = self._extract_tables_from_page(page)
                if page_tables:
                    current_tables.extend(page_tables)
                
                # Extract text
                text = page.extract_text()
                if not text:
                    continue
                
                # Split text into paragraphs
                paragraphs = text.split('\n\n')
                
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue
                    
                    # If paragraph is too long, split it into sentences
                    if len(paragraph) > self.max_context_length:
                        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                        for sentence in sentences:
                            if current_length + len(sentence) > self.max_context_length:
                                if current_chunk:
                                    chunks.append((current_page, '\n\n'.join(current_chunk), current_tables))
                                    current_chunk = []
                                    current_length = 0
                                    current_tables = []
                            current_chunk.append(sentence)
                            current_length += len(sentence)
                    else:
                        if current_length + len(paragraph) > self.max_context_length:
                            if current_chunk:
                                chunks.append((current_page, '\n\n'.join(current_chunk), current_tables))
                                current_chunk = []
                                current_length = 0
                                current_tables = []
                        current_chunk.append(paragraph)
                        current_length += len(paragraph)
                
                current_page = page_num
                
                # Add remaining content as a chunk
                if current_chunk:
                    chunks.append((current_page, '\n\n'.join(current_chunk), current_tables))
                    current_chunk = []
                    current_length = 0
                    current_tables = []
            
            # Merge small chunks
            merged_chunks = []
            current_merged = []
            current_merged_length = 0
            current_merged_tables = []
            
            for page, chunk, tables in chunks:
                if current_merged_length + len(chunk) <= self.max_context_length:
                    current_merged.append((page, chunk))
                    current_merged_tables.extend(tables)
                    current_merged_length += len(chunk)
                else:
                    if current_merged:
                        merged_chunks.append(self._merge_chunks(current_merged, current_merged_tables))
                    current_merged = [(page, chunk)]
                    current_merged_tables = tables
                    current_merged_length = len(chunk)
            
            if current_merged:
                merged_chunks.append(self._merge_chunks(current_merged, current_merged_tables))
            
            return merged_chunks
            
        except Exception as e:
            self.logger.log_error(e, "pdf_chunking")
            return []

    def _merge_chunks(self, chunks: List[Tuple[int, str]], tables: List[pd.DataFrame]) -> Tuple[int, str, List[pd.DataFrame]]:
        """Merge chunks with overlap handling and table preservation"""
        if not chunks:
            return (0, "", [])
        
        # Get the page number of the first chunk
        page = chunks[0][0]
        
        # Merge chunks with overlap
        merged_text = []
        for i, (_, chunk) in enumerate(chunks):
            if i > 0:
                # Add overlap from previous chunk
                prev_chunk = chunks[i-1][1]
                overlap = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                merged_text.append(overlap)
            merged_text.append(chunk)
        
        return (page, '\n\n'.join(merged_text), tables)

    def _extract_tables_from_page(self, page) -> List[pd.DataFrame]:
        """Extract tables from a single page using multiple methods"""
        tables = []
        try:
            # Try pdfplumber first
            with pdfplumber.open(BytesIO(page.get_contents())) as pdf:
                page_tables = pdf.pages[0].extract_tables()
                if page_tables:
                    for table in page_tables:
                        if table and len(table) > 1:
                            try:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                if not df.empty and len(df.columns) > 1:
                                    tables.append(df)
                            except Exception:
                                pass
            
            # Try camelot if no tables found
            if not tables:
                with BytesIO(page.get_contents()) as temp_pdf:
                    camelot_tables = camelot.read_pdf(temp_pdf, flavor='lattice', pages='1')
                    for table in camelot_tables:
                        if not table.df.empty and len(table.df.columns) > 1:
                            tables.append(table.df)
            
            # Validate and filter tables
            validated_tables = []
            for table in tables:
                metrics = self.validate_table_structure(table)
                if metrics["is_valid"]:
                    validated_tables.append(table)
            
            return validated_tables
            
        except Exception as e:
            self.logger.warning(f"Error extracting tables from page: {str(e)}")
            return []

    def process_pdf_content(self, pdf_content: bytes, source_name: str) -> Dict[str, Any]:
        """Process PDF content from memory with detailed logging"""
        try:
            self.logger.info(f"Starting PDF content processing for {source_name}")
            
            # Step 1: Create PDF reader
            self.logger.info("Step 1: Creating PDF reader")
            pdf_file = BytesIO(pdf_content)
            reader = PdfReader(pdf_file)
            
            # Step 2: Split into chunks
            self.logger.info("Step 2: Splitting PDF into chunks")
            chunks = self._chunk_pdf_content(reader)
            self.logger.info(f"Split PDF into {len(chunks)} chunks")
            
            # Step 3: Process chunks
            self.logger.info("Step 3: Processing chunks")
            results = []
            for i, (page, chunk, tables) in enumerate(chunks):
                self.logger.info(f"Processing chunk {i+1}/{len(chunks)} (page {page+1}) with {len(tables)} tables")
                
                # Process tables
                self.logger.info("Processing tables")
                financial_statements = self.identify_financial_statements(tables)
                metrics = self.extract_financial_metrics(financial_statements)
                
                # Process text
                self.logger.info("Processing text content")
                risk_factors = self.extract_risk_factors_from_text(chunk)
                compensation = self.extract_executive_compensation_from_text(chunk)
                
                # Convert tables to string
                self.logger.info("Converting tables to string format")
                table_strings = []
                for table_type, table in financial_statements.items():
                    table_strings.append(f"{table_type}:\n{table.to_string()}")
                
                # Combine results
                result = {
                    "metadata": self.extract_metadata_from_text(chunk, source_name),
                    "financial_statements": financial_statements,
                    "metrics": metrics,
                    "risk_factors": risk_factors,
                    "executive_compensation": compensation,
                    "source": source_name,
                    "chunk": i,
                    "page": page,
                    "content": chunk,
                    "tables": '\n\n'.join(table_strings)
                }
                
                results.append(result)
            
            # Step 4: Merge results
            self.logger.info("Step 4: Merging results")
            final_result = self._merge_results(results)
            
            # Step 5: Prepare vector content
            self.logger.info("Step 5: Preparing vector content")
            vector_content = self._prepare_vector_content(final_result)
            
            self.logger.info(f"Completed PDF content processing for {source_name}")
            return {
                "data": final_result,
                "vector_content": vector_content
            }
            
        except Exception as e:
            self.logger.log_error(e, "pdf_content_processing", {"source_name": source_name})
            return {}

    def _prepare_vector_content(self, result: Dict[str, Any]) -> str:
        """Prepare content for vector store by converting all data to string format"""
        try:
            content_parts = []
            
            # Add metadata
            if "metadata" in result:
                content_parts.append("METADATA:")
                for key, value in result["metadata"].items():
                    content_parts.append(f"{key}: {value}")
            
            # Add financial statements
            if "financial_statements" in result:
                content_parts.append("\nFINANCIAL STATEMENTS:")
                for statement_type, table in result["financial_statements"].items():
                    content_parts.append(f"\n{statement_type.upper()}:")
                    content_parts.append(table.to_string())
            
            # Add metrics
            if "metrics" in result:
                content_parts.append("\nFINANCIAL METRICS:")
                for metric, value in result["metrics"].items():
                    content_parts.append(f"{metric}: {value}")
            
            # Add risk factors
            if "risk_factors" in result:
                content_parts.append("\nRISK FACTORS:")
                for category, factors in result["risk_factors"].items():
                    content_parts.append(f"\n{category.upper()}:")
                    for factor in factors:
                        content_parts.append(f"- {factor}")
            
            # Add executive compensation
            if "executive_compensation" in result:
                content_parts.append("\nEXECUTIVE COMPENSATION:")
                if "executives" in result["executive_compensation"]:
                    content_parts.append("\nEXECUTIVES:")
                    for executive in result["executive_compensation"]["executives"]:
                        content_parts.append(f"- {executive}")
                if "summary" in result["executive_compensation"]:
                    content_parts.append("\nCOMPENSATION SUMMARY:")
                    for key, value in result["executive_compensation"]["summary"].items():
                        content_parts.append(f"{key}: {value}")
            
            return '\n'.join(content_parts)
            
        except Exception as e:
            self.logger.log_error(e, "vector_content_preparation")
            return ""

    def _merge_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results from multiple chunks"""
        if not results:
            return {}
        
        # Initialize final result with first chunk's data
        final_result = results[0].copy()
        
        # Merge financial statements
        for chunk_result in results[1:]:
            # Merge financial statements
            for statement_type, table in chunk_result["financial_statements"].items():
                if statement_type not in final_result["financial_statements"]:
                    final_result["financial_statements"][statement_type] = table
            
            # Merge metrics (take the most recent values)
            for metric, value in chunk_result["metrics"].items():
                if metric not in final_result["metrics"]:
                    final_result["metrics"][metric] = value
            
            # Merge risk factors
            for category, factors in chunk_result["risk_factors"].items():
                if category not in final_result["risk_factors"]:
                    final_result["risk_factors"][category] = []
                final_result["risk_factors"][category].extend(
                    factor for factor in factors 
                    if factor not in final_result["risk_factors"][category]
                )
            
            # Merge executive compensation
            for executive in chunk_result["executive_compensation"]["executives"]:
                if executive not in final_result["executive_compensation"]["executives"]:
                    final_result["executive_compensation"]["executives"].append(executive)
            
            # Update compensation summary
            for key, value in chunk_result["executive_compensation"]["summary"].items():
                if key not in final_result["executive_compensation"]["summary"]:
                    final_result["executive_compensation"]["summary"][key] = value
        
        return final_result

    def extract_metadata_from_reader(self, reader: PdfReader, source_name: str) -> Dict[str, Any]:
        """Extract metadata from PDF reader object"""
        metadata = {
            "company_name": "",
            "form_type": "",
            "filing_date": "",
            "cik": "",
            "period_end": "",
            "fiscal_year": "",
            "source": source_name
        }
        
        try:
            first_page = reader.pages[0].extract_text()
            
            # Extract company name
            company_match = re.search(r"COMPANY\s+CONFORMED\s+NAME:\s+(.*?)\n", first_page)
            if company_match:
                metadata["company_name"] = company_match.group(1).strip()
            
            # Extract form type
            for form_type in self.form_types:
                if form_type in first_page:
                    metadata["form_type"] = form_type
                    break
            
            # Extract filing date
            date_match = re.search(r"FILED\s+AS\s+OF\s+DATE:\s+(\d{8})", first_page)
            if date_match:
                date_str = date_match.group(1)
                metadata["filing_date"] = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
            
            # Extract CIK
            cik_match = re.search(r"CENTRAL\s+INDEX\s+KEY:\s+(\d+)", first_page)
            if cik_match:
                metadata["cik"] = cik_match.group(1)
            
            # Extract period end date
            period_match = re.search(r"CONFORMED\s+PERIOD\s+OF\s+REPORT:\s+(\d{8})", first_page)
            if period_match:
                date_str = period_match.group(1)
                metadata["period_end"] = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
                metadata["fiscal_year"] = datetime.strptime(date_str, "%Y%m%d").strftime("%Y")
            
        except Exception as e:
            self.logger.log_error(e, "metadata_extraction", {"source_name": source_name})
        
        return metadata

    def extract_tables_from_reader(self, reader: PdfReader) -> List[pd.DataFrame]:
        """Extract tables from PDF reader object using multiple methods with fallbacks"""
        tables = []
        try:
            # First validate the PDF
            if not self._validate_pdf(reader):
                self.logger.error("Invalid PDF structure detected")
                return self._extract_tables_basic(reader)
            
            # Save PDF content temporarily with memory optimization
            with BytesIO() as temp_pdf:
                # Write the entire PDF content with chunking
                chunk_size = 1024 * 1024  # 1MB chunks
                total_chunks = sum(len(page.get_contents()) // chunk_size + 1 for page in reader.pages)
                current_chunk = 0
                
                for page in reader.pages:
                    page_content = page.get_contents()
                    if page_content:
                        if isinstance(page_content, bytes):
                            for i in range(0, len(page_content), chunk_size):
                                temp_pdf.write(page_content[i:i + chunk_size])
                                current_chunk += 1
                                self.logger.info(f"PDF content chunking progress: {current_chunk}/{total_chunks}")
                        else:
                            # If it's an EncodedStreamObject, get its raw data
                            data = page_content.get_data()
                            for i in range(0, len(data), chunk_size):
                                temp_pdf.write(data[i:i + chunk_size])
                                current_chunk += 1
                                self.logger.info(f"PDF content chunking progress: {current_chunk}/{total_chunks}")
                temp_pdf.seek(0)
                
                # Create a copy of the PDF content for parallel processing
                pdf_content = temp_pdf.getvalue()
                
                # Process tables in parallel using ThreadPoolExecutor with memory limits
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="table_extraction"
                ) as executor:
                    # Submit all table extraction methods with timeout
                    future_to_method = {
                        executor.submit(self._extract_tables_pdfplumber, pdf_content): "pdfplumber",
                        executor.submit(self._extract_tables_camelot, pdf_content): "camelot"
                    }
                    
                    # Collect results from all methods with timeout
                    for future in concurrent.futures.as_completed(future_to_method):
                        method = future_to_method[future]
                        try:
                            method_tables = future.result(timeout=300)  # 5-minute timeout
                            if method_tables:
                                tables.extend(method_tables)
                                self.logger.info(f"Tables found with {method}: {len(method_tables)}")
                        except concurrent.futures.TimeoutError:
                            self.logger.warning(f"Timeout in {method} table extraction")
                        except Exception as e:
                            self.logger.warning(f"Error in {method} table extraction: {str(e)}")
                
                # Clear memory
                del pdf_content
                
                # If no tables found, try basic extraction
                if not tables:
                    self.logger.warning("No tables found with pdfplumber or camelot. Falling back to basic extraction.")
                    tables = self._extract_tables_basic(reader)
                
                # Validate and filter tables in parallel with memory limits
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="table_validation"
                ) as executor:
                    # Submit table validation tasks with timeout
                    future_to_table = {
                        executor.submit(self._validate_and_type_table, table): table 
                        for table in tables
                    }
                    
                    # Collect validated tables
                    validated_tables = []
                    total_tables = len(tables)
                    processed_tables = 0
                    
                    for future in concurrent.futures.as_completed(future_to_table):
                        try:
                            result = future.result(timeout=60)  # 1-minute timeout
                            if result:
                                validated_tables.append(result)
                            processed_tables += 1
                            self.logger.info(f"Table validation progress: {processed_tables}/{total_tables}")
                        except concurrent.futures.TimeoutError:
                            self.logger.warning("Timeout in table validation")
                        except Exception as e:
                            self.logger.warning(f"Error in table validation: {str(e)}")
                
                # Sort tables by validation metrics
                validated_tables.sort(key=lambda x: (
                    x[2]["header_confidence"],
                    x[2]["row_consistency"],
                    x[2]["numeric_cols"]
                ), reverse=True)
                
                # Log final results
                self.logger.info(f"Total tables found: {len(tables)}")
                self.logger.info(f"Validated tables: {len(validated_tables)}")
                for table_type in set(table[1] for table in validated_tables):
                    count = sum(1 for table in validated_tables if table[1] == table_type)
                    self.logger.info(f"Tables of type {table_type}: {count}")
                
                # Clear memory
                del tables
                
                return [table[0] for table in validated_tables]
                
        except Exception as e:
            self.logger.log_error(e, "table_extraction")
            return []

    def _validate_pdf(self, reader: PdfReader) -> bool:
        """Validate PDF structure and content"""
        try:
            # Check if PDF has pages
            if not reader.pages:
                return False
            
            # Check if first page has content
            first_page = reader.pages[0]
            if not first_page.get_contents():
                return False
            
            # Check if PDF has a root object
            if not hasattr(reader, 'root_object'):
                return False
            
            return True
        except Exception:
            return False

    def _extract_tables_basic(self, reader: PdfReader) -> List[pd.DataFrame]:
        """Basic table extraction using PyPDF2 text extraction with improved pattern matching"""
        tables = []
        try:
            for page in reader.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                # Look for table-like patterns
                lines = text.split('\n')
                table_data = []
                current_table = []
                in_table = False
                
                for line in lines:
                    # Check for table start patterns
                    if not in_table and any(pattern in line for pattern in ['$', '(', ')', '|', '\t']):
                        in_table = True
                    
                    # If in table, check if line contains multiple values
                    if in_table:
                        if len(line.split()) > 2 or any(char in line for char in ['$', '(', ')', '|', '\t']):
                            current_table.append(line.split())
                        else:
                            # End of table found
                            if len(current_table) > 1:  # At least 2 rows
                                try:
                                    # Clean up the table data
                                    cleaned_table = []
                                    for row in current_table:
                                        if len(row) > 1:  # Only include rows with multiple columns
                                            cleaned_table.append(row)
                                    
                                    if len(cleaned_table) > 1:
                                        df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
                                        if not df.empty and len(df.columns) > 1:
                                            tables.append(df)
                                except Exception:
                                    pass
                            current_table = []
                            in_table = False
                
                # Handle last table if exists
                if current_table and len(current_table) > 1:
                    try:
                        cleaned_table = []
                        for row in current_table:
                            if len(row) > 1:
                                cleaned_table.append(row)
                        
                        if len(cleaned_table) > 1:
                            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
                            if not df.empty and len(df.columns) > 1:
                                tables.append(df)
                    except Exception:
                        pass
                    
        except Exception as e:
            self.logger.log_error(e, "basic_table_extraction")
            
        return tables

    def _validate_and_type_table(self, table: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, str, dict]]:
        """Validate table and identify its type"""
        try:
            metrics = self.validate_table_structure(table)
            if metrics["is_valid"]:
                table_type = self.identify_financial_table_type(table)
                if table_type != "unknown":
                    return (table, table_type, metrics)
            return None
        except Exception as e:
            self.logger.warning(f"Error in table validation: {str(e)}")
            return None

    def _extract_tables_pdfplumber(self, file_path: str) -> List[pd.DataFrame]:
        """Extract tables using pdfplumber."""
        try:
            tables = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table in page_tables:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(df)
            return tables
        except Exception as e:
            self.logger.log_error(e, "PDFPlumber extraction", "TABLE_EXTRACTION")
            return []

    def _extract_tables_camelot(self, file_path: str) -> List[pd.DataFrame]:
        """Extract tables using camelot."""
        try:
            tables = []
            # Try both lattice and stream methods
            for flavor in ['lattice', 'stream']:
                try:
                    camelot_tables = camelot.read_pdf(file_path, flavor=flavor, pages='all')
                    for table in camelot_tables:
                        tables.append(table.df)
                except Exception as e:
                    self.logger.log_error(e, f"Camelot {flavor} extraction", "TABLE_EXTRACTION")
            return tables
        except Exception as e:
            self.logger.log_error(e, "Camelot extraction", "TABLE_EXTRACTION")
            return []

    def identify_financial_statements(self, tables: List[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Identify and categorize financial statements from tables"""
        financial_statements = {}
        
        try:
            for table in tables:
                if table.empty:
                    continue
                    
                # Convert table to string for pattern matching
                table_text = table.to_string()
                
                for statement_type, pattern in self.financial_patterns.items():
                    if re.search(pattern, table_text, re.IGNORECASE):
                        financial_statements[statement_type] = table
                        break
        except Exception as e:
            self.logger.log_error(e, "financial_statement_identification")
        
        return financial_statements

    def extract_financial_metrics(self, financial_statements: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Extract key financial metrics from financial statements"""
        metrics = {}
        
        try:
            for statement_type, table in financial_statements.items():
                for metric_name, pattern in self.metric_patterns.items():
                    # Search for metric in table
                    for col in table.columns:
                        if re.search(pattern, str(col), re.IGNORECASE):
                            # Get the most recent value
                            value = table[col].iloc[-1]
                            if pd.notna(value):
                                metrics[metric_name] = float(value)
                                break
        except Exception as e:
            self.logger.log_error(e, "financial_metric_extraction")
        
        return metrics

    def extract_risk_factors_from_reader(self, reader: PdfReader) -> List[Dict[str, Any]]:
        """Extract risk factors from PDF reader object."""
        try:
            risk_factors = []
            risk_section = False
            current_risk = {"title": "", "description": ""}
            
            for page in reader.pages:
                text = page.extract_text()
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check for risk section start
                    if "risk factors" in line.lower() or "risk factors" in line.lower():
                        risk_section = True
                        continue
                        
                    # Check for risk section end
                    if risk_section and ("item" in line.lower() or "table of contents" in line.lower()):
                        risk_section = False
                        if current_risk["title"] and current_risk["description"]:
                            risk_factors.append(current_risk)
                            current_risk = {"title": "", "description": ""}
                        break
                        
                    if risk_section:
                        # Check for new risk factor (usually starts with a number or bullet)
                        if re.match(r'^\d+\.|^•|^[A-Z]\.', line):
                            if current_risk["title"] and current_risk["description"]:
                                risk_factors.append(current_risk)
                                current_risk = {"title": "", "description": ""}
                            current_risk["title"] = line
                        else:
                            current_risk["description"] += line + " "
                            
            # Add the last risk factor if exists
            if current_risk["title"] and current_risk["description"]:
                risk_factors.append(current_risk)
                
            return risk_factors
            
        except Exception as e:
            self.logger.log_error(e, "Risk factor extraction", "RISK_FACTOR_EXTRACTION")
            return []

    def extract_risk_factors_from_text(self, content: Union[str, bytes]) -> List[Dict[str, Any]]:
        """
        Extract risk factors from PDF content.
        
        Args:
            content: PDF content as either string or bytes
            
        Returns:
            List of dictionaries containing risk factor information
        """
        try:
            # Convert string to bytes if needed
            if isinstance(content, str):
                content = content.encode('utf-8')
            
            # Validate PDF content
            if not content or len(content) < 100:  # Minimum size check
                self.logger.warning("Invalid or empty PDF content")
                return []
            
            # Create a BytesIO object from the content
            pdf_file = BytesIO(content)
            
            try:
                # Create a PDF reader with validation
                reader = PdfReader(pdf_file)
                if not reader.pages:
                    self.logger.warning("PDF has no pages")
                    return []
            except Exception as e:
                self.logger.error(f"Error creating PDF reader: {str(e)}")
                return []
            
            # Use the existing method to extract risk factors
            return self.extract_risk_factors_from_reader(reader)
            
        except Exception as e:
            self.logger.error(f"Error extracting risk factors from text: {str(e)}")
            return []

    def extract_executive_compensation_from_reader(self, reader: PdfReader) -> Dict[str, Any]:
        """Extract executive compensation from PDF reader object"""
        compensation = {
            "executives": [],
            "summary": {}
        }
        
        try:
            comp_section = ""
            for page in reader.pages:
                text = page.extract_text()
                if "EXECUTIVE COMPENSATION" in text:
                    comp_section = text
                    break
            
            if not comp_section:
                return compensation
            
            # Extract compensation tables
            tables = self.extract_tables_from_reader(reader)
            comp_tables = []
            
            for table in tables:
                if table.empty:
                    continue
                
                # Look for compensation-related columns
                cols = [str(col).lower() for col in table.columns]
                if any(keyword in ' '.join(cols) for keyword in ['salary', 'bonus', 'stock', 'option', 'compensation']):
                    comp_tables.append(table)
            
            # Process compensation tables
            for table in comp_tables:
                # Look for executive names
                for col in table.columns:
                    if 'name' in str(col).lower():
                        executives = table[col].dropna().tolist()
                        compensation["executives"].extend(executives)
                
                # Extract compensation details
                for col in table.columns:
                    col_name = str(col).lower()
                    if 'salary' in col_name:
                        compensation["summary"]["salary"] = table[col].mean()
                    elif 'bonus' in col_name:
                        compensation["summary"]["bonus"] = table[col].mean()
                    elif 'stock' in col_name:
                        compensation["summary"]["stock_awards"] = table[col].mean()
                    elif 'option' in col_name:
                        compensation["summary"]["option_awards"] = table[col].mean()
        
        except Exception as e:
            self.logger.log_error(e, "executive_compensation_extraction")
        
        return compensation

    def extract_executive_compensation_from_text(self, content: Union[str, bytes]) -> Dict[str, Any]:
        """
        Extract executive compensation from PDF content.
        
        Args:
            content: PDF content as either string or bytes
            
        Returns:
            Dictionary containing executive compensation information
        """
        try:
            # Convert string to bytes if needed
            if isinstance(content, str):
                content = content.encode('utf-8')
            
            # Create a BytesIO object from the content
            pdf_file = BytesIO(content)
            
            # Create a PDF reader
            reader = PdfReader(pdf_file)
            
            # Use the existing method to extract executive compensation
            return self.extract_executive_compensation_from_reader(reader)
            
        except Exception as e:
            self.logger.error(f"Error extracting executive compensation from text: {str(e)}")
            return {
                "executives": [],
                "summary": {}
            }

    def process_gcs_directory(self, prefix: str) -> List[Dict[str, Any]]:
        """Process all PDF files in a GCS directory sequentially"""
        results = []
        
        try:
            self.logger.info(f"Starting sequential processing of GCS directory: {prefix}")
            
            # List all PDF blobs
            blobs = self.bucket.list_blobs(prefix=prefix)
            pdf_blobs = [blob for blob in blobs if blob.name.lower().endswith('.pdf')]
            
            if not pdf_blobs:
                self.logger.warning(f"No PDF files found in GCS path: {prefix}")
                return results
            
            total_files = len(pdf_blobs)
            self.logger.info(f"Found {total_files} PDF files to process")
            
            # Process files sequentially
            for i, blob in enumerate(pdf_blobs, 1):
                self.logger.info(f"Processing file {i}/{total_files}: {blob.name}")
                
                try:
                    # Process single file
                    result = self.process_gcs_pdf(blob.name)
                    
                    if result:
                        results.append(result)
                        self.logger.info(f"Successfully completed processing of {blob.name}")
                        
                        # Log completion status
                        self.logger.log_document_processing(
                            blob.name,
                            "COMPLETED",
                            f"Processed {i}/{total_files} files",
                            "DOCUMENT_PROCESSING"
                        )
                    else:
                        self.logger.warning(f"Processing returned no result for: {blob.name}")
                        
                        # Log failure status
                        self.logger.log_document_processing(
                            blob.name,
                            "FAILED",
                            f"No result generated for {blob.name}",
                            "DOCUMENT_PROCESSING"
                        )
                        
                except Exception as e:
                    self.logger.log_error(
                        e,
                        f"Error processing {blob.name}",
                        "DOCUMENT_PROCESSING"
                    )
                    
                    # Log error status
                    self.logger.log_document_processing(
                        blob.name,
                        "ERROR",
                        f"Error processing {blob.name}: {str(e)}",
                        "DOCUMENT_PROCESSING"
                    )
                    
                    # Continue with next file
                    continue
            
            # Log final summary
            self.logger.info(f"Completed processing of {len(results)}/{total_files} files successfully")
            self.logger.log_document_processing(
                "SUMMARY",
                "COMPLETED",
                f"Processed {len(results)}/{total_files} files successfully",
                "DOCUMENT_PROCESSING"
            )
            
        except Exception as e:
            self.logger.log_error(e, "gcs_directory_processing", {"prefix": prefix})
        
        return results

    def validate_table_structure(self, df: pd.DataFrame) -> dict:
        """Validate table structure and return metrics"""
        metrics = {
            "is_valid": False,
            "row_count": len(df),
            "col_count": len(df.columns),
            "numeric_cols": 0,
            "header_confidence": 0.0,
            "row_consistency": 0.0
        }
        
        try:
            # Check basic dimensions
            if (metrics["row_count"] < self.table_validation_metrics["min_rows"] or 
                metrics["col_count"] < self.table_validation_metrics["min_cols"]):
                return metrics
            
            # Count numeric columns
            numeric_cols = sum(df[col].dtype in ['int64', 'float64'] for col in df.columns)
            metrics["numeric_cols"] = numeric_cols
            numeric_ratio = numeric_cols / metrics["col_count"]
            
            # Check header confidence
            header_confidence = self._calculate_header_confidence(df)
            metrics["header_confidence"] = header_confidence
            
            # Check row consistency
            row_consistency = self._calculate_row_consistency(df)
            metrics["row_consistency"] = row_consistency
            
            # Determine if table is valid
            metrics["is_valid"] = (
                numeric_ratio >= self.table_validation_metrics["numeric_cols_ratio"] and
                header_confidence >= self.table_validation_metrics["header_confidence"] and
                row_consistency >= self.table_validation_metrics["row_consistency"]
            )
            
        except Exception as e:
            self.logger.log_error(e, "table_validation")
            
        return metrics

    def _calculate_header_confidence(self, df: pd.DataFrame) -> float:
        """Calculate confidence score for header row"""
        try:
            header = df.columns.tolist()
            confidence = 0.0
            
            # Check for common financial terms
            financial_terms = [
                'amount', 'balance', 'revenue', 'income', 'expense', 
                'asset', 'liability', 'equity', 'cash', 'flow',
                'earnings', 'profit', 'loss', 'margin', 'ratio',
                'debt', 'capital', 'investment', 'dividend', 'interest',
                'depreciation', 'amortization', 'tax', 'net', 'gross',
                'operating', 'financial', 'cost', 'price', 'value',
                'share', 'stock', 'option', 'warrant', 'security',
                'derivative', 'hedge', 'lease', 'pension', 'benefit'
            ]
            term_matches = sum(1 for term in financial_terms 
                             if any(term in str(col).lower() for col in header))
            
            # Check for numeric indicators
            numeric_indicators = ['$', '(', ')', '%', 'USD', 'EUR', 'GBP', 'JPY']
            indicator_matches = sum(1 for col in header 
                                  if any(ind in str(col) for ind in numeric_indicators))
            
            # Check for date patterns
            date_patterns = [
                r'\d{4}',  # Year
                r'Q\d',    # Quarter
                r'FY\d{4}', # Fiscal Year
                r'YTD',    # Year to Date
                r'TTM'     # Trailing Twelve Months
            ]
            date_matches = sum(1 for col in header 
                             if any(re.search(pattern, str(col)) for pattern in date_patterns))
            
            # Calculate confidence score
            confidence = (
                (term_matches * 0.5) + 
                (indicator_matches * 0.3) + 
                (date_matches * 0.2)
            ) / len(header)
            
            return min(confidence, 1.0)
            
        except Exception:
            return 0.0

    def _calculate_row_consistency(self, df: pd.DataFrame) -> float:
        """Calculate consistency score for row structure"""
        try:
            if len(df) < 2:
                return 0.0
                
            # Check for consistent data types in columns
            type_consistency = 0.0
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    # Check for consistent numeric patterns
                    if df[col].notna().all():
                        # Check for financial number patterns
                        if df[col].astype(str).str.match(r'^-?\d+(,\d{3})*(\.\d+)?$').all():
                            type_consistency += 1.0
                else:
                    # Check for consistent string patterns
                    if df[col].str.match(r'^[A-Za-z\s\-\(\)\$\%]+$').all():
                        type_consistency += 1.0
            
            return type_consistency / len(df.columns)
            
        except Exception:
            return 0.0

    def identify_financial_table_type(self, df: pd.DataFrame) -> str:
        """Identify the type of financial table"""
        try:
            # Convert table to string for pattern matching
            table_text = df.to_string()
            
            # Check each table type pattern
            for table_type, patterns in self.financial_table_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, table_text, re.IGNORECASE):
                        return table_type
            
            return "unknown"
            
        except Exception as e:
            self.logger.log_error(e, "table_type_identification")
            return "unknown"

    def process_filing(self, file_path: str) -> Dict[str, Any]:
        """Process a local PDF file."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            if not content:
                self.logger.log_error(ValueError("Empty file"), f"Processing {file_path}", "DOCUMENT_PROCESSING")
                return {}
            
            return self._process_pdf_content(content, file_path)
        except Exception as e:
            self.logger.log_error(e, f"Processing {file_path}", "DOCUMENT_PROCESSING")
            return {}
    
    def _process_pdf_content(self, content: bytes, doc_id: str) -> Dict[str, Any]:
        """Process PDF content and extract information."""
        start_time = time.time()
        
        try:
            # Create temporary file for parallel processing
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            # Process in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_pdfplumber = executor.submit(self._extract_tables_pdfplumber, temp_path)
                future_camelot = executor.submit(self._extract_tables_camelot, temp_path)
                
                tables_pdfplumber = future_pdfplumber.result(timeout=30)
                tables_camelot = future_camelot.result(timeout=30)
            
            # Combine and validate tables
            all_tables = tables_pdfplumber + tables_camelot
            validated_tables = [t for t in all_tables if self.validate_table_structure(t)]
            
            # Sort tables by confidence metrics
            sorted_tables = sorted(validated_tables, key=lambda t: (
                self._calculate_header_confidence(t),
                self._calculate_row_consistency(t)
            ), reverse=True)
            
            # Identify table types
            financial_statements = {}
            for table in sorted_tables:
                table_type = self.identify_financial_table_type(table)
                if table_type:
                    financial_statements[table_type] = table
            
            # Extract risk factors using the correct method
            pdf_file = BytesIO(content)
            reader = PdfReader(pdf_file)
            risk_factors = self.extract_risk_factors_from_reader(reader)
            
            # Clean up
            os.unlink(temp_path)
            
            # Log results
            duration = time.time() - start_time
            self.logger.log_performance(f"Processed {doc_id}", duration, "DOCUMENT_PROCESSING")
            self.logger.log_table_extraction(
                doc_id,
                len(validated_tables),
                len(validated_tables) > 0,
                f"Found {len(financial_statements)} financial statements",
                "TABLE_EXTRACTION"
            )
            
            return {
                "financial_statements": financial_statements,
                "metrics": self._extract_metrics(sorted_tables),
                "risk_factors": risk_factors,
                "executives": self._extract_executives(content)
            }
            
        except Exception as e:
            self.logger.log_error(e, f"Processing {doc_id}", "DOCUMENT_PROCESSING")
            return {} 