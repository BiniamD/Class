"""
Enhanced Document Processor for SEC Filings
Handles structured data extraction, table processing, and metadata identification
"""

import os
import re
from typing import Dict, List, Any, Optional, Generator
import pandas as pd
from PyPDF2 import PdfReader
import tabula
from datetime import datetime
import json
import concurrent.futures
from google.cloud import storage
from google.api_core import retry
from google.api_core.exceptions import GoogleAPIError
from io import BytesIO
from logger import FinancialIQLogger
from functools import lru_cache

class EnhancedSECFilingProcessor:
    """Enhanced processor for SEC filings with improved table and structured data extraction"""
    
    def __init__(self, project_id: str, bucket_name: str, log_dir: str = "logs", max_workers: int = 4):
        self.logger = FinancialIQLogger(log_dir)
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.max_workers = max_workers
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
        
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
        """Process a PDF file directly from Google Cloud Storage"""
        try:
            self.logger.info(f"Processing GCS document: {blob_name}")
            
            # Check cache first
            cached_result = self.get_cached_metadata(blob_name)
            if cached_result:
                self.logger.info(f"Using cached result for {blob_name}")
                return cached_result
            
            # Stream and validate PDF content
            pdf_content = b""
            for chunk in self.get_blob_content(blob_name):
                pdf_content += chunk
                
            if not self.validate_pdf_content(pdf_content):
                self.logger.error(f"Invalid PDF content for {blob_name}")
                return {}
            
            # Process PDF content
            result = self.process_pdf_content(pdf_content, blob_name)
            
            if result:
                # Store results in GCS
                result_blob = self.bucket.blob(f"processed/{blob_name}.json")
                result_blob.upload_from_string(
                    json.dumps(result),
                    content_type="application/json"
                )
                
                # Cache metadata
                cache_blob = self.bucket.blob(f"cache/metadata/{blob_name}.json")
                cache_blob.upload_from_string(
                    json.dumps(result),
                    content_type="application/json"
                )
            
            self.logger.info(f"Successfully processed and stored results for {blob_name}")
            return result
            
        except Exception as e:
            self.logger.log_error(e, "gcs_pdf_processing", {"blob_name": blob_name})
            return {}

    def process_pdf_content(self, pdf_content: bytes, source_name: str) -> Dict[str, Any]:
        """Process PDF content from memory"""
        try:
            # Create PDF reader from bytes
            pdf_file = BytesIO(pdf_content)
            reader = PdfReader(pdf_file)
            
            # Extract metadata
            metadata = self.extract_metadata_from_reader(reader, source_name)
            
            # Extract tables
            tables = self.extract_tables_from_reader(reader)
            
            # Process tables and extract information
            financial_statements = self.identify_financial_statements(tables)
            metrics = self.extract_financial_metrics(financial_statements)
            
            # Extract risk factors and compensation
            risk_factors = self.extract_risk_factors_from_reader(reader)
            compensation = self.extract_executive_compensation_from_reader(reader)
            
            # Combine results
            result = {
                "metadata": metadata,
                "financial_statements": financial_statements,
                "metrics": metrics,
                "risk_factors": risk_factors,
                "executive_compensation": compensation,
                "source": source_name
            }
            
            return result
            
        except Exception as e:
            self.logger.log_error(e, "pdf_content_processing", {"source_name": source_name})
            return {}

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
        """Extract tables from PDF reader object"""
        try:
            # Save PDF content temporarily to process with tabula
            with BytesIO() as temp_pdf:
                for page in reader.pages:
                    temp_pdf.write(page.get_contents())
                temp_pdf.seek(0)
                
                tables = tabula.read_pdf(
                    temp_pdf,
                    pages='all',
                    multiple_tables=True,
                    lattice=True,
                    stream=True
                )
            return tables
        except Exception as e:
            self.logger.log_error(e, "table_extraction")
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

    def extract_risk_factors_from_reader(self, reader: PdfReader) -> Dict[str, List[str]]:
        """Extract risk factors from PDF reader object"""
        risk_factors = {category: [] for category in self.risk_categories.keys()}
        
        try:
            risk_section = ""
            for page in reader.pages:
                text = page.extract_text()
                if "RISK FACTORS" in text:
                    risk_section = text
                    break
            
            if not risk_section:
                return risk_factors
            
            # Extract individual risk factors
            risk_items = re.split(r'\n\s*\d+\.\s*', risk_section)
            
            for item in risk_items:
                if not item.strip():
                    continue
                
                # Categorize risk factor
                for category, keywords in self.risk_categories.items():
                    if any(keyword.lower() in item.lower() for keyword in keywords):
                        risk_factors[category].append(item.strip())
                        break
        
        except Exception as e:
            self.logger.log_error(e, "risk_factor_extraction")
        
        return risk_factors

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

    def process_gcs_directory(self, prefix: str) -> List[Dict[str, Any]]:
        """Process all PDF files in a GCS directory with parallel processing"""
        results = []
        
        try:
            self.logger.info(f"Processing GCS directory: {prefix}")
            
            # List all PDF blobs
            blobs = self.bucket.list_blobs(prefix=prefix)
            pdf_blobs = [blob for blob in blobs if blob.name.lower().endswith('.pdf')]
            
            if not pdf_blobs:
                self.logger.warning(f"No PDF files found in GCS path: {prefix}")
                return results
            
            # Process PDFs in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_blob = {
                    executor.submit(self.process_gcs_pdf, blob.name): blob 
                    for blob in pdf_blobs
                }
                
                for future in concurrent.futures.as_completed(future_to_blob):
                    blob = future_to_blob[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        self.logger.log_error(
                            e, 
                            "parallel_processing", 
                            {"blob_name": blob.name}
                        )
            
            self.logger.info(f"Processed {len(results)} documents from GCS path: {prefix}")
            
        except Exception as e:
            self.logger.log_error(e, "gcs_directory_processing", {"prefix": prefix})
        
        return results 