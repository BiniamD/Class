"""
SEC Filing Processor for FinancialIQ
Handles PDF processing, metadata extraction, and financial data analysis
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import re
from datetime import datetime
import pdfplumber
import pandas as pd
from pydantic import BaseModel, Field
from google.cloud import storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings

class SECFilingMetadata(BaseModel):
    """Metadata model for SEC filings"""
    company_name: str
    filing_date: datetime
    form_type: str
    cik: str
    accession_number: str
    file_number: Optional[str] = None
    sic: Optional[str] = None
    fiscal_year_end: Optional[str] = None

class FinancialData(BaseModel):
    """Financial data model"""
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None
    assets: Optional[float] = None
    liabilities: Optional[float] = None
    equity: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None

class DocumentChunk(BaseModel):
    """Document chunk model"""
    text: str
    metadata: Dict[str, Any]
    page_number: int
    section: Optional[str] = None

class SECFilingProcessor:
    """Processes SEC filings and extracts relevant information"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_documents: int = 1000,
        test_mode: bool = False
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_documents = max_documents
        self.test_mode = test_mode
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings if not in test mode
        if not test_mode:
            self.embeddings = VertexAIEmbeddings(
                model_name="textembedding-gecko@001",
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION")
            )

    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a PDF file and extract all relevant information"""
        try:
            # Extract text and metadata
            text = self._extract_text(pdf_path)
            metadata = self._extract_metadata(pdf_path)
            
            # Extract financial data
            financial_data = self._extract_financial_data(text)
            
            # Chunk the document
            chunks = self._chunk_document(text, metadata)
            
            return {
                "text": text,
                "metadata": metadata.dict(),
                "financial_data": financial_data.dict(),
                "chunks": [chunk.dict() for chunk in chunks]
            }
        except Exception as e:
            raise Exception(f"Error processing PDF {pdf_path}: {str(e)}")

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    def _extract_metadata(self, pdf_path: Path) -> SECFilingMetadata:
        """Extract metadata from PDF file"""
        text = self._extract_text(pdf_path)
        
        # Extract company name (usually in first few lines)
        company_name = self._extract_company_name(text)
        
        # Extract filing date
        filing_date = self._extract_filing_date(text)
        
        # Extract form type
        form_type = self._extract_form_type(text)
        
        # Extract CIK
        cik = self._extract_cik(text)
        
        # Extract accession number
        accession_number = self._extract_accession_number(text)
        
        return SECFilingMetadata(
            company_name=company_name,
            filing_date=filing_date,
            form_type=form_type,
            cik=cik,
            accession_number=accession_number
        )

    def _extract_financial_data(self, text: str) -> FinancialData:
        """Extract financial data from text"""
        # Initialize financial data
        financial_data = FinancialData()
        
        # Extract revenue
        revenue_match = re.search(r"Total Revenue\s*\$?\s*([\d,]+\.?\d*)", text)
        if revenue_match:
            financial_data.revenue = float(revenue_match.group(1).replace(",", ""))
        
        # Extract net income
        net_income_match = re.search(r"Net Income\s*\$?\s*([\d,]+\.?\d*)", text)
        if net_income_match:
            financial_data.net_income = float(net_income_match.group(1).replace(",", ""))
        
        # Extract EPS
        eps_match = re.search(r"Earnings Per Share\s*\$?\s*([\d,]+\.?\d*)", text)
        if eps_match:
            financial_data.eps = float(eps_match.group(1).replace(",", ""))
        
        # Extract assets
        assets_match = re.search(r"Total Assets\s*\$?\s*([\d,]+\.?\d*)", text)
        if assets_match:
            financial_data.assets = float(assets_match.group(1).replace(",", ""))
        
        # Extract liabilities
        liabilities_match = re.search(r"Total Liabilities\s*\$?\s*([\d,]+\.?\d*)", text)
        if liabilities_match:
            financial_data.liabilities = float(liabilities_match.group(1).replace(",", ""))
        
        return financial_data

    def _chunk_document(self, text: str, metadata: SECFilingMetadata) -> List[DocumentChunk]:
        """Split document into chunks with metadata"""
        chunks = []
        sections = self._split_into_sections(text)
        
        for section_name, section_text in sections.items():
            # Split section into chunks
            section_chunks = self.text_splitter.split_text(section_text)
            
            # Create chunk objects with metadata
            for i, chunk_text in enumerate(section_chunks):
                chunk = DocumentChunk(
                    text=chunk_text,
                    metadata={
                        "company_name": metadata.company_name,
                        "filing_date": metadata.filing_date,
                        "form_type": metadata.form_type,
                        "cik": metadata.cik,
                        "section": section_name
                    },
                    page_number=i + 1,
                    section=section_name
                )
                chunks.append(chunk)
        
        return chunks

    def _extract_company_name(self, text: str) -> str:
        """Extract company name from text"""
        # Look for common patterns in SEC filings
        patterns = [
            r"COMPANY NAME:\s*(.*?)\n",
            r"Name of Registrant:\s*(.*?)\n",
            r"\(Exact name of registrant as specified in its charter\)\s*(.*?)\n"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Unknown Company"

    def _extract_filing_date(self, text: str) -> datetime:
        """Extract filing date from text"""
        # Look for common date patterns
        patterns = [
            r"Date of Report \(Date of earliest event reported\):\s*(\d{1,2}/\d{1,2}/\d{4})",
            r"Filing Date:\s*(\d{1,2}/\d{1,2}/\d{4})",
            r"Date:\s*(\d{1,2}/\d{1,2}/\d{4})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return datetime.strptime(match.group(1), "%m/%d/%Y")
                except ValueError:
                    continue
        
        return datetime.now()  # Default to current date if not found

    def _extract_form_type(self, text: str) -> str:
        """Extract form type from text"""
        match = re.search(r"FORM\s+([A-Z0-9-]+)", text)
        return match.group(1) if match else "UNKNOWN"

    def _extract_cik(self, text: str) -> str:
        """Extract CIK number from text"""
        match = re.search(r"CIK:\s*(\d+)", text)
        return match.group(1) if match else "0000000000"

    def _extract_accession_number(self, text: str) -> str:
        """Extract accession number from text"""
        match = re.search(r"Accession Number:\s*(\d{10}-\d{2}-\d{6})", text)
        return match.group(1) if match else "0000000000-00-000000"

    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split document into sections"""
        sections = {
            "RISK_FACTORS": "",
            "MANAGEMENT_DISCUSSION": "",
            "FINANCIAL_STATEMENTS": "",
            "OTHER": ""
        }
        
        # Define section patterns
        patterns = {
            "RISK_FACTORS": r"ITEM\s*1A\.?\s*RISK\s*FACTORS(.*?)(?=ITEM\s*1B)",
            "MANAGEMENT_DISCUSSION": r"ITEM\s*7\.?\s*MANAGEMENT'S\s*DISCUSSION(.*?)(?=ITEM\s*8)",
            "FINANCIAL_STATEMENTS": r"ITEM\s*8\.?\s*FINANCIAL\s*STATEMENTS(.*?)(?=ITEM\s*9)"
        }
        
        # Extract sections
        for section, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section] = match.group(1).strip()
        
        # Put remaining text in OTHER section
        other_text = text
        for section_text in sections.values():
            other_text = other_text.replace(section_text, "")
        sections["OTHER"] = other_text.strip()
        
        return sections 