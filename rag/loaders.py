import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import io
import zipfile
import tempfile
import os

from langchain.document_loaders import (
    CSVLoader, PyPDFLoader, UnstructuredFileLoader, 
    DataFrameLoader, TextLoader
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class FinancialCSVLoader:
    """Loader for financial CSV files with specialized parsing"""
    
    def __init__(self, file_path: Union[str, io.BytesIO]):
        self.file_path = file_path
        
    def load(self) -> List[Document]:
        """Load financial CSV data into documents"""
        try:
            if isinstance(self.file_path, io.BytesIO):
                df = pd.read_csv(self.file_path)
            else:
                df = pd.read_csv(self.file_path)
            
            # Convert DataFrame to documents
            documents = []
            for _, row in df.iterrows():
                content = "\n".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                metadata = {
                    "source": "csv_financial_data",
                    "row_index": _,
                    "columns": list(df.columns)
                }
                documents.append(Document(page_content=content, metadata=metadata))
            
            return documents
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return []

class FinancialPDFLoader:
    """Loader for financial PDF documents"""
    
    def __init__(self, file_path: Union[str, io.BytesIO]):
        self.file_path = file_path
        
    def load(self) -> List[Document]:
        """Load PDF financial documents"""
        try:
            if isinstance(self.file_path, io.BytesIO):
                # Save bytes to temp file for PDF loader
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(self.file_path.getvalue())
                    tmp_path = tmp.name
                
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                
                # Clean up temp file
                os.unlink(tmp_path)
            else:
                loader = PyPDFLoader(self.file_path)
                documents = loader.load()
            
            # Add financial metadata
            for doc in documents:
                doc.metadata.update({
                    "document_type": "financial_pdf",
                    "source_type": "bank_statement"
                })
            
            return documents
            
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return []

class MultiFormatFinancialLoader:
    """Loader that handles multiple financial document formats"""
    
    def __init__(self):
        self.loaders = {
            'csv': FinancialCSVLoader,
            'pdf': FinancialPDFLoader
        }
    
    def load_documents(self, file_path: str, file_type: str = None) -> List[Document]:
        """Load documents based on file type"""
        if file_type is None:
            file_type = self._detect_file_type(file_path)
        
        if file_type in self.loaders:
            loader = self.loaders[file_type](file_path)
            return loader.load()
        else:
            print(f"Unsupported file type: {file_type}")
            return []
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension"""
        if isinstance(file_path, io.BytesIO):
            # For in-memory files, we need to handle differently
            return 'csv'  # Default assumption for streamlit uploads
        
        ext = Path(file_path).suffix.lower()
        if ext == '.csv':
            return 'csv'
        elif ext == '.pdf':
            return 'pdf'
        elif ext == '.xlsx':
            return 'csv'  # Treat Excel as CSV for now
        elif ext == '.zip':
            return 'zip'
        else:
            return 'unknown'

def load_financial_documents(file_paths: List[str]) -> List[Document]:
    """Load multiple financial documents"""
    loader = MultiFormatFinancialLoader()
    all_documents = []
    
    for file_path in file_paths:
        documents = loader.load_documents(file_path)
        all_documents.extend(documents)
    
    return all_documents

def load_knowledge_base(knowledge_base_path: str) -> List[Document]:
    """Load documents from knowledge base directory"""
    documents = []
    knowledge_path = Path(knowledge_base_path)
    
    if knowledge_path.exists() and knowledge_path.is_dir():
        for file_path in knowledge_path.glob('**/*'):
            if file_path.suffix.lower() in ['.txt', '.md', '.pdf', '.csv']:
                loader = MultiFormatFinancialLoader()
                docs = loader.load_documents(str(file_path))
                documents.extend(docs)
    
    return documents
