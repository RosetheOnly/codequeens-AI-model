"""
Document loaders for RAG system in CreditAI application.
Supports various document formats for credit-related information.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import csv
from dataclasses import dataclass

@dataclass
class Document:
    """Document class to store content and metadata."""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None

class BaseLoader:
    """Base class for document loaders."""
    
    def load(self, file_path: str) -> List[Document]:
        raise NotImplementedError

class CSVLoader(BaseLoader):
    """Loader for CSV files containing credit data."""
    
    def __init__(self, content_column: str = "content", metadata_columns: List[str] = None):
        self.content_column = content_column
        self.metadata_columns = metadata_columns or []
    
    def load(self, file_path: str) -> List[Document]:
        documents = []
        df = pd.read_csv(file_path)
        
        for idx, row in df.iterrows():
            # Use specified content column or concatenate all columns
            if self.content_column in df.columns:
                content = str(row[self.content_column])
            else:
                content = " ".join([f"{col}: {val}" for col, val in row.items()])
            
            # Extract metadata
            metadata = {"source": file_path, "row_id": idx}
            for col in self.metadata_columns:
                if col in df.columns:
                    metadata[col] = row[col]
            
            documents.append(Document(
                content=content,
                metadata=metadata,
                doc_id=f"{file_path}_{idx}"
            ))
        
        return documents

class TextLoader(BaseLoader):
    """Loader for plain text files."""
    
    def load(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return [Document(
            content=content,
            metadata={"source": file_path, "type": "text"},
            doc_id=file_path
        )]

class JSONLoader(BaseLoader):
    """Loader for JSON files containing credit policies or guidelines."""
    
    def __init__(self, content_key: str = "content"):
        self.content_key = content_key
    
    def load(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        documents = []
        
        if isinstance(data, list):
            for idx, item in enumerate(data):
                content = self._extract_content(item)
                metadata = {"source": file_path, "item_id": idx}
                metadata.update({k: v for k, v in item.items() if k != self.content_key})
                
                documents.append(Document(
                    content=content,
                    metadata=metadata,
                    doc_id=f"{file_path}_{idx}"
                ))
        else:
            content = self._extract_content(data)
            metadata = {"source": file_path, "type": "json"}
            metadata.update({k: v for k, v in data.items() if k != self.content_key})
            
            documents.append(Document(
                content=content,
                metadata=metadata,
                doc_id=file_path
            ))
        
        return documents
    
    def _extract_content(self, item: Dict) -> str:
        if self.content_key in item:
            return str(item[self.content_key])
        else:
            return json.dumps(item, indent=2)

class CreditDataLoader(BaseLoader):
    """Specialized loader for credit-related data."""
    
    def __init__(self):
        self.risk_mapping = {0: "Low Risk", 1: "High Risk"}
    
    def load(self, file_path: str) -> List[Document]:
        """Load and format credit data into documents."""
        documents = []
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        for idx, row in df.iterrows():
            # Create structured content for credit records
            content_parts = []
            
            if 'Amount' in row:
                content_parts.append(f"Transaction Amount: ${row['Amount']:.2f}")
            if 'Balance' in row:
                content_parts.append(f"Account Balance: ${row['Balance']:.2f}")
            if 'Punctuality_Score' in row:
                content_parts.append(f"Payment Punctuality Score: {row['Punctuality_Score']:.2f}")
            if 'Sector_Code' in row:
                content_parts.append(f"Business Sector Code: {row['Sector_Code']}")
            if 'Risk_Label' in row:
                risk_text = self.risk_mapping.get(row['Risk_Label'], "Unknown")
                content_parts.append(f"Risk Assessment: {risk_text}")
            
            content = " | ".join(content_parts)
            
            metadata = {
                "source": file_path,
                "record_id": idx,
                "data_type": "credit_record"
            }
            
            # Add all row data as metadata
            for col, val in row.items():
                metadata[col] = val
            
            documents.append(Document(
                content=content,
                metadata=metadata,
                doc_id=f"credit_record_{idx}"
            ))
        
        return documents

class DocumentLoader:
    """Main document loader that automatically selects appropriate loader."""
    
    def __init__(self):
        self.loaders = {
            '.csv': CSVLoader(),
            '.txt': TextLoader(),
            '.json': JSONLoader(),
            '.credit': CreditDataLoader()  # Custom extension for credit data
        }
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load documents from multiple file paths."""
        all_documents = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            file_ext = Path(file_path).suffix.lower()
            
            # Special handling for credit data files
            if 'credit' in file_path.lower() or 'risk' in file_path.lower():
                loader = CreditDataLoader()
            elif file_ext in self.loaders:
                loader = self.loaders[file_ext]
            else:
                print(f"Warning: No loader found for file type: {file_ext}")
                continue
            
            try:
                documents = loader.load(file_path)
                all_documents.extend(documents)
                print(f"Loaded {len(documents)} documents from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        return all_documents

def load_credit_policies(policy_dir: str = "data/policies") -> List[Document]:
    """Load credit policy documents from a directory."""
    loader = DocumentLoader()
    policy_files = []
    
    if os.path.exists(policy_dir):
        for file in os.listdir(policy_dir):
            if file.endswith(('.txt', '.json', '.csv')):
                policy_files.append(os.path.join(policy_dir, file))
    
    return loader.load_documents(policy_files)
