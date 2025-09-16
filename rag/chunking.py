from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, 
    SentenceTransformersTokenTextSplitter,
    CharacterTextSplitter
)
import re
from sentence_transformers import SentenceTransformer
import numpy as np

class FinancialTextSplitter:
    """Text splitter optimized for financial documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split financial documents into chunks"""
        return self.splitter.split_documents(documents)
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        return self.splitter.split_text(text)

class SemanticFinancialChunker:
    """Semantic chunker for financial content using sentence transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', threshold: float = 0.85):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
    
    def semantic_chunking(self, documents: List[Document]) -> List[Document]:
        """Chunk documents based on semantic similarity"""
        chunks = []
        
        for doc in documents:
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', doc.page_content)
            
            current_chunk = []
            previous_embedding = None
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                current_embedding = self.model.encode([sentence])[0]
                
                if previous_embedding is not None:
                    similarity = np.dot(previous_embedding, current_embedding) / (
                        np.linalg.norm(previous_embedding) * np.linalg.norm(current_embedding)
                    )
                    
                    if similarity < self.threshold:
                        # Start new chunk
                        if current_chunk:
                            chunk_text = " ".join(current_chunk)
                            chunks.append(Document(
                                page_content=chunk_text,
                                metadata=doc.metadata.copy()
                            ))
                        current_chunk = [sentence]
                    else:
                        current_chunk.append(sentence)
                else:
                    current_chunk.append(sentence)
                
                previous_embedding = current_embedding
            
            # Add the last chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy()
                ))
        
        return chunks

class AdaptiveFinancialChunker:
    """Adaptive chunker that adjusts based on content type"""
    
    def __init__(self):
        self.financial_splitter = FinancialTextSplitter()
        self.semantic_chunker = SemanticFinancialChunker()
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Adaptively chunk documents based on content type"""
        chunks = []
        
        for doc in documents:
            content_type = self._detect_content_type(doc.page_content)
            
            if content_type == "structured_financial":
                # Use smaller chunks for structured data
                splitter = FinancialTextSplitter(chunk_size=500, chunk_overlap=100)
                doc_chunks = splitter.split_documents([doc])
            elif content_type == "narrative_financial":
                # Use semantic chunking for narrative content
                doc_chunks = self.semantic_chunker.semantic_chunking([doc])
            else:
                # Default chunking
                doc_chunks = self.financial_splitter.split_documents([doc])
            
            chunks.extend(doc_chunks)
        
        return chunks
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the type of financial content"""
        # Check for structured data patterns
        structured_patterns = [
            r'\d+/\d+/\d+',  # Dates
            r'\$\d+\.\d+',   # Currency amounts
            r'Account Number:\s*\w+',
            r'Transaction ID:\s*\w+'
        ]
        
        narrative_patterns = [
            r'financial statement',
            r'annual report',
            r'management discussion',
            r'risk factors'
        ]
        
        structured_count = sum(1 for pattern in structured_patterns if re.search(pattern, text, re.IGNORECASE))
        narrative_count = sum(1 for pattern in narrative_patterns if re.search(pattern, text, re.IGNORECASE))
        
        if structured_count > 2:
            return "structured_financial"
        elif narrative_count > 1:
            return "narrative_financial"
        else:
            return "general"

def chunk_financial_documents(documents: List[Document], method: str = "adaptive") -> List[Document]:
    """Chunk financial documents using specified method"""
    if method == "adaptive":
        chunker = AdaptiveFinancialChunker()
        return chunker.chunk_documents(documents)
    elif method == "semantic":
        chunker = SemanticFinancialChunker()
        return chunker.semantic_chunking(documents)
    else:
        splitter = FinancialTextSplitter()
        return splitter.split_documents(documents)

def optimize_chunks_for_retrieval(chunks: List[Document], min_length: int = 50, max_length: int = 1500) -> List[Document]:
    """Optimize chunks for better retrieval performance"""
    optimized_chunks = []
    
    for chunk in chunks:
        content = chunk.page_content.strip()
        
        # Filter by length
        if len(content) < min_length or len(content) > max_length:
            continue
        
        # Clean up content
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n+', ' ', content)
        
        if content:
            optimized_chunk = Document(
                page_content=content,
                metadata=chunk.metadata.copy()
            )
            optimized_chunks.append(optimized_chunk)
    
    return optimized_chunks
