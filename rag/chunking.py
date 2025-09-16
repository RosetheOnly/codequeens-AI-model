"""
Document chunking module for RAG system in CreditAI application.
Implements various chunking strategies for credit-related documents.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loaders import Document

@dataclass
class Chunk:
    """Chunk class to store chunked content and metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_index: int = 0
    end_index: int = 0

class BaseChunker:
    """Base class for document chunkers."""
    
    def chunk(self, document: Document) -> List[Chunk]:
        raise NotImplementedError

class FixedSizeChunker(BaseChunker):
    """Chunks documents into fixed-size pieces."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, document: Document) -> List[Chunk]:
        content = document.content
        chunks = []
        
        start = 0
        chunk_num = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            
            # Try to end at a sentence boundary
            if end < len(content):
                # Look for sentence endings within the last 100 characters
                last_period = content.rfind('.', start, end)
                last_newline = content.rfind('\n', start, end)
                boundary = max(last_period, last_newline)
                
                if boundary > start + self.chunk_size * 0.7:  # Don't make chunks too small
                    end = boundary + 1
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    "chunk_number": chunk_num,
                    "start_index": start,
                    "end_index": end,
                    "chunk_size": len(chunk_content)
                })
                
                chunks.append(Chunk(
                    content=chunk_content,
                    metadata=chunk_metadata,
                    chunk_id=f"{document.doc_id}_chunk_{chunk_num}",
                    start_index=start,
                    end_index=end
                ))
                
                chunk_num += 1
            
            start = end - self.overlap
            if start >= end:  # Prevent infinite loop
                break
        
        return chunks

class SemanticChunker(BaseChunker):
    """Chunks documents based on semantic boundaries."""
    
    def __init__(self, max_chunk_size: int = 1500, min_chunk_size: int = 300):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, document: Document) -> List[Chunk]:
        content = document.content
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = ""
        chunk_num = 0
        start_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed max size, create a chunk
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                if len(current_chunk) >= self.min_chunk_size:
                    chunk_metadata = document.metadata.copy()
                    chunk_metadata.update({
                        "chunk_number": chunk_num,
                        "start_index": start_index,
                        "end_index": start_index + len(current_chunk),
                        "chunk_size": len(current_chunk)
                    })
                    
                    chunks.append(Chunk(
                        content=current_chunk.strip(),
                        metadata=chunk_metadata,
                        chunk_id=f"{document.doc_id}_semantic_{chunk_num}",
                        start_index=start_index,
                        end_index=start_index + len(current_chunk)
                    ))
                    
                    chunk_num += 1
                    start_index += len(current_chunk)
                
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "chunk_number": chunk_num,
                "start_index": start_index,
                "end_index": start_index + len(current_chunk),
                "chunk_size": len(current_chunk)
            })
            
            chunks.append(Chunk(
                content=current_chunk.strip(),
                metadata=chunk_metadata,
                chunk_id=f"{document.doc_id}_semantic_{chunk_num}",
                start_index=start_index,
                end_index=start_index + len(current_chunk)
            ))
        
        return chunks

class CreditRecordChunker(BaseChunker):
    """Specialized chunker for credit records."""
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk credit records by meaningful sections."""
        content = document.content
        chunks = []
        
        # For credit records, we might want to chunk by different aspects
        sections = self._identify_credit_sections(content)
        
        for i, (section_name, section_content) in enumerate(sections.items()):
            if section_content.strip():
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    "section": section_name,
                    "chunk_number": i,
                    "chunk_type": "credit_section"
                })
                
                chunks.append(Chunk(
                    content=section_content.strip(),
                    metadata=chunk_metadata,
                    chunk_id=f"{document.doc_id}_{section_name}",
                    start_index=0,
                    end_index=len(section_content)
                ))
        
        # If no sections found, return the whole document as one chunk
        if not chunks:
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "section": "full_record",
                "chunk_number": 0,
                "chunk_type": "complete_record"
            })
            
            chunks.append(Chunk(
                content=content,
                metadata=chunk_metadata,
                chunk_id=f"{document.doc_id}_full",
                start_index=0,
                end_index=len(content)
            ))
        
        return chunks
    
    def _identify_credit_sections(self, content: str) -> Dict[str, str]:
        """Identify different sections in credit record content."""
        sections = {}
        
        # Split by common delimiters in credit records
        if " | " in content:
            parts = content.split(" | ")
            for part in parts:
                if ":" in part:
                    key, value = part.split(":", 1)
                    sections[key.strip().lower().replace(" ", "_")] = value.strip()
                else:
                    sections[f"section_{len(sections)}"] = part.strip()
        else:
            # If no clear structure, treat as one section
            sections["full_content"] = content
        
        return sections

class RecursiveChunker(BaseChunker):
    """Recursively chunks documents using multiple strategies."""
    
    def __init__(self, separators: List[str] = None, chunk_size: int = 1000, overlap: int = 200):
        self.separators = separators or ["\n\n", "\n", ". ", " "]
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, document: Document) -> List[Chunk]:
        return self._split_text_recursive(document.content, document)
    
    def _split_text_recursive(self, text: str, document: Document, depth: int = 0) -> List[Chunk]:
        chunks = []
        
        if len(text) <= self.chunk_size:
            # Text is small enough, create a chunk
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "chunk_number": depth,
                "chunk_size": len(text),
                "split_depth": depth
            })
            
            chunks.append(Chunk(
                content=text.strip(),
                metadata=chunk_metadata,
                chunk_id=f"{document.doc_id}_recursive_{depth}",
                start_index=0,
                end_index=len(text)
            ))
            return chunks
        
        # Try to split with current separator
        if depth < len(self.separators):
            separator = self.separators[depth]
            splits = text.split(separator)
            
            if len(splits) > 1:
                current_chunk = ""
                chunk_num = 0
                
                for split in splits:
                    if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                        if current_chunk:
                            current_chunk += separator + split
                        else:
                            current_chunk = split
                    else:
                        if current_chunk:
                            # Process current chunk
                            sub_chunks = self._split_text_recursive(current_chunk, document, depth + 1)
                            for sub_chunk in sub_chunks:
                                sub_chunk.chunk_id = f"{document.doc_id}_recursive_{depth}_{chunk_num}"
                                chunks.extend([sub_chunk])
                            chunk_num += 1
                        current_chunk = split
                
                # Process remaining chunk
                if current_chunk:
                    sub_chunks = self._split_text_recursive(current_chunk, document, depth + 1)
                    for sub_chunk in sub_chunks:
                        sub_chunk.chunk_id = f"{document.doc_id}_recursive_{depth}_{chunk_num}"
                        chunks.extend([sub_chunk])
                
                return chunks
        
        # If we can't split further, use fixed-size chunking
        fixed_chunker = FixedSizeChunker(self.chunk_size, self.overlap)
        return fixed_chunker.chunk(document)

class DocumentChunker:
    """Main chunker that selects appropriate strategy based on document type."""
    
    def __init__(self):
        self.chunkers = {
            "credit_record": CreditRecordChunker(),
            "policy": SemanticChunker(max_chunk_size=2000, min_chunk_size=500),
            "text": RecursiveChunker(),
            "default": FixedSizeChunker(chunk_size=1000, overlap=200)
        }
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk a list of documents using appropriate strategies."""
        all_chunks = []
        
        for document in documents:
            # Determine chunking strategy based on document metadata
            doc_type = self._determine_document_type(document)
            chunker = self.chunkers.get(doc_type, self.chunkers["default"])
            
            try:
                chunks = chunker.chunk(document)
                all_chunks.extend(chunks)
                print(f"Chunked document {document.doc_id} into {len(chunks)} chunks using {doc_type} strategy")
            except Exception as e:
                print(f"Error chunking document {document.doc_id}: {str(e)}")
                # Fallback to default chunker
                try:
                    chunks = self.chunkers["default"].chunk(document)
                    all_chunks.extend(chunks)
                except Exception as fallback_error:
                    print(f"Fallback chunking also failed for {document.doc_id}: {str(fallback_error)}")
        
        return all_chunks
    
    def _determine_document_type(self, document: Document) -> str:
        """Determine the appropriate chunking strategy for a document."""
        metadata = document.metadata
        
        # Check metadata for document type hints
        if metadata.get("data_type") == "credit_record":
            return "credit_record"
        elif "policy" in metadata.get("source", "").lower():
            return "policy"
        elif metadata.get("type") == "text":
            return "text"
        
        # Check content for patterns
        content_lower = document.content.lower()
        if any(term in content_lower for term in ["risk assessment", "credit score", "balance", "transaction"]):
            return "credit_record"
        elif any(term in content_lower for term in ["policy", "guideline", "procedure", "regulation"]):
            return "policy"
        
        return "default"
