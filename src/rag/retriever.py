"""
Document Retriever using FAISS vector search
"""

import numpy as np
from typing import List, Dict, Optional
import logging

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    logging.error("Required packages not installed. Run: pip install sentence-transformers faiss-cpu")
    raise

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    Retrieve relevant documents using semantic search
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize retriever
        
        Args:
            embedding_model: SentenceTransformer model name
        """
        logger.info(f"Loading embedding model: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)
        
        self.chunks = []
        self.embeddings = None
        self.index = None
        
        logger.info("Retriever initialized")
    
    def index_documents(self, chunks: List[Dict]) -> None:
        """
        Create FAISS index from document chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
        """
        logger.info(f"Indexing {len(chunks)} documents...")
        
        self.chunks = chunks
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings (this may take a few minutes)...")
        self.embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"✓ Indexed {len(chunks)} documents (dimension={dimension})")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-k most relevant documents
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            results: List of relevant chunks with scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call index_documents() first.")
        
        # Encode query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            results.append({
                'rank': i + 1,
                'chunk': self.chunks[idx],
                'text': self.chunks[idx]['text'],
                'distance': float(dist),
                'similarity': 1 / (1 + dist)  # Convert distance to similarity score
            })
        
        logger.debug(f"Retrieved {len(results)} documents for query: '{query[:50]}...'")
        
        return results
    
    def retrieve_with_context(self, query: str, patient_state: Dict, 
                             top_k: int = 3) -> List[Dict]:
        """
        Retrieve documents with patient context
        
        Args:
            query: Base query
            patient_state: Current patient state
            top_k: Number of documents
            
        Returns:
            results: Retrieved documents
        """
        # Enhance query with patient context
        enhanced_query = self._enhance_query(query, patient_state)
        
        logger.info(f"Enhanced query: {enhanced_query}")
        
        return self.retrieve(enhanced_query, top_k)
    
    def _enhance_query(self, query: str, patient_state: Dict) -> str:
        """
        Enhance query with patient context
        
        Args:
            query: Original query
            patient_state: Patient state
            
        Returns:
            enhanced_query: Query with context
        """
        glucose = patient_state.get('current_glucose', 0)
        stress = patient_state.get('stress_level', 0)
        
        context_parts = []
        
        if glucose > 180:
            context_parts.append("hiperglikemia")
        elif glucose < 70:
            context_parts.append("hipoglikemia")
        
        if stress > 7:
            context_parts.append("stress tinggi")
        
        if context_parts:
            enhanced = f"{query}. Pasien mengalami {', '.join(context_parts)}."
        else:
            enhanced = query
        
        return enhanced
    
    def save_index(self, filepath: str) -> None:
        """
        Save FAISS index to disk
        
        Args:
            filepath: Path to save index
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        faiss.write_index(self.index, filepath)
        logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str, chunks: List[Dict]) -> None:
        """
        Load FAISS index from disk
        
        Args:
            filepath: Path to index file
            chunks: Associated chunks (must match index order)
        """
        self.index = faiss.read_index(filepath)
        self.chunks = chunks
        
        logger.info(f"Index loaded from {filepath}")