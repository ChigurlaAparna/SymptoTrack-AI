"""
RAG Engine Module for SymptoTrack AI
Hybrid RAG implementation using FAISS and Sentence Transformers
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


class RAGEngine:
    """
    Hybrid RAG Engine combining vector search (FAISS) with keyword matching
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = None):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent.parent / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.index = None
        self.documents = []
        self.metadata = []
        self.embeddings = None
        
    def _load_model(self):
        """Load the sentence transformer model"""
        if not SBERT_AVAILABLE:
            raise ImportError("sentence-transformers is not installed. Run: pip install sentence-transformers")
        
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            
    def _load_index(self):
        """Load FAISS index if available"""
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is not installed. Run: pip install faiss-cpu")
    
    def create_documents(self, df: pd.DataFrame) -> List[Dict]:
        """Create documents from dataframe"""
        self.documents = []
        self.metadata = []
        
        for _, row in df.iterrows():
            # Create comprehensive document text
            symptoms = str(row.get("symptoms", "")).replace(",", "; ")
            document = f"""
            Disease: {row['disease']}
            Symptoms: {symptoms}
            Severity: {row['severity']}
            Category: {row['category']}
            Description: {row['description']}
            Treatment: {row['treatment']}
            Duration: {row['duration']}
            """.strip()
            
            self.documents.append(document)
            self.metadata.append({
                "disease": row["disease"],
                "symptoms": symptoms,
                "severity": row["severity"],
                "category": row["category"],
                "description": row["description"],
                "treatment": row["treatment"],
                "duration": row["duration"]
            })
        
        return self.documents
    
    def build_index(self, force_rebuild: bool = False) -> bool:
        """Build FAISS index from documents"""
        self._load_model()
        self._load_index()
        
        index_path = self.cache_dir / "faiss_index.bin"
        docs_path = self.cache_dir / "documents.pkl"
        meta_path = self.cache_dir / "metadata.pkl"
        
        # Load from cache if available
        if not force_rebuild and index_path.exists() and docs_path.exists() and meta_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                print(f"Loaded index from cache with {len(self.documents)} documents")
                return True
            except Exception as e:
                print(f"Failed to load cache: {e}")
        
        if not self.documents:
            raise ValueError("No documents to index. Call create_documents() first.")
        
        # Generate embeddings
        print(f"Generating embeddings for {len(self.documents)} documents...")
        self.embeddings = self.model.encode(self.documents, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors = cosine similarity
        self.index.add(self.embeddings.astype(np.float32))
        
        # Save to cache
        faiss.write_index(self.index, str(index_path))
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        
        print(f"Built index with {self.index.ntotal} documents")
        return True
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents using hybrid approach"""
        self._load_model()
        
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.model.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Combine results with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(scores[0][i])
                result["document"] = self.documents[idx]
                results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: -x["score"])
        return results[:top_k]
    
    def search_by_symptoms(self, symptoms: List[str], top_k: int = 5) -> List[Dict]:
        """Search for diseases matching symptoms"""
        query = " ".join(symptoms)
        return self.search(query, top_k)
    
    def get_disease_details(self, disease_name: str) -> Optional[Dict]:
        """Get details for a specific disease"""
        for meta in self.metadata:
            if meta["disease"].lower() == disease_name.lower():
                return meta
        return None
    
    def suggest_related_questions(self, query: str, n: int = 3) -> List[str]:
        """Suggest related questions based on query"""
        suggestions = [
            "What are the common symptoms?",
            "What treatments are recommended?",
            "How long does it last?",
            "What precautions should I take?",
            "When should I see a doctor?",
            "Is this contagious?"
        ]
        
        # Simple keyword-based filtering
        query_lower = query.lower()
        filtered = []
        
        for s in suggestions:
            s_lower = s.lower()
            if any(word in query_lower for word in ["symptom", "sign", "feel"]):
                if "symptom" in s_lower:
                    filtered.append(s)
            elif any(word in query_lower for word in ["treatment", "cure", "medicine", "medication"]):
                if "treatment" in s_lower or "precaution" in s_lower:
                    filtered.append(s)
            elif any(word in query_lower for word in ["duration", "last", "time", "recover"]):
                if "last" in s_lower:
                    filtered.append(s)
            elif any(word in query_lower for word in ["precaution", "prevent", "avoid"]):
                if "precaution" in s_lower or "doctor" in s_lower:
                    filtered.append(s)
        
        # If no specific matches, return generic suggestions
        if not filtered:
            filtered = suggestions[:n]
        
        return filtered[:n]


class SymptomMatcher:
    """Keyword-based symptom matching for fallback"""
    
    def __init__(self):
        self.symptom_keywords = {
            "fever": ["fever", "high temperature", "febrile"],
            "cough": ["cough", "coughing"],
            "headache": ["headache", "head pain"],
            "fatigue": ["fatigue", "tired", "exhaustion", "weakness"],
            "nausea": ["nausea", "nauseous", "sick to stomach"],
            "pain": ["pain", "ache", "soreness"],
            "breathing": ["breathing", "shortness of breath", "dyspnea"],
            "skin": ["rash", "skin", "redness", "itching"],
        }
    
    def extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text"""
        text_lower = text.lower()
        found_symptoms = []
        
        for symptom, keywords in self.symptom_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_symptoms.append(symptom)
                    break
        
        return found_symptoms
    
    def match_score(self, text: str, symptoms: List[str]) -> float:
        """Calculate match score between text and symptoms"""
        extracted = self.extract_symptoms(text)
        if not extracted:
            return 0.0
        
        matches = sum(1 for s in symptoms if s.lower() in extracted)
        return matches / len(symptoms) if symptoms else 0.0


if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Test RAG engine
    loader = DataLoader()
    df = loader.load_data()
    
    rag = RAGEngine()
    rag.create_documents(df)
    rag.build_index()
    
    # Test search
    results = rag.search("fever headache cough")
    for r in results:
        print(f"Disease: {r['disease']}, Score: {r['score']:.3f}")
