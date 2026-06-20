"""
Data Loader Module for SymptoTrack AI
Handles loading and preprocessing healthcare data
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """Loads and preprocesses healthcare data"""
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            self.data_path = Path(__file__).parent.parent / "data" / "healthcare_data.csv"
        else:
            self.data_path = Path(data_path)
        
        self.df = None
        self.disease_info = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load healthcare data from CSV"""
        self.df = pd.read_csv(self.data_path)
        self._preprocess_data()
        self._build_disease_info()
        return self.df
    
    def _preprocess_data(self):
        """Preprocess the dataframe"""
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        # Fill NaN values
        self.df = self.df.fillna("")
        
        # Create severity scores
        severity_map = {"mild": 1, "moderate": 2, "high": 3}
        self.df["severity_score"] = self.df["severity"].str.lower().map(severity_map).fillna(1)
        
    def _build_disease_info(self):
        """Build disease information dictionary"""
        for _, row in self.df.iterrows():
            disease = row["disease"].strip()
            if disease not in self.disease_info:
                self.disease_info[disease] = {
                    "symptoms": [],
                    "severity": row["severity"],
                    "severity_score": row["severity_score"],
                    "category": row["category"],
                    "description": row["description"],
                    "treatment": row["treatment"],
                    "duration": row["duration"]
                }
            
            # Collect all symptoms
            symptoms = [s.strip().lower() for s in str(row["symptoms"]).split(",")]
            self.disease_info[disease]["symptoms"].extend(symptoms)
            self.disease_info[disease]["symptoms"] = list(set(self.disease_info[disease]["symptoms"]))
    
    def get_all_symptoms(self) -> list:
        """Get all unique symptoms from the dataset"""
        all_symptoms = set()
        for info in self.disease_info.values():
            all_symptoms.update(info["symptoms"])
        return sorted(list(all_symptoms))
    
    def get_disease_info(self, disease: str) -> dict:
        """Get information about a specific disease"""
        return self.disease_info.get(disease.strip(), None)
    
    def get_diseases_by_category(self, category: str) -> list:
        """Get all diseases in a specific category"""
        return [
            disease for disease, info in self.disease_info.items()
            if info["category"].lower() == category.lower()
        ]
    
    def get_categories(self) -> list:
        """Get all disease categories"""
        categories = set()
        for info in self.disease_info.values():
            categories.add(info["category"])
        return sorted(list(categories))
    
    def get_all_diseases(self) -> list:
        """Get all diseases"""
        return list(self.disease_info.keys())
    
    def search_by_symptoms(self, symptoms: list) -> list:
        """Find diseases matching given symptoms"""
        symptoms_lower = [s.lower().strip() for s in symptoms]
        matches = []
        
        for disease, info in self.disease_info.items():
            disease_symptoms = [s.lower() for s in info["symptoms"]]
            
            # Count matching symptoms
            matching = sum(1 for s in symptoms_lower if s in disease_symptoms)
            total_symptoms = len(disease_symptoms)
            
            if matching > 0:
                match_percentage = matching / total_symptoms * 100
                matches.append({
                    "disease": disease,
                    "match_count": matching,
                    "total_symptoms": total_symptoms,
                    "match_percentage": match_percentage,
                    "severity": info["severity"],
                    "severity_score": info["severity_score"],
                    "category": info["category"]
                })
        
        # Sort by match percentage and severity
        matches.sort(key=lambda x: (-x["match_percentage"], -x["severity_score"]))
        return matches
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the preprocessed dataframe"""
        return self.df


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_data()
    print(f"Loaded {len(df)} records")
    print(f"Unique diseases: {len(loader.get_all_diseases())}")
    print(f"Unique symptoms: {len(loader.get_all_symptoms())}")
    print(f"Categories: {loader.get_categories()}")
