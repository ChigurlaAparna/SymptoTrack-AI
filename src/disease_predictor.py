"""
Disease Predictor Module for SymptoTrack AI
Predicts diseases based on symptoms using hybrid RAG
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Prediction:
    """Disease prediction result"""
    disease: str
    confidence: float
    matched_symptoms: List[str]
    severity: str
    category: str
    description: str
    treatment: str
    duration: str


class DiseasePredictor:
    """
    Predicts diseases based on symptom input
    Uses RAG engine for semantic search
    """
    
    def __init__(self, rag_engine, data_loader):
        self.rag_engine = rag_engine
        self.data_loader = data_loader
        self.symptom_variations = self._build_symptom_variations()
    
    def _build_symptom_variations(self) -> Dict[str, List[str]]:
        """Build symptom variations mapping"""
        return {
            "fever": ["fever", "high temperature", "febrile", "chills", "shivering"],
            "cough": ["cough", "coughing", "dry cough", "wet cough"],
            "headache": ["headache", "head pain", "migraine", "head ache"],
            "fatigue": ["fatigue", "tiredness", "tired", "exhaustion", "exhausted", "weakness", "lethargy"],
            "nausea": ["nausea", "nauseous", "feeling sick", "queasy"],
            "vomiting": ["vomiting", "throwing up", "being sick"],
            "diarrhea": ["diarrhea", "loose stools", "watery stools"],
            "sore throat": ["sore throat", "throat pain", "painful throat"],
            "runny nose": ["runny nose", "nasal discharge", "rhinorrhea"],
            "congestion": ["congestion", "stuffy nose", "blocked nose", "nasal congestion"],
            "body aches": ["body aches", "body pain", "muscle pain", "muscle ache", "aches"],
            "shortness of breath": ["shortness of breath", "difficulty breathing", "breathlessness", "dyspnea"],
            "chest pain": ["chest pain", "chest tightness"],
            "dizziness": ["dizziness", "dizzy", "lightheaded", "light-headedness"],
            "sweating": ["sweating", "sweaty", "excessive sweating", "night sweats"],
            "rash": ["rash", "skin rash", "redness"],
            "itching": ["itching", "itchy", "pruritus"],
            "loss of appetite": ["loss of appetite", "decreased appetite", "not hungry"],
            "weight loss": ["weight loss", "losing weight"],
            "insomnia": ["insomnia", "sleeplessness", "can't sleep", "trouble sleeping"],
            "anxiety": ["anxiety", "anxious", "worry", "worried", "nervous"],
            "depression": ["depression", "depressed", "sad", "feeling down"],
            "joint pain": ["joint pain", "arthralgia", "painful joints"],
            "stiffness": ["stiffness", "stiff", "morning stiffness"],
            "swelling": ["swelling", "swollen", "edema"],
            "numbness": ["numbness", "numb", "tingling", "pins and needles"],
            "blurred vision": ["blurred vision", "blurry vision", "vision problems"],
            "dry cough": ["dry cough"],
            "wet cough": ["wet cough", "productive cough"],
            "sneezing": ["sneezing", "sneeze"],
            "wheezing": ["wheezing", "wheeze"],
            "abdominal pain": ["abdominal pain", "stomach pain", "belly pain", "stomach ache"],
            "back pain": ["back pain", "lower back pain"],
            "muscle weakness": ["muscle weakness", "weak muscles"],
            "pale skin": ["pale skin", "pallor"],
            "frequent urination": ["frequent urination", "urinating often"],
            "burning urination": ["burning urination", "painful urination", "dysuria"],
            "blood in urine": ["blood in urine", "hematuria"],
            "cloudy urine": ["cloudy urine"],
            "high blood pressure": ["high blood pressure", "hypertension"],
            "palpitations": ["palpitations", "racing heart", "heart racing"],
            "loss of taste": ["loss of taste", "ageusia", "can't taste"],
            "loss of smell": ["loss of smell", "anosmia", "can't smell"],
        }
    
    def normalize_symptoms(self, symptoms: List[str]) -> List[str]:
        """Normalize and expand symptom list"""
        normalized = []
        seen = set()
        
        for symptom in symptoms:
            symptom_lower = symptom.lower().strip()
            
            # Check if this matches a known variation
            for canonical, variations in self.symptom_variations.items():
                if symptom_lower in variations or symptom_lower == canonical:
                    if canonical not in seen:
                        normalized.append(canonical)
                        seen.add(canonical)
                    break
            else:
                # Keep original if no match found
                if symptom_lower not in seen:
                    normalized.append(symptom_lower)
                    seen.add(symptom_lower)
        
        return normalized
    
    def predict(self, symptoms: List[str], top_k: int = 3) -> List[Prediction]:
        """
        Predict diseases based on symptoms
        
        Args:
            symptoms: List of symptoms
            top_k: Number of predictions to return
        
        Returns:
            List of Prediction objects
        """
        # Normalize symptoms
        normalized_symptoms = self.normalize_symptoms(symptoms)
        
        # Search using RAG
        query = " ".join(symptoms)
        results = self.rag_engine.search(query, top_k=top_k * 2)
        
        predictions = []
        seen_diseases = set()
        
        for result in results:
            if result["disease"] in seen_diseases:
                continue
            
            # Calculate matched symptoms
            result_symptoms = result["symptoms"].lower().split("; ")
            matched = [s for s in normalized_symptoms if any(s in rs or rs in s for rs in result_symptoms)]
            
            # Calculate confidence based on match
            if normalized_symptoms:
                confidence = len(matched) / len(normalized_symptoms)
            else:
                confidence = result.get("score", 0.5)
            
            # Adjust confidence based on semantic similarity
            confidence = min(confidence * 0.7 + result.get("score", 0.5) * 0.3, 1.0)
            
            prediction = Prediction(
                disease=result["disease"],
                confidence=confidence,
                matched_symptoms=matched,
                severity=result["severity"],
                category=result["category"],
                description=result["description"],
                treatment=result["treatment"],
                duration=result["duration"]
            )
            
            predictions.append(prediction)
            seen_diseases.add(result["disease"])
            
            if len(predictions) >= top_k:
                break
        
        # Sort by confidence
        predictions.sort(key=lambda x: -x.confidence)
        return predictions
    
    def get_alternative_diagnoses(self, symptoms: List[str], exclude: List[str], top_k: int = 2) -> List[Prediction]:
        """Get alternative diagnoses excluding certain diseases"""
        all_predictions = self.predict(symptoms, top_k=top_k * 3)
        return [p for p in all_predictions if p.disease not in exclude][:top_k]


class SymptomExtractor:
    """Extract symptoms from natural language"""
    
    # Words to exclude from being considered symptoms
    EXCLUDE_WORDS = {
        'have', 'has', 'had', 'having', 'i', 'my', 'me', 'a', 'an', 'the', 'and', 'or',
        'with', 'for', 'this', 'that', 'these', 'those', 'since', 'back', 'severe',
        'last', 'days', 'day', 'week', 'weeks', 'month', 'months', 'now', 'also',
        'some', 'lot', 'lots', 'much', 'more', 'most', 'very', 'really', 'quite',
        'feeling', 'feel', 'felt', 'getting', 'got', 'get', 'been', 'being',
        'high', 'low', 'off', 'on', 'out', 'up', 'down', 'around', 'like',
        'am', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'done', 'doing',
        'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom',
        'no', 'not', 'any', 'all', 'both', 'each', 'every', 'either', 'neither',
        'about', 'after', 'before', 'between', 'during', 'under', 'over', 'through',
        'because', 'although', 'though', 'if', 'unless', 'until', 'while', 'once',
    }
    
    def __init__(self):
        self.symptom_patterns = {
            "fever": r"(?:have\s+)?(?:a\s+)?fever|high?\s+temperature|febrile|chills?(?!\s+with)|high\s+fever|low\s+grade\s+fever",
            "cough": r"cough(?:ing)?(?!\s+up)|dry\s+cough|wet\s+cough|productive\s+cough",
            "headache": r"headache(?:s)?|head\s+pain|head\s+ache|migraine(?:s)?|head\s+hurt",
            "fatigue": r"fatigue[sd]?|tired(?:ness)?|exhaust(?:ed|ion)(?! breath)|weak(?:ness)?|letharg|y(?:all)?|no\s+energy",
            "nausea": r"nausea(?!ted)|nauseous|feel(?:s|ing)?\s+sick(?!\s+of\s+it)|queasy",
            "vomiting": r"vomit(?:ing)?|throw(?:ing)?\s+up|threw\s+up|being\s+sick",
            "diarrhea": r"diarrhea|loose\s+stool|watery\s+stool|runny\s+stool",
            "sore_throat": r"sore\s+throat|throat\s+pain|throat\s+irritation|scratchy\s+throat",
            "runny_nose": r"runny\s+nose|nasal\s+discharge|rhinorrhea|nose\s+drip",
            "congestion": r"congestion|stuffy\s+nose|blocked\s+nose|nasal\s+congestion",
            "body_aches": r"body\s+ache[sd]?|body\s+pain|muscle\s+(?:pain|ache)|body\s+hurts?|all\s+over\s+ache",
            "breathing": r"(?:shortness|difficulty)\s+of\s+breath|breathless|dyspnea|trouble\s+breathing|cannot\s+breathe",
            "chest_pain": r"chest\s+(?:pain|tightness|discomfort|pressure)",
            "dizziness": r"dizz(?:y|iness)|lightheaded|light\s+headed|vertigo|room\s+spinning",
            "rash": r"rash(?:es)?|skin\s+rash|redness|skin\s+irritation",
            "itching": r"itch(?:ing|y)(?!\s+eye)|pruritus|skin\s+itch",
            "sneezing": r"sneezing|sneeze[sd]?",
            "wheezing": r"wheezing|wheeze[sd]?",
            "abdominal_pain": r"(?:abdominal|stomach|belly)\s+pain|belly\s+ache|stomach\s+ache|stomach\s+hurts?",
            "back_pain": r"back\s+pain(?!\s+of)|lower\s+back\s+pain|backache",
            "joint_pain": r"joint\s+pain|arthralgia|joints\s+hurt",
            "swelling": r"swelling|swollen(?!\s+ankles?\s+or)|edema|puffy",
            "numbness": r"numb(?:ness)?(?!\s+in)|tingling|pins\s+and\s+needles|numb\s+(?:fingers?|toes?|hands?|feet?)",
            "blurred_vision": r"blurred?\s+vision|vision\s+problem|vision\s+blurry|can't\s+see\s+clearly",
            "loss_of_taste": r"loss\s+of\s+taste|ageusia|cannot\s+taste",
            "loss_of_smell": r"loss\s+of\s+smell|anosmia|cannot\s+smell",
            "frequent_urination": r"frequent\s+urination|urinating\s+often|peeing\s+a\s+lot",
            "burning_urination": r"burning\s+(?:when\s+)?urinating|painful\s+urination|dysuria",
            "anxiety": r"anxi(?:ety|ous)(?!ous)|worried?|nervous|panic",
            "depression": r"depress(?:ed|ion)|sad(?:ness)?|feeling\s+down|hopeless",
            "insomnia": r"insomnia|can't\s+sleep|trouble\s+sleeping|sleepless|sleep\s+problems?",
            "weight_loss": r"weight\s+loss|lost\s+weight|unintentional\s+weight\s+loss",
            "night_sweats": r"night\s+sweats?s?|sweating\s+at\s+night",
            "chills": r"chills?(?:\s+and\s+fever)?|shivering",
            "sore muscles": r"sore\s+(?:muscles?|body)|muscle\s+soreness",
            "loss of appetite": r"loss\s+of\s+appetite|no\s+appetite|not\s+hungry|appetite\s+loss",
        }
    
    def extract(self, text: str) -> List[str]:
        """Extract symptoms from text"""
        text_lower = text.lower()
        found_symptoms = []
        
        for symptom, pattern in self.symptom_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                # Convert snake_case to display format
                display_name = symptom.replace("_", " ")
                found_symptoms.append(display_name)
        
        return found_symptoms


if __name__ == "__main__":
    from data_loader import DataLoader
    from rag_engine import RAGEngine
    
    loader = DataLoader()
    df = loader.load_data()
    
    rag = RAGEngine()
    rag.create_documents(df)
    rag.build_index()
    
    predictor = DiseasePredictor(rag, loader)
    
    # Test
    symptoms = ["fever", "cough", "headache"]
    predictions = predictor.predict(symptoms)
    
    print(f"\nSymptoms: {symptoms}")
    print(f"\nPredictions:")
    for p in predictions:
        print(f"  {p.disease}: {p.confidence:.2%} confidence")
        print(f"    Matched: {p.matched_symptoms}")
        print(f"    Severity: {p.severity}")
