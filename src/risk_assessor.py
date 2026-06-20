"""
Risk Assessor Module for SymptoTrack AI
Assesses health risk levels based on symptoms and conditions
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAssessment:
    """Risk assessment result"""
    risk_level: RiskLevel
    risk_score: float
    risk_factors: List[str]
    recommendations: List[str]
    urgency: str
    follow_up: str


class RiskAssessor:
    """
    Assesses health risk based on symptoms and patient factors
    """
    
    def __init__(self):
        # High-risk symptom combinations
        self.critical_combinations = [
            {"symptoms": ["chest pain", "shortness of breath", "dizziness"], "condition": "possible heart attack"},
            {"symptoms": ["high fever", "stiff neck", "sensitivity to light"], "condition": "possible meningitis"},
            {"symptoms": ["difficulty breathing", "swelling", "rash"], "condition": "possible anaphylaxis"},
            {"symptoms": ["severe headache", "vomiting", "blurred vision"], "condition": "possible stroke"},
            {"symptoms": ["high fever", "confusion", "seizures"], "condition": "possible encephalitis"},
        ]
        
        # High-risk individual symptoms
        self.high_risk_symptoms = {
            "high fever": 2.0,
            "chest pain": 2.5,
            "difficulty breathing": 2.5,
            "severe headache": 2.0,
            "confusion": 2.5,
            "seizures": 3.0,
            "unconsciousness": 3.0,
            "blood in urine": 2.0,
            "vomiting blood": 3.0,
            "black stools": 2.5,
            "sudden vision loss": 3.0,
            "numbness": 2.0,
            "tingling": 1.5,
        }
        
        # Moderate risk symptoms
        self.moderate_risk_symptoms = {
            "fever": 1.0,
            "persistent cough": 1.0,
            "abdominal pain": 1.0,
            "fatigue": 0.5,
            "weight loss": 1.0,
            "night sweats": 1.0,
            "swelling": 1.0,
            "rash": 0.5,
        }
        
        # Severity multipliers
        self.severity_multipliers = {
            "mild": 1.0,
            "moderate": 1.5,
            "high": 2.0,
        }
        
        # Duration risk factors
        self.duration_risk = {
            "acute (< 1 week)": 1.0,
            "subacute (1-4 weeks)": 1.5,
            "chronic (> 4 weeks)": 2.0,
        }
    
    def assess(
        self,
        symptoms: List[str],
        severity: str = "moderate",
        duration_days: Optional[int] = None,
        age_group: str = "adult",
        comorbidities: Optional[List[str]] = None
    ) -> RiskAssessment:
        """
        Assess health risk based on symptoms and factors
        
        Args:
            symptoms: List of symptoms
            severity: Overall severity level
            duration_days: Duration of symptoms in days
            age_group: Age group (child, adult, elderly)
            comorbidities: List of pre-existing conditions
        
        Returns:
            RiskAssessment object
        """
        symptoms_lower = [s.lower() for s in symptoms]
        risk_factors = []
        risk_score = 0.0
        recommendations = []
        
        # Check for critical symptom combinations
        for combo in self.critical_combinations:
            combo_symptoms = [s.lower() for s in combo["symptoms"]]
            match_count = sum(1 for s in combo_symptoms if any(cs in s or s in cs for cs in symptoms_lower))
            
            if match_count >= 2:
                risk_score += 3.0
                risk_factors.append(f"Critical: {combo['condition']}")
        
        # Check high-risk symptoms
        for symptom, score in self.high_risk_symptoms.items():
            for s in symptoms_lower:
                if symptom in s or s in symptom:
                    risk_score += score
                    risk_factors.append(f"High-risk symptom: {symptom}")
                    break
        
        # Check moderate symptoms
        for symptom, score in self.moderate_risk_symptoms.items():
            for s in symptoms_lower:
                if symptom in s or s in symptom:
                    risk_score += score
                    break
        
        # Apply severity multiplier
        severity_mult = self.severity_multipliers.get(severity.lower(), 1.5)
        risk_score *= severity_mult
        
        # Age group risk
        if age_group == "elderly":
            risk_score *= 1.3
            risk_factors.append("Age group: Elderly (increased risk)")
        elif age_group == "child":
            risk_score *= 1.2
            risk_factors.append("Age group: Child (special monitoring needed)")
        
        # Comorbidity risk
        if comorbidities:
            high_risk_comorbidities = [
                "diabetes", "heart disease", "cancer", "hiv",
                "immunocompromised", "liver disease", "kidney disease",
                "lung disease", "stroke history"
            ]
            for comorbidity in comorbidities:
                comorbidity_lower = comorbidity.lower()
                if any(hrc in comorbidity_lower for hrc in high_risk_comorbidities):
                    risk_score *= 1.5
                    risk_factors.append(f"Comorbidity: {comorbidity}")
        
        # Duration risk
        if duration_days is not None:
            if duration_days > 14:
                risk_score *= 1.5
                risk_factors.append(f"Prolonged symptoms: {duration_days} days")
            elif duration_days > 7:
                risk_score *= 1.2
                risk_factors.append(f"Extended symptoms: {duration_days} days")
        
        # Normalize risk score
        risk_score = min(risk_score / 10, 1.0)
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW
        
        # Generate recommendations based on risk level
        recommendations = self._generate_recommendations(risk_level, symptoms_lower, risk_factors)
        
        # Determine urgency
        urgency = self._get_urgency(risk_level)
        
        # Follow-up timing
        follow_up = self._get_follow_up_timing(risk_level, duration_days)
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            risk_factors=risk_factors,
            recommendations=recommendations,
            urgency=urgency,
            follow_up=follow_up
        )
    
    def _generate_recommendations(self, risk_level: RiskLevel, symptoms: List[str], risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on risk level"""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "🚨 SEEK IMMEDIATE MEDICAL ATTENTION",
                "Call emergency services or go to the nearest emergency room immediately",
                "Do not delay seeking care - this requires urgent evaluation",
                "If available, contact your healthcare provider while en route"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "⚠️ Consult a healthcare provider within 24 hours",
                "Consider visiting an urgent care center if symptoms worsen",
                "Monitor symptoms closely and prepare for potential emergency",
                "Keep a list of all medications and medical history ready"
            ])
        elif risk_level == RiskLevel.MODERATE:
            recommendations.extend([
                "📋 Schedule an appointment with your healthcare provider",
                "Rest and stay hydrated",
                "Monitor symptoms for changes or worsening",
                "Consider over-the-counter remedies for symptom relief"
            ])
        else:  # LOW
            recommendations.extend([
                "🏠 Rest and monitor symptoms",
                "Stay well hydrated",
                "Use home remedies as appropriate",
                "Consult a doctor if symptoms persist beyond 1 week"
            ])
        
        return recommendations
    
    def _get_urgency(self, risk_level: RiskLevel) -> str:
        """Get urgency text based on risk level"""
        urgency_map = {
            RiskLevel.CRITICAL: "EMERGENCY - Seek immediate medical care",
            RiskLevel.HIGH: "URGENT - See a healthcare provider within 24 hours",
            RiskLevel.MODERATE: "SEMI-URGENT - Schedule an appointment within 1-2 days",
            RiskLevel.LOW: "NON-URGENT - Self-care with optional medical consultation"
        }
        return urgency_map.get(risk_level, "Unknown urgency")
    
    def _get_follow_up_timing(self, risk_level: RiskLevel, duration_days: Optional[int]) -> str:
        """Get recommended follow-up timing"""
        if risk_level == RiskLevel.CRITICAL:
            return "Immediate emergency care required"
        elif risk_level == RiskLevel.HIGH:
            return "Follow up within 24-48 hours"
        elif duration_days and duration_days > 7:
            return "Follow up if no improvement within 1 week"
        else:
            return "Follow up as needed or if symptoms change"


class SymptomTracker:
    """Track symptom progression over time"""
    
    def __init__(self):
        self.records = []
    
    def add_record(self, symptoms: List[str], severity: str, notes: str = ""):
        """Add a symptom record"""
        from datetime import datetime
        self.records.append({
            "timestamp": datetime.now(),
            "symptoms": symptoms,
            "severity": severity,
            "notes": notes
        })
    
    def get_history(self) -> List[Dict]:
        """Get symptom history"""
        return self.records
    
    def analyze_trend(self) -> str:
        """Analyze symptom trend"""
        if len(self.records) < 2:
            return "Insufficient data for trend analysis"
        
        severity_map = {"mild": 1, "moderate": 2, "high": 3, "critical": 4}
        scores = [severity_map.get(r["severity"].lower(), 2) for r in self.records]
        
        if scores[-1] > scores[0]:
            return "Symptoms appear to be worsening"
        elif scores[-1] < scores[0]:
            return "Symptoms appear to be improving"
        else:
            return "Symptoms remain stable"


if __name__ == "__main__":
    assessor = RiskAssessor()
    
    # Test case
    assessment = assessor.assess(
        symptoms=["fever", "cough", "chest pain", "shortness of breath"],
        severity="moderate",
        duration_days=3,
        age_group="adult"
    )
    
    print(f"Risk Level: {assessment.risk_level.value}")
    print(f"Risk Score: {assessment.risk_score:.2%}")
    print(f"\nRisk Factors:")
    for factor in assessment.risk_factors:
        print(f"  - {factor}")
    print(f"\nRecommendations:")
    for rec in assessment.recommendations:
        print(f"  {rec}")
    print(f"\nUrgency: {assessment.urgency}")
