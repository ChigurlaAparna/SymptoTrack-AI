"""
Precautions Database Module for SymptoTrack AI
Contains general and disease-specific precautions
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Precaution:
    """Precaution information"""
    category: str
    general: List[str]
    disease_specific: Dict[str, List[str]]
    lifestyle: List[str]
    emergency_signs: List[str]


class PrecautionsDatabase:
    """
    Database of health precautions and recommendations
    """
    
    def __init__(self):
        self.general_precautions = self._get_general_precautions()
        self.disease_precautions = self._get_disease_precautions()
        self.emergency_signs = self._get_emergency_signs()
        self.nutrition_advice = self._get_nutrition_advice()
        self.lifestyle_advice = self._get_lifestyle_advice()
    
    def _get_general_precautions(self) -> List[str]:
        """Get general health precautions"""
        return [
            "Wash hands frequently with soap for at least 20 seconds",
            "Avoid touching your face, especially eyes, nose, and mouth",
            "Maintain social distancing in crowded places",
            "Cover your mouth and nose when coughing or sneezing",
            "Get adequate sleep (7-9 hours for adults)",
            "Exercise regularly for at least 30 minutes daily",
            "Stay hydrated by drinking 8-10 glasses of water daily",
            "Avoid smoking and limit alcohol consumption",
            "Manage stress through meditation or relaxation techniques",
            "Keep your living spaces clean and well-ventilated",
            "Get regular health check-ups and screenings",
            "Stay up to date with vaccinations",
        ]
    
    def _get_disease_precautions(self) -> Dict[str, List[str]]:
        """Get disease-specific precautions"""
        return {
            "Common Cold": [
                "Rest and avoid strenuous activities",
                "Stay home to prevent spreading",
                "Use separate towels and utensils",
                "Avoid close contact with others",
                "Keep the environment humid",
            ],
            "Flu": [
                "Get an annual flu vaccine",
                "Avoid crowds during flu season",
                "Disinfect commonly touched surfaces",
                "Avoid sharing personal items",
                "Rest and stay home for at least 24 hours after fever resolves",
            ],
            "COVID-19": [
                "Wear a mask in public places",
                "Maintain 6 feet distance from others",
                "Get vaccinated and boosted",
                "Test if exposed or symptomatic",
                "Quarantine if infected",
            ],
            "Migraine": [
                "Identify and avoid triggers (certain foods, lights, sounds)",
                "Maintain a regular sleep schedule",
                "Stay hydrated and don't skip meals",
                "Manage stress through relaxation techniques",
                "Keep a migraine diary to track triggers",
            ],
            "Gastroenteritis": [
                "Practice good hand hygiene",
                "Avoid preparing food for others while ill",
                "Disinfect contaminated surfaces",
                "Don't share utensils or towels",
                "Rehydrate with oral rehydration solutions",
            ],
            "Hypertension": [
                "Reduce sodium intake to less than 2,300mg daily",
                "Exercise regularly (150 minutes per week)",
                "Maintain a healthy weight",
                "Limit alcohol consumption",
                "Manage stress levels",
            ],
            "Diabetes": [
                "Monitor blood sugar levels regularly",
                "Follow a balanced diet with controlled carbohydrates",
                "Exercise regularly",
                "Take medications as prescribed",
                "Check feet daily for any wounds or changes",
            ],
            "Asthma": [
                "Avoid known triggers (dust, pollen, smoke)",
                "Keep rescue inhaler accessible at all times",
                "Follow an asthma action plan",
                "Get annual flu vaccine",
                "Avoid cold air if it triggers symptoms",
            ],
            "Pneumonia": [
                "Get pneumococcal vaccines if recommended",
                "Avoid smoking and second-hand smoke",
                "Practice good hand hygiene",
                "Stay away from sick people",
                "Get adequate rest during recovery",
            ],
            "Bronchitis": [
                "Avoid smoking and irritants",
                "Use a humidifier",
                "Stay hydrated to loosen mucus",
                "Get plenty of rest",
                "Avoid air pollutants",
            ],
            "Anemia": [
                "Eat iron-rich foods (red meat, spinach, beans)",
                "Take iron supplements as prescribed",
                "Consume vitamin C to enhance iron absorption",
                "Avoid tea/coffee with meals",
                "Regular blood tests to monitor levels",
            ],
            "Allergies": [
                "Identify and avoid allergens",
                "Keep windows closed during high pollen seasons",
                "Use air purifiers",
                "Wash bedding regularly in hot water",
                "Consider allergy medications or immunotherapy",
            ],
        }
    
    def _get_emergency_signs(self) -> Dict[str, List[str]]:
        """Get emergency warning signs"""
        return {
            "general": [
                "Difficulty breathing or shortness of breath",
                "Chest pain or pressure",
                "Sudden confusion or inability to arouse",
                "Severe or persistent vomiting",
                "Blood in vomit, stool, or urine",
                "High fever that doesn't respond to medication",
                "Seizures",
                "Loss of consciousness",
            ],
            "respiratory": [
                "Severe difficulty breathing",
                "Blue lips or fingertips",
                "Wheezing that doesn't improve with medication",
                "Coughing up blood",
            ],
            "cardiac": [
                "Chest pain radiating to arm, jaw, or back",
                "Rapid or irregular heartbeat",
                "Severe shortness of breath",
                "Fainting or near-fainting",
            ],
            "neurological": [
                "Sudden severe headache",
                "Difficulty speaking or understanding",
                "Numbness or weakness on one side",
                "Vision changes",
                "Loss of balance",
            ],
        }
    
    def _get_nutrition_advice(self) -> Dict[str, List[str]]:
        """Get nutrition advice by condition"""
        return {
            "general": [
                "Eat a variety of fruits and vegetables",
                "Choose whole grains over refined grains",
                "Include lean proteins in your diet",
                "Limit processed foods and added sugars",
                "Stay hydrated with water and natural beverages",
            ],
            "immune_boost": [
                "Citrus fruits for vitamin C",
                "Leafy greens for vitamins and minerals",
                "Nuts and seeds for healthy fats",
                "Garlic and ginger for anti-inflammatory properties",
                "Yogurt for probiotics",
            ],
            "recovery": [
                "Protein-rich foods for tissue repair",
                "Complex carbohydrates for energy",
                "Vitamin C for immune support",
                "Zinc-rich foods for healing",
                "Plenty of fluids for hydration",
            ],
        }
    
    def _get_lifestyle_advice(self) -> Dict[str, List[str]]:
        """Get lifestyle advice"""
        return {
            "sleep": [
                "Maintain consistent sleep schedule",
                "Avoid screens 1 hour before bed",
                "Keep bedroom cool and dark",
                "Limit caffeine after 2 PM",
                "Aim for 7-9 hours of sleep",
            ],
            "exercise": [
                "Start slowly and gradually increase intensity",
                "Warm up before and cool down after exercise",
                "Listen to your body and rest when needed",
                "Include both cardio and strength training",
                "Stay hydrated during exercise",
            ],
            "stress": [
                "Practice deep breathing exercises",
                "Engage in regular physical activity",
                "Maintain social connections",
                "Set boundaries and learn to say no",
                "Seek professional help if needed",
            ],
        }
    
    def get_precautions(self, disease: Optional[str] = None) -> Dict:
        """Get precautions for a specific disease or general precautions"""
        result = {
            "general": self.general_precautions,
            "nutrition": self.nutrition_advice["general"],
            "lifestyle": self.lifestyle_advice["sleep"] + self.lifestyle_advice["exercise"],
        }
        
        if disease:
            disease_lower = disease.lower()
            # Find matching disease
            for key, precautions in self.disease_precautions.items():
                if key.lower() in disease_lower or disease_lower in key.lower():
                    result["disease_specific"] = precautions
                    break
            else:
                result["disease_specific"] = [
                    "Follow your healthcare provider's recommendations",
                    "Take medications as prescribed",
                    "Rest and avoid strenuous activities",
                    "Monitor symptoms and seek care if they worsen",
                ]
        
        return result
    
    def get_emergency_signs(self, category: Optional[str] = None) -> List[str]:
        """Get emergency warning signs"""
        if category and category in self.emergency_signs:
            return self.emergency_signs[category]
        return self.emergency_signs["general"]
    
    def get_nutrition_advice(self, purpose: str = "general") -> List[str]:
        """Get nutrition advice"""
        if purpose in self.nutrition_advice:
            return self.nutrition_advice[purpose]
        return self.nutrition_advice["general"]
    
    def get_lifestyle_advice(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """Get lifestyle advice"""
        if category and category in self.lifestyle_advice:
            return {category: self.lifestyle_advice[category]}
        return self.lifestyle_advice
    
    def get_symptom_tips(self, symptoms: List[str]) -> List[str]:
        """Get tips based on symptoms"""
        tips = []
        
        symptom_tips = {
            "fever": [
                "Stay hydrated with water, clear broths, or electrolyte solutions",
                "Rest and avoid strenuous activities",
                "Use a light blanket - don't try to sweat it out",
                "Take fever-reducing medication if needed",
            ],
            "cough": [
                "Honey can help soothe throat irritation",
                "Stay hydrated to thin mucus",
                "Use a humidifier to add moisture to air",
                "Avoid smoking and irritants",
            ],
            "headache": [
                "Rest in a quiet, dark room",
                "Apply a cold or warm compress to your head",
                "Stay hydrated",
                "Consider over-the-counter pain relievers",
            ],
            "fatigue": [
                "Ensure you're getting adequate sleep",
                "Check for nutritional deficiencies",
                "Take short breaks throughout the day",
                "Light exercise can boost energy levels",
            ],
            "nausea": [
                "Eat small, frequent meals",
                "Avoid strong odors",
                "Try ginger or peppermint tea",
                "Stay hydrated with small sips",
            ],
            "sore throat": [
                "Gargle with warm salt water",
                "Drink warm liquids like tea with honey",
                "Avoid irritants like smoke",
                "Use throat lozenges for relief",
            ],
            "body aches": [
                "Rest and avoid strenuous activities",
                "Apply heat or ice to affected areas",
                "Take over-the-counter pain relievers",
                "Gentle stretching may help",
            ],
            "congestion": [
                "Use a humidifier or steam inhalation",
                "Try nasal saline spray",
                "Stay hydrated",
                "Elevate head while sleeping",
            ],
        }
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for key, symptom_tip_list in symptom_tips.items():
                if key in symptom_lower:
                    tips.extend(symptom_tip_list)
                    break
        
        if not tips:
            tips = [
                "Rest and take care of yourself",
                "Stay hydrated",
                "Monitor your symptoms",
                "Consult a healthcare provider if symptoms persist",
            ]
        
        return tips[:4]  # Return top 4 tips


class HealthTips:
    """Generate contextual health tips"""
    
    @staticmethod
    def get_symptom_tips(symptoms: List[str]) -> List[str]:
        """Get tips based on symptoms"""
        tips = []
        
        symptom_tips = {
            "fever": [
                "Stay hydrated with water, clear broths, or electrolyte solutions",
                "Rest and avoid strenuous activities",
                "Use a light blanket - don't try to sweat it out",
                "Take fever-reducing medication if needed",
            ],
            "cough": [
                "Honey can help soothe throat irritation",
                "Stay hydrated to thin mucus",
                "Use a humidifier to add moisture to air",
                "Avoid smoking and irritants",
            ],
            "headache": [
                "Rest in a quiet, dark room",
                "Apply a cold or warm compress to your head",
                "Stay hydrated",
                "Consider over-the-counter pain relievers",
            ],
            "fatigue": [
                "Ensure you're getting adequate sleep",
                "Check for nutritional deficiencies",
                "Take short breaks throughout the day",
                "Light exercise can boost energy levels",
            ],
            "nausea": [
                "Eat small, frequent meals",
                "Avoid strong odors",
                "Try ginger or peppermint tea",
                "Stay hydrated with small sips",
            ],
        }
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for key, symptom_tip_list in symptom_tips.items():
                if key in symptom_lower:
                    tips.extend(symptom_tip_list)
                    break
        
        if not tips:
            tips = [
                "Rest and take care of yourself",
                "Stay hydrated",
                "Monitor your symptoms",
                "Consult a healthcare provider if symptoms persist",
            ]
        
        return tips[:4]  # Return top 4 tips
    
    @staticmethod
    def get_prevention_tips(disease_category: str) -> List[str]:
        """Get prevention tips for a disease category"""
        prevention_tips = {
            "respiratory": [
                "Wash hands frequently",
                "Avoid touching your face",
                "Stay away from sick people",
                "Get vaccinated if available",
            ],
            "digestive": [
                "Practice food safety",
                "Wash hands before eating",
                "Avoid contaminated water",
                "Cook food thoroughly",
            ],
            "cardiovascular": [
                "Exercise regularly",
                "Maintain healthy diet",
                "Monitor blood pressure",
                "Manage stress",
            ],
            "mental_health": [
                "Practice self-care",
                "Stay connected with others",
                "Seek professional help when needed",
                "Establish routines",
            ],
        }
        
        return prevention_tips.get(disease_category.lower(), prevention_tips["respiratory"])


if __name__ == "__main__":
    db = PrecautionsDatabase()
    
    print("General Precautions:")
    for p in db.get_precautions()["general"][:3]:
        print(f"  - {p}")
    
    print("\nCOVID-19 Specific Precautions:")
    for p in db.get_precautions("COVID-19")["disease_specific"]:
        print(f"  - {p}")
    
    print("\nEmergency Signs:")
    for sign in db.get_emergency_signs()[:3]:
        print(f"  - {sign}")
