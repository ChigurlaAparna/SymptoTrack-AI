"""
Chatbot Interface Module for SymptoTrack AI
Handles chat interactions and conversation flow
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MessageType(Enum):
    """Message type enumeration"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Message:
    """Chat message"""
    content: str
    message_type: MessageType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict] = None


class ChatbotInterface:
    """
    Chat interface for health chatbot
    Manages conversation flow and responses
    """
    
    def __init__(self, predictor, risk_assessor, precautions_db):
        self.predictor = predictor
        self.risk_assessor = risk_assessor
        self.precautions_db = precautions_db
        self.conversation_history = []
        self.current_symptoms = []
        self.conversation_state = "initial"  # initial, symptoms_collected, assessed, follow_up
        
        # Import and initialize symptom extractor
        from .disease_predictor import SymptomExtractor
        self.symptom_extractor = SymptomExtractor()
        
        # Response templates
        self.greeting = """
👋 **Welcome to SymptoTrack AI!**

I'm here to help you understand your symptoms and provide guidance. 

**Important:** I can provide general health information but am not a substitute for professional medical advice.

To get started, please describe your symptoms (e.g., "I have a fever and headache").
"""
        
        self.clarification_prompt = """
Could you please provide more details? For example:
- What symptoms are you experiencing?
- How long have you had these symptoms?
- How severe are they (mild, moderate, severe)?
"""
        
        self.thanks_response = """
You're welcome! Take care of yourself. 💙

If you have any more questions or want to track new symptoms, I'm here to help.

**Quick tips:**
- Stay hydrated
- Get adequate rest
- Monitor your symptoms
"""
    
    def reset_conversation(self):
        """Reset the conversation state"""
        self.conversation_history = []
        self.current_symptoms = []
        self.conversation_state = "initial"
    
    def process_message(self, user_input: str) -> Message:
        """
        Process user input and generate response
        
        Args:
            user_input: User's message
        
        Returns:
            Message object with response
        """
        user_input_lower = user_input.lower().strip()
        
        # Handle special commands
        if self._is_greeting(user_input_lower):
            return self._handle_greeting()
        
        if self._is_thanks(user_input_lower):
            return self._handle_thanks()
        
        if self._is_bye(user_input_lower):
            return self._handle_bye()
        
        if self._is_help_request(user_input_lower):
            return self._handle_help()
        
        # Extract symptoms from input
        extracted_symptoms = self.predictor.normalize_symptoms(
            self.symptom_extractor.extract(user_input)
        )
        
        # Also check for any symptoms mentioned directly
        words = user_input_lower.replace("?", "").replace(",", " ").split()
        direct_symptoms = [w for w in words if len(w) > 3]
        extracted_symptoms = list(set(extracted_symptoms + direct_symptoms))
        
        if not extracted_symptoms:
            return Message(
                content=self.clarification_prompt,
                message_type=MessageType.ASSISTANT
            )
        
        # Add to current symptoms
        self.current_symptoms.extend(extracted_symptoms)
        self.current_symptoms = list(set(self.current_symptoms))
        
        # Generate assessment
        return self._generate_assessment()
    
    def _is_greeting(self, text: str) -> bool:
        """Check if input is a greeting"""
        greetings = ["hi", "hello", "hey", "start", "help", "reset"]
        return any(text.startswith(g) for g in greetings)
    
    def _is_thanks(self, text: str) -> bool:
        """Check if input is thanks"""
        return "thank" in text or "thanks" in text
    
    def _is_bye(self, text: str) -> bool:
        """Check if input is goodbye"""
        return "bye" in text or "goodbye" in text or "exit" in text
    
    def _is_help_request(self, text: str) -> bool:
        """Check if input is a help request"""
        return "help" in text or "what can you do" in text
    
    def _handle_greeting(self) -> Message:
        """Handle greeting"""
        self.reset_conversation()
        return Message(
            content=self.greeting,
            message_type=MessageType.ASSISTANT
        )
    
    def _handle_thanks(self) -> Message:
        """Handle thanks"""
        return Message(
            content=self.thanks_response,
            message_type=MessageType.ASSISTANT
        )
    
    def _handle_bye(self) -> Message:
        """Handle goodbye"""
        return Message(
            content="Goodbye! Take care of your health. 👋\n\nFeel free to return if you have any more questions.",
            message_type=MessageType.ASSISTANT
        )
    
    def _handle_help(self) -> Message:
        """Handle help request"""
        help_text = """
🔍 **What I can help with:**

**1. Symptom Analysis**
Describe your symptoms, and I'll analyze them and suggest possible conditions.

**2. Risk Assessment**
I'll evaluate the urgency and severity of your situation.

**3. Precautions & Tips**
Get recommendations for self-care and when to seek medical help.

**4. Disease Information**
Learn more about potential conditions.

**How to use:**
Simply describe your symptoms in plain language, for example:
- "I have fever and cough"
- "Headache and fatigue for 3 days"
- "Stomach pain with nausea"

**Important:** Always consult a healthcare professional for proper diagnosis and treatment.
"""
        return Message(
            content=help_text,
            message_type=MessageType.INFO
        )
    
    def _generate_assessment(self) -> Message:
        """Generate health assessment based on symptoms"""
        symptoms = self.current_symptoms
        
        if not symptoms:
            return Message(
                content=self.clarification_prompt,
                message_type=MessageType.ASSISTANT
            )
        
        # Get predictions
        predictions = self.predictor.predict(symptoms, top_k=3)
        
        if not predictions:
            return Message(
                content=f"I couldn't find matches for the symptoms: {', '.join(symptoms)}\n\n{self.clarification_prompt}",
                message_type=MessageType.ASSISTANT
            )
        
        # Get risk assessment
        risk = self.risk_assessor.assess(symptoms, severity=predictions[0].severity)
        
        # Get precautions
        precautions = self.precautions_db.get_precautions(predictions[0].disease if predictions else None)
        
        # Get symptom tips
        tips = self.precautions_db.get_symptom_tips(symptoms)
        
        # Build response
        response = self._build_assessment_response(predictions, risk, precautions, tips)
        
        return Message(
            content=response,
            message_type=MessageType.ASSISTANT,
            metadata={
                "symptoms": symptoms,
                "predictions": [(p.disease, p.confidence) for p in predictions],
                "risk_level": risk.risk_level.value
            }
        )
    
    def _build_assessment_response(
        self,
        predictions: List,
        risk,
        precautions: Dict,
        tips: List[str]
    ) -> str:
        """Build comprehensive assessment response"""
        
        response_parts = []
        
        # Header
        response_parts.append("## 🏥 Health Assessment\n")
        
        # Symptoms recognized
        response_parts.append(f"**Your Symptoms:** {', '.join(self.current_symptoms)}\n")
        
        # Risk level indicator
        risk_emoji = {
            "low": "🟢",
            "moderate": "🟡",
            "high": "🟠",
            "critical": "🔴"
        }
        response_parts.append(f"**Risk Level:** {risk_emoji.get(risk.risk_level.value, '⚪')} {risk.risk_level.value.upper()}\n")
        response_parts.append(f"⏰ *{risk.urgency}*\n")
        
        # Possible conditions
        response_parts.append("\n### 📋 Possible Conditions\n")
        for i, pred in enumerate(predictions[:3], 1):
            confidence_pct = pred.confidence * 100
            response_parts.append(f"{i}. **{pred.disease}** ({confidence_pct:.0f}% match)")
            response_parts.append(f"   - Severity: {pred.severity}")
            response_parts.append(f"   - Category: {pred.category}\n")
        
        # Disease details
        if predictions:
            top = predictions[0]
            response_parts.append(f"### 📖 About {top.disease}\n")
            response_parts.append(f"{top.description}\n")
            response_parts.append(f"\n**Treatment:** {top.treatment}")
            response_parts.append(f"\n**Duration:** {top.duration}\n")
        
        # Risk factors
        if risk.risk_factors:
            response_parts.append("\n### ⚠️ Risk Factors\n")
            for factor in risk.risk_factors:
                response_parts.append(f"- {factor}")
            response_parts.append("")
        
        # Recommendations
        response_parts.append("\n### 💡 Recommendations\n")
        for rec in risk.recommendations[:4]:
            response_parts.append(f"- {rec}")
        response_parts.append("")
        
        # Self-care tips
        response_parts.append("\n### 🏠 Self-Care Tips\n")
        for tip in tips:
            response_parts.append(f"- {tip}")
        response_parts.append("")
        
        # General precautions
        response_parts.append("\n### 🛡️ General Precautions\n")
        for prec in precautions["general"][:4]:
            response_parts.append(f"- {prec}")
        response_parts.append("")
        
        # Disclaimer
        response_parts.append("\n---\n")
        response_parts.append("⚕️ **Disclaimer:** This information is for educational purposes only and should not replace professional medical advice. Always consult a healthcare provider for proper diagnosis and treatment.\n")
        
        return "\n".join(response_parts)
    
    def get_quick_symptoms(self) -> List[str]:
        """Get list of common symptoms for quick selection"""
        return [
            "Fever",
            "Cough",
            "Headache",
            "Fatigue",
            "Sore throat",
            "Body aches",
            "Nausea",
            "Dizziness",
            "Shortness of breath",
            "Chest pain",
            "Abdominal pain",
            "Diarrhea",
            "Runny nose",
            "Congestion",
            "Rash",
            "Vomiting",
            "Back pain",
            "Joint pain",
            "Anxiety",
            "Insomnia"
        ]
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of current conversation"""
        return {
            "state": self.conversation_state,
            "symptoms": self.current_symptoms,
            "message_count": len(self.conversation_history)
        }


if __name__ == "__main__":
    from data_loader import DataLoader
    from rag_engine import RAGEngine
    from disease_predictor import DiseasePredictor
    from risk_assessor import RiskAssessor
    from precautions import PrecautionsDatabase
    
    # Initialize components
    loader = DataLoader()
    df = loader.load_data()
    
    rag = RAGEngine()
    rag.create_documents(df)
    rag.build_index()
    
    predictor = DiseasePredictor(rag, loader)
    risk_assessor = RiskAssessor()
    precautions = PrecautionsDatabase()
    
    # Create chatbot
    chatbot = ChatbotInterface(predictor, risk_assessor, precautions)
    
    # Test conversation
    print("=== SymptoTrack AI Chatbot ===\n")
    
    # Greeting
    response = chatbot.process_message("hi")
    print(f"Bot: {response.content}\n")
    
    # Symptom input
    response = chatbot.process_message("I have fever, cough, and headache")
    print(f"Bot: {response.content}\n")
