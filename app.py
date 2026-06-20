"""
SymptoTrack AI - Hybrid RAG Healthcare Chatbot
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.rag_engine import RAGEngine
from src.disease_predictor import DiseasePredictor, SymptomExtractor
from src.risk_assessor import RiskAssessor, RiskLevel
from src.precautions import PrecautionsDatabase
from src.chatbot import ChatbotInterface, MessageType

# Page configuration
st.set_page_config(
    page_title="SymptoTrack AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #22d3ee;
        --background: #0d0d0f;
        --card-bg: #141418;
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.6);
        --border-color: rgba(255, 255, 255, 0.08);
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: rgba(255, 255, 255, 0.6);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .stCard {
        background: #141418;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
    }
    
    /* Risk levels */
    .risk-low { color: #22c55e; }
    .risk-moderate { color: #eab308; }
    .risk-high { color: #f97316; }
    .risk-critical { color: #ef4444; }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
    }
    
    .bot-message {
        background: #1e1e24;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
    }
    
    /* Symptom pills */
    .symptom-pill {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
        padding: 0.4rem 0.8rem;
        border-radius: 50px;
        font-size: 0.85rem;
        margin: 0.25rem;
        display: inline-block;
    }
    
    /* Prediction cards */
    .prediction-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(34, 211, 238, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0d0d0f;
    }
    ::-webkit-scrollbar-thumb {
        background: #6366f1;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all models and initialize components"""
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    
    # Build RAG engine
    rag = RAGEngine()
    rag.create_documents(df)
    rag.build_index(force_rebuild=False)
    
    # Initialize components
    predictor = DiseasePredictor(rag, loader)
    risk_assessor = RiskAssessor()
    precautions_db = PrecautionsDatabase()
    chatbot = ChatbotInterface(predictor, risk_assessor, precautions_db)
    
    return {
        "loader": loader,
        "rag": rag,
        "predictor": predictor,
        "risk_assessor": risk_assessor,
        "precautions_db": precautions_db,
        "chatbot": chatbot
    }


def render_header():
    """Render page header"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">🏥 SymptoTrack AI</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Your intelligent health companion powered by Hybrid RAG</p>', unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="text-align: right; padding-top: 1rem;">
            <span style="color: #22c55e;">●</span> System Online
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with quick actions"""
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        
        # Quick symptom buttons
        st.markdown("#### Common Symptoms")
        quick_symptoms = st.columns(2)
        
        common_symptoms = [
            "Fever", "Cough", "Headache", "Fatigue",
            "Sore throat", "Body aches", "Nausea", "Dizziness"
        ]
        
        selected_symptoms = []
        for i, symptom in enumerate(common_symptoms):
            with quick_symptoms[i % 2]:
                if st.button(symptom, key=f"quick_{i}"):
                    selected_symptoms.append(symptom)
        
        st.markdown("---")
        
        # Info section
        st.markdown("""
        ### ℹ️ About
        
        SymptoTrack AI uses **Hybrid RAG** technology combining:
        
        - **FAISS** for vector similarity search
        - **Sentence Transformers** for semantic understanding
        - **Comprehensive** medical knowledge base
        
        ### ⚠️ Disclaimer
        
        This is for educational purposes only. Always consult healthcare professionals.
        """)
        
        return selected_symptoms


def render_chat_interface(components):
    """Render chat interface"""
    chatbot = components["chatbot"]
    predictor = components["predictor"]
    risk_assessor = components["risk_assessor"]
    precautions_db = components["precautions_db"]
    
    st.markdown("### 💬 Health Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = chatbot.process_message("start")
        st.session_state.messages = [
            {"role": "assistant", "content": welcome_msg.content}
        ]
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Symptom input
    st.markdown("#### Enter Your Symptoms")
    
    # Symptom chips
    available_symptoms = chatbot.get_quick_symptoms()
    cols = st.columns(5)
    
    for i, symptom in enumerate(available_symptoms[:10]):
        with cols[i % 5]:
            if st.button(f"+ {symptom}", key=f"chip_{i}", use_container_width=True):
                # Add to input
                if "symptom_input" not in st.session_state:
                    st.session_state.symptom_input = ""
                st.session_state.symptom_input = symptom
    
    # Text input
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Describe your symptoms:",
            placeholder="e.g., I have fever, cough, and headache...",
            key="symptom_input"
        )
    with col2:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if user_input and analyze_btn:
        # Process user input
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Get chatbot response
        response = chatbot.process_message(user_input)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.content
        })
        
        st.rerun()


def render_symptom_checker(components):
    """Render symptom checker section"""
    st.markdown("### 🔍 Symptom Checker")
    
    predictor = components["predictor"]
    risk_assessor = components["risk_assessor"]
    precautions_db = components["precautions_db"]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Multi-select for symptoms
        available_symptoms = predictor.normalize_symptoms(
            components["loader"].get_all_symptoms()
        )
        
        selected_symptoms = st.multiselect(
            "Select your symptoms:",
            options=available_symptoms[:50],
            default=[]
        )
        
        # Custom symptom input
        custom_symptom = st.text_input("Or enter a custom symptom:", placeholder="Type here...")
        
        if custom_symptom:
            if custom_symptom not in selected_symptoms:
                selected_symptoms.append(custom_symptom)
    
    with col2:
        st.markdown("#### Selected Symptoms")
        if selected_symptoms:
            for s in selected_symptoms:
                st.markdown(f'<span class="symptom-pill">✓ {s}</span>', unsafe_allow_html=True)
        else:
            st.info("No symptoms selected")
    
    # Analyze button
    if selected_symptoms and st.button("🔬 Analyze Symptoms", type="primary", use_container_width=True):
        with st.spinner("Analyzing symptoms..."):
            # Get predictions
            predictions = predictor.predict(selected_symptoms, top_k=3)
            
            # Get risk assessment
            if predictions:
                risk = risk_assessor.assess(
                    selected_symptoms,
                    severity=predictions[0].severity
                )
            else:
                risk = risk_assessor.assess(selected_symptoms, severity="moderate")
            
            # Display results
            st.markdown("---")
            st.markdown("#### 📊 Analysis Results")
            
            # Risk assessment
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_color = {
                    RiskLevel.LOW: "🟢 Low",
                    RiskLevel.MODERATE: "🟡 Moderate",
                    RiskLevel.HIGH: "🟠 High",
                    RiskLevel.CRITICAL: "🔴 Critical"
                }
                st.metric(
                    "Risk Level",
                    risk_color.get(risk.risk_level, "⚪ Unknown")
                )
            
            with col2:
                st.metric(
                    "Urgency",
                    risk.urgency.split(" - ")[0] if " - " in risk.urgency else risk.urgency
                )
            
            with col3:
                if predictions:
                    st.metric(
                        "Top Match",
                        f"{predictions[0].disease[:20]}..."
                    )
            
            # Risk factors
            if risk.risk_factors:
                st.markdown("**⚠️ Risk Factors:**")
                for factor in risk.risk_factors:
                    st.markdown(f"- {factor}")
            
            # Predictions
            if predictions:
                st.markdown("#### 🏥 Possible Conditions")
                
                for i, pred in enumerate(predictions):
                    confidence = pred.confidence * 100
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>{i+1}. {pred.disease}</h4>
                            <p><strong>Match:</strong> {confidence:.0f}% | <strong>Severity:</strong> {pred.severity} | <strong>Category:</strong> {pred.category}</p>
                            <p>{pred.description[:200]}...</p>
                            <p><strong>Treatment:</strong> {pred.treatment}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("#### 💡 Recommendations")
            for rec in risk.recommendations[:3]:
                st.markdown(f"- {rec}")
            
            # Precautions
            if predictions:
                precautions = precautions_db.get_precautions(predictions[0].disease)
                
                st.markdown("#### 🛡️ Precautions")
                for prec in precautions["general"][:4]:
                    st.markdown(f"- {prec}")
            
            # Emergency signs
            st.markdown("""
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 1rem; margin-top: 1rem;">
                <h4 style="color: #ef4444;">🚨 When to Seek Emergency Care</h4>
                <ul style="color: rgba(255,255,255,0.8);">
                    <li>Difficulty breathing or shortness of breath</li>
                    <li>Chest pain or pressure</li>
                    <li>Severe or persistent vomiting</li>
                    <li>High fever not responding to medication</li>
                    <li>Sudden confusion or inability to arouse</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


def render_disease_database(components):
    """Render disease database explorer"""
    st.markdown("### 🗄️ Disease Database")
    
    loader = components["loader"]
    
    # Filters
    col1, col2 = st.columns([1, 2])
    
    with col1:
        categories = ["All"] + loader.get_categories()
        selected_category = st.selectbox("Category:", categories)
    
    with col2:
        search = st.text_input("Search diseases:", placeholder="Search by name or symptom...")
    
    # Get diseases
    if selected_category == "All":
        diseases = loader.get_all_diseases()
    else:
        diseases = loader.get_diseases_by_category(selected_category)
    
    # Filter by search
    if search:
        diseases = [d for d in diseases if search.lower() in d.lower()]
    
    # Display diseases
    st.markdown(f"**Found {len(diseases)} conditions**")
    
    # Pagination
    page_size = 10
    page = st.number_input("Page", min_value=1, max_value=max(1, (len(diseases) + page_size - 1) // page_size), value=1)
    
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(diseases))
    
    for disease in diseases[start_idx:end_idx]:
        info = loader.get_disease_info(disease)
        if info:
            with st.expander(f"**{disease}** ({info['severity']})"):
                st.markdown(f"**Category:** {info['category']}")
                st.markdown(f"**Symptoms:** {', '.join(info['symptoms'][:10])}")
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Treatment:** {info['treatment']}")
                st.markdown(f"**Duration:** {info['duration']}")


def render_about():
    """Render about section"""
    st.markdown("""
    ### ℹ️ About SymptoTrack AI
    
    **SymptoTrack AI** is a hybrid RAG-powered healthcare chatbot designed to help users
    understand their symptoms and provide preliminary health guidance.
    
    #### Technology Stack
    
    | Component | Technology |
    |-----------|------------|
    | Frontend | Streamlit |
    | Vector Search | FAISS |
    | Embeddings | Sentence Transformers |
    | Data Processing | Pandas |
    | NLP | Custom symptom extraction |
    
    #### Features
    
    - ✅ **Symptom Input** - Natural language symptom description
    - ✅ **Disease Prediction** - AI-powered condition matching
    - ✅ **Risk Assessment** - Urgency evaluation
    - ✅ **Precautions** - Personalized health recommendations
    - ✅ **Chat Interface** - Conversational AI interaction
    - ✅ **Disease Database** - Comprehensive medical knowledge
    
    #### How It Works
    
    1. **Symptom Extraction** - Natural language processing extracts symptoms
    2. **Vector Search** - FAISS searches semantic matches in knowledge base
    3. **Hybrid Matching** - Combines keyword and semantic similarity
    4. **Risk Assessment** - Evaluates urgency based on symptoms
    5. **Recommendations** - Provides precautions and guidance
    
    ---
    
    ⚠️ **Important Disclaimer**
    
    This application is for **educational and informational purposes only**. It is not
    intended to be a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the advice of your physician or other qualified health provider with any
    questions you may have regarding a medical condition.
    """)


def main():
    """Main application"""
    # Load models
    try:
        components = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure all dependencies are installed and the data file exists.")
        return
    
    # Header
    render_header()
    
    # Sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat",
        "🔍 Symptom Checker",
        "🏥 Diseases",
        "ℹ️ About"
    ])
    
    with tab1:
        render_chat_interface(components)
    
    with tab2:
        render_symptom_checker(components)
    
    with tab3:
        render_disease_database(components)
    
    with tab4:
        render_about()


if __name__ == "__main__":
    main()
