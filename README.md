# SymptoTrack AI 🏥

A Hybrid RAG-based Healthcare Chatbot that provides symptom analysis, disease prediction, risk assessment, and health precautions.

## Features

- **💬 Chat Interface** - Conversational AI interaction for health queries
- **🔍 Symptom Checker** - Multi-select and natural language symptom input
- **🏥 Disease Prediction** - AI-powered condition matching using Hybrid RAG
- **⚠️ Risk Assessment** - Urgency evaluation based on symptoms
- **🛡️ Precautions** - Personalized health recommendations
- **🏗️ Disease Database** - Comprehensive medical knowledge base

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Vector Search | FAISS |
| Embeddings | Sentence Transformers |
| Data Processing | Pandas |
| NLP | Custom symptom extraction |

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-repo/symtotrack-ai.git
cd symtotrack-ai
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

1. **Run the application**
```bash
streamlit run app.py
```

2. **Open in browser**
The app will open at `http://localhost:8501`

## Project Structure

```
symtotrack_ai/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/
│   └── healthcare_data.csv # Medical knowledge base
├── models/                # Cached FAISS index and embeddings
└── src/
    ├── __init__.py
    ├── chatbot.py          # Chat interface and conversation flow
    ├── data_loader.py      # Data loading and preprocessing
    ├── disease_predictor.py # Disease prediction logic
    ├── precautions.py      # Health precautions database
    ├── rag_engine.py       # Hybrid RAG implementation
    └── risk_assessor.py    # Risk assessment logic
```

## How It Works

### Hybrid RAG Architecture

1. **Data Ingestion**: Healthcare data is loaded from CSV and preprocessed
2. **Document Creation**: Each disease record is converted to a rich text document
3. **Embedding Generation**: Sentence Transformers generate semantic embeddings
4. **Vector Indexing**: FAISS indexes all document embeddings for fast similarity search
5. **Query Processing**: User symptoms are embedded and searched against the index
6. **Hybrid Matching**: Combines vector similarity with keyword matching
7. **Response Generation**: Results are formatted with predictions, risks, and precautions

### Risk Assessment

The risk assessor evaluates:
- Symptom severity and combinations
- High-risk symptom patterns
- Patient factors (age, comorbidities)
- Symptom duration

Risk levels: 🟢 Low → 🟡 Moderate → 🟠 High → 🔴 Critical

## Screenshots

### Chat Interface
- Conversational AI interaction
- Natural language symptom input
- Quick symptom buttons

### Symptom Checker
- Multi-select symptom interface
- Detailed analysis results
- Risk assessment display

### Disease Database
- Category filtering
- Search functionality
- Disease details

## Disclaimer

⚠️ **Important**: This application is for educational and informational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - See LICENSE file for details

---

Built with ❤️ using Streamlit, FAISS, and Sentence Transformers
