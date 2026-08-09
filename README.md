
# ✈️ AeroResolve AI

### AI-Powered Flight Delay Prediction & Operational Risk Assessment

AeroResolve AI is an **AI-powered aviation decision-support system** that predicts flight delay risk and provides operational insights using  **Machine Learning, Agentic AI, real-time weather data, RAG, and LLMs** .

## 🚀 Key Features

* ✈️ **Flight Delay Prediction** using XGBoost
* 🌦️ **Live Weather Analysis** using Open-Meteo
* 🤖 **Multi-Agent AI** for flight, weather, route, and risk analysis
* 📚 **RAG** using FAA aviation protocols
* 🔎 **FAISS Vector Database** for knowledge retrieval
* 🧠 **Llama 3.3 70B** through Groq for AI-generated assessments
* 📊 **Streamlit Dashboard** for interactive risk visualization

## 🏗️ Architecture

```text
Flight Input
     ↓
Flight Validation Agent
     ↓
Weather + Route Analysis
     ↓
XGBoost Risk Prediction
     ↓
FAA Protocol RAG + FAISS
     ↓
Llama 3.3 70B
     ↓
Final Risk Assessment
     ↓
Operational Recommendation
```

## 🛠️ Tech Stack

| Category   | Technologies          |
| ---------- | --------------------- |
| Language   | Python                |
| Dashboard  | Streamlit             |
| ML         | XGBoost, Scikit-learn |
| AI Agents  | LangChain             |
| LLM        | Llama 3.3 70B, Groq   |
| RAG        | LangChain, FAISS      |
| Embeddings | HuggingFace           |
| Data       | Pandas, NumPy         |
| Weather    | Open-Meteo            |

## 📁 Project Structure

```text
AeroResolve_AI/
│
├── app.py
├── agents.py
├── requirements.txt
├── models/
├── faiss_aviation_index/
├── rag/
└── src/
```

## 🚀 Run Locally

### 1. Clone

```bash
git clone https://github.com/siddhantankar18/AeroResolve_AI.git
cd AeroResolve_AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add API Key

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

### 4. Run

```bash
streamlit run app.py
```

## 🎯 Objective

AeroResolve AI combines **prediction, real-time intelligence, regulatory knowledge, and generative AI** to help aviation teams identify potential risks and make faster, better-informed operational decisions.

> **Predict Risk. Understand the Cause. Decide with Confidence.**

## 👨‍💻 Author

**Siddhant Ankar**

Data Science | Machine Learning | Generative AI | Agentic AI
