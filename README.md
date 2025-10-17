# Furniture Product Recommendation System 🛋️

A full-stack ML-driven web application for intelligent furniture product recommendations using NLP, Computer Vision, and Generative AI.

## 🎯 Project Overview

This application combines multiple AI domains to create a comprehensive product recommendation system:
- **Machine Learning**: Hybrid recommendation engine (content-based + collaborative filtering)
- **NLP**: Text analysis and semantic search using transformer models
- **Computer Vision**: Image-based product classification and similarity
- **GenAI**: Creative product descriptions using LangChain
- **Vector Database**: Semantic search with Pinecone

## 🏗️ Architecture

```
├── backend/                # FastAPI backend
│   ├── main.py            # API endpoints
│   ├── models/            # ML models and vectorizers
│   └── utils/             # Helper functions
├── frontend/              # React application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Recommendation & Analytics pages
│   │   └── services/      # API integration
├── notebooks/             # Jupyter notebooks
│   ├── data_analytics.ipynb
│   └── model_training.ipynb
├── data/                  # Dataset
└── models/                # Trained models
```

## 📊 Dataset

- **Products**: 312 furniture items
- **Features**: title, brand, description, price, categories, images, dimensions, material, color
- **Source**: E-commerce furniture catalog

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+
- Pinecone account (for vector database)
- OpenAI API key (for GenAI)

### Backend Setup

1. **Clone and navigate to project**
```bash
cd Ikarus3d_AI
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Run Jupyter notebooks** (in order)
```bash
jupyter notebook
# Run: 1. data_analytics.ipynb
#      2. model_training.ipynb
```

6. **Start backend server**
```bash
cd backend
python main.py
# OR
uvicorn main:app --reload --port 8000
```

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Start development server**
```bash
npm run dev
```

4. **Open browser**
```
http://localhost:3000
```

## 📓 Notebooks

### 1. Data Analytics (`data_analytics.ipynb`)
- Exploratory Data Analysis (EDA)
- Missing value analysis
- Price, category, brand distributions
- Geographic analysis
- Data export for dashboard

### 2. Model Training (`model_training.ipynb`)
- Data preprocessing
- Text embeddings (NLP)
- Image feature extraction (CV)
- Recommendation model training
- Vector database integration (Pinecone)
- GenAI integration with LangChain
- Model evaluation

## 🔧 Tech Stack

### Backend
- **Framework**: FastAPI
- **ML/DL**: PyTorch, scikit-learn, sentence-transformers
- **NLP**: HuggingFace Transformers, spaCy
- **CV**: torchvision, OpenCV
- **Vector DB**: Pinecone
- **GenAI**: LangChain + OpenAI

### Frontend
- **Framework**: React
- **Routing**: React Router
- **UI**: Tailwind CSS / Material-UI
- **Charts**: Recharts / Chart.js
- **HTTP**: Axios

## 📡 API Endpoints

### Recommendations
```http
POST /api/recommend
Content-Type: application/json

{
  "query": "modern dining chair under $100",
  "num_recommendations": 5
}
```

### Analytics
```http
GET /api/analytics
```

### Product Details
```http
GET /api/products/{product_id}
```

## 🎨 Features

### 1. Conversational Recommendations
- Chat-based interface for product discovery
- Natural language query understanding
- Contextual recommendations

### 2. Analytics Dashboard
- Product distribution visualizations
- Price analysis and trends
- Category and brand insights
- Geographic distribution

### 3. Intelligent Search
- Semantic search using vector embeddings
- Multi-modal search (text + image)
- Hybrid recommendation (content + collaborative)

### 4. AI-Generated Descriptions
- Creative product descriptions
- Feature highlighting
- Style recommendations

## 🔬 ML Pipeline

1. **Data Preprocessing**
   - Text cleaning and normalization
   - Price parsing and standardization
   - Category extraction

2. **Feature Engineering**
   - TF-IDF vectorization
   - Sentence embeddings (sentence-transformers)
   - Image features (ResNet/ViT)

3. **Model Training**
   - Content-based filtering (cosine similarity)
   - Collaborative filtering (matrix factorization)
   - Hybrid model ensemble

4. **Vector Database**
   - Embedding storage in Pinecone
   - Semantic search capabilities
   - Fast similarity retrieval

5. **GenAI Integration**
   - LangChain for description generation
   - Prompt engineering for product features
   - Context-aware responses

## 📈 Model Performance

- **Recommendation Accuracy**: TBD after training
- **Search Latency**: < 100ms (with Pinecone)
- **Description Quality**: Evaluated manually

## 🛠️ Development

### Project Structure
```
backend/
  ├── main.py              # FastAPI application
  ├── models/              # Model artifacts
  │   ├── text_vectorizer.pkl
  │   ├── product_embeddings.npy
  │   └── image_features.npy
  ├── utils/
  │   ├── pinecone_client.py
  │   ├── genai_generator.py
  │   └── preprocessor.py

frontend/
  ├── src/
  │   ├── App.jsx
  │   ├── pages/
  │   │   ├── RecommendationPage.jsx
  │   │   └── AnalyticsPage.jsx
  │   ├── components/
  │   │   ├── ChatInterface.jsx
  │   │   ├── ProductCard.jsx
  │   │   └── AnalyticsCharts.jsx
  │   └── services/
  │       └── api.js
```

## 🧪 Testing

```bash
# Backend tests
pytest tests/

# Frontend tests
npm test
```

## 📝 Notes

- Ensure all API keys are properly configured
- Vector database initialization required before first use
- Model training can take 30-60 minutes depending on hardware
- Images are fetched from URLs (ensure internet connection)

## 🤝 Contributing

This is an internship assignment project. For questions or issues, please contact the project maintainer.

## 📄 License

This project is created for educational purposes as part of an internship assignment.

## 🙏 Acknowledgments

- HuggingFace for transformer models
- Pinecone for vector database
- OpenAI for GenAI capabilities
- FastAPI and React communities

---

**Developed as part of Ikarus3D AI Internship Assignment**
