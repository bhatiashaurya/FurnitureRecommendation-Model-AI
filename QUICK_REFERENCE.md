# 🚀 Quick Reference Card - Furniture Recommendation System

## 📁 Project Structure at a Glance

```
Ikarus3d_AI/
├── 📊 data_analytics.ipynb           # EDA & Analytics
├── 📁 notebooks/
│   └── model_training.ipynb          # ML Training
├── 🔧 backend/
│   ├── main.py                       # FastAPI Server
│   └── utils/
│       ├── pinecone_client.py        # Vector DB
│       ├── genai_generator.py        # LangChain GenAI
│       └── preprocessor.py           # Data Utils
├── 🎨 frontend/
│   └── src/
│       ├── App.jsx                   # Main App
│       ├── pages/
│       │   ├── RecommendationPage.jsx
│       │   └── AnalyticsPage.jsx
│       └── components/
├── 📝 README.md                       # Project Overview
├── 📋 SETUP_GUIDE.md                  # Installation Guide
├── 🎓 MODEL_TRAINING_GUIDE.md         # ML Training Guide
└── 📦 requirements.txt                # Dependencies
```

## ⚡ Quick Commands

### Setup
```powershell
# Quick setup
.\quickstart.ps1

# Manual setup
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run Backend
```powershell
cd backend
python main.py
# OR
uvicorn main:app --reload --port 8000
```

### Run Frontend
```powershell
cd frontend
npm install
npm run dev
```

### Run Notebooks
```powershell
jupyter notebook data_analytics.ipynb
jupyter notebook notebooks/model_training.ipynb
```

## 🔑 Environment Variables (.env)

```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=furniture-products
PINECONE_DIMENSION=384
```

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/recommend` | Get recommendations |
| GET | `/api/analytics` | Dashboard data |
| GET | `/api/products/{id}` | Product details |

## 📊 Tech Stack Overview

| Domain | Technology |
|--------|------------|
| **Backend** | FastAPI, Python 3.13 |
| **Frontend** | React, Vite, Tailwind CSS |
| **ML/AI** | PyTorch, scikit-learn, Transformers |
| **NLP** | sentence-transformers, HuggingFace |
| **CV** | torchvision, ResNet |
| **GenAI** | LangChain, OpenAI GPT |
| **Vector DB** | Pinecone |
| **Viz** | Plotly, Recharts, Matplotlib |

## 🎯 Key Files & What They Do

### Backend
- `main.py` → FastAPI server & API routes
- `pinecone_client.py` → Vector database operations
- `genai_generator.py` → AI description generation
- `preprocessor.py` → Data cleaning utilities

### Frontend
- `App.jsx` → Main application & routing
- `RecommendationPage.jsx` → Chat interface
- `AnalyticsPage.jsx` → Dashboard with charts
- `ProductCard.jsx` → Product display component

### Notebooks
- `data_analytics.ipynb` → EDA & visualizations
- `model_training.ipynb` → ML model training

## 🔄 Typical Workflow

1. **Setup Environment**
   ```
   .\quickstart.ps1
   Edit .env with API keys
   ```

2. **Run Data Analysis**
   ```
   jupyter notebook data_analytics.ipynb
   Execute all cells
   ```

3. **Train Models**
   ```
   jupyter notebook notebooks/model_training.ipynb
   Follow MODEL_TRAINING_GUIDE.md
   ```

4. **Start Services**
   ```
   Terminal 1: cd backend && python main.py
   Terminal 2: cd frontend && npm run dev
   ```

5. **Test Application**
   ```
   Frontend: http://localhost:3000
   Backend API: http://localhost:8000/docs
   ```

## 🐛 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| **Import errors** | Activate venv: `.venv\Scripts\Activate.ps1` |
| **Pinecone error** | Check API key in .env |
| **OpenAI error** | Verify API key & quota |
| **Port in use** | Change port or kill process |
| **CORS error** | Check backend CORS config |
| **Module not found** | `pip install -r requirements.txt` |

## 📈 Model Training Pipeline

```
Data → Preprocess → Text Embeddings → Image Features →
  ↓                    (BERT)          (ResNet)
Combine → Normalize → Upload to Pinecone → Test Search
           ↓
      Save Models (pkl/npy)
```

## 🎨 Frontend Routes

- `/` → Recommendation Page (Chat Interface)
- `/analytics` → Analytics Dashboard

## 🔧 Utility Functions

```python
# Data Preprocessing
from backend.utils.preprocessor import DataPreprocessor
preprocessor = DataPreprocessor()
df_clean = preprocessor.preprocess_dataframe(df)

# Pinecone Client
from backend.utils.pinecone_client import PineconeClient
client = PineconeClient()
client.create_index()
results = client.search(embedding, top_k=5)

# GenAI Generator
from backend.utils.genai_generator import ProductDescriptionGenerator
gen = ProductDescriptionGenerator()
desc = gen.generate_description(product_data)
```

## 📦 Package Installation Order

1. Core: `pandas numpy scikit-learn`
2. Viz: `matplotlib seaborn plotly`
3. ML: `torch torchvision transformers`
4. NLP: `sentence-transformers`
5. Web: `fastapi uvicorn`
6. AI: `langchain langchain-openai openai`
7. DB: `pinecone-client`
8. Utils: `python-dotenv pillow requests`

## 🎯 Testing Checklist

- [ ] Virtual environment activated
- [ ] .env configured with API keys
- [ ] Data analytics notebook runs
- [ ] Model training completes
- [ ] Pinecone index created
- [ ] Backend starts (port 8000)
- [ ] Frontend starts (port 3000)
- [ ] Can send chat messages
- [ ] Recommendations display
- [ ] Analytics dashboard loads
- [ ] Charts render correctly

## 📚 Documentation Links

- **Setup**: `SETUP_GUIDE.md`
- **Training**: `MODEL_TRAINING_GUIDE.md`
- **Overview**: `README.md`
- **Summary**: `PROJECT_SUMMARY.md`
- **API Docs**: `http://localhost:8000/docs` (when running)

## 🚨 Important Notes

⚠️ **Before Training**:
- Configure .env with valid API keys
- Ensure 2GB+ free disk space for models
- Check internet connection for image downloads

⚠️ **Before Running**:
- Backend must start before frontend
- Pinecone index must be created
- Models must be trained and saved

⚠️ **API Rate Limits**:
- OpenAI: Monitor usage
- Pinecone: Free tier limits apply

## 🎉 Quick Success Test

```powershell
# 1. Check backend
curl http://localhost:8000/api/health

# 2. Check frontend
# Open browser: http://localhost:3000

# 3. Test recommendation
# Type in chat: "modern dining chair"
# Should see product recommendations
```

## 📞 Need Help?

1. Check `SETUP_GUIDE.md` for detailed instructions
2. Review `PROJECT_SUMMARY.md` for overview
3. See `MODEL_TRAINING_GUIDE.md` for ML help
4. Check inline comments in code
5. Review error messages in terminal

---

**Project Status**: ✅ Complete & Ready  
**Last Updated**: October 2025  
**Quick Start Time**: ~15 minutes
