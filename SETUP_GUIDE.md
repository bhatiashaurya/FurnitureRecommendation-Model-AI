# Project Setup Guide

## Complete Installation and Setup Instructions

### Step 1: Environment Setup

1. **Clone or Navigate to Project Directory**
```powershell
cd D:\Projects\Ikarus3d_AI
```

2. **Create and Activate Virtual Environment**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. **Install Core Dependencies**
```powershell
pip install pandas numpy scikit-learn matplotlib seaborn plotly jupyter ipykernel
```

4. **Install Deep Learning Libraries**
```powershell
pip install torch torchvision transformers sentence-transformers
```

5. **Install FastAPI and Web Framework**
```powershell
pip install fastapi uvicorn[standard] python-multipart python-dotenv
```

6. **Install LangChain and GenAI Tools**
```powershell
pip install langchain langchain-openai langchain-community openai
```

7. **Install Vector Database Client**
```powershell
pip install pinecone-client
```

8. **Install Additional Tools**
```powershell
pip install pillow opencv-python requests tqdm
```

### Step 2: Environment Configuration

1. **Create .env file**
```powershell
Copy-Item .env.example .env
```

2. **Edit .env with your API keys**
```
OPENAI_API_KEY=sk-your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=furniture-products
PINECONE_DIMENSION=384
```

### Step 3: Run Data Analytics Notebook

1. **Start Jupyter**
```powershell
jupyter notebook
```

2. **Open `data_analytics.ipynb`**
   - Run all cells in order
   - This will:
     - Analyze the dataset
     - Generate visualizations
     - Export analytics data for the dashboard

### Step 4: Run Model Training Notebook

1. **Open `notebooks/model_training.ipynb`**
   - Run all cells in order
   - This will:
     - Preprocess the data
     - Train recommendation models
     - Generate embeddings
     - Upload to Pinecone
     - Save models to `models/` directory

### Step 5: Start Backend Server

```powershell
cd backend
python main.py
# OR
uvicorn main:app --reload --port 8000
```

Backend will be available at: `http://localhost:8000`

### Step 6: Setup and Run Frontend

1. **Navigate to frontend directory**
```powershell
cd frontend
```

2. **Install Node.js dependencies**
```powershell
npm install
```

3. **Start development server**
```powershell
npm run dev
```

Frontend will be available at: `http://localhost:3000`

## Project Structure

```
Ikarus3d_AI/
│
├── data/
│   └── intern_data_ikarus.csv          # Product dataset
│
├── backend/
│   ├── main.py                         # FastAPI application
│   ├── utils/
│   │   ├── pinecone_client.py         # Vector DB integration
│   │   ├── genai_generator.py         # LangChain GenAI
│   │   └── preprocessor.py            # Data preprocessing
│   └── models/                         # Saved ML models
│       ├── text_vectorizer.pkl
│       ├── product_embeddings.npy
│       └── image_features.npy
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                    # Main React app
│   │   ├── pages/
│   │   │   ├── RecommendationPage.jsx # Chat interface
│   │   │   └── AnalyticsPage.jsx      # Analytics dashboard
│   │   └── components/
│   │       ├── ProductCard.jsx
│   │       └── ChatMessage.jsx
│   ├── package.json
│   └── vite.config.js
│
├── notebooks/
│   ├── data_analytics.ipynb           # EDA notebook
│   └── model_training.ipynb           # ML training notebook
│
├── .env                               # Environment variables
├── requirements.txt                   # Python dependencies
└── README.md                          # Documentation
```

## Key Features Implementation

### 1. Machine Learning (ML)
- **Location**: `notebooks/model_training.ipynb`
- **Techniques**: 
  - Content-based filtering using TF-IDF and cosine similarity
  - Hybrid recommendation combining text and image features
  - K-means clustering for product grouping

### 2. Natural Language Processing (NLP)
- **Location**: `notebooks/model_training.ipynb` + `backend/utils/`
- **Tools**: 
  - sentence-transformers for semantic embeddings
  - Text preprocessing and cleaning
  - Category extraction and analysis

### 3. Computer Vision (CV)
- **Location**: `notebooks/model_training.ipynb`
- **Models**: 
  - Pre-trained ResNet or ViT for image feature extraction
  - Image-based product similarity
  - Product category classification from images

### 4. Generative AI (GenAI)
- **Location**: `backend/utils/genai_generator.py`
- **Framework**: LangChain + OpenAI
- **Features**: 
  - Creative product descriptions
  - Recommendation summaries
  - Context-aware responses

### 5. Vector Database
- **Location**: `backend/utils/pinecone_client.py`
- **Database**: Pinecone
- **Features**: 
  - Semantic search
  - Fast similarity retrieval
  - Metadata filtering

### 6. Frontend (React)
- **Location**: `frontend/src/`
- **Features**: 
  - Conversational recommendation interface
  - Real-time chat with AI
  - Product cards with generated descriptions
  - Analytics dashboard with charts

### 7. Analytics Dashboard
- **Location**: `frontend/src/pages/AnalyticsPage.jsx`
- **Visualizations**: 
  - Price distribution
  - Category breakdown
  - Brand analytics
  - Product statistics

## API Endpoints

### Health Check
```http
GET /api/health
```

### Get Recommendations
```http
POST /api/recommend
Content-Type: application/json

{
  "query": "modern dining chair under $100",
  "num_recommendations": 5
}
```

### Get Analytics
```http
GET /api/analytics
```

### Get Product Details
```http
GET /api/products/{product_id}
```

## Testing the Application

### 1. Test Backend
```powershell
# Health check
curl http://localhost:8000/api/health

# Get analytics
curl http://localhost:8000/api/analytics
```

### 2. Test Frontend
- Open `http://localhost:3000`
- Navigate to Recommendations page
- Type a query like "comfortable office chair"
- View generated recommendations
- Navigate to Analytics page to see dashboard

### 3. Test Vector Search (in notebook)
```python
from backend.utils.pinecone_client import PineconeClient

client = PineconeClient()
results = client.search(query_embedding, top_k=5)
```

## Troubleshooting

### Common Issues

1. **Pinecone Connection Error**
   - Verify API key in .env
   - Check Pinecone region/environment
   - Ensure index is created

2. **OpenAI API Error**
   - Verify API key
   - Check API quota/credits
   - Ensure correct model name

3. **Module Not Found**
   - Activate virtual environment
   - Reinstall requirements: `pip install -r requirements.txt`

4. **Frontend Won't Start**
   - Delete `node_modules` and `package-lock.json`
   - Run `npm install` again
   - Check Node.js version (16+)

5. **CORS Errors**
   - Ensure backend is running on port 8000
   - Check CORS configuration in `backend/main.py`

## Performance Optimization

1. **Vector Search**: Use appropriate `top_k` values (5-10)
2. **Image Loading**: Implement lazy loading for product images
3. **API Caching**: Cache frequent queries
4. **Batch Processing**: Process embeddings in batches

## Next Steps / Enhancements

1. **User Authentication**: Add user login and preferences
2. **Real-time Updates**: Implement WebSocket for live recommendations
3. **Advanced Filters**: Price range, material, color filters
4. **Recommendation History**: Track and display user's search history
5. **A/B Testing**: Test different recommendation algorithms
6. **Mobile App**: React Native version
7. **Image Search**: Upload image to find similar products
8. **Voice Search**: Speech-to-text for queries

## Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **LangChain Docs**: https://python.langchain.com/
- **Pinecone Docs**: https://docs.pinecone.io/
- **React Docs**: https://react.dev/
- **Sentence Transformers**: https://www.sbert.net/

## Support

For issues or questions:
1. Check this setup guide
2. Review error logs in terminal
3. Check API documentation
4. Review notebook comments

---

**Happy Coding! 🚀**
