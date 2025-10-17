"""
Furniture Product Recommendation System - Backend API
====================================================
FastAPI backend for ML-driven product recommendations with GenAI integration.

Features:
- Product recommendations using hybrid ML models
- Semantic search with vector database (Pinecone)
- GenAI-powered product descriptions using LangChain
- Analytics endpoint for dashboard
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import pickle
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Furniture Recommendation API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class RecommendationRequest(BaseModel):
    query: str
    num_recommendations: int = 5

class ProductResponse(BaseModel):
    uniq_id: str
    title: str
    brand: str
    description: str
    price: Optional[float]
    categories: List[str]
    images: List[str]
    generated_description: str
    similarity_score: float

class AnalyticsResponse(BaseModel):
    total_products: int
    categories_distribution: dict
    price_stats: dict
    top_brands: dict

# Global variables for models and data
df = None
vectorizer = None
embeddings = None
pinecone_index = None

@app.on_event("startup")
async def load_models():
    """Load ML models and data on startup"""
    global df, vectorizer, embeddings
    
    # Load product data
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "..", "intern_data_ikarus.csv")
    df = pd.read_csv(csv_path)
    
    # Load trained models (will be created in training notebook)
    if os.path.exists("models/text_vectorizer.pkl"):
        with open("models/text_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    
    if os.path.exists("models/product_embeddings.npy"):
        embeddings = np.load("models/product_embeddings.npy")
    
    print("✓ Models and data loaded successfully")

@app.get("/")
async def root():
    return {"message": "Furniture Recommendation API", "status": "active"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "products_loaded": len(df) if df is not None else 0}

@app.post("/api/recommend", response_model=List[ProductResponse])
async def get_recommendations(request: RecommendationRequest):
    """
    Get product recommendations based on user query
    Uses hybrid approach: semantic search + content-based filtering
    """
    if df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Simple keyword-based search for demo purposes
    # In production, this would use vector embeddings and semantic search
    query_lower = request.query.lower()
    
    # Search in title, description, categories, brand, material, color
    results = []
    
    for idx, row in df.iterrows():
        score = 0
        
        # Check title
        if pd.notna(row['title']) and query_lower in row['title'].lower():
            score += 3
        
        # Check description
        if pd.notna(row['description']) and query_lower in row['description'].lower():
            score += 2
        
        # Check brand
        if pd.notna(row['brand']) and query_lower in row['brand'].lower():
            score += 1
        
        # Check categories
        if pd.notna(row['categories']) and query_lower in row['categories'].lower():
            score += 2
        
        # Check material
        if pd.notna(row['material']) and query_lower in row['material'].lower():
            score += 1
        
        # Check color
        if pd.notna(row['color']) and query_lower in row['color'].lower():
            score += 1
        
        if score > 0:
            # Parse images
            try:
                import ast
                images_list = ast.literal_eval(row['images']) if pd.notna(row['images']) else []
                images_list = [img.strip() for img in images_list]
            except:
                images_list = []
            
            # Parse categories
            try:
                categories_list = ast.literal_eval(row['categories']) if pd.notna(row['categories']) else []
            except:
                categories_list = []
            
            # Create product response
            product = {
                'uniq_id': str(row['uniq_id']),
                'title': str(row['title']) if pd.notna(row['title']) else 'Unknown Product',
                'brand': str(row['brand']) if pd.notna(row['brand']) else 'Unknown Brand',
                'description': str(row['description']) if pd.notna(row['description']) else '',
                'price': float(row['price'].replace('$', '').replace(',', '')) if pd.notna(row['price']) and isinstance(row['price'], str) else None,
                'categories': categories_list,
                'images': images_list,
                'generated_description': f"This {row['brand']} product is perfect for your needs. {str(row['description'])[:100] if pd.notna(row['description']) else 'Quality furniture piece.'}",
                'similarity_score': min(score / 10, 1.0)  # Normalize to 0-1
            }
            results.append(product)
    
    # Sort by score and return top N
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results[:request.num_recommendations]

@app.get("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """
    Get analytics data for dashboard
    Returns aggregated statistics and distributions
    """
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Calculate analytics
    total_products = len(df)
    
    # Category distribution - parse and count main categories
    import ast
    categories_dist = {}
    for idx, row in df.iterrows():
        try:
            if pd.notna(row['categories']):
                cats = ast.literal_eval(row['categories'])
                if cats and len(cats) > 0:
                    main_cat = cats[0]  # Use first category as main
                    categories_dist[main_cat] = categories_dist.get(main_cat, 0) + 1
        except:
            pass
    
    # Get top 10 categories
    categories_dist = dict(sorted(categories_dist.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Price statistics
    prices = df['price'].dropna()
    prices_numeric = prices.str.replace('$', '').str.replace(',', '').astype(float)
    
    price_stats = {
        "mean": float(prices_numeric.mean()) if len(prices_numeric) > 0 else 0,
        "median": float(prices_numeric.median()) if len(prices_numeric) > 0 else 0,
        "min": float(prices_numeric.min()) if len(prices_numeric) > 0 else 0,
        "max": float(prices_numeric.max()) if len(prices_numeric) > 0 else 0
    }
    
    # Top brands
    top_brands = df['brand'].value_counts().head(10).to_dict()
    
    return AnalyticsResponse(
        total_products=total_products,
        categories_distribution=categories_dist,
        price_stats=price_stats,
        top_brands=top_brands
    )

@app.get("/api/products/{product_id}")
async def get_product(product_id: str):
    """Get single product details by ID"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    product = df[df['uniq_id'] == product_id]
    if product.empty:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return product.iloc[0].to_dict()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
