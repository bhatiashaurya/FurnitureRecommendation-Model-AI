"""
Lightweight Furniture Product Recommendation API for Vercel
============================================================
Optimized FastAPI backend with minimal dependencies for serverless deployment.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from pathlib import Path

app = FastAPI(
    title="Furniture Recommendation API",
    description="AI-powered furniture product recommendations",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class RecommendationRequest(BaseModel):
    query: str
    num_recommendations: int = 5

class AnalyticsResponse(BaseModel):
    total_products: int
    categories_distribution: dict
    top_brands: dict
    price_range: dict

# Load data once at startup
products_data = None

def load_data():
    """Load products data - try Python data file first, then CSV"""
    global products_data
    if products_data is not None:
        return products_data
    
    # Method 1: Try loading from Python data file (most reliable for Vercel)
    try:
        from products_data import PRODUCTS_DATA
        products_data = PRODUCTS_DATA
        print(f"✓ Loaded {len(products_data)} products from products_data.py")
        return products_data
    except ImportError as e:
        print(f"⚠ products_data.py not found: {e}")
    except Exception as e:
        print(f"⚠ Error loading from products_data.py: {e}")
    
    # Method 2: Fallback to CSV file
    try:
        # Try multiple possible paths for the CSV file
        possible_paths = [
            Path(__file__).parent / "intern_data_ikarus.csv",  # Same directory (Vercel)
            Path(__file__).parent.parent / "intern_data_ikarus.csv",  # Parent directory (local)
            Path("intern_data_ikarus.csv"),  # Current working directory
            Path("/var/task/intern_data_ikarus.csv"),  # Vercel serverless path
            Path("/var/task/backend/intern_data_ikarus.csv"),  # Vercel backend path
        ]
        
        csv_path = None
        for path in possible_paths:
            if path.exists():
                csv_path = path
                print(f"✓ Found CSV at: {csv_path}")
                break
        
        if csv_path is None:
            print(f"❌ CSV file not found. Tried paths: {[str(p) for p in possible_paths]}")
            return []
        
        print(f"Loading CSV from: {csv_path}")
        
        # Use proper CSV parsing
        import csv
        products = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                products.append(dict(row))
        
        products_data = products
        print(f"✓ Successfully loaded {len(products)} products")
        return products
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return []

@app.on_event("startup")
async def startup_event():
    """Initialize data on startup"""
    load_data()
    print(f"✓ Loaded {len(products_data) if products_data else 0} products")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Furniture Recommendation API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "recommend": "/api/recommend (POST)",
            "analytics": "/api/analytics"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint with detailed diagnostics"""
    products = load_data()
    
    # Check if products_data.py exists
    import os
    from pathlib import Path
    
    diagnostics = {
        "status": "healthy" if products and len(products) > 0 else "unhealthy",
        "products_loaded": len(products) if products else 0,
        "data_source": "products_data.py" if products_data else "none",
        "current_dir": str(Path.cwd()),
        "file_dir": str(Path(__file__).parent),
        "files_in_dir": [f for f in os.listdir(Path(__file__).parent) if f.endswith(('.py', '.csv'))],
    }
    
    if products and len(products) > 0:
        diagnostics["sample_product"] = {
            "title": products[0].get('title', 'N/A'),
            "brand": products[0].get('brand', 'N/A')
        }
    
    return diagnostics

@app.post("/api/recommend")
async def recommend_products(request: RecommendationRequest):
    """
    Recommend products based on query.
    Uses simple keyword matching without heavy ML dependencies.
    """
    products = load_data()
    
    if not products:
        raise HTTPException(status_code=500, detail="Products data not loaded")
    
    query_lower = request.query.lower()
    query_words = query_lower.split()
    
    # Score products based on keyword matching
    scored_products = []
    for product in products:
        score = 0
        
        # Check title
        title = product.get('title', '').lower()
        if any(word in title for word in query_words):
            score += 3
        
        # Check description
        description = product.get('description', '').lower()
        if description and any(word in description for word in query_words):
            score += 2
        
        # Check brand
        brand = product.get('brand', '').lower()
        if any(word in brand for word in query_words):
            score += 1
        
        # Check categories
        categories = product.get('categories', '').lower()
        if any(word in categories for word in query_words):
            score += 2
        
        # Check material
        material = product.get('material', '').lower()
        if material and any(word in material for word in query_words):
            score += 1
        
        # Check color
        color = product.get('color', '').lower()
        if color and any(word in color for word in query_words):
            score += 1
        
        if score > 0:
            product_copy = product.copy()
            product_copy['score'] = score
            scored_products.append(product_copy)
    
    # Sort by score and return top N
    scored_products.sort(key=lambda x: x['score'], reverse=True)
    top_products = scored_products[:request.num_recommendations]
    
    # Clean up response
    for product in top_products:
        product.pop('score', None)
    
    return top_products

@app.get("/api/analytics")
async def get_analytics():
    """
    Get analytics data for dashboard.
    Returns aggregated statistics about products.
    """
    products = load_data()
    
    if not products:
        raise HTTPException(status_code=500, detail="Products data not loaded")
    
    # Count categories
    categories_count = {}
    brand_count = {}
    prices = []
    
    for product in products:
        # Categories
        categories_str = product.get('categories', '')
        if categories_str:
            # Simple parsing of category list
            cats = categories_str.replace('[', '').replace(']', '').replace("'", "").split(',')
            for cat in cats:
                cat = cat.strip()
                if cat:
                    categories_count[cat] = categories_count.get(cat, 0) + 1
        
        # Brands
        brand = product.get('brand', '').strip()
        if brand:
            brand_count[brand] = brand_count.get(brand, 0) + 1
        
        # Prices
        price_str = product.get('price', '')
        if price_str:
            try:
                # Extract numeric value
                price_clean = price_str.replace('$', '').replace(',', '').strip()
                if price_clean:
                    price = float(price_clean)
                    prices.append(price)
            except:
                pass
    
    # Get top categories and brands
    top_categories = dict(sorted(categories_count.items(), key=lambda x: x[1], reverse=True)[:10])
    top_brands = dict(sorted(brand_count.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Price statistics
    price_stats = {}
    if prices:
        prices.sort()
        price_stats = {
            "min": min(prices),
            "max": max(prices),
            "average": sum(prices) / len(prices),
            "median": prices[len(prices) // 2]
        }
    
    return {
        "total_products": len(products),
        "categories_distribution": top_categories,
        "top_brands": top_brands,
        "price_range": price_stats
    }

# Vercel serverless function handler
app = app
