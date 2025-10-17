# Vercel Deployment Guide

This guide will help you deploy your Furniture Recommendation AI application to Vercel.

## Overview

Your application consists of:
- **Frontend**: React + Vite (static site)
- **Backend**: FastAPI (serverless functions)

We'll deploy them as separate projects on Vercel for optimal performance.

---

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI** (optional but recommended):
   ```powershell
   npm install -g vercel
   ```
3. **GitHub Repository**: Your code is already pushed to GitHub

---

## Deployment Option 1: Deploy via Vercel Dashboard (Recommended)

### Step 1: Deploy the Backend (FastAPI)

1. Go to [vercel.com/new](https://vercel.com/new)
2. Click **Import Project** → **Import Git Repository**
3. Select your repository: `bhatiashaurya/FurnitureRecommendation-Model-AI`
4. Configure the project:
   - **Project Name**: `furniture-recommendation-backend`
   - **Framework Preset**: `Other`
   - **Root Directory**: `backend`
   - **Build Command**: Leave empty
   - **Output Directory**: Leave empty

5. **Environment Variables** (Add these in Settings):
   ```
   OPENAI_API_KEY=your_openai_key_here
   PINECONE_API_KEY=your_pinecone_key_here
   PINECONE_ENVIRONMENT=your_pinecone_env_here
   ```

6. Click **Deploy**

7. After deployment, note your backend URL (e.g., `https://furniture-recommendation-backend.vercel.app`)

### Step 2: Deploy the Frontend (React)

1. Go to [vercel.com/new](https://vercel.com/new) again
2. Click **Import Project** → **Import Git Repository**
3. Select your repository again: `bhatiashaurya/FurnitureRecommendation-Model-AI`
4. Configure the project:
   - **Project Name**: `furniture-recommendation-frontend`
   - **Framework Preset**: `Vite`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`

5. **Environment Variables**:
   ```
   VITE_API_URL=https://your-backend-url.vercel.app
   ```
   Replace with your actual backend URL from Step 1.

6. Click **Deploy**

### Step 3: Update Frontend API Calls

After getting your backend URL, you need to update the frontend to use the production API:

1. Create a `.env.production` file in the `frontend` directory:
   ```env
   VITE_API_URL=https://your-backend-url.vercel.app
   ```

2. Update `frontend/src/pages/RecommendationPage.jsx` and `AnalyticsPage.jsx` to use the environment variable:
   ```javascript
   const API_URL = import.meta.env.VITE_API_URL || '';
   
   // In your axios calls:
   axios.post(`${API_URL}/api/recommend`, ...)
   ```

3. Redeploy the frontend (Vercel will auto-redeploy on git push)

---

## Deployment Option 2: Deploy via Vercel CLI

### Deploy Backend

```powershell
cd D:\Projects\Ikarus3d_AI\backend
vercel

# Follow the prompts:
# - Set up and deploy? Yes
# - Which scope? Your account
# - Link to existing project? No
# - Project name? furniture-recommendation-backend
# - Directory? ./
# - Override settings? No
```

### Deploy Frontend

```powershell
cd D:\Projects\Ikarus3d_AI\frontend
vercel

# Follow the prompts:
# - Set up and deploy? Yes
# - Which scope? Your account
# - Link to existing project? No
# - Project name? furniture-recommendation-frontend
# - Directory? ./
# - Override settings? Yes
# - Build Command? npm run build
# - Output Directory? dist
# - Development Command? npm run dev
```

---

## Important Notes

### 1. **Serverless Function Limitations**

Vercel serverless functions have limitations:
- **Execution timeout**: 10 seconds (Hobby plan) / 60 seconds (Pro plan)
- **Memory**: 1024 MB max
- **Payload size**: 4.5 MB request / 4.5 MB response

If your ML models are too large or slow, consider these alternatives:
- Use external ML API services (OpenAI, Hugging Face)
- Deploy backend on Railway, Render, or AWS Lambda with higher limits
- Use Vercel Edge Functions for lighter operations

### 2. **CSV Data File**

The `intern_data_ikarus.csv` file will be included in the deployment. Make sure it's not too large (Vercel has a 50MB limit per file).

### 3. **Environment Variables**

For production, you'll need to set:
- `OPENAI_API_KEY` - For GenAI descriptions
- `PINECONE_API_KEY` - For vector database
- `PINECONE_ENVIRONMENT` - Your Pinecone environment

### 4. **CORS Configuration**

Update your `backend/main.py` to allow your frontend domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend-domain.vercel.app"  # Add this
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Alternative: Deploy as Monorepo (Single Project)

If you want to deploy both frontend and backend in one project:

1. Use the root `vercel.json` configuration provided
2. Deploy from the root directory
3. Vercel will handle both builds automatically

However, this approach is less flexible and harder to scale.

---

## Recommended: Split Deployment + Alternative Backend Hosting

For production-grade deployment, I recommend:

1. **Frontend on Vercel**: Fast, free, great CDN
2. **Backend on Railway/Render**: Better for FastAPI, no serverless limitations

### Deploy Backend on Railway (Alternative)

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click **New Project** → **Deploy from GitHub repo**
4. Select your repository
5. Set root directory to `backend`
6. Add environment variables
7. Railway will auto-detect Python and deploy

Railway gives you:
- Always-on server (no cold starts)
- More memory and longer timeouts
- Better for ML workloads

---

## Testing Your Deployment

After deployment:

1. Test backend health:
   ```
   https://your-backend.vercel.app/api/health
   ```

2. Test recommendations:
   ```
   POST https://your-backend.vercel.app/api/recommend
   Body: {"query": "modern sofa"}
   ```

3. Visit your frontend:
   ```
   https://your-frontend.vercel.app
   ```

---

## Troubleshooting

### Issue: "Module not found" errors
**Solution**: Make sure `backend/requirements.txt` includes all dependencies

### Issue: Cold starts are slow
**Solution**: Vercel serverless functions have cold starts. Consider Railway/Render for backend.

### Issue: API calls fail
**Solution**: Check CORS settings and ensure frontend has correct backend URL

### Issue: Build fails
**Solution**: Check build logs in Vercel dashboard, ensure all dependencies are in package.json

---

## Post-Deployment Checklist

- [ ] Backend health endpoint working
- [ ] Frontend loads correctly
- [ ] Recommendations API working
- [ ] Analytics dashboard displaying data
- [ ] Environment variables set correctly
- [ ] CORS configured for production domain
- [ ] Custom domain added (optional)
- [ ] Analytics/monitoring enabled

---

## Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Deploy FastAPI on Vercel](https://vercel.com/guides/python-fastapi-on-vercel)
- [Deploy Vite on Vercel](https://vercel.com/docs/frameworks/vite)
- [Vercel CLI Reference](https://vercel.com/docs/cli)

---

## Questions?

If you encounter issues, check:
1. Vercel deployment logs
2. Browser console for frontend errors
3. Network tab for API call failures
4. Vercel function logs for backend errors

Good luck with your deployment! 🚀
