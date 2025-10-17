# Quick Vercel Deployment Steps

## ✨ Fastest Way to Deploy (5 minutes)

### Step 1: Deploy Backend
1. Go to **https://vercel.com/new**
2. Click **Import Git Repository**
3. Select: `bhatiashaurya/FurnitureRecommendation-Model-AI`
4. Configure:
   - **Root Directory**: `backend`
   - Click **Deploy**
5. Copy your backend URL (e.g., `https://xxx.vercel.app`)

### Step 2: Update Frontend Configuration
1. Open `frontend/.env.production`
2. Replace with your backend URL:
   ```
   VITE_API_URL=https://your-backend-url.vercel.app
   ```
3. Commit and push:
   ```powershell
   git add frontend/.env.production
   git commit -m "Update production API URL"
   git push
   ```

### Step 3: Deploy Frontend
1. Go to **https://vercel.com/new** again
2. Import same repository
3. Configure:
   - **Root Directory**: `frontend`
   - **Framework**: Vite
   - Click **Deploy**
4. Done! 🎉

## 🔗 Your Live URLs
- **Frontend**: `https://your-frontend.vercel.app`
- **Backend**: `https://your-backend.vercel.app`

## 📖 Full Guide
See **VERCEL_DEPLOYMENT.md** for detailed instructions and troubleshooting.

## ⚡ Alternative: Use Deployment Script
```powershell
.\deploy-vercel.ps1
```

## 🚨 Important Notes
- Vercel free tier has limitations for ML workloads
- For production, consider Railway/Render for backend
- Add API keys in Vercel dashboard → Settings → Environment Variables
