# Monorepo Deployment Guide

## 🔄 Deploy Frontend + Backend Together (Monorepo)

If you want **one URL** for both frontend and backend, follow these steps:

### Current Situation:
- ✅ Backend already deployed at: `https://furniture-recommendation-model-ai-3a8c-al6n6fqp6.vercel.app`

### Two Approaches:

---

## Approach 1: Keep Separate Deployments (Recommended) ⭐

**This is the standard approach and what I recommend.**

### What you have now:
- Backend: `https://furniture-recommendation-model-ai-3a8c-al6n6fqp6.vercel.app`

### What to do:
1. Go to https://vercel.com/new
2. Import **same repository** again
3. Set **Root Directory**: `frontend`
4. Add environment variable:
   - `VITE_API_URL`: `https://furniture-recommendation-model-ai-3a8c-al6n6fqp6.vercel.app`
5. Deploy

### Result:
- Backend: `https://furniture-recommendation-model-ai-3a8c-al6n6fqp6.vercel.app` (existing)
- Frontend: `https://furniture-recommendation-frontend.vercel.app` (new)

**Advantages:**
- ✅ Both services independent
- ✅ Can update frontend without touching backend
- ✅ Better performance
- ✅ Easier to manage

---

## Approach 2: Redeploy as Single Project (Monorepo)

**If you want ONE URL for everything.**

### Step 1: Delete Current Backend Deployment

1. Go to Vercel Dashboard
2. Select your backend project: `furniture-recommendation-model-ai-3a8c`
3. Go to **Settings** → **General**
4. Scroll down and click **"Delete Project"**

### Step 2: Deploy from Root

1. Go to https://vercel.com/new
2. Import repository: `bhatiashaurya/FurnitureRecommendation-Model-AI`
3. **Important**: Leave **Root Directory** as `.` (root, not frontend or backend)
4. Framework: Select **"Other"**
5. Build settings:
   - Build Command: `cd frontend && npm install && npm run build`
   - Output Directory: `frontend/dist`
6. Deploy

### Step 3: Access Your App

After deployment, you'll have ONE URL:
- Frontend: `https://your-app.vercel.app/`
- Backend API: `https://your-app.vercel.app/api/recommend`
- Analytics: `https://your-app.vercel.app/api/analytics`

### Configuration Files Updated:
- ✅ `vercel.json` (root) - Routes configured for monorepo
- ✅ Both backend and frontend in one deployment

**Advantages:**
- ✅ Single URL for everything
- ✅ Simpler to share (one link)

**Disadvantages:**
- ⚠️ Larger deployment
- ⚠️ Must redeploy both if either changes
- ⚠️ Backend limitations still apply

---

## 🎯 My Recommendation

**Use Approach 1 (Separate Deployments)** because:

1. ✅ Your backend is already deployed and working
2. ✅ No need to delete and redeploy
3. ✅ Standard industry practice
4. ✅ Better scalability
5. ✅ Can update frontend independently

Just deploy the frontend as a **new project** pointing to your existing backend!

---

## 🚀 Quick Decision Guide

### Choose Separate Deployments if:
- You want to keep existing backend deployment
- You want flexibility to update independently
- You want best performance
- **This is what I recommend!**

### Choose Monorepo if:
- You want ONE single URL
- You don't mind redeploying both together
- You're okay with deleting current backend

---

## What Should You Do?

**I recommend Approach 1.** Just tell me:

**"Deploy frontend separately"** → I'll guide you through deploying frontend as new project

**"Redeploy as monorepo"** → I'll help you delete backend and redeploy everything together

Which approach do you prefer? 🤔
