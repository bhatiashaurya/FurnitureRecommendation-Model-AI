# Quick Deployment Script for Vercel
# Run this script to prepare your project for Vercel deployment

Write-Host "🚀 Preparing Furniture Recommendation AI for Vercel Deployment..." -ForegroundColor Cyan
Write-Host ""

# Check if Vercel CLI is installed
$vercelInstalled = Get-Command vercel -ErrorAction SilentlyContinue
if (-not $vercelInstalled) {
    Write-Host "⚠️  Vercel CLI not found. Installing..." -ForegroundColor Yellow
    npm install -g vercel
}

Write-Host "✅ Vercel CLI is ready" -ForegroundColor Green
Write-Host ""

# Build frontend locally to test
Write-Host "🔨 Building frontend..." -ForegroundColor Cyan
Set-Location frontend
npm run build
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Frontend build successful" -ForegroundColor Green
} else {
    Write-Host "❌ Frontend build failed" -ForegroundColor Red
    exit 1
}
Set-Location ..
Write-Host ""

# Display deployment instructions
Write-Host "📋 Deployment Instructions:" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""
Write-Host "Option 1: Deploy via Vercel Dashboard (Recommended)" -ForegroundColor Yellow
Write-Host "  1. Go to https://vercel.com/new" -ForegroundColor White
Write-Host "  2. Import your GitHub repository" -ForegroundColor White
Write-Host "  3. Follow VERCEL_DEPLOYMENT.md for detailed steps" -ForegroundColor White
Write-Host ""
Write-Host "Option 2: Deploy via CLI" -ForegroundColor Yellow
Write-Host "  Backend:  cd backend && vercel" -ForegroundColor White
Write-Host "  Frontend: cd frontend && vercel" -ForegroundColor White
Write-Host ""
Write-Host "📖 For complete instructions, see: VERCEL_DEPLOYMENT.md" -ForegroundColor Cyan
Write-Host ""

# Ask if user wants to deploy now
$deploy = Read-Host "Do you want to deploy the backend now? (y/n)"
if ($deploy -eq "y" -or $deploy -eq "Y") {
    Write-Host "🚀 Deploying backend..." -ForegroundColor Cyan
    Set-Location backend
    vercel --prod
    Set-Location ..
    
    Write-Host ""
    Write-Host "✅ Backend deployed! Now update the frontend with the backend URL." -ForegroundColor Green
    Write-Host "   Edit frontend/.env.production with your backend URL" -ForegroundColor Yellow
    Write-Host ""
    
    $deployFrontend = Read-Host "Deploy frontend now? (y/n)"
    if ($deployFrontend -eq "y" -or $deployFrontend -eq "Y") {
        Write-Host "🚀 Deploying frontend..." -ForegroundColor Cyan
        Set-Location frontend
        vercel --prod
        Set-Location ..
        Write-Host ""
        Write-Host "✅ Deployment complete! 🎉" -ForegroundColor Green
    }
} else {
    Write-Host "ℹ️  You can deploy later using the instructions above." -ForegroundColor Blue
}

Write-Host ""
Write-Host "✨ All done! Check VERCEL_DEPLOYMENT.md for more details." -ForegroundColor Green
