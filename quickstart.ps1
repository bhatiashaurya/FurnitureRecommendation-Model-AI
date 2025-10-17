# Quick Start Script
# Run this script to quickly set up and test the project

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Furniture Recommendation System Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (Test-Path ".venv") {
    Write-Host "✓ Virtual environment found" -ForegroundColor Green
    .venv\Scripts\Activate.ps1
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Install basic packages
Write-Host ""
Write-Host "Installing core packages..." -ForegroundColor Yellow
pip install pandas numpy scikit-learn matplotlib seaborn plotly jupyter ipykernel -q

# Check if .env exists
if (Test-Path ".env") {
    Write-Host "✓ .env file found" -ForegroundColor Green
} else {
    Write-Host "⚠ .env file not found. Creating from template..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "⚠ Please edit .env with your API keys!" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "1. Edit .env file with your API keys" -ForegroundColor White
Write-Host "2. Install remaining packages:" -ForegroundColor White
Write-Host "   pip install -r requirements.txt" -ForegroundColor Gray
Write-Host "3. Run data analytics notebook:" -ForegroundColor White
Write-Host "   jupyter notebook data_analytics.ipynb" -ForegroundColor Gray
Write-Host "4. Run model training notebook:" -ForegroundColor White
Write-Host "   jupyter notebook notebooks/model_training.ipynb" -ForegroundColor Gray
Write-Host "5. Start backend:" -ForegroundColor White
Write-Host "   cd backend && python main.py" -ForegroundColor Gray
Write-Host "6. Start frontend:" -ForegroundColor White
Write-Host "   cd frontend && npm install && npm run dev" -ForegroundColor Gray
Write-Host ""
Write-Host "For detailed instructions, see SETUP_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
