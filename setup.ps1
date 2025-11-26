# setup.ps1 - Force setup with Python 3.11

Write-Host "🔄 Removing old virtual environment (if exists)..."
if (Test-Path "venv") {
    Remove-Item -Recurse -Force venv
}

Write-Host "📦 Creating new virtual environment with Python 3.11..."
py -3.11 -m venv venv

Write-Host "✅ Activating virtual environment..."
& venv\Scripts\Activate.ps1

Write-Host "⚡ Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

Write-Host "📥 Installing project requirements..."
pip install -r requirements.txt

Write-Host "`n✅ Setup complete!"
Write-Host "Test with: python src/test_setup.py"
