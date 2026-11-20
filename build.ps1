# Builds dist\IVIM-DKI\IVIM-DKI.exe (Windows, CPU) with PyInstaller.
# Usage (from this folder):  powershell -ExecutionPolicy Bypass -File build.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    py -3 -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt pyinstaller
}

# tkinter inside a virtualenv may not locate the base installation's Tcl/Tk; point it there so PyInstaller bundles it
$base = & .\.venv\Scripts\python.exe -c "import sys; print(sys.base_prefix)"
$env:TCL_LIBRARY = Join-Path $base 'tcl\tcl8.6'
$env:TK_LIBRARY = Join-Path $base 'tcl\tk8.6'
& .\.venv\Scripts\python.exe -c "import tkinter; tkinter.Tcl(); print('Tcl/Tk found')"

& .\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --windowed --name IVIM-DKI `
    --paths src --collect-all jax --collect-all jaxlib `
    --hidden-import matplotlib.backends.backend_tkagg `
    src\app.py

Copy-Item -Recurse -Force test_data dist\IVIM-DKI\test_data
Copy-Item -Force README.md, LICENSE dist\IVIM-DKI\
Write-Host "Built dist\IVIM-DKI\IVIM-DKI.exe"
