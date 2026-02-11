# setup.ps1 — Verify Windows benchmark environment
# Run this first to check everything is ready.

$ErrorActionPreference = "Continue"

Write-Host "=== Windows Benchmark Setup Check ===" -ForegroundColor Cyan
Write-Host ""

# 1. Python
Write-Host "[1/5] Python..." -NoNewline
try {
    $pyVer = python --version 2>&1
    Write-Host " OK ($pyVer)" -ForegroundColor Green
} catch {
    Write-Host " MISSING — install Python 3.10+ from python.org" -ForegroundColor Red
}

# 2. nvidia-smi
Write-Host "[2/5] nvidia-smi..." -NoNewline
try {
    $smi = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    Write-Host " OK ($smi)" -ForegroundColor Green
} catch {
    Write-Host " MISSING — NVIDIA drivers not installed?" -ForegroundColor Red
}

# 3. Check GPU device indices
Write-Host "[3/5] GPU devices..." -ForegroundColor White
nvidia-smi -L 2>&1 | ForEach-Object { Write-Host "       $_" }
Write-Host "       If multiple GPUs: check Vulkan/CUDA device index!" -ForegroundColor Yellow

# 4. llama-bench binaries
$benchDir = "$PSScriptRoot\..\..\localscore-bench"
Write-Host "[4/5] llama-bench binaries..." -ForegroundColor White

$vulkan = "$benchDir\backends\vulkan\llama-bench.exe"
$cuda   = "$benchDir\backends\cuda\llama-bench.exe"

if (Test-Path $vulkan) {
    Write-Host "       Vulkan: OK ($vulkan)" -ForegroundColor Green
} else {
    Write-Host "       Vulkan: MISSING — download from ggml-org/llama.cpp releases" -ForegroundColor Red
    Write-Host "       Place at: $vulkan" -ForegroundColor DarkGray
}

if (Test-Path $cuda) {
    Write-Host "       CUDA:   OK ($cuda)" -ForegroundColor Green
} else {
    Write-Host "       CUDA:   MISSING — download from ai-dock/llama.cpp-cuda releases" -ForegroundColor Red
    Write-Host "       Place at: $cuda" -ForegroundColor DarkGray
}

# 5. localscore-bench
Write-Host "[5/5] localscore-bench..." -NoNewline
if (Test-Path "$benchDir\main.py") {
    Write-Host " OK ($benchDir)" -ForegroundColor Green
} else {
    Write-Host " MISSING" -ForegroundColor Red
    Write-Host "       git clone https://github.com/bauagonzo/localscore-bench $benchDir" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "=== Directory Layout Expected ===" -ForegroundColor Cyan
Write-Host @"
llm-bench-lab\
  tools\win\          <-- you are here
  results\
localscore-bench\     <-- sibling directory
  main.py
  backends\
    vulkan\llama-bench.exe
    cuda\llama-bench.exe
"@

Write-Host ""
Write-Host "When ready: .\run-all-scaling.ps1 -ModelsDir D:\models" -ForegroundColor Green
