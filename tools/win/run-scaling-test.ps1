# run-scaling-test.ps1 — Run a single model through Vulkan + CUDA scaling test
# Usage: .\run-scaling-test.ps1 -ModelPath <path.gguf> -OutputName <name> [-Mode both|vulkan|cuda]
param(
    [Parameter(Mandatory)][string]$ModelPath,
    [Parameter(Mandatory)][string]$OutputName,
    [ValidateSet("both","vulkan","cuda")][string]$Mode = "both"
)

$ErrorActionPreference = "Stop"

# --- Configuration (edit these paths) ---
$BenchDir    = "$PSScriptRoot\..\..\localscore-bench"
$VulkanBench = "$BenchDir\backends\vulkan\llama-bench.exe"
$CudaBench   = "$BenchDir\backends\cuda\llama-bench.exe"
$ResultsDir  = "$PSScriptRoot\..\results\psyche-win\2026-02-11\scaling-test"
# -----------------------------------------

if (!(Test-Path $ResultsDir)) { New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null }

function Run-Bench {
    param(
        [string]$Backend,
        [string]$BenchBin,
        [int]$GpuIndex,
        [string]$Suffix
    )

    $outJson = Join-Path $ResultsDir "$OutputName-$Suffix.json"
    $outCsv  = Join-Path $ResultsDir "$OutputName-$Suffix.csv"

    Write-Host ""
    Write-Host ("=" * 50) -ForegroundColor Cyan
    Write-Host "  $OutputName - $Backend" -ForegroundColor Cyan
    Write-Host ("=" * 50) -ForegroundColor Cyan

    if (!(Test-Path $BenchBin)) {
        Write-Host "ERROR: llama-bench not found at $BenchBin" -ForegroundColor Red
        return $false
    }

    Push-Location $BenchDir
    try {
        python main.py `
            -m $ModelPath `
            --llama-bench $BenchBin `
            -i $GpuIndex `
            --quick `
            -n `
            --save-json $outJson `
            --monitor-gpu $outCsv `
            --monitor-interval 200 `
            -o console

        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARNING: Benchmark exited with code $LASTEXITCODE" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
    finally {
        Pop-Location
    }

    Write-Host "  OK: $outJson" -ForegroundColor Green
    return $true
}

# Detect GPU index — on Windows with single NVIDIA GPU, usually index 0
# Vulkan may differ if integrated GPU exists. Check with: llama-bench.exe --list-devices
$VulkanGpuIndex = 0
$CudaGpuIndex   = 0

if ($Mode -ne "cuda") {
    Run-Bench -Backend "Vulkan" -BenchBin $VulkanBench -GpuIndex $VulkanGpuIndex -Suffix "vulkan"
}

if ($Mode -ne "vulkan") {
    Run-Bench -Backend "CUDA 13.1" -BenchBin $CudaBench -GpuIndex $CudaGpuIndex -Suffix "cuda13"
}

Write-Host ""
Write-Host "Done: $OutputName" -ForegroundColor Green
