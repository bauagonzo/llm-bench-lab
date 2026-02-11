# run-scaling-test.ps1 -Run a single model through Vulkan + CUDA scaling test
# Usage: .\run-scaling-test.ps1 -ModelPath <path.gguf> -OutputName <name> [-Mode both|vulkan|cuda]
param(
    [Parameter(Mandatory)][string]$ModelPath,
    [Parameter(Mandatory)][string]$OutputName,
    [ValidateSet("both","vulkan","cuda")][string]$Mode = "both",
    [string]$RunName = ""
)

$ErrorActionPreference = "Stop"

# --- Configuration (edit these paths) ---
$BenchDir    = "$PSScriptRoot\..\..\localscore-bench"
$VulkanBench = "$BenchDir\backends\vulkan\llama-bench.exe"
$CudaBench   = "$BenchDir\backends\cuda\llama-bench.exe"
$DateStr     = Get-Date -Format "yyyy-MM-dd"
if ($RunName -eq "") { $RunName = "run1" }
$ResultsDir  = "$PSScriptRoot\..\results\psyche-win\${DateStr}_${RunName}_${Mode}\scaling-test"
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

# Detect GPU index -on Windows with single NVIDIA GPU, usually index 0
# Vulkan may differ if integrated GPU exists. Check with: llama-bench.exe --list-devices
$VulkanGpuIndex = 1
$CudaGpuIndex   = 0

function Test-GpuHealth {
    <# Returns $true if the NVIDIA GPU is healthy, $false if lost/crashed #>
    try {
        $smiOut = & nvidia-smi --query-gpu=gpu_name,temperature.gpu,power.draw --format=csv,noheader 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: nvidia-smi failed (exit code $LASTEXITCODE)" -ForegroundColor Red
            Write-Host "  Output: $smiOut" -ForegroundColor Red
            return $false
        }
        if ($smiOut -match "GPU is lost|Unknown Error|ERR!") {
            Write-Host "ERROR: GPU is in a bad state! Output: $smiOut" -ForegroundColor Red
            return $false
        }
        Write-Host "  GPU health OK: $($smiOut.Trim())" -ForegroundColor DarkGray
        return $true
    }
    catch {
        Write-Host "ERROR: nvidia-smi not found or crashed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Pre-flight GPU check
Write-Host "Checking GPU health before starting..." -ForegroundColor Cyan
if (!(Test-GpuHealth)) {
    Write-Host "FATAL: GPU is not healthy. Reboot the system and try again." -ForegroundColor Red
    exit 1
}

if ($Mode -ne "cuda") {
    if (!(Test-GpuHealth)) {
        Write-Host "FATAL: GPU lost before Vulkan run. Aborting." -ForegroundColor Red
        exit 1
    }
    Run-Bench -Backend "Vulkan" -BenchBin $VulkanBench -GpuIndex $VulkanGpuIndex -Suffix "vulkan"
}

if ($Mode -ne "vulkan") {
    if (!(Test-GpuHealth)) {
        Write-Host "FATAL: GPU lost before CUDA run. Aborting." -ForegroundColor Red
        exit 1
    }
    Run-Bench -Backend "CUDA 13.1" -BenchBin $CudaBench -GpuIndex $CudaGpuIndex -Suffix "cuda13"
}

Write-Host ""
Write-Host "Done: $OutputName" -ForegroundColor Green
