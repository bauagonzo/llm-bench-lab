# run-all-scaling.ps1 — Run all scaling test models (smallest first)
# Usage: .\run-all-scaling.ps1 -ModelsDir D:\models [-Mode both|vulkan|cuda] [-StartFrom <number>]
param(
    [Parameter(Mandatory)][string]$ModelsDir,
    [ValidateSet("both","vulkan","cuda")][string]$Mode = "both",
    [int]$StartFrom = 1
)

$ErrorActionPreference = "Stop"

# Model list — ordered smallest to largest (crash-safe order)
$models = @(
    @{ Num=1;  Name="gemma-3-1b";       File="gemma-3-1b-it-Q4_K_M.gguf" }
    @{ Num=2;  Name="llama-3.2-1b";     File="Llama-3.2-1B-Instruct-Q4_K_M.gguf" }
    @{ Num=3;  Name="phi-4-mini-3.8b";  File="Phi-4-mini-instruct-Q4_K_M.gguf" }
    @{ Num=4;  Name="ministral-8b";     File="Ministral-8B-Instruct-2410-Q4_K_M.gguf" }
    @{ Num=5;  Name="gemma-3-12b";      File="gemma-3-12b-it-Q4_K_M.gguf" }
    @{ Num=6;  Name="mistral-nemo-12b"; File="Mistral-Nemo-Instruct-2407-Q4_K_M.gguf" }
    @{ Num=7;  Name="qwen3-32b";        File="Qwen3-32B-Q4_K_M.gguf" }
    @{ Num=8;  Name="llama-3.3-70b";    File="Llama-3.3-70B-Instruct-Q4_K_M.gguf" }
)

# Record system info
Write-Host "=== System Info ===" -ForegroundColor Cyan
Write-Host "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "OS: $([System.Environment]::OSVersion.VersionString)"
Write-Host "Windows Build: $((Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').DisplayVersion) ($((Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').CurrentBuild))"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
Write-Host ""

$failed  = @()
$skipped = @()
$passed  = @()

foreach ($m in $models) {
    if ($m.Num -lt $StartFrom) {
        Write-Host "Skipping $($m.Name) (before StartFrom=$StartFrom)" -ForegroundColor DarkGray
        continue
    }

    $modelPath = Join-Path $ModelsDir $m.File

    if (!(Test-Path $modelPath)) {
        Write-Host "SKIP: $($m.File) not found in $ModelsDir" -ForegroundColor Yellow
        $skipped += $m.Name
        continue
    }

    Write-Host ""
    Write-Host ("*" * 60) -ForegroundColor Magenta
    Write-Host "  [$($m.Num)/8] $($m.Name)" -ForegroundColor Magenta
    Write-Host ("*" * 60) -ForegroundColor Magenta

    try {
        & "$PSScriptRoot\run-scaling-test.ps1" -ModelPath $modelPath -OutputName $m.Name -Mode $Mode
        $passed += $m.Name
    }
    catch {
        Write-Host "FAILED: $($m.Name) — $($_.Exception.Message)" -ForegroundColor Red
        $failed += $m.Name
    }
}

# Summary
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "  SUMMARY" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "Passed:  $($passed.Count)/8 — $($passed -join ', ')" -ForegroundColor Green
if ($skipped.Count -gt 0) {
    Write-Host "Skipped: $($skipped.Count) — $($skipped -join ', ')" -ForegroundColor Yellow
}
if ($failed.Count -gt 0) {
    Write-Host "Failed:  $($failed.Count) — $($failed -join ', ')" -ForegroundColor Red
}
Write-Host ""
Write-Host "Results in: tools\..\results\psyche-win\2026-02-11\scaling-test\" -ForegroundColor Cyan
