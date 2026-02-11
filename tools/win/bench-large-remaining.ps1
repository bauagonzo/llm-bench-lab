$ErrorActionPreference = "Stop"
$ScriptDir = "D:\llm-bench-lab\tools\win"
$ModelsDir = "D:\models"

$allLarge = @(
    @{ Name="gemma-3-12b";      File="gemma-3-12b-it-Q4_K_M.gguf" }
    @{ Name="mistral-nemo-12b"; File="Mistral-Nemo-Instruct-2407-Q4_K_M.gguf" }
    @{ Name="qwen3-32b";        File="Qwen3-32B-Q4_K_M.gguf" }
    @{ Name="llama-3.3-70b";    File="Llama-3.3-70B-Instruct-Q4_K_M.gguf" }
)

# run4: only the remaining 2 models
$run4remaining = @(
    @{ Name="qwen3-32b";   File="Qwen3-32B-Q4_K_M.gguf" }
    @{ Name="llama-3.3-70b"; File="Llama-3.3-70B-Instruct-Q4_K_M.gguf" }
)

$runs = @(
    @{ RunName="run4"; Models=$run4remaining }
    @{ RunName="run5"; Models=$allLarge }
    @{ RunName="run6"; Models=$allLarge }
)

foreach ($run in $runs) {
    Write-Host "`n`n########################################" -ForegroundColor Yellow
    Write-Host "  STARTING $($run.RunName) (large models)" -ForegroundColor Yellow
    Write-Host "########################################`n" -ForegroundColor Yellow

    foreach ($m in $run.Models) {
        # GPU health check
        $smiOut = & nvidia-smi --query-gpu=gpu_name,temperature.gpu,power.draw --format=csv,noheader 2>&1
        if ($LASTEXITCODE -ne 0 -or "$smiOut" -match "GPU is lost|Unknown Error|ERR!") {
            Write-Host "FATAL: GPU dead before $($m.Name) in $($run.RunName). Aborting." -ForegroundColor Red
            exit 1
        }
        Write-Host "GPU OK: $($smiOut.Trim())" -ForegroundColor DarkGray

        $modelPath = Join-Path $ModelsDir $m.File
        & "$ScriptDir\run-scaling-test.ps1" -ModelPath $modelPath -OutputName $m.Name -Mode cuda -RunName $run.RunName
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FATAL: $($m.Name) failed in $($run.RunName). Aborting." -ForegroundColor Red
            exit 1
        }

        # Cooldown between models
        Write-Host "  Cooling down 30s..." -ForegroundColor DarkGray
        Start-Sleep -Seconds 30
    }
    Write-Host "`nCompleted $($run.RunName) successfully!" -ForegroundColor Green
}

Write-Host "`n`nALL REMAINING LARGE RUNS COMPLETE!" -ForegroundColor Green
