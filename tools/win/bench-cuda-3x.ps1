$ErrorActionPreference = "Stop"
$ScriptDir = "D:\llm-bench-lab\tools\win"
$ModelsDir = "D:\models"

$models = @(
    @{ Name="gemma-3-1b";      File="gemma-3-1b-it-Q4_K_M.gguf" }
    @{ Name="llama-3.2-1b";    File="Llama-3.2-1B-Instruct-Q4_K_M.gguf" }
    @{ Name="phi-4-mini-3.8b"; File="Phi-4-mini-instruct-Q4_K_M.gguf" }
    @{ Name="ministral-8b";    File="Ministral-8B-Instruct-2410-Q4_K_M.gguf" }
    @{ Name="gemma-3-12b";     File="gemma-3-12b-it-Q4_K_M.gguf" }
    @{ Name="mistral-nemo-12b"; File="Mistral-Nemo-Instruct-2407-Q4_K_M.gguf" }
    @{ Name="qwen3-32b";       File="Qwen3-32B-Q4_K_M.gguf" }
)

foreach ($run in @("run7","run8","run9")) {
    Write-Host "`n`n########################################" -ForegroundColor Yellow
    Write-Host "  STARTING $run (CUDA, driver 582.32)" -ForegroundColor Yellow
    Write-Host "########################################`n" -ForegroundColor Yellow

    foreach ($m in $models) {
        $smiOut = & nvidia-smi --query-gpu=gpu_name,temperature.gpu,power.draw --format=csv,noheader 2>&1
        if ($LASTEXITCODE -ne 0 -or "$smiOut" -match "GPU is lost|Unknown Error|ERR!") {
            Write-Host "FATAL: GPU dead before $($m.Name) in $run. Aborting." -ForegroundColor Red
            exit 1
        }
        Write-Host "GPU OK: $($smiOut.Trim())" -ForegroundColor DarkGray

        $modelPath = Join-Path $ModelsDir $m.File
        & "$ScriptDir\run-scaling-test.ps1" -ModelPath $modelPath -OutputName $m.Name -Mode cuda -RunName $run
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FATAL: $($m.Name) failed in $run. Aborting." -ForegroundColor Red
            exit 1
        }

        Start-Sleep -Seconds 30
        Write-Host "  Cooldown done." -ForegroundColor DarkGray
    }
    Write-Host "`nCompleted $run successfully!" -ForegroundColor Green
}

Write-Host "`n`nALL 3 CUDA RUNS COMPLETE!" -ForegroundColor Green
