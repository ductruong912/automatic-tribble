<#
.SYNOPSIS
    Khởi chạy llama-server với model Qwen3-8B GGUF Q4_K_M.
#>

param (
    [int]$Port = 8080,
    [string]$HostAddress = "127.0.0.1",
    [int]$CtxSize = 8192,
    [int]$GpuLayers = 99
)

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$LlamaServer = Join-Path $ProjectRoot "bin\llama.cpp\llama-server.exe"
$ModelPath = Join-Path $ProjectRoot "data\models\llm\Qwen3-8B.Q4_K_M.gguf"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   KHOI DONG LOCAL LLM SERVER (LLAMA.CPP - QWEN3-8B)        " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (-not (Test-Path $LlamaServer)) {
    Write-Host "❌ Không tìm thấy llama-server.exe tại: $LlamaServer" -ForegroundColor Red
    Write-Host "Vui lòng chạy trước: .\scripts\setup_llama_and_model.ps1" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $ModelPath)) {
    Write-Host "❌ Không tìm thấy model tại: $ModelPath" -ForegroundColor Red
    Write-Host "Vui lòng chạy trước: .\scripts\setup_llama_and_model.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "• Host       : $HostAddress"
Write-Host "• Port       : $Port"
Write-Host "• Context    : $CtxSize tokens"
Write-Host "• GPU Layers : $GpuLayers"
Write-Host "• Model      : $ModelPath"
Write-Host "• Endpoint   : http://${HostAddress}:${Port}/v1" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Đang khởi động server... (Nhấn Ctrl+C để dừng)" -ForegroundColor Yellow

& "$LlamaServer" -m "$ModelPath" -c $CtxSize --port $Port --host $HostAddress -ngl $GpuLayers -np 1 -fa 1 --alias qwen3-8b --alias default --alias gpt-4.1-mini --alias gpt-4o-mini
