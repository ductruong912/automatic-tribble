<#
.SYNOPSIS
    Tự động tải llama.cpp binary cho Windows và model Qwen3-8B GGUF Q4_K_M từ Hugging Face.
.DESCRIPTION
    Script thực hiện:
    1. Thiết lập bảng mã UTF-8 để hiển thị font chữ tiếng Việt chuẩn xác trong PowerShell.
    2. Kiểm tra cấu hình phần cứng (NVIDIA GPU / CPU).
    3. Tải bản build mới nhất của llama.cpp giải nén vào bin\llama.cpp\
    4. Tải model Qwen3-8B Q4_K_M (~4.9GB) từ Hugging Face vào data\models\llm\ (hỗ trợ resume).
.EXAMPLE
    .\scripts\setup_llama_and_model.ps1
    .\scripts\setup_llama_and_model.ps1 -GpuMode CPU
#>

[CmdletBinding()]
param (
    [ValidateSet("Auto", "CUDA", "CPU")]
    [string]$GpuMode = "Auto",

    [string]$ModelUrl = "https://huggingface.co/mradermacher/Qwen3-8B-GGUF/resolve/main/Qwen3-8B.Q4_K_M.gguf",
    [string]$ModelFileName = "Qwen3-8B.Q4_K_M.gguf",

    [switch]$SkipModel,
    [switch]$SkipLlama
)

# Thiết lập UTF-8 cho console để không bị vỡ font tiếng Việt
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 > $null

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$BinDir = Join-Path $ProjectRoot "bin\llama.cpp"
$ModelDir = Join-Path $ProjectRoot "data\models\llm"
$ModelPath = Join-Path $ModelDir $ModelFileName
$LlamaServerExe = Join-Path $BinDir "llama-server.exe"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   SETUP LOCAL LLM (LLAMA.CPP + QWEN3-8B GGUF Q4_K_M)       " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Thư mục dự án: $ProjectRoot"
Write-Host ""

# Đảm bảo các thư mục tồn tại
if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
}
if (-not (Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
}

# =============================================================================
# 1. TẢI VÀ GIẢI NÉN LLAMA.CPP
# =============================================================================
if (-not $SkipLlama) {
    if (Test-Path $LlamaServerExe) {
        Write-Host " [OK] Đã tìm thấy llama-server.exe tại: $LlamaServerExe" -ForegroundColor Green
    } else {
        Write-Host "--- Bước 1: Chuẩn bị llama.cpp ---" -ForegroundColor Yellow

        # Kiểm tra GPU
        $hasNvidia = $false
        if ($GpuMode -eq "CUDA" -or $GpuMode -eq "Auto") {
            try {
                $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
                if ($gpus -match "NVIDIA") {
                    $hasNvidia = $true
                    $gpuName = ($gpus -match "NVIDIA") -join ", "
                    Write-Host " Phát hiện GPU: $gpuName" -ForegroundColor Green
                }
            } catch {}
        }

        Write-Host " Đang tìm bản build llama.cpp tối ưu nhất cho Windows..." -ForegroundColor Cyan
        
        # Danh sách download fallback mặc định ổn định
        $tag = "b10997"
        $downloadItems = @()

        try {
            $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5" -Headers @{"User-Agent"="PowerShell"} -TimeoutSec 10
            # Lọc bản release thực sự (bNNNN) có assets Windows
            $buildRelease = $releases | Where-Object { $_.tag_name -match "^b\d+" -and ($_.assets | Where-Object { $_.name -match "win" }) } | Select-Object -First 1
            if ($buildRelease) {
                $tag = $buildRelease.tag_name
            }
        } catch {
            Write-Host " (Sử dụng tag phát hành mặc định: $tag)" -ForegroundColor Gray
        }

        Write-Host " Phiên bản llama.cpp được chọn: $tag" -ForegroundColor Green

        if ($hasNvidia -and $GpuMode -ne "CPU") {
            Write-Host " Tải bản GPU CUDA 12.4 tối ưu cho card NVIDIA..." -ForegroundColor Cyan
            $downloadItems += "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-win-cuda-12.4-x64.zip"
            $downloadItems += "https://github.com/ggml-org/llama.cpp/releases/download/$tag/cudart-llama-bin-win-cuda-12.4-x64.zip"
        } else {
            Write-Host " Tải bản CPU x64 phổ thông..." -ForegroundColor Cyan
            $downloadItems += "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-win-cpu-x64.zip"
        }

        foreach ($url in $downloadItems) {
            $fileName = [System.IO.Path]::GetFileName($url)
            $zipPath = Join-Path $ProjectRoot "bin\$fileName"
            Write-Host " Đang tải: $fileName..." -ForegroundColor Cyan

            if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
                & curl.exe -L -o "$zipPath" "$url" --progress-bar
            } else {
                Invoke-WebRequest -Uri $url -OutFile $zipPath
            }

            if (Test-Path $zipPath) {
                Write-Host " Đang giải nén $fileName..." -ForegroundColor Cyan
                Expand-Archive -Path $zipPath -DestinationPath $BinDir -Force
                Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
            }
        }

        if (Test-Path $LlamaServerExe) {
            Write-Host " [OK] Cài đặt llama.cpp thành công!" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Chưa tìm thấy llama-server.exe. Vui lòng kiểm tra lại kết nối mạng khi tải." -ForegroundColor Red
        }
    }
}

Write-Host ""

# =============================================================================
# 2. TẢI MODEL QWEN3-8B GGUF Q4_K_M (~4.9 GB)
# =============================================================================
if (-not $SkipModel) {
    Write-Host "--- Bước 2: Chuẩn bị Model Qwen3-8B GGUF Q4_K_M ---" -ForegroundColor Yellow
    
    $needDownload = $true
    if (Test-Path $ModelPath) {
        $fileSize = (Get-Item $ModelPath).Length
        # 4.0 GB = 4294967296 bytes
        if ($fileSize -gt 4294967296) {
            $sizeGB = [math]::Round($fileSize / 1GB, 2)
            Write-Host " [OK] Đã có sẵn model: $ModelPath ($sizeGB GB)" -ForegroundColor Green
            $needDownload = $false
        } else {
            Write-Host "⚠️ File model chưa tải hoàn tất ($([math]::Round($fileSize / 1MB, 2)) MB). Sẽ tiếp tục tải..." -ForegroundColor Yellow
        }
    }

    if ($needDownload) {
        Write-Host " Đang tải model từ Hugging Face:" -ForegroundColor Cyan
        Write-Host " URL : $ModelUrl"
        Write-Host " Đích: $ModelPath"
        Write-Host " Dung lượng: ~4.9 GB (Có hỗ trợ Resume tiếp tục nếu rớt mạng)" -ForegroundColor Yellow
        Write-Host ""

        if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
            & curl.exe -L -C - -o "$ModelPath" "$ModelUrl" --progress-bar
        } else {
            Start-BitsTransfer -Source $ModelUrl -Destination $ModelPath -DisplayName "Downloading Qwen3-8B GGUF"
        }

        if (Test-Path $ModelPath) {
            $finalSize = (Get-Item $ModelPath).Length
            Write-Host " [OK] Tải model hoàn tất! Dung lượng: $([math]::Round($finalSize / 1GB, 2)) GB" -ForegroundColor Green
        } else {
            Write-Host "❌ Tải model thất bại. Vui lòng kiểm tra lại kết nối mạng." -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "                      SETUP HOÀN TẤT!                       " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Tiếp theo bạn có thể làm:"
Write-Host " 1. Khởi động server local bằng:"
Write-Host "    .\scripts\run_local_llm.bat" -ForegroundColor Cyan
Write-Host ""
Write-Host " 2. Kiểm tra kết nối và tốc độ sinh token bằng:"
Write-Host "    python .\scripts\test_llm_connection.py" -ForegroundColor Cyan
Write-Host ""
Write-Host " 3. Cấu hình trong file .env để chuyển dự án sang dùng Local:"
Write-Host "    OPENAI_BASE_URL=http://127.0.0.1:8080/v1" -ForegroundColor Yellow
Write-Host "    OPENAI_MODEL=qwen3-8b" -ForegroundColor Yellow
Write-Host "    OPENAI_API_KEY=llama.cpp" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Green
