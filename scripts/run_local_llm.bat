@echo off
setlocal enabledelayedexpansion

title Local LLM Server (llama.cpp - Qwen3-8B)

set "PROJECT_ROOT=%~dp0.."
set "LLAMA_SERVER=%PROJECT_ROOT%\bin\llama.cpp\llama-server.exe"
set "MODEL_PATH=%PROJECT_ROOT%\data\models\llm\Qwen3-8B.Q4_K_M.gguf"
set "PORT=8080"
set "HOST=127.0.0.1"
set "CTX_SIZE=8192"

echo ============================================================
echo   KHOI DONG LOCAL LLM SERVER (LLAMA.CPP - QWEN3-8B)
echo ============================================================
echo.

if not exist "%LLAMA_SERVER%" (
    echo [LOI] Khong tim thay llama-server.exe tai:
    echo       %LLAMA_SERVER%
    echo.
    echo Vui long chay script cai dat truoc:
    echo   powershell -ExecutionPolicy Bypass -File scripts\setup_llama_and_model.ps1
    echo.
    pause
    exit /b 1
)

if not exist "%MODEL_PATH%" (
    echo [LOI] Khong tim thay model tai:
    echo       %MODEL_PATH%
    echo.
    echo Vui long chay script tai model truoc:
    echo   powershell -ExecutionPolicy Bypass -File scripts\setup_llama_and_model.ps1
    echo.
    pause
    exit /b 1
)

echo [OK] Da tim thay llama-server va model.
echo.
echo Thong so khoi dong:
echo - Host     : %HOST%
echo - Port     : %PORT%
echo - Context  : %CTX_SIZE% tokens
echo - Model    : %MODEL_PATH%
echo - GPU Offload: -ngl 99 (Neu co NVIDIA GPU se offload toan bo)
echo.
echo API Endpoint OpenAI-compatible:
echo   http://%HOST%:%PORT%/v1
echo.
echo [Luu y] De cua so nay chay de backend FastAPI ket noi.
echo Nhan Ctrl+C de dung server bat cu luc nao.
echo ============================================================
echo.

"%LLAMA_SERVER%" -m "%MODEL_PATH%" -c %CTX_SIZE% --port %PORT% --host %HOST% -ngl 99 -np 1 -fa 1 --alias qwen3-8b --alias default --alias gpt-4.1-mini --alias gpt-4o-mini %*

if errorlevel 1 (
    echo.
    echo Server bi dung hoac xay ra loi.
    pause
)
