@echo off
setlocal

set "ROOT=%~dp0"
set "COMFY=%ROOT%ComfyUI_windows_portable"

if not exist "%COMFY%\run_nvidia_gpu.bat" (
    echo [ERROR] ComfyUI not found at %COMFY%
    echo         Run setup_comfyui.bat first.
    pause
    exit /b 1
)

echo ============================================================
echo   Launching ComfyUI
echo   UI will open at: http://127.0.0.1:8188
echo ============================================================
echo.
echo   Workflows are in: %ROOT%workflows\
echo     - flux_dev_fp8_t2i.json   (Text to Image)
echo     - wan22_5b_i2v.json       (Image to Video)
echo.
echo   Output goes to:  %ROOT%output\
echo     - output\images\           (generated images)
echo     - output\videos\           (generated videos)
echo.
echo   Load workflows via: Workflows ^> Open (Ctrl+O)
echo ============================================================
echo.

timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8188"

set "OUTPUT_DIR=%ROOT%output"
mkdir "%OUTPUT_DIR%\images" 2>nul
mkdir "%OUTPUT_DIR%\videos" 2>nul

cd /d "%COMFY%\ComfyUI"
..\python_embeded\python.exe main.py --output-directory "%OUTPUT_DIR%" --input-directory "%OUTPUT_DIR%"
