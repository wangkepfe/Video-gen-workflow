@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Model Downloader — FLUX.1 Dev FP8 + Wan2.2 5B TI2V
echo   Total download: ~33 GB
echo ============================================================
echo.

set "ROOT=%~dp0"
set "COMFY=%ROOT%ComfyUI_windows_portable\ComfyUI"

if not exist "%COMFY%\main.py" (
    echo [ERROR] ComfyUI not found at %COMFY%
    echo         Run setup_comfyui.bat first.
    pause
    exit /b 1
)

set "CHECKPOINTS=%COMFY%\models\checkpoints"
set "DIFFUSION=%COMFY%\models\diffusion_models"
set "TEXTENCODERS=%COMFY%\models\text_encoders"
set "VAE=%COMFY%\models\vae"

mkdir "%CHECKPOINTS%" 2>nul
mkdir "%DIFFUSION%" 2>nul
mkdir "%TEXTENCODERS%" 2>nul
mkdir "%VAE%" 2>nul

set "FAIL=0"

REM -------------------------------------------------------
REM  FLUX.1 Dev FP8 checkpoint (17.2 GB)
REM -------------------------------------------------------
echo [1/4] FLUX.1 Dev FP8 checkpoint (17.2 GB)
set "F1=%CHECKPOINTS%\flux1-dev-fp8.safetensors"
if exist "%F1%" (
    echo       Already exists, skipping.
) else (
    echo       Downloading...
    curl -L -C - --progress-bar -o "%F1%.tmp" ^
        "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"
    if !errorlevel! neq 0 (
        echo       [WARN] Download failed or incomplete. Re-run this script to resume.
        set "FAIL=1"
    ) else (
        move /y "%F1%.tmp" "%F1%" >nul
        echo       Done.
    )
)
echo.

REM -------------------------------------------------------
REM  Wan2.2 TI2V 5B diffusion model (10.7 GB)
REM -------------------------------------------------------
echo [2/4] Wan2.2 TI2V 5B diffusion model (10.7 GB)
set "F2=%DIFFUSION%\wan2.2_ti2v_5B_fp16.safetensors"
if exist "%F2%" (
    echo       Already exists, skipping.
) else (
    echo       Downloading...
    curl -L -C - --progress-bar -o "%F2%.tmp" ^
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
    if !errorlevel! neq 0 (
        echo       [WARN] Download failed or incomplete. Re-run this script to resume.
        set "FAIL=1"
    ) else (
        move /y "%F2%.tmp" "%F2%" >nul
        echo       Done.
    )
)
echo.

REM -------------------------------------------------------
REM  UMT5-XXL text encoder fp8 (4.9 GB)
REM -------------------------------------------------------
echo [3/4] UMT5-XXL text encoder fp8 (4.9 GB)
set "F3=%TEXTENCODERS%\umt5_xxl_fp8_e4m3fn_scaled.safetensors"
if exist "%F3%" (
    echo       Already exists, skipping.
) else (
    echo       Downloading...
    curl -L -C - --progress-bar -o "%F3%.tmp" ^
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    if !errorlevel! neq 0 (
        echo       [WARN] Download failed or incomplete. Re-run this script to resume.
        set "FAIL=1"
    ) else (
        move /y "%F3%.tmp" "%F3%" >nul
        echo       Done.
    )
)
echo.

REM -------------------------------------------------------
REM  Wan2.2 VAE (~200 MB)
REM -------------------------------------------------------
echo [4/4] Wan2.2 VAE (~200 MB)
set "F4=%VAE%\wan2.2_vae.safetensors"
if exist "%F4%" (
    echo       Already exists, skipping.
) else (
    echo       Downloading...
    curl -L -C - --progress-bar -o "%F4%.tmp" ^
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors"
    if !errorlevel! neq 0 (
        echo       [WARN] Download failed or incomplete. Re-run this script to resume.
        set "FAIL=1"
    ) else (
        move /y "%F4%.tmp" "%F4%" >nul
        echo       Done.
    )
)
echo.

REM -------------------------------------------------------
REM  Summary
REM -------------------------------------------------------
echo ============================================================
if "%FAIL%"=="1" (
    echo   Some downloads failed. Run this script again to resume.
) else (
    echo   All models downloaded successfully!
    echo.
    echo   Models installed:
    echo     - flux1-dev-fp8.safetensors          [checkpoints]
    echo     - wan2.2_ti2v_5B_fp16.safetensors    [diffusion_models]
    echo     - umt5_xxl_fp8_e4m3fn_scaled.safetensors [text_encoders]
    echo     - wan2.2_vae.safetensors             [vae]
    echo.
    echo   Next step: run run_comfyui.bat to launch ComfyUI.
)
echo ============================================================
echo.
pause
exit /b %FAIL%
