@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Model Downloader - FLUX.1 Dev FP8 + Wan2.2 14B I2V (MoE)
echo   Total download: ~50 GB (FLUX 17 + 2x14B + UMT5 5 + VAE)
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
echo [1/5] FLUX.1 Dev FP8 checkpoint (17.2 GB)
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
REM  Wan2.2 I2V 14B HIGH-NOISE expert fp8 (~14 GB)
REM -------------------------------------------------------
echo [2/5] Wan2.2 I2V 14B high-noise expert fp8 (~14 GB)
set "F2=%DIFFUSION%\wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
if exist "%F2%" (
    echo       Already exists, skipping.
) else (
    echo       Downloading...
    curl -L -C - --progress-bar -o "%F2%.tmp" ^
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
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
REM  Wan2.2 I2V 14B LOW-NOISE expert fp8 (~14 GB)
REM -------------------------------------------------------
echo [3/5] Wan2.2 I2V 14B low-noise expert fp8 (~14 GB)
set "F3=%DIFFUSION%\wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
if exist "%F3%" (
    echo       Already exists, skipping.
) else (
    echo       Downloading...
    curl -L -C - --progress-bar -o "%F3%.tmp" ^
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
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
REM  UMT5-XXL text encoder fp8 (4.9 GB)
REM -------------------------------------------------------
echo [4/5] UMT5-XXL text encoder fp8 (4.9 GB)
set "F4=%TEXTENCODERS%\umt5_xxl_fp8_e4m3fn_scaled.safetensors"
if exist "%F4%" (
    echo       Already exists, skipping.
) else (
    echo       Downloading...
    curl -L -C - --progress-bar -o "%F4%.tmp" ^
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
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
REM  Wan2.2 VAE (~200 MB)
REM -------------------------------------------------------
echo [5/5] Wan2.2 VAE (~200 MB)
set "F5=%VAE%\wan2.2_vae.safetensors"
if exist "%F5%" (
    echo       Already exists, skipping.
) else (
    echo       Downloading...
    curl -L -C - --progress-bar -o "%F5%.tmp" ^
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors"
    if !errorlevel! neq 0 (
        echo       [WARN] Download failed or incomplete. Re-run this script to resume.
        set "FAIL=1"
    ) else (
        move /y "%F5%.tmp" "%F5%" >nul
        echo       Done.
    )
)
echo.

REM -------------------------------------------------------
REM  Cleanup: remove obsolete 5B model if present
REM -------------------------------------------------------
set "OLD5B=%DIFFUSION%\wan2.2_ti2v_5B_fp16.safetensors"
if exist "%OLD5B%" (
    echo Removing obsolete Wan2.2 5B model...
    del /f /q "%OLD5B%"
    echo       Removed.
    echo.
)

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
    echo     - flux1-dev-fp8.safetensors                          [checkpoints]
    echo     - wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors   [diffusion_models]
    echo     - wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors    [diffusion_models]
    echo     - umt5_xxl_fp8_e4m3fn_scaled.safetensors             [text_encoders]
    echo     - wan2.2_vae.safetensors                             [vae]
    echo.
    echo   Next step: run web_video.bat to launch the web app.
)
echo ============================================================
echo.
pause
exit /b %FAIL%
