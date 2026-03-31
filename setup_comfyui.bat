@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   ComfyUI Portable (NVIDIA) Setup
echo ============================================================
echo.

set "ROOT=%~dp0"
set "ARCHIVE=%ROOT%ComfyUI_windows_portable_nvidia.7z"
set "DEST=%ROOT%ComfyUI_windows_portable"
set "URL=https://github.com/Comfy-Org/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z"

if exist "%DEST%\ComfyUI\main.py" (
    echo [OK] ComfyUI is already installed at:
    echo      %DEST%
    echo.
    echo If you want to reinstall, delete the ComfyUI_windows_portable folder first.
    goto :done
)

echo [1/2] Downloading ComfyUI portable (~1.9 GB)...
echo       Source: %URL%
echo.

curl -L -C - -o "%ARCHIVE%" "%URL%"
if %errorlevel% neq 0 (
    echo [ERROR] Download failed. Check your internet connection and try again.
    echo         The script supports resuming — just run it again.
    goto :error
)

echo.
echo [2/2] Extracting archive...

set "SEVENZIP="
where 7z >nul 2>&1
if %errorlevel% equ 0 (
    set "SEVENZIP=7z"
)

if not defined SEVENZIP (
    echo       7z not in PATH, checking common locations...
    if exist "C:\Program Files\7-Zip\7z.exe" (
        set "SEVENZIP=C:\Program Files\7-Zip\7z.exe"
    ) else if exist "C:\Program Files (x86)\7-Zip\7z.exe" (
        set "SEVENZIP=C:\Program Files (x86)\7-Zip\7z.exe"
    )
)

if not defined SEVENZIP (
    echo       7-Zip not found. Downloading portable 7zr.exe...
    curl -L -o "%ROOT%7zr.exe" "https://www.7-zip.org/a/7zr.exe"
    if !errorlevel! equ 0 (
        set "SEVENZIP=%ROOT%7zr.exe"
    ) else (
        echo       [ERROR] Could not download 7zr.exe.
        echo       Please install 7-Zip from https://www.7-zip.org/ and re-run,
        echo       or manually extract %ARCHIVE% into %ROOT%
        goto :error
    )
)

echo       Using: !SEVENZIP!
"!SEVENZIP!" x "%ARCHIVE%" -o"%ROOT%" -y
if !errorlevel! neq 0 (
    echo       [ERROR] Extraction failed.
    echo       Please install 7-Zip from https://www.7-zip.org/ and re-run,
    echo       or manually extract %ARCHIVE% into %ROOT%
    goto :error
)

if not exist "%DEST%\ComfyUI\main.py" (
    echo [ERROR] Extraction failed or unexpected folder structure.
    echo         Please manually extract %ARCHIVE% into %ROOT%
    goto :error
)

echo.
echo [OK] ComfyUI installed successfully at:
echo      %DEST%
echo.

del "%ARCHIVE%" 2>nul
echo Cleaned up archive file.

:done
echo.
echo Next step: run download_models.bat to fetch the AI models.
echo.
pause
exit /b 0

:error
echo.
echo Setup did not complete. See errors above.
pause
exit /b 1
