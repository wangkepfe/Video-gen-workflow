@echo off
set "ROOT=%~dp0"
set "PYTHON=%ROOT%ComfyUI_windows_portable\python_embeded\python.exe"
"%PYTHON%" "%ROOT%web_video.py" --low-priority %*
pause