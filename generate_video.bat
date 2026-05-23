@echo off
set "ROOT=%~dp0"
set "PYTHON=%ROOT%ComfyUI_windows_portable\python_embeded\python.exe"
@REM "%PYTHON%" "%ROOT%generate_video.py" %*
"%PYTHON%" "%ROOT%generate_video.py" output\images\6\zimage_20260401_020754.png --prompt-file prompts\video_sofa.txt --negative-file prompts\video_negative_example.txt --low-priority
