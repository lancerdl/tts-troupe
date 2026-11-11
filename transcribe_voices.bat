@echo off
cd /d "%~dp0"
echo === Running Voice Library Transcriber ===
.venv\Scripts\python.exe -m server.transcribe_voices %*
pause
