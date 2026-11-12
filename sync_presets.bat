@echo off
cd /d "%~dp0"
echo === Syncing Model Preset Voices ===
echo Stop the TTS server before running this, then restart it afterwards.
echo.
.venv\Scripts\python.exe -m server.sync_presets %*
echo.
echo Restart the server to pick up preset changes.
pause
