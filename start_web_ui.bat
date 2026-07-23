@echo off
REM Launch the speech-to-speech web dashboard on Windows.
REM
REM This script:
REM   1. Ensures uv is installed (https://docs.astral.sh/uv/).
REM   2. Runs `uv sync` on first run to install dependencies.
REM   3. Starts the dashboard at http://localhost:8050 and opens it in your browser.
REM
REM The dashboard wraps the existing `speech-to-speech` CLI. It does NOT
REM modify any code in src\ -- it spawns the pipeline as a subprocess.

setlocal

cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: 'uv' is not installed.
    echo Install it with:  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo Or see:          https://docs.astral.sh/uv/getting-started/installation/
    echo.
    pause
    exit /b 1
)

uv sync
if errorlevel 1 (
    echo.
    echo ERROR: 'uv sync' failed. See the output above.
    pause
    exit /b 1
)

echo.
echo ===========================================
echo   speech-to-speech dashboard
echo   opening http://localhost:8050
echo   (close this window to stop the server)
echo ===========================================
echo.

REM Best-effort browser auto-open after a brief delay so the server has time to start.
start "" /B cmd /c "timeout /t 3 /nobreak >nul && start "" http://localhost:8050"

uv run speech-to-speech-web
