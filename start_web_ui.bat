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

REM Install dependencies only when the lockfile has changed since last
REM sync. pyproject.toml pins ``torch>=2.4.0`` (no upper bound), so an
REM unconditional uv sync on every launch can pull a different torch +
REM different NVIDIA wheels each time. The .venv\Scripts\.lock-stamp
REM file records the last-seen uv.lock mtime; we skip uv sync when it
REM matches. (Windows shell can't reliably read mtime, so we use a
REM portable hash of the lockfile contents instead.)
set "STAMP=.venv\Scripts\.lock-stamp"
set "NEED_SYNC=1"
if exist "%STAMP%" if exist "uv.lock" (
    REM hash the lockfile -- if the stamp's hash matches, lockfile is unchanged.
    powershell -NoProfile -Command "exit ('{0}' -eq (Get-FileHash -Algorithm SHA256 'uv.lock' -ErrorAction SilentlyContinue).Hash)" >nul 2>&1
    if not errorlevel 1 set "NEED_SYNC=0"
)
if "%NEED_SYNC%"=="1" (
    uv sync
    if exist "uv.lock" (
        powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 'uv.lock').Hash | Out-File -Encoding ascii '%STAMP%'" >nul 2>&1
    )
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
