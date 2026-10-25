@echo off
setlocal EnableDelayedExpansion

title Real-Time PPE Compliance System Launcher
echo ============================================================
echo   REAL-TIME PPE COMPLIANCE ^& MONITORING SYSTEM
echo ============================================================
echo.

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

:: Check virtual environment
if not exist "%ROOT_DIR%.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\
    echo Please create it first:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: Check frontend dependencies
if not exist "%ROOT_DIR%frontend\node_modules" (
    echo [INFO] Installing frontend node_modules...
    cd /d "%ROOT_DIR%frontend"
    call npm install
    cd /d "%ROOT_DIR%"
)

echo [1/3] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "PPE Backend (FastAPI)" cmd /k "cd /d "%ROOT_DIR%" && call .venv\Scripts\activate.bat && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/3] Starting React Dashboard on http://localhost:5173 ...
start "PPE Frontend (Vite)" cmd /k "cd /d "%ROOT_DIR%frontend" && npm run dev"

echo [3/3] Waiting for AI models and backend server to initialize...
set "BACKEND_READY=0"
for /l %%i in (1,1,45) do (
    curl.exe -s --connect-timeout 1 http://127.0.0.1:8000/health >nul 2>&1
    if !errorlevel! equ 0 (
        set "BACKEND_READY=1"
        goto :backend_ready
    )
    <nul set /p =.
    ping 127.0.0.1 -n 2 >nul
)

:backend_ready
echo.
if "!BACKEND_READY!"=="1" (
    echo [INFO] Backend and AI models initialized successfully.
) else (
    echo [WARNING] Backend initialization is taking longer than expected. Proceeding anyway...
)

echo.
echo Launching default browser to http://localhost:5173 ...
start http://localhost:5173

echo.
echo ============================================================
echo   SYSTEM IS NOW RUNNING!
echo   ----------------------------------------------------------
echo   - Web Dashboard : http://localhost:5173
echo   - Backend API   : http://127.0.0.1:8000
echo   - Swagger Docs  : http://127.0.0.1:8000/docs
echo   - WebSocket     : ws://127.0.0.1:8000/ws/monitoring
echo.
echo   To stop all servers, run 'stop.bat' or close the terminal
echo   windows for Backend and Frontend.
echo ============================================================
echo.
