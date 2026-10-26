@echo off
setlocal EnableDelayedExpansion

title Real-Time PPE Compliance - Public Online Launcher
echo ============================================================
echo   PPE COMPLIANCE SYSTEM - ONLINE PUBLIC DEPLOYMENT
echo ============================================================
echo.

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

:: 1. Verify virtual environment
if not exist "%ROOT_DIR%.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\
    pause
    exit /b 1
)

:: 2. Check cloudflared.exe
if not exist "%ROOT_DIR%cloudflared.exe" (
    echo [INFO] Downloading cloudflared for public HTTPS tunneling...
    powershell -Command "curl.exe -L -o cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
)

:: 3. Start Backend
echo [1/3] Starting FastAPI Backend on port 8000 ...
start "PPE Backend (Port 8000)" cmd /k "cd /d "%ROOT_DIR%" && call .venv\Scripts\activate.bat && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

:: 4. Start Frontend
echo [2/3] Starting React Dashboard on port 5173 ...
start "PPE Frontend (Port 5173)" cmd /k "cd /d "%ROOT_DIR%frontend" && npm run dev"

:: 5. Wait for servers to be active
echo [3/3] Waiting for servers to initialize...
for /l %%i in (1,1,30) do (
    curl.exe -s --connect-timeout 1 http://127.0.0.1:8000/health >nul 2>&1
    if !errorlevel! equ 0 goto :servers_ready
    <nul set /p =.
    ping 127.0.0.1 -n 2 >nul
)

:servers_ready
echo.
echo [INFO] Local servers are ready!
echo.
echo ============================================================
echo   GENERATING YOUR PUBLIC ONLINE HTTPS LINK (CLOUDFLARE)
echo ============================================================
echo.
echo [NOTE] A new window 'Cloudflare Public Tunnel' will open.
echo Look for the link ending in: .trycloudflare.com
echo.
echo You can send that exact https://... link to your Mam,
echo and she can open it on her phone, laptop, or tablet!
echo ============================================================
echo.

start "Cloudflare Public Tunnel" cmd /k "cd /d "%ROOT_DIR%" && echo Connecting to Cloudflare Global Edge... && echo. && .\cloudflared.exe tunnel --url http://localhost:5173"

echo Opening local dashboard in your browser...
start http://localhost:5173

echo.
echo Done! Check the 'Cloudflare Public Tunnel' window for your public URL.
pause
