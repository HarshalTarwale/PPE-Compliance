@echo off
title Stop Real-Time PPE Compliance System
echo ============================================================
echo   STOPPING PPE COMPLIANCE SERVERS
echo ============================================================
echo.

echo Terminating backend and frontend processes...

taskkill /F /FI "WINDOWTITLE eq PPE Backend (FastAPI)*" /T >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq PPE Frontend (Vite)*" /T >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Cloudflare Public Tunnel*" /T >nul 2>&1
taskkill /F /IM cloudflared.exe >nul 2>&1

:: Also kill by listening port in case child processes remain active
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5173" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo [OK] All backend and frontend processes have been stopped.
echo ============================================================
timeout /t 2 >nul
