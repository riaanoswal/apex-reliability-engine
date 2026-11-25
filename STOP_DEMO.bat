@echo off
title APEX - stop
echo Stopping APEX...
taskkill /FI "WINDOWTITLE eq APEX Backend*"   /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq APEX Dashboard*" /T /F >nul 2>&1
echo Done. Both windows have been closed.
timeout /t 3 /nobreak >nul
