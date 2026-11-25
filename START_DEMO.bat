@echo off
REM Double-click wrapper. Everything is done by run.py, so there is one
REM source of truth whether you double-click this or type the command:
REM
REM     python run.py
REM
title APEX Demo Launcher
cd /d "%~dp0"

set "PY=python"
where python >nul 2>&1
if errorlevel 1 set "PY=py"
%PY% -c "import sys" >nul 2>&1
if errorlevel 1 (
  echo.
  echo [X] Python was not found on this computer.
  echo     Install Python 3.10 or newer from python.org, tick "Add python.exe to PATH",
  echo     then run this file again.
  echo.
  pause
  exit /b 1
)

%PY% run.py %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo The launcher exited with code %RC%. The reason is in the output above.
  pause
)
exit /b %RC%
