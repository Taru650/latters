@echo off
REM ---------------------------------------------------------------------
REM  latters -- start the drafting pages. Double-click this.
REM
REM  Lives in the DATA folder, not the install folder, so `%~dp0` is the
REM  corpus directory and no `cd` to somewhere else is needed. The eight
REM  errors this project's own setup notes open with were all one mistake:
REM  running the commands from the wrong directory.
REM ---------------------------------------------------------------------
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "MODEL=gemma3:1b"
set "PORT=8765"

if not exist corpus.db (
  echo.
  echo   There is no corpus.db in this folder.
  echo.
  echo   Build one first -- from THIS folder:
  echo       latters segment archive --db corpus.db
  echo       latters classify --db corpus.db --write
  echo       latters templates --db corpus.db -o skeletons
  echo.
  pause
  exit /b 1
)

REM  Ollama normally runs as a service after install. Starting it again is
REM  harmless if it is already up, and the alternative -- the app failing
REM  with "cannot reach Ollama" -- sends a non-technical user nowhere.
where ollama >nul 2>&1
if not errorlevel 1 (
  tasklist /fi "imagename eq ollama.exe" | findstr /i ollama.exe >nul
  if errorlevel 1 (
    echo   starting Ollama...
    start "" /min ollama serve
    REM  Give it a moment to bind before the app tries to reach it. On the
    REM  spinning disk this machine has, the first model load is slow
    REM  regardless; this is only about the socket being there.
    timeout /t 3 /nobreak >nul
  )
)

echo.
echo   Opening http://127.0.0.1:%PORT%/ in your browser.
echo   Leave THIS window open. Closing it stops the application.
echo.
start "" http://127.0.0.1:%PORT%/

latters serve --db corpus.db --skeletons skeletons --model %MODEL% --port %PORT%

REM  If serve exits immediately the browser is already open on a dead page,
REM  so hold the window: the reason is the last thing printed above it.
echo.
echo   The application has stopped.
pause
endlocal
