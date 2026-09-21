@echo off
REM ---------------------------------------------------------------------
REM  latters -- offline install. Run this from the USB stick.
REM
REM  Everything installs from files in this folder. Nothing is downloaded,
REM  because the office has no internet and an installer that quietly
REM  needs it fails at the worst possible moment.
REM
REM  `setlocal` keeps the PATH edits and variables out of the user's shell.
REM  No `cd` at the end, so the window stays where they started it.
REM ---------------------------------------------------------------------
setlocal EnableExtensions
chcp 65001 >nul
REM  chcp 65001 first: every message below contains a path that may hold
REM  Hindi, and the default console codepage mangles it.

set "BUNDLE=%~dp0"
set "TARGET=%USERPROFILE%\latters-data"

echo.
echo   latters -- offline install
echo   from: %BUNDLE%
echo   data: %TARGET%
echo.

REM --- 1. Python -------------------------------------------------------
where py >nul 2>&1
if errorlevel 1 (
  where python >nul 2>&1
  if errorlevel 1 (
    echo   [X] Python is not installed.
    echo       Install Python 3.11 or newer from python.org, tick
    echo       "Add python.exe to PATH" on the first screen, then run
    echo       this file again.
    goto :fail
  )
  set "PY=python"
) else (
  set "PY=py -3"
)
%PY% --version
if errorlevel 1 goto :fail

REM --- 2. the application ----------------------------------------------
echo.
echo   [1/5] installing the application from %BUNDLE%wheels
REM  --no-index is the whole point: pip must not try the network. Without
REM  it, pip reaches for PyPI, hangs on a machine with no route out, and
REM  eventually fails with a timeout that reads like a broken package.
%PY% -m pip install --no-index --find-links "%BUNDLE%wheels" latters[web]
if errorlevel 1 (
  echo   [X] install failed. If it mentions a platform tag, the bundle was
  echo       built for the wrong Python or the wrong Windows architecture.
  goto :fail
)

REM --- 3. Ollama -------------------------------------------------------
echo.
echo   [2/5] Ollama
where ollama >nul 2>&1
if errorlevel 1 (
  if not exist "%BUNDLE%OllamaSetup.exe" (
    echo   [X] Ollama is not installed and OllamaSetup.exe is not in this
    echo       folder. The bundle is incomplete.
    goto :fail
  )
  echo       running OllamaSetup.exe -- accept the prompts
  "%BUNDLE%OllamaSetup.exe" /SILENT
  REM  The installer puts ollama.exe here but does not refresh THIS shell's
  REM  PATH, so the next command would fail on a first install.
  set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
) else (
  echo       already installed
)

REM --- 4. the model ----------------------------------------------------
echo.
echo   [3/5] the model
if not exist "%BUNDLE%model\MODEL_NAME" (
  echo   [X] no model in the bundle. The application will still run with
  echo       --stub, but it cannot write letters.
  goto :fail
)
set /p MODEL=<"%BUNDLE%model\MODEL_NAME"
REM  NOT wrapped in an if(...) block. Two traps live in that shape and the
REM  first version of this file fell into both:
REM    * %ERRORLEVEL% inside a block expands when the block is PARSED, so it
REM      reports the code from before the command in it ever ran;
REM    * `if errorlevel` after popd tests popd, not `ollama create`, so a
REM      failed create reported success and the office got an installer that
REM      finished with no model.
REM  `if errorlevel` is a runtime test, so at statement level it is correct.
ollama list | findstr /i /c:"%MODEL%" >nul && goto :model_ready
echo       creating %MODEL% from the bundled GGUF -- a few minutes
pushd "%BUNDLE%model"
ollama create %MODEL% -f Modelfile
if errorlevel 1 (popd & goto :fail)
popd
goto :model_done
:model_ready
echo       %MODEL% already present
:model_done

REM --- 4b. OCR (optional, for scans and PDFs) --------------------------
echo.
echo   [3b/5] OCR tools
REM  Only needed to read PDFs and scans in the admin page. A .docx archive
REM  needs neither, so a missing one is a warning and not a failure.
where tesseract >nul 2>&1
if errorlevel 1 (
  if exist "%BUNDLE%tesseract-setup.exe" (
    echo       installing Tesseract -- tick the Hindi language pack
    "%BUNDLE%tesseract-setup.exe" /SILENT
    set "PATH=%PATH%;%ProgramFiles%\Tesseract-OCR"
  ) else (
    echo       [!] not in the bundle: scans and photographs cannot be read.
    echo           .docx and PDFs with a text layer still work.
  )
) else (
  echo       Tesseract already installed
)
REM  The Hindi pack is a separate file and the installer's default does NOT
REM  include it. Without it, OCR returns Latin gibberish for Devanagari.
if exist "%BUNDLE%tessdata\hin.traineddata" (
  if defined ProgramFiles (
    copy /y "%BUNDLE%tessdata\*.traineddata" ^
            "%ProgramFiles%\Tesseract-OCR\tessdata\" >nul 2>&1
  )
)

where pdftotext >nul 2>&1
if errorlevel 1 (
  if exist "%BUNDLE%poppler\bin\pdftotext.exe" (
    echo       installing poppler to %TARGET%\poppler
    xcopy /e /i /y /q "%BUNDLE%poppler" "%TARGET%\poppler" >nul
    echo       [!] add %TARGET%\poppler\bin to PATH, or PDFs will be skipped
  ) else (
    echo       [!] poppler not in the bundle: PDFs cannot be read.
  )
) else (
  echo       poppler already installed
)

REM --- 5. the working folder -------------------------------------------
echo.
echo   [4/5] working folder
if not exist "%TARGET%"           mkdir "%TARGET%"
if not exist "%TARGET%\archive"   mkdir "%TARGET%\archive"
if not exist "%TARGET%\skeletons" mkdir "%TARGET%\skeletons"
if not exist "%TARGET%\backups"   mkdir "%TARGET%\backups"
copy /y "%BUNDLE%start.bat"  "%TARGET%\start.bat"  >nul
copy /y "%BUNDLE%backup.bat" "%TARGET%\backup.bat" >nul

echo.
echo   Done.
echo.
echo   Next:
echo     1. copy the office's old letters into  %TARGET%\archive
echo     2. open a Command Prompt there and run:
echo          latters segment archive --db corpus.db
echo          latters classify --db corpus.db --write
echo          latters templates --db corpus.db -o skeletons
echo     3. double-click  %TARGET%\start.bat
echo.
goto :done

:fail
echo.
echo   Install did not finish. Nothing was half-configured that a second
echo   run will not fix -- read the [X] line above, then run this again.
endlocal
exit /b 1

:done
endlocal
exit /b 0
