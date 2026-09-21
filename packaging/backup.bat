@echo off
REM ---------------------------------------------------------------------
REM  latters -- back up the corpus. Run this at the end of the day, or
REM  point Task Scheduler at it.
REM
REM  It does NOT do `copy corpus.db backups\`. The database runs in WAL
REM  mode, so a plain file copy of a database the server is writing gives
REM  you a file that is missing the most recent transactions and reports
REM  no error at all. `latters backup` uses SQLite's own backup API, which
REM  takes a consistent snapshot of a live database, and then opens the
REM  result and checks it.
REM ---------------------------------------------------------------------
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

if not exist corpus.db (
  echo   No corpus.db in %~dp0 -- nothing to back up.
  pause
  exit /b 1
)

latters backup --db corpus.db -o backups --keep 14
if errorlevel 1 (
  echo.
  echo   BACKUP FAILED. Do not ignore this: read the message above.
  pause
  exit /b 1
)

REM  The skeletons are the office's edited letterheads. They are not in the
REM  database and they are the one thing here nobody can reconstruct from
REM  the archive, because they hold corrections a person made by hand.
if exist skeletons (
  robocopy skeletons "backups\skeletons" /MIR /NJH /NJS /NP /NDL >nul
  echo   skeletons\ copied too
)

echo.
echo   Copy the backups\ folder to a second drive. A backup on the same
echo   disk as the original protects against a mistake, not against the
echo   disk. This machine has a spinning disk.
echo.
pause
endlocal
