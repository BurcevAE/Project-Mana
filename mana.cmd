@echo off
rem MANA from a shell, without the two things that just broke it.
rem
rem   * PowerShell treats a quoted path at the start of a line as a string,
rem     not a command -- "C:\...\python.exe" -m mana.cli is a parse error,
rem     and the fix (the & call operator) is not something anyone should
rem     have to know to run their own program.
rem   * `python -m mana.cli` only finds the package when the current
rem     directory happens to be the checkout. Run from System32 it fails
rem     with ModuleNotFoundError, which says nothing about the cause.
rem
rem So: cd to wherever this file lives, prefer the checkout's venv, and
rem pass everything through.
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m mana.cli %*
) else (
  python -m mana.cli %*
)
endlocal
