@echo off
cd /d %~dp0\experiments\zk_vos
call npm install || exit /b 1
call scripts\run_full_pipeline.bat || exit /b 1
