@echo off
setlocal
cd /d %~dp0\..
echo [Clean install] Removing node_modules and package-lock.json...
if exist node_modules rmdir /s /q node_modules
if exist package-lock.json del /q package-lock.json
echo [Clean install] Installing stable Hardhat 2 dependencies...
call npm install --registry=https://registry.npmmirror.com || exit /b 1
echo [DONE] Hardhat 2 dependency install complete.
