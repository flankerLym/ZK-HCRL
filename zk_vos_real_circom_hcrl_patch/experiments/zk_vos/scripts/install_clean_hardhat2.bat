@echo off
setlocal
cd /d %~dp0\..

echo [1/4] Removing mixed Hardhat dependencies...
if exist node_modules rmdir /S /Q node_modules
if exist package-lock.json del /Q package-lock.json

echo [2/4] Setting npm registry mirror for better connectivity...
call npm config set registry https://registry.npmmirror.com
call npm cache verify

echo [3/4] Installing Hardhat 2 + ethers v5 stack without EDR/toolbox...
call npm install --legacy-peer-deps || exit /b 1

echo [4/4] Installed dependency versions:
call npm ls hardhat ethers @nomiclabs/hardhat-ethers hardhat-gas-reporter snarkjs

echo.
echo [DONE] Clean Hardhat 2 environment is ready.
