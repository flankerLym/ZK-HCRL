@echo off
setlocal
rem Optional helper if nvm symlink is broken. Adjust NODE_HOME if needed.
set NODE_HOME=E:\Develop\node\nvm\v20.9.0
set RUST_BIN=E:\keyan\DevelopTool\rust\.cargo\bin
if exist "%NODE_HOME%\node.exe" set PATH=%NODE_HOME%;%RUST_BIN%;%PATH%

echo ===== Tool versions =====
node -v
npm -v
npx -v
circom --version

echo ===== Install clean Hardhat 2 stack =====
call scripts\install_clean_hardhat2.bat || exit /b 1

echo ===== Run full real proof pipeline =====
call scripts\run_real_proof_pipeline.bat || exit /b 1
pause
