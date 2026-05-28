@echo off
setlocal
REM Optional helper when nvm use succeeds but node/npm/npx are not in PATH.
REM Edit NODE_HOME/RUST_BIN if your paths differ.
set NODE_HOME=E:\Develop\node\nvm\v20.9.0
set RUST_BIN=E:\keyan\DevelopTool\rust\.cargo\bin
set PATH=%NODE_HOME%;%RUST_BIN%;%PATH%

echo ===== Tool versions =====
node -v
npm -v
npx -v
circom --version

echo ===== Run real-trace ZK batch experiment =====
call scripts\run_hcrl_trace_zk_batch.bat
pause
