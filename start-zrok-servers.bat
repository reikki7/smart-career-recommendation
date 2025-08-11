@echo off

echo cd "D:\Software Development\Projects\Jobify\backend" > "%TEMP%\run_node.ps1"
echo .\venv\Scripts\Activate.ps1 >> "%TEMP%\run_node.ps1"
echo python server.py >> "%TEMP%\run_node.ps1"

wt ^
  --title "Jobify Server" ^
  powershell.exe -NoExit -File "%TEMP%\run_node.ps1" ^
  ; new-tab ^
  --title "Zrok Shares" ^
  powershell.exe -NoExit -Command "zrok share reserved jobifybackend" ^
  ; split-pane -H ^
  powershell.exe -NoExit -Command "zrok share reserved lmstudioserver"