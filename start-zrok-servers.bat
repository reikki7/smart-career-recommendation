@echo off

echo cd "D:\Web Development\Projects\Jobify\backend" > "%TEMP%\run_node.ps1"
echo node server.js >> "%TEMP%\run_node.ps1"

wt ^
  --title "Jobify Server" ^
  powershell.exe -NoExit -File "%TEMP%\run_node.ps1" ^
  ; new-tab ^
  --title "Zrok Shares" ^
  powershell.exe -NoExit -Command "zrok share reserved backendjobify" ^
  ; split-pane -H ^
  powershell.exe -NoExit -Command "zrok share reserved lmstudiojobify"