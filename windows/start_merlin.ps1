$dir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $dir
& "$dir\venv\Scripts\Activate.ps1"
python merlin.py
