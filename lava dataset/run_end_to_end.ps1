param(
    [string]$Location = "amsterdam",
    [string]$Split = "test",
    [string]$Host = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3 -m venv .venv
}

$python = Join-Path $root ".venv\Scripts\python.exe"

& $python -m pip install -r requirements.txt
& $python -m lava_search bootstrap --locations $Location --splits $Split --include-videos --profile strongest
& $python -m lava_search serve-api --host $Host --port $Port
