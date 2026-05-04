param(
    [int]$Port = 7861,
    [int]$MaxPort = 7899
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

function Test-PortFree {
    param([int]$PortToCheck)

    $connection = Get-NetTCPConnection -LocalPort $PortToCheck -ErrorAction SilentlyContinue |
        Where-Object { $_.State -eq "Listen" }
    return -not $connection
}

function Resolve-Python {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return $py.Source
    }

    throw "Python was not found on PATH."
}

$SelectedPort = $null
foreach ($Candidate in $Port..$MaxPort) {
    if (Test-PortFree -PortToCheck $Candidate) {
        $SelectedPort = $Candidate
        break
    }
}

if (-not $SelectedPort) {
    throw "No free port found in range $Port-$MaxPort."
}

$PythonExe = Resolve-Python
$Url = "http://127.0.0.1:$SelectedPort"

Write-Host ""
Write-Host "Starting Backgrounder..." -ForegroundColor Cyan
Write-Host "Folder : $Root"
Write-Host "URL    : $Url"
Write-Host ""
Write-Host "Keep this window open while using the app. Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

& $PythonExe "app.py" "--port" "$SelectedPort"
