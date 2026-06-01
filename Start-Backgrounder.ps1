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
    # Prefer a project virtualenv that has the full model stack (incl. SAM 3.1).
    # Falls back to PATH python only if no venv is present. This avoids the common
    # mistake of launching with the system interpreter, where heavy experts such
    # as sam3 are not installed and silently get skipped.
    $venvCandidates = @(
        Join-Path $Root ".venv-sam3\Scripts\python.exe",
        Join-Path $Root ".venv\Scripts\python.exe",
        Join-Path $Root "venv\Scripts\python.exe"
    )
    foreach ($candidate in $venvCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            Write-Host "Using project venv: $candidate" -ForegroundColor Green
            return $candidate
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        Write-Host "No project venv found; falling back to PATH python: $($python.Source)" -ForegroundColor Yellow
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
