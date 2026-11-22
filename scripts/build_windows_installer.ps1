# Build the Inno Setup installer from an already-packaged Windows folder.
#
# Usage (from the repository root, in PowerShell):
#   scripts\build_windows_installer.ps1
#   scripts\build_windows_installer.ps1 -SourceDir dist\openvrcardboard -Version 1.2.2
#
# The portable folder must exist: run scripts\package.py first (CI does this).

[CmdletBinding()]
param(
    [string]$SourceDir = "dist\openvrcardboard",
    [string]$Version = "",
    [string]$Iscc = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not $Version) {
    $init = Get-Content "desktop\openvrcardboard\__init__.py" -Raw
    if ($init -match '__version__\s*=\s*"([^"]+)"') {
        $Version = $Matches[1]
    } else {
        throw "could not read the version from desktop\openvrcardboard\__init__.py"
    }
}

if (-not (Test-Path (Join-Path $SourceDir "openvrcardboard.exe"))) {
    throw "missing $SourceDir\openvrcardboard.exe; run scripts\package.py first"
}

if (-not $Iscc) {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        (Get-Command ISCC.exe -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
    ) | Where-Object { $_ -and (Test-Path $_) }
    if (-not $candidates) {
        throw "Inno Setup 6 was not found; install it from https://jrsoftware.org/isinfo.php"
    }
    $Iscc = @($candidates)[0]
}

$source = (Resolve-Path $SourceDir).Path
Write-Host "Building OpenVR Cardboard $Version installer from $source"
& $Iscc "/DAppVersion=$Version" "/DSourceDir=$source" "packaging\windows\openvrcardboard.iss"
if ($LASTEXITCODE -ne 0) {
    throw "ISCC failed with exit code $LASTEXITCODE"
}

$target = Join-Path $Root "dist\OpenVR-Cardboard-$Version-Setup.exe"
if (-not (Test-Path $target)) {
    throw "ISCC reported success but $target is missing"
}
$size = [math]::Round((Get-Item $target).Length / 1MB, 1)
Write-Host "Wrote $target ($size MB)"
