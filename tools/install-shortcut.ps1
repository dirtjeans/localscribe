<#
.SYNOPSIS
    Puts LocalScribe in the Start Menu.

.DESCRIPTION
    Creates a per-user Start Menu shortcut pointing at a published build. Per-user on purpose:
    it needs no administrator, and this app is not installed system-wide.

    The icon comes from the executable itself, which carries it as a resource, so the shortcut
    keeps working if the Assets folder is not around.

.PARAMETER Path
    The publish directory holding LocalScribe.App.exe. Defaults to publish/win-arm64 beside the
    repository root.

.PARAMETER Desktop
    Also place a shortcut on the desktop.

.PARAMETER Remove
    Delete the shortcuts instead of creating them.

.EXAMPLE
    ./tools/install-shortcut.ps1
    ./tools/install-shortcut.ps1 -Desktop
    ./tools/install-shortcut.ps1 -Remove
#>
[CmdletBinding()]
param(
    [string]$Path,
    [switch]$Desktop,
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Path) { $Path = Join-Path $repoRoot 'publish\win-arm64' }

$startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
$targets = @(Join-Path $startMenu 'LocalScribe.lnk')
if ($Desktop) { $targets += Join-Path ([Environment]::GetFolderPath('Desktop')) 'LocalScribe.lnk' }

if ($Remove) {
    foreach ($t in $targets) {
        if (Test-Path $t) { Remove-Item $t -Force; "removed $t" }
        else { "not present: $t" }
    }
    return
}

$exe = Join-Path $Path 'LocalScribe.App.exe'
if (-not (Test-Path $exe)) {
    throw "No LocalScribe.App.exe under '$Path'. Publish first:`n" +
          "  dotnet publish src/LocalScribe.App/LocalScribe.App.csproj -c Release -r win-arm64 --self-contained true -p:Platform=arm64 -o publish/win-arm64"
}

# Shortcuts do not follow a moved target, so resolve to a full path rather than storing whatever
# relative form the caller happened to pass.
$exe = (Resolve-Path $exe).Path
$workingDirectory = (Resolve-Path $Path).Path

# The models directory is resolved relative to the executable, not the working directory, so a
# shortcut cannot fix a missing one. Say so now rather than letting the app fail on first click.
$models = Join-Path $workingDirectory 'models'
if (-not (Test-Path $models)) {
    Write-Warning "No models directory beside the executable. The app will start but cannot transcribe until one exists. Run: localscribe-doctor --fetch-models"
}

$shell = New-Object -ComObject WScript.Shell

foreach ($t in $targets) {
    $shortcut = $shell.CreateShortcut($t)
    $shortcut.TargetPath = $exe
    $shortcut.WorkingDirectory = $workingDirectory
    $shortcut.IconLocation = "$exe,0"
    $shortcut.Description = 'Offline transcription on the Snapdragon NPU'
    $shortcut.Save()

    "created $t"
}

"Search the Start Menu for 'LocalScribe'."
