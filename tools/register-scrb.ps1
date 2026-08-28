<#
.SYNOPSIS
Gives saved transcripts their icon and their double-click.

.DESCRIPTION
Associates .scrb (and the older .lscribe) with LocalScribe for the current user: Explorer shows
the transcript icon on the files, and opening one launches the app with the file on its command
line, which the app already understands.

Written to HKCU\Software\Classes, so it needs no administrator prompt and touches nobody else's
account. Run it again after moving the app; run with -Remove to take the association away.

The app deliberately does not write this itself. A program that edits file associations on
first launch is doing something to the machine that nobody asked for; a script the user runs is
the same change with consent.

.PARAMETER AppPath
Where LocalScribe.App.exe lives. Defaults to the publish folder next to this script's repo.

.PARAMETER Remove
Take the association away instead of creating it.
#>
param(
    [string]$AppPath = (Join-Path (Split-Path $PSScriptRoot -Parent) "publish\win-arm64\LocalScribe.App.exe"),
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

$progId = "LocalScribe.Transcript"
$classes = "HKCU:\Software\Classes"
$extensions = @(".scrb", ".lscribe")

function Refresh-Shell {
    # Tells Explorer the associations changed, so icons update without logging off.
    $signature = '[System.Runtime.InteropServices.DllImport("shell32.dll")] public static extern void SHChangeNotify(int wEventId, uint uFlags, System.IntPtr dwItem1, System.IntPtr dwItem2);'
    $shell = Add-Type -MemberDefinition $signature -Name "Shell" -Namespace "Refresh" -PassThru
    $shell::SHChangeNotify(0x08000000, 0, [System.IntPtr]::Zero, [System.IntPtr]::Zero)
}

if ($Remove) {
    foreach ($extension in $extensions) {
        if (Test-Path "$classes\$extension") { Remove-Item "$classes\$extension" -Recurse -Force }
    }
    if (Test-Path "$classes\$progId") { Remove-Item "$classes\$progId" -Recurse -Force }

    Refresh-Shell
    Write-Host "Removed the .scrb association."
    exit 0
}

if (-not (Test-Path $AppPath)) {
    Write-Error "No app at $AppPath. Pass -AppPath with the location of LocalScribe.App.exe."
}

$icon = Join-Path (Split-Path $AppPath -Parent) "Assets\scrb.ico"

if (-not (Test-Path $icon)) {
    Write-Error "No icon at $icon. Publish the app first; the icon ships beside the exe."
}

New-Item -Path "$classes\$progId" -Force | Out-Null
Set-ItemProperty -Path "$classes\$progId" -Name "(default)" -Value "LocalScribe transcript"
Set-ItemProperty -Path "$classes\$progId" -Name "FriendlyTypeName" -Value "LocalScribe transcript"

New-Item -Path "$classes\$progId\DefaultIcon" -Force | Out-Null
Set-ItemProperty -Path "$classes\$progId\DefaultIcon" -Name "(default)" -Value "`"$icon`""

New-Item -Path "$classes\$progId\shell\open\command" -Force | Out-Null
Set-ItemProperty -Path "$classes\$progId\shell\open\command" -Name "(default)" -Value "`"$AppPath`" `"%1`""

foreach ($extension in $extensions) {
    New-Item -Path "$classes\$extension" -Force | Out-Null
    Set-ItemProperty -Path "$classes\$extension" -Name "(default)" -Value $progId
}

Refresh-Shell
Write-Host "Saved transcripts (.scrb, .lscribe) now open with $AppPath and carry the transcript icon."
