<#
.SYNOPSIS
    Builds LocalScribe for this machine.

.DESCRIPTION
    Produces a self-contained localscribe-doctor you can run without a .NET install.

    Architecture is the one thing that must be decided at build time: the ONNX Runtime QNN
    native libraries are architecture-specific, and an emulated x64 process cannot reach the
    Hexagon NPU at all. Everything else -- model size, execution providers, thread budget --
    is detected at startup by DeviceProbe and AcceleratorPlanner, because things like whether
    the laptop is on battery change while the app is running.

.PARAMETER Runtime
    Target runtime identifier. Defaults to the architecture of the machine you are on.

.PARAMETER App
    Also build the WinUI app. Requires the Windows App SDK workload, and has never been
    compiled successfully -- see docs/handoff.md before relying on it.
#>
param(
    [string]$Runtime = $(if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'win-arm64' } else { 'win-x64' }),
    [switch]$App
)

$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot

try {
    Write-Host "Building for $Runtime" -ForegroundColor Cyan

    if ($Runtime -ne 'win-arm64' -and $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        Write-Warning "Building $Runtime on an ARM64 machine. The NPU will be unreachable under emulation."
    }

    dotnet test --nologo
    if ($LASTEXITCODE -ne 0) { throw "Tests failed." }

    dotnet publish src/LocalScribe.Doctor -c Release -r $Runtime --self-contained true -o "publish/doctor-$Runtime"
    if ($LASTEXITCODE -ne 0) { throw "Doctor publish failed." }

    # Shipped by the ONNX Runtime package and never used at runtime. Dropping them saves
    # about 15 MB, which matters when this gets copied onto a machine over a network.
    Get-ChildItem "publish/doctor-$Runtime" -Include onnx_test_runner.exe,onnxruntime_perf_test.exe,*.lib `
        -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force

    if ($App) {
        dotnet publish src/LocalScribe.App -c Release -r $Runtime --self-contained true -o "publish/app-$Runtime"
        if ($LASTEXITCODE -ne 0) { throw "App publish failed. See docs/handoff.md -- this has never built." }
    }

    Write-Host ""
    Write-Host "Done. Run the doctor with:" -ForegroundColor Green
    Write-Host "  .\publish\doctor-$Runtime\localscribe-doctor.exe"
    Write-Host "  .\publish\doctor-$Runtime\localscribe-doctor.exe --install"
}
finally {
    Pop-Location
}
