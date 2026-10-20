param(
    [ValidateSet('debug', 'asan')]
    [string]$Preset = 'debug'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'dev-shell.ps1')
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    New-Item -ItemType Directory -Force results | Out-Null
    & cmake --preset $Preset 2>&1 | Tee-Object -FilePath "results/m1-$Preset-configure.txt"
    if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
    & cmake --build --preset $Preset 2>&1 | Tee-Object -FilePath "results/m1-$Preset-build.txt"
    if ($LASTEXITCODE -ne 0) { throw 'CMake build failed.' }
    & ctest --preset $Preset --verbose --no-tests=error 2>&1 |
        Tee-Object -FilePath "results/m1-$Preset-tests.txt"
    if ($LASTEXITCODE -ne 0) { throw 'CTest failed.' }
}
finally {
    Pop-Location
}
