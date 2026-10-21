param(
    [ValidateSet('debug', 'asan')]
    [string]$Preset = 'debug',
    [switch]$Pretrained
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'dev-shell.ps1')
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    & cmake --preset $Preset 2>&1 | Tee-Object "results/m3-$Preset-configure.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Configure failed' }
    & cmake --build --preset $Preset 2>&1 | Tee-Object "results/m3-$Preset-build.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
    & python -m pytest "--basetemp=build/pytest-m3-$Preset" -v 2>&1 |
        Tee-Object "results/m3-$Preset-python.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Python tests failed' }
    & ctest --preset $Preset --verbose --no-tests=error 2>&1 |
        Tee-Object "results/m3-$Preset-tests.txt"
    if ($LASTEXITCODE -ne 0) { throw 'CTest failed' }
    & "build/$Preset/nanoinfer_tests.exe" --seed=1234 2>&1 |
        Tee-Object "results/m3-$Preset-seed1234.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Alternate-seed unit tests failed' }
    Push-Location "build/$Preset"
    try {
        & .\nanoinfer_cache_tests.exe --seed=1234 2>&1 |
            Tee-Object "../../results/m3-$Preset-cache-seed1234.txt"
        if ($LASTEXITCODE -ne 0) { throw 'Alternate-seed cache tests failed' }
    } finally { Pop-Location }
    Copy-Item -LiteralPath "build/$Preset/cached-golden-report.json" `
        -Destination "results/m3-$Preset-tiny.json"
    if ($Pretrained) {
        & python -m tests.golden_cached --executable "build/$Preset/gpt2cachecheck.exe" `
            --model models/gpt2.bin --golden golden --directory "build/$Preset/cached-pretrained" `
            --jobs 3 --timeout 7200 --report "results/m3-$Preset-pretrained.json" 2>&1 |
            Tee-Object "results/m3-$Preset-pretrained.txt"
        if ($LASTEXITCODE -ne 0) { throw 'Pretrained cached comparison failed' }
    }
}
finally {
    & (Join-Path $PSScriptRoot 'sanitize-results.ps1')
    Pop-Location
}
