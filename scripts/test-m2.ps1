param(
    [ValidateSet('debug', 'asan')]
    [string]$Preset = 'debug',
    [switch]$Pretrained
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'dev-shell.ps1')
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    & cmake --preset $Preset 2>&1 | Tee-Object "results/m2-$Preset-configure.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Configure failed' }
    & cmake --build --preset $Preset 2>&1 | Tee-Object "results/m2-$Preset-build.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
    & ctest --preset $Preset --verbose --no-tests=error 2>&1 |
        Tee-Object "results/m2-$Preset-tests.txt"
    if ($LASTEXITCODE -ne 0) { throw 'CTest failed' }
    Copy-Item -LiteralPath "build/$Preset/negative-report.json" `
        -Destination "results/m2-$Preset-negative.json"
    & "build/$Preset/nanoinfer_tests.exe" --seed=1234 2>&1 |
        Tee-Object "results/m2-$Preset-seed1234.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Alternate-seed unit tests failed' }
    if ($Pretrained) {
        & python -m tests.golden_forward --executable "build/$Preset/gpt2ref.exe" `
            --model models/gpt2.bin --golden golden --directory "build/$Preset/pretrained" `
            --jobs 3 --report "results/m2-$Preset-pretrained.json" 2>&1 |
            Tee-Object "results/m2-$Preset-pretrained.txt"
        if ($LASTEXITCODE -ne 0) { throw 'Pretrained golden comparison failed' }
        & python scripts/hidden_error_stats.py --golden golden `
            --captures "build/$Preset/pretrained" --output "results/m2-$Preset-hidden-stats.json"
        if ($LASTEXITCODE -ne 0) { throw 'Hidden-state statistics failed' }
    }
}
finally {
    & (Join-Path $PSScriptRoot 'sanitize-results.ps1')
    Pop-Location
}
