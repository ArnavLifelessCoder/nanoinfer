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
    if (Test-Path -LiteralPath 'models/gpt2.bin') {
        & python tests/roundtrip.py --inspector "build/$Preset/gpt2inspect.exe" `
            --model models/gpt2.bin --manifest models/gpt2.json 2>&1 |
            Tee-Object -FilePath "results/m1-$Preset-gpt2.txt"
        if ($LASTEXITCODE -ne 0) { throw 'Pretrained GPT-2 round-trip failed.' }
    }
}
finally {
    & (Join-Path $PSScriptRoot 'sanitize-results.ps1')
    Pop-Location
}
