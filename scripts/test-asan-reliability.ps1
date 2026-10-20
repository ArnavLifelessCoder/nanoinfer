param(
    [string]$BuildDirectory = 'build/asan-m3-reliability',
    [string]$Label = 'm3-reliability'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'dev-shell.ps1')
Push-Location $projectRoot
$previousOptions = $env:ASAN_OPTIONS
try {
    if (Test-Path -LiteralPath $BuildDirectory) { throw 'A fresh build directory is required.' }
    # Owner-approved local trial. Do not put these options in CI or persistent settings.
    $env:ASAN_OPTIONS = 'windows_hook_legacy_allocators=false:symbolize=false'
    & cmake --preset asan -B $BuildDirectory `
        "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$projectRoot/build/debug/_deps/googletest-src" `
        *> "build/diagnostics/$Label-configure.txt"
    if ($LASTEXITCODE) { throw 'Configure failed.' }
    & cmake --build $BuildDirectory *> "build/diagnostics/$Label-build.txt"
    if ($LASTEXITCODE) { throw 'Build failed.' }
    $runs = @()
    foreach ($index in 1..5) {
        $runLabel = "$Label-$index"
        & (Get-Process -Id $PID).Path -NoProfile -File `
            (Join-Path $PSScriptRoot 'diagnose-asan.ps1') `
            -BuildDirectory $BuildDirectory -Label $runLabel
        $code = $LASTEXITCODE
        $report = Get-Content -Raw "build/diagnostics/$runLabel-report.json" | ConvertFrom-Json
        $runs += $report
        $summary = [ordered]@{
            asan_options = $env:ASAN_OPTIONS
            fresh_build = $BuildDirectory
            requested_suites = 5
            completed_suites = $runs.Count
            passed = ($runs.Count -eq 5 -and $code -eq 0 -and
                @($runs | Where-Object { $_.ctest_exit_code -ne 0 -or $_.captures.Count -ne 0 }).Count -eq 0)
            runs = $runs
        }
        $summary | ConvertTo-Json -Depth 8 |
            Set-Content -Encoding utf8 "build/diagnostics/$Label-summary.json"
        if ($code -ne 0 -or $report.captures.Count -ne 0) {
            throw "Stopped at suite $index. Preserve the failure and investigate; do not retry."
        }
    }
}
finally {
    $env:ASAN_OPTIONS = $previousOptions
    Pop-Location
}
