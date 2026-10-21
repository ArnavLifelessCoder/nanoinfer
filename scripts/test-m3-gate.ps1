# Owner-approved local M3 gate: fresh ASan build, full suite under idle-process
# capture, deliberate memory-error probes, and Debug CTest. Any stall must be
# classified from its dump; retries are not evidence.
param([Parameter(Mandatory)][string]$Label)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'dev-shell.ps1')
Push-Location $projectRoot
try {
    if ($Label -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Label must be letters, digits, _ or -.' }
    $asan = "build/asan-$Label"
    if (Test-Path -LiteralPath $asan) { throw "Use a new label; $asan already exists." }
    $googletest = "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$projectRoot/build/debug/_deps/googletest-src"
    & cmake --preset asan -B $asan $googletest *> "results/m3-gate-$Label-asan-configure.txt"
    if ($LASTEXITCODE -ne 0) { throw 'ASan configure failed' }
    & cmake --build $asan *> "results/m3-gate-$Label-asan-build.txt"
    if ($LASTEXITCODE -ne 0) { throw 'ASan build failed' }
    & (Join-Path $PSScriptRoot 'diagnose-asan.ps1') -BuildDirectory $asan -Label "m3-gate-$Label"
    $asanExit = $LASTEXITCODE
    Copy-Item "build/diagnostics/m3-gate-$Label-ctest.txt" "results/m3-gate-$Label-asan-tests.txt"
    $capture = Get-Content -Raw "build/diagnostics/m3-gate-$Label-report.json" | ConvertFrom-Json

    $probe = "build/diagnostics/error-probe-$Label"
    & cmake -S tests/diagnostics/asan_exit -B $probe -G Ninja -DCMAKE_BUILD_TYPE=Debug `
        "-DPython3_EXECUTABLE=$projectRoot/.venv/Scripts/python.exe" *> "$probe-configure.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Probe configure failed' }
    & cmake --build $probe --target asan_error_probe *> "$probe-build.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Probe build failed' }
    & python tests/diagnostics/asan_exit/repro.py --check-errors `
        --executable "$probe/asan_error_probe.exe" --directory "$probe/evidence" `
        --procdump build/diagnostics/tools/procdump/procdump64.exe
    $probeExit = $LASTEXITCODE
    Copy-Item "$probe/evidence/report.json" "results/m3-gate-$Label-error-probes.json"

    & cmake --build --preset debug *> "results/m3-gate-$Label-debug-build.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Debug build failed' }
    & ctest --preset debug -j 1 --timeout 600 --no-tests=error *> "results/m3-gate-$Label-debug-tests.txt"
    $debugExit = $LASTEXITCODE

    $summary = [ordered]@{
        label = $Label
        git_head = (git rev-parse HEAD)
        asan_ctest_exit = $asanExit
        asan_idle_captures = @($capture.captures)
        error_probes_exit = $probeExit
        debug_ctest_exit = $debugExit
        gate_passed = ($asanExit -eq 0 -and $probeExit -eq 0 -and $debugExit -eq 0)
        note = 'A nonzero ASan exit passes only after its dump matches the documented shutdown signature.'
    }
    $summary | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 "results/m3-gate-$Label.json"
    Write-Output "ASan CTest=$asanExit probes=$probeExit Debug CTest=$debugExit captures=$(@($capture.captures).Count)"
}
finally {
    & (Join-Path $PSScriptRoot 'sanitize-results.ps1')
    # The sanitizer handles .txt logs; redact this gate's JSON reports the same way.
    foreach ($report in Get-ChildItem -LiteralPath results -Filter "m3-gate-$Label*.json") {
        $text = [IO.File]::ReadAllText($report.FullName)
        foreach ($pair in @(@($projectRoot, '<project>'), @($env:USERPROFILE, '<user>'))) {
            foreach ($form in @($pair[0], $pair[0].Replace('\', '\\'), $pair[0].Replace('\', '/'))) {
                $text = $text.Replace($form, $pair[1])
            }
        }
        [IO.File]::WriteAllText($report.FullName, $text)
    }
    Pop-Location
}
