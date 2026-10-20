$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$fixture = Join-Path $projectRoot 'results/sanitize-regression-fixture.txt'
if (Test-Path -LiteralPath $fixture) { throw 'Regression fixture already exists.' }
try {
    [IO.File]::WriteAllText($fixture, "https://example.com/run`nhttp://example.com/run`nC:\temp\file`nD:/temp/file`n")
    & (Join-Path $projectRoot 'scripts/sanitize-results.ps1')
    $actual = [IO.File]::ReadAllText($fixture)
    $expected = "https://example.com/run`nhttp://example.com/run`n<drive>/temp\file`n<drive>/temp/file`n"
    if ($actual -cne $expected) { throw "URL/path sanitization regression: $actual" }
    & (Join-Path $projectRoot 'scripts/sanitize-results.ps1')
    if ([IO.File]::ReadAllText($fixture) -cne $expected) { throw 'Sanitization is not idempotent.' }
    Write-Output 'PASS sanitization preserves HTTP(S) URLs, redacts drive paths and is idempotent'
}
finally {
    Remove-Item -LiteralPath $fixture
}
